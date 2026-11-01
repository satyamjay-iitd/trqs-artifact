# Evaluation suite

The 24-query workload for the evaluation section, and everything needed to run,
check, and typeset it. `../query_suite_plan.md` is the write-up — findings,
measurements, and what is left; this file is just how to use the code.

## Running

Everything goes through `run_suite.py`, which loops the sub-datasets `trql-cli`
can only take one at a time (106 soccernet clips, 850 nuscenes scenes, 24 i24
slices) and reports the distribution.

```bash
./run_suite.py                                  # everything, run mode
./run_suite.py --dataset soccernet              # one family
./run_suite.py --query SN-Q2 CL-Q3              # named queries
./run_suite.py --limit 20                       # sample 20 sub-datasets per query
./run_suite.py --bench --results-db ../../plotter/dbs/vldb27.db --run-repeat 5
```

*run* mode (default) executes each query once per sub-dataset and reports match
counts and query time — the cheap pass, for checking a query still matches after
an edit. `--bench` goes through `trql-cli bench`, which reloads the database per
repeat and samples peak RSS, and appends every run to the results db the
`papers/plotter/` scripts read.

Needs `cargo build --release -p trql-cli` first, and the datasets venv:
`datasets/.venv/bin/python`.

## Files

| file | what it is |
|---|---|
| `queries.py` | the 24 queries — arity, operators, prose, and DSL in one place |
| `run_suite.py` | the driver |
| `probe.py` | authoring aid: run one query on one sub-dataset, print the count |
| `compile_sql.py` | compile each query to DuckDB SQL; `--check` diffs it against the engine |
| `head_to_head.py` | time \sys against its own compiled SQL, one core each |
| `frames.py` | natural units → stored units, read from each sub-dataset's metadata |
| `scenes.py` | per-scene class sets and generated regions (nuScenes) |
| `slice_i24.py` | cut i24's 4-hour sub-datasets into fixed-length slices |
| `gen_workload_tex.py` | generate `../tables/workload.tex` and `workload-full.tex` |
| `sql/` | generated `<ID>.sql` listings, the paper's appendix material |

## Benchmarking

`trql-cli` is single-threaded, so DuckDB gets `SET threads=1` — `compile_sql.py` and
`head_to_head.py` both issue it. A DuckDB left on its default takes one thread per core
(27–29 on this machine) and measures parallelism rather than the two evaluation
strategies; report that separately and label it. \sys is always the release build.

Check that a DuckDB run actually *finished*: on the largest i24 queries it is OOM-killed
at ~101 GB, and a timing line alone does not distinguish that from success.

## Three things that will bite you

**Write regions in natural units.** i24 stores centi-feet shifted non-negative,
so eastbound lane 1 is `[0, 21100, 2400000, 22200]`. Do not type that. Use
`frames.Frame.of("i24_motion", "post1").rect(309000, 11, 333000, 22)` and let it
read the dataset's own recorded frame. Pixel datasets get the identity frame, so
the same code works everywhere.

**An absent class is a parse error, not an empty result.** The engine rejects an
unknown enum value outright, so a query constraining on `vehicle.truck` *fails*
on the 151 nuScenes scenes without one. Declare `requires=[…]` on the query and
the driver skips those sub-datasets.

**On a concatenated dataset, every slot must appear in the first appearance group.**
`clevrer_merged` is 10,000 independent videos as 128-frame fid blocks. Co-appearance keeps
a single group inside one video, but a slot introduced *only* after a top-level `THEN` sits
in a second group that never has to co-appear with the first -- so it can bind an object
from the next video. CL-Q6 does this and 51 % of its matches are cross-video. Bounding the
`THEN` gap below 128 does not help: any positive gap crosses near a block boundary.

**Two slots can bind the same object across a `THEN`.** Within one appearance
group the selector rejects a repeated object as a self-join, so `CONTAINS`,
`OVERLAP` and `DURATION` give distinctness for free. A top-level `THEN` puts its
operands in *different* groups, where the same object may fill both slots. If a
query needs two distinct same-kind objects either side of a `THEN`, give them
different classes — there is no variable-inequality constraint in the language.

## Adding a query

Append a `Query` to the right list in `queries.py`, then:

```bash
./probe.py <dataset> <sub-dataset> "<dsl>"      # does it parse, does it match
./run_suite.py --query <ID>                     # does it match across the family
./compile_sql.py --query <ID> --check           # does the SQL agree
./gen_workload_tex.py && ./gen_workload_tex.py --listing
```

Check the match count across the *family*, not one sub-dataset. Three of the
original 24 matched nowhere and had to be redesigned; each looked fine on the
clip it was written against.

## The deduplicated DuckDB baseline

The frame-expanded encoding writes one row per object per frame however the
object behaves, and the obvious relational answer to that is to stop writing
rows that repeat the previous frame. `build_dedup.py` builds that copy of the
two clips `plots/compression.png` draws: a row survives only where the box moved
or the object had been absent, and carries `e_fid`, the last frame through which
it holds. Every surviving row is a real detection of the source at its own
frame, and the build asserts that the surviving intervals cover every source row
exactly once, so nothing is lost.

`sql-dedup/<ID>.sql` is `sql-ref/<ID>.sql` over that copy: co-appearance becomes
an interval overlap instead of a join on `fid`, phases are coalesced before the
DURATION minimum rather than grouped by consecutive frame, and nothing else
changes. `bench_dedup.py` runs both at `threads=1`, tiled the way `trql-cli
--db-repeat` tiles, and requires the two answers to be identical before
reporting a time — they are, on both clips.

```bash
uv run build_detrac_px.py                 # integer-pixel DETRAC (see below)
uv run build_dedup.py
uv run bench_dedup.py --repeat 10         # writes results-dedup.json
```

`papers/plotter` draws the result as the second bar of each group in
`plots/compression.png` (`baselines = "../vldb27/suite/results-dedup.json"` in
its manifest).

## Why the figure pairs VIRAT with DETRAC

The two clips are the ends of the redundancy scale. In the VIRAT parking lot
`S_010109_04`, 98 % of frames repeat the previous frame's box, so deduplication
removes 47.6 of every 48 rows; in the DETRAC motorway junction `MVI_40191`,
0.3 % do, and it removes nothing. All 315 VIRAT clips were measured looking for
a busier one (`scan_virat_motion.py`): none drops below 58 % repeated frames,
and the ones that come closest are pedestrian scenes with two or three cars,
where the query matches nothing. The contrast had to come from another dataset.

Both clips run the same query, `queries/{VIRAT,DETRAC}-meet.trql` — two vehicles
more than 200 px apart, then within 70 px — because DETRAC annotates vehicles
only, so a query naming a person cannot run on it. Both slots are `class=car`,
which both datasets have.

**Why 70 px and not 80.** At an 80 px near threshold one DETRAC answer rests on
two exact ties at once: its near phase runs exactly 30 frames against a minimum
of 30, and that run's first frame has `|dx|` of exactly 80.0 against a `<= 80`
test. Any budget above zero drops that frame, the run falls one short, and the
pair is lost — one answer in 26, which on an answer set that small reads as
recall 0.962 from eps=1 onwards. It is the dataset's knife edge, not the
engine's fit. At 70 px no answer sits on a tie and the engine is exact through
eps=16 on the parking lot and eps=4 on the motorway. The sweep behind the choice
is in the session notes: thresholds of 60, 70, 75, 78, 82, 85, 88 and 92 px were
measured, and 70 is the one with no boundary-degenerate answer on either clip.

**DETRAC is read from an integer-pixel copy.** UA-DETRAC annotates boxes as
floats; `trql-storage` truncates each coordinate on ingest (`convert.rs:197`)
and evaluates `cx = (x1 + x2) / 2.0` over the result. A reference query reading
the original doubles therefore asks a question of data the engine never sees,
and on `MVI_40191` that costs exactly one pair in fifty at ε=0 — a near phase of
29 frames against a minimum of 30. `build_detrac_px.py` applies the same
truncation to the source once, which puts all three encodings on the same
numbers; with it the engine and the reference SQL agree exactly at ε=0.

## Rekall head-to-head

Rekall is the closest published system to this paper's query model — labels with
spatial and temporal extent, composed with joins under Allen-style predicates —
and its own evaluation reports no query-execution cost: "the Rekall
implementation contains minimal optimization … final queries ran in less than
thirty seconds". `bench_rekall.py` measures that cost, on Rekall's own tutorial
query and its own released detections, beside the two encodings the rest of the
suite already compares.

```bash
cargo build --release -p trql-cli              # default features; the ablation leaves others
uv run build_rekall_cydet.py                   # raw: their data as published
uv run build_rekall_cydet.py --track           # tracked: the same boxes, with identity
uv run bench_rekall.py --verify-encoding       # one-time fidelity check
uv run bench_rekall.py --tile 1 2 5 10 25 50 100 --errs 0 1 2 4
uv run bench_rekall.py --report                # re-print results-rekall.json
```

`build_rekall_cydet.py` downloads the Mask R-CNN detections behind tutorial 01
("Cyclist Detection") into `datasets/raw/rekall_cydet/` and writes them as a
processed dataset. Two videos, 6,695 frames each at 1242×375 and 10 fps; 52,904
detections on `0001`, which is the one measured. Coordinates are truncated to
integers on write, for the `build_detrac_px.py` reason: the engine casts with
`as u32` on ingest, so storing the source doubles would have the three systems
answering slightly different questions. Every detected class is kept, not just
the two the query names, because the tutorial filters by class *inside* the
query — so the filter is part of what each system is timed on, and pre-filtering
would remove that work from all three and shrink the input ninefold.

**The query** is `queries/CYDET-cyclist.trql`: a person's box intersects a
bicycle's, no duration constraint. `sql-ref/CYDET-cyclist.sql` derives the same
thing relationally and explains why Rekall's `overlaps()` reduces to a strict
interval overlap on this data (the build asserts there are no degenerate boxes,
which is what the reduction needs).

**Identity is ours, and is reported both ways.** Their release is per-frame
detections with no track ids, which is all Rekall needs — it never uses
identity. TrQS fits segments per object, so without ids every trajectory is one
frame long and there is nothing to compress. `raw` is therefore TrQS's floor and
`tracked` the like-for-like comparison; both feed all three systems identical
rows. The tracker is a greedy per-class IoU association (≥ 0.3, closed after two
missed frames) costing 180 ms on 52,904 boxes, and 0.3 is the conservative end
of its range: looser thresholds give longer trajectories and a better TrQS
number. On `tracked` it yields 5.28 boxes per object.

**Two Rekall configurations, because one of them measures the wrong thing.**
`IntervalSet.join` copies a list slice of the other set per outer interval
(`interval_set.py:236`), which is quadratic in set size and has nothing to do
with the query. `whole` is the tutorial's own shape, one `IntervalSet` for the
video; `frame` is an `IntervalSetMapping` keyed by frame, which makes each join
see one frame's boxes — same operators, same predicate, same `window=0.0`, and
keying a mapping by domain is Rekall's own idiom. The two answers must agree and
the script asserts it. **`frame` is the number to quote.** `window=0.0` is
passed on every call, as the tutorial does; left to default it would be 1/100 of
the temporal span and the `whole` figure would be measuring that instead.

### What it found

Query time on `tracked`, at ε = 0, best of three (ms):

| boxes | Rekall `whole` | Rekall `frame` | DuckDB | TrQS |
|---:|---:|---:|---:|---:|
| 52,904 | 27.1 | 22.6 | 5.2 | 5.4 |
| 529,040 | 728.1 | 231.0 | 30.7 | 5.4 |
| 2,645,200 | 23,332.4 | 1,370.3 | 62.1 | 17.5 |
| 5,290,400 | 102,161.8 | 2,618.2 | 124.6 | 36.2 |

The finding is the slopes, not the top row. Over 100× more data the tutorial's
own shape costs 3,774× more (quadratic), the frame-keyed one 116× (linear),
DuckDB 24× and TrQS 6.7×. At the largest size TrQS is 72× faster than the
strongest Rekall configuration, 2,826× faster than the tutorial's, and 3.4×
faster than the frame-expanded SQL. At the smallest it is level with DuckDB:
5.4 ms is the engine's fixed per-query cost, which is most of a 52,904-box
query. On `raw` — their data untouched, one segment per detection — TrQS still
wins, by 24.5× over `frame` and 1.7× over DuckDB, which is the honest floor.

Three things reported but never summed into a query time, because they sit in
different places in a system's lifecycle: Rekall's `IntervalSetMapping` build
(1.6 s and 462 MB of live Python objects at ×100, rebuilt every session, with no
on-disk form), TrQS's `dataset build` (a compile step producing a reloadable
`.trql`), and DuckDB's, which is nothing.

### What was checked before any timing was believed

Every one of the 56 (variant, scale, ε) points passes three gates, and the
script refuses to record a timing otherwise:

1. Rekall's two decompositions return the same answer, so the frame keying is
   semantics-preserving rather than a different query.
2. Rekall's answer equals the relational one, as `(person, bicycle, frame)`
   triples — the finest projection the three share, insensitive to each one's
   interval-merging convention.
3. At ε = 0 the engine reproduces *Rekall's* answer exactly. The Rekall output
   is written as the ground-truth CSV `bench --gt` scores against, in engine
   index space, and precision, recall and mean IoU must all be 1.0 with zero
   interval mismatches. They are, at every scale, on both variants.

Two further checks, run once each:

- **`--verify-encoding`.** The benchmark gives Rekall frame-index time and pixel
  coordinates where the tutorial uses seconds and width-normalised bounds. Every
  predicate here compares two coordinates of the same kind, so the rescaling
  should not matter — and the check confirms it, 1,650 identical (pair, frame)
  rows under both encodings. It also removes the query's only float-equality
  test and puts Rekall on exactly the integers the engine stores.
- **`--verbatim-merge`.** The one departure from the tutorial's code is
  `merge_op`, which must carry the pair of object ids or the answers cannot be
  compared at all. Timing the tutorial's payload-dropping version alongside
  shows no measurable difference at any scale — 2,618 ms against 2,348 ms at
  ×100, inside the run-to-run spread.

Past ε = 0 the fit is lossy by construction and the budgets are a measurement
rather than a gate. On this query it barely moves: at ×100 on `tracked`, ε = 4
drops the segment count from 2.31 M to 1.81 M for a mean IoU of 0.9997 and no
change in query time, because box-overlap is a coarse enough predicate that a
four-pixel fit error rarely flips it.
