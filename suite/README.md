# Evaluation suite

Query execution, SQL checks, and baseline experiments. `queries.py` defines a
24-query development workload; the paper's six-query comparison uses the named
files in `queries/` and their hand-written SQL in `sql-ref/`.

## Setup

Follow the root README for [building](../README.md#building-and-preparing-inputs)
and the [converter README](../converters/README.md) for raw-data preparation.
Run commands below from the repository root:

```bash
source .trqs-pyenv/bin/activate
export TRQS_DATA_DIR="$PWD/datasets/processed"
export TRQS_RAW_DIR="$PWD/datasets/raw"
export TRQS_OUTPUT_DIR="$PWD/results/suite"
mkdir -p "$TRQS_OUTPUT_DIR"
```

Build each selected sub-dataset with `--max-gap 10 --max-error-px 0 dataset build`;
the suite drivers discover `10_0.trql` artifacts. Paths default to the values above.

SQL checks and head-to-head timing require the **DuckDB CLI 1.5.5** on `PATH`.
Download the `duckdb_cli` archive for your platform from
[DuckDB releases](https://github.com/duckdb/duckdb/releases), extract it, and
check `duckdb --version`. The Python DuckDB package is also needed for metadata
lookups and other scripts; it does not install the executable.

## Run queries

```bash
python suite/run_suite.py --dataset soccernet
python suite/run_suite.py --query SN-Q2 --limit 20
python suite/run_suite.py --bench --results-db results/suite/results.db --run-repeat 5
```

Omit filters to run all available sub-datasets. Default mode reports match counts
and query times. `--bench` loads each database once, repeats the query, and records
peak memory. `--db-repeat` is the dataset replication factor. Both modes write
`results/suite/results.json`, including per-sub-dataset results and failures.

The driver does not supply reference SQL or ground truth. Use the root README's
[paired benchmark command](../README.md#re-running-the-measurements) for those
comparisons. Unscored precision/recall fields are not measured accuracy.

| SQLite output | CLI flags | Contents and consumer |
|---|---|---|
| `results.db` | `--results-db <path>` | Per-system timing, memory, settings, and optional ground-truth scores; evaluator-compatible `systems`, `queries`, `runs` tables |
| `bench.db` | `--sql <reference.sql> --bench-db <path>` | Paired TrQS/DuckDB timings, answer agreement, dataset statistics, and optional ground-truth scores; plotter's `Dataset`, `Query`, `Run` tables |

Both flags can be supplied together. Filenames are conventions; the flags select
the schema. Plotting recorded or new measurements is documented in the
[plotter README](../plotter/README.md#recorded-inputs-and-new-measurements).

## Check or add a query

Add a `Query` to `queries.py`, then check it across its dataset family:

```bash
python suite/probe.py <dataset> <sub-dataset> "<dsl>"
python suite/run_suite.py --query <ID>
python suite/compile_sql.py --query <ID> --check --strict
```

`compile_sql.py` builds the `trql-sql` compilation example on demand and writes
SQL to `results/suite/sql/`. `--check` compares object tuples and interval spans;
`--strict` returns a failure status on disagreement. Without `--strict`, differences
are printed but do not fail the command. Gap smoothing and coordinate rounding
can cause differences from frame-expanded SQL. `head_to_head.py` times the engine
against its compiled SQL. Both scripts set DuckDB to one thread.

`CL-Q2` and `CL-Q3` have unresolved compiled-SQL span differences and should not
be treated as validated correctness baselines.

Query authoring conventions:

- Use `frames.Frame` to convert natural coordinates into stored units using dataset metadata.
- Declare `requires=[...]` for constrained classes: an absent class otherwise causes a parse error.
- Object distinctness is enforced within an appearance group. Slots in different groups across a top-level `THEN` can bind the same object.

| Helper | Purpose |
|---|---|
| `frames.py` | Coordinate and time conversion |
| `scenes.py` | Class sets and generated scene regions |
| `slice_i24.py` | Fixed-length I24 slices |
| `gen_workload_tex.py` | LaTeX workload descriptions and DSL listings |

Generate the optional workload tables with:

```bash
mkdir -p results/suite/tables
python suite/gen_workload_tex.py --out results/suite/tables/workload.tex
python suite/gen_workload_tex.py --listing --out results/suite/tables/workload-full.tex
```

## Deduplicated DuckDB baseline

This compares frame-expanded boxes with an interval encoding that removes
consecutive repeated boxes. `build_dedup.py` checks that the intervals cover
all source rows exactly once; `bench_dedup.py` requires identical query answers
before recording timings.

```bash
uv run suite/build_detrac_px.py
uv run suite/build_dedup.py
uv run suite/bench_dedup.py --repeat 10
```

The inputs are VIRAT `S_010109_04` and DETRAC `MVI_40191`. Both use the meet
query: two cars separate by more than 200 pixels, then approach within 70 pixels.
DETRAC uses integer-truncated coordinates so SQL and the engine see the same
boxes. `--repeat 10` is the tiling factor; `--runs` controls timed repeats.
The report is `results/suite/results-dedup.json`.

## Rekall comparison

`bench_rekall.py` compares Rekall, DuckDB, and TrQS on the cyclist-detection
tutorial query: a person's box intersects a bicycle's box.

```bash
uv run suite/build_rekall_cydet.py
uv run suite/build_rekall_cydet.py --track
uv run suite/bench_rekall.py --verify-encoding
uv run suite/bench_rekall.py --tile 1 2 5 10 25 50 100 --errs 0 1 2 4
uv run suite/bench_rekall.py --report
```

The builder downloads released detections into `datasets/raw/rekall_cydet/`.
All classes are retained and coordinates are truncated to integers. The `raw`
variant has one object per detection; `tracked` adds identity through per-class
IoU association (threshold 0.3, two missed frames allowed). All three systems
receive the same rows for each variant.

Rekall is measured both as one video-wide interval set (`whole`) and keyed by
frame (`frame`). Use `frame` as the stronger baseline. The benchmark checks
that both decompositions agree with SQL and, at zero fitting error, that TrQS
matches Rekall's answer. Higher error budgets measure approximation.
`--verify-encoding` compares frame/pixel and tutorial coordinate conventions;
`--verbatim-merge` measures the tutorial's payload-dropping merge separately.

Outputs are `results/suite/results-rekall.json` and `results/suite/bench.db`.
Python dependencies, including the pinned Rekall revision, are declared in the
script. See [vlm/README.md](vlm/README.md) for the VLM comparison.
