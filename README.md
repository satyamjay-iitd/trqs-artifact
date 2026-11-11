# TrQS artifact

Source and evaluation harness for the TrQS spatio-temporal query engine.
**Datasets are not included.** Obtain them separately and follow their licences;
use the supplied converters to prepare inputs.

## Layout

| path | what |
|---|---|
| `engine/` | the Rust workspace: algebra, DSL, polynomial fitting, selector, storage, CLI |
| `suite/` | the evaluation workload: TrQL queries, hand-written reference SQL, drivers |
| `suite/sql-ref/` | reference SQL per query, written against the DSL's denotation, not generated from the engine -- this is what the exactness gate compares against |
| `suite/vlm/` | the Gemini comparison: harness, prompts, raw responses, scoring, and the built `ytstream/uav_0006` artifact it queries |
| `plotter/` | plot renderers, recorded auxiliary measurements, and the preserved MIRIS PDF |
| `scripts/detect_track.py` | the ingest worker: detector + tracker -> the `objects`/`bbox` schema |
| `converters/` | 20 dataset converters, shared helpers, and projection dependencies; see its README for coverage |
| `lean-proof/` | interval-set algebra proofs and pinned Lean/mathlib dependencies |
| `bench.db` | the measurements behind the manifest's plots (SQLite): the paper's figures and the supplementary DETRAC recall plots, filtered to the queries and error budgets they read |

## Building and preparing inputs

The engine requires Rust/Cargo 1.92+, a C/C++ compiler toolchain, and the
**DuckDB 1.5.5 shared library** (`libduckdb.so` on Linux). DuckDB is not bundled;
install the matching library before building. It is the `libduckdb-<platform>.zip`
asset of the [v1.5.5 release](https://github.com/duckdb/duckdb/releases/tag/v1.5.5)
(`libduckdb-linux-amd64.zip` on x86-64 Linux). For a library outside the system
search paths, set `DUCKDB_LIB_DIR` to its directory when building and add that
directory to `LD_LIBRARY_PATH` when running the binaries on Linux.
Python 3.13+, uv, and FFmpeg (`ffmpeg` and `ffprobe`) are used by the converters,
suite, and plotter.

Run from the repository root:

```sh
cargo build --locked --release --manifest-path engine/Cargo.toml \
  -p trql-cli -p trql-tui -p trql-server
python3 -m venv .trqs-pyenv
.trqs-pyenv/bin/python -m pip install -r converters/requirements.txt
```

The binaries are in `engine/target/release/`. Follow the
[converter README](converters/README.md) for supported datasets, raw input
layouts, and conversion commands. Each processed sub-dataset contains
`duckdb.db`, `metadata.json`, and a video when available. SoccerNet is
distinct from SoccerTrack v2, which `converters/soccertrack_convert.py` converts.
The merged sub-datasets the experiments use are built with
`converters/paper_merges.sh`.

## Reproducing the figures from recorded measurements

The root `bench.db` and `plotter/data/` contain the measurements needed by the
figures. No datasets or API keys are needed to render them:

```sh
cd plotter
uv run --locked benchplot render
```

Outputs go to `plotter/plots/`, Figure 9 included.

Figure 5 labels its six queries Q1–Q6:

| Label | Query | Dataset |
|---|---|---|
| Q1 | `I24-journey4` | I-24 MOTION |
| Q2 | `I24-convoy` | I-24 MOTION |
| Q3 | `CL-overtake` | CLEVRER |
| Q4 | `NS-crossing` | nuScenes |
| Q5 | `ST-swap-img` (`suite/queries/ST-swap.trql`) | SoccerTrack v2 |
| Q6 | `SD-carpass` | Stanford Drone |

The VLM comparison's Q1 and Q2 are its own two queries; see its
[README](suite/vlm/README.md#queries-and-scoring). See the [plotter README](plotter/README.md) for
individual figures and new measurements. The MIRIS figure is a preserved PDF;
its generator and execution repository are not included.

## Re-running the measurements

### One command

`reproduce_paper.sh` re-measures every point that Figures 4a, 5 and 8 draw from
`bench.db`, re-measures the Figure 9 ablations, then redraws those figures from
the new measurements:

```sh
./reproduce_paper.sh --dry-run          # the plan and the exact commands; runs nothing
./reproduce_paper.sh                    # 4a, 5, 8, 9a, 9b
./reproduce_paper.sh --plots compression --data-dir /path/to/processed
./reproduce_paper.sh --plots "degree_ablation interval_ablation"   # Figure 9 only
```

The plan comes from `plotter/manifest.toml`, applied to the recorded `bench.db`.
That database is opened read-only, so the plan holds exactly the (query,
sub-dataset, budget, gap, scale) points the figures draw: 39 runs for the three
figures. The I24-Motion queries were measured at `--max-gap 10`, the others at
`--max-gap 1`. Each run uses the query in `suite/queries/` and the reference SQL
in `suite/sql-ref/`, matched by content to what was recorded.

Outputs go to `results/repro/`: `bench.db`, `plots/`, `logs/` and `plan.tsv`.
**The recorded `bench.db` is never written.** The script refuses an output that
resolves to it and checks its SHA-256 before and after every run. Re-running the
script resumes: points already in `results/repro/bench.db` are skipped
(`--no-resume` re-measures them). Points whose sub-dataset is absent from
`--data-dir` (default `$TRQS_DATA_DIR` or `datasets/processed`) are skipped and
listed. A figure is drawn only when every point it needs has been measured,
except Figure 5: a query whose dataset is absent is drawn as a marked
"not measured" row, and the summary names it. The engine is built if
`engine/target/release/trql-cli` is absent.

Figure 9 compares engine builds rather than budgets, so its two manifest entries
(`degree_ablation`, `interval_ablation`) carry their own matrix: the query,
sub-dataset, budgets and one cargo feature set per variant. The script builds
each feature set once into `results/repro/ablation/bin/`. Each degree gets its own
data directory, because a `.trql` file's name does not record the polynomial
degree. A degree's instrumented build reuses that directory's fitted files,
which are byte-identical (the manifest's `fit` key). The figures are drawn
from `results/repro/ablation/<plot>/`.

| Figure | Variants (cargo features of `trql-cli`) | Runs |
|---|---|---|
| 9a degree | `linear`, default (quadratic), `cubic`, each again with `instrument` | I24-convoy on `i24_motion/post1`, gap 10, budgets 30–487, gated against the reference SQL |
| 9b interval sets | `instrument`; `instrument,naive-intervals` | CL-overtake on `clevrer_merged/video_00000+99more` (`converters/paper_merges.sh --only clevrer-100`), gap 1, budget 0 |

#### Time and resources

Measured on one machine: Intel i9-14900K (32 threads), 123 GB RAM, NVMe, with
the converted datasets already in place. Treat the times as an order of
magnitude for your hardware.

| Stage | First run (fits every `.trql`) | Re-run (`.trql` kept) |
|---|---|---|
| Engine release build | 35 s | — |
| Figures 4a, 5, 8: 39 runs and drawing | 25 min | 4 min |
| — of which I24-convoy (8 budgets) | 19 min | 1 min |
| Figure 9: 7 engine builds, 32 runs | about 42 min | — |
| — of which 9a, 30 runs, 15 fits | about 40 min | — |
| — of which 9b, 2 runs, 1 fit | about 2 min | — |
| Everything (`./reproduce_paper.sh`) | about 1 h 10 min | — |

The Figure 9 times are derived from a measured run in which every build fitted
its own files: 9a took 70 min, about 2 min per point, almost all of it fitting
(the timed query work was 11 min in total). Builds that share fitted files skip
15 of those 30 fits.

Fitting I24-Motion `post1` dominates: 830k objects and 280 M boxes, refitted
for each budget, and for Figure 9a once per budget and degree. Each 9a
run also times DuckDB three times, about 7 s each. Interrupted runs resume from
the last finished point.

Memory: DuckDB peaks at about 1.8 GB (I24-convoy) and TrQS at about 1.1 GB
(SoccerTrack); the naive interval build of Figure 9b needs about 4.5 GB. Disk:
the eight converted sub-datasets the figures read take about 6.6 GB (6.2 GB of
it I24), and the fitted `.trql` files add about 3 GB. Figure 9a adds 1.5–2.2 GB
of fitted files per degree, about 5.3 GB for all three, under
`results/repro/ablation/`. Converting I24-Motion `post1` from its 19 GB JSON
takes about 10 min, 7.6 GB of RAM and about 14 GB of temporary space.

Figure 4a's deduplicated-DuckDB bars still come from the recorded
`plotter/data/results-dedup.json`. Re-measure them with the
[deduplicated baseline](suite/README.md#deduplicated-duckdb-baseline).

### Individual runs

Needs the datasets. Each is converted into a common schema --
`objects(oid, class)` and `bbox(oid, fid, x1, y1, x2, y2)` -- then fitted into a
`<max_gap>_<max_error_px>.trql` artifact:

```sh
mkdir -p results/suite
engine/target/release/trql-cli --dataset <family> --sub-dataset <name> \
         --max-gap 1 --max-error-px 0 dataset build
engine/target/release/trql-cli --dataset <family> --sub-dataset <name> \
         --max-gap 1 --max-error-px 0 bench <id> "<TrQL query>" 1 3 \
         --sql suite/sql-ref/<id>.sql --bench-db results/suite/bench.db
```

Always pass `--bench-db` for new measurements: the CLI otherwise defaults to
the supplied root `bench.db`. The commands above preserve that database and
`plotter/data/`; plotting new measurements requires selecting their paths as
described in the [plotter README](plotter/README.md#recorded-inputs-and-new-measurements).

The final `1 3` means dataset replication factor 1 and three query executions.
`--sql` runs the hand-written reference in DuckDB on the same input and reports
tuple and interval agreement. At `--max-gap 1 --max-error-px 0`, a disagreement
causes an error; higher gaps or error budgets can intentionally change answers.
`--gt answers.csv` additionally scores query precision, recall, F1, and temporal
IoU. The CSV must use object IDs aligned to the engine's dense indices.

See [suite/README.md](suite/README.md) for the development workload and baselines,
[suite/vlm/README.md](suite/vlm/README.md) for the recorded Gemini comparison,
and [lean-proof/README.md](lean-proof/README.md) for checking the proofs.

## Running detection and tracking

The annotation-based benchmarks above do not need a detector rerun. To generate
predictions, first prepare a sub-dataset containing a source `video.mp4` and its
ground-truth `duckdb.db`. Create a separate inference environment:

```sh
python3 -m venv .venv-detect
.venv-detect/bin/python -m pip install -r scripts/requirements-detect.txt
.venv-detect/bin/python -c 'import torch; print(torch.cuda.is_available())'
```

The supplied presets use FP16 and are intended for a compatible NVIDIA GPU and
PyTorch installation. Model weights are downloaded on first use. List the
presets, then run one prepared clip. This uses the recorded DETRAC configuration:
YOLO11x, BoT-SORT, confidence 0.25, image size 960, FP16, COCO classes 2/5/7.

```sh
DATA_ROOT=datasets/processed
FAMILY=detrac
CLIP=MVI_40191
PRESET=yolo11l-bytetrack-traffic

engine/target/release/trql-cli --data-dir "$DATA_ROOT" \
  --dataset "$FAMILY" detect configs
engine/target/release/trql-cli --data-dir "$DATA_ROOT" \
  --dataset "$FAMILY" detect run --sub "$CLIP" --config "$PRESET" \
  --model yolo11x.pt --tracker botsort.yaml --imgsz 960 --classes 2 5 7 \
  --python .venv-detect/bin/python --script scripts/detect_track.py
```

This calls `scripts/detect_track.py`, aligns predicted track IDs to the source
annotations, and writes box-level precision, recall, and F1 to `metrics.json`.
These localization scores are distinct from query-result scores. Outputs live
under `$DATA_ROOT/predicted/$FAMILY/$CLIP/<reported-config-id>/`: `duckdb.db`, video link,
metadata, `detect.log`, `alignment.json`, and `metrics.json`. Failed jobs are
reported per clip; inspect their errors rather than relying on the batch exit
status alone. Existing predictions are skipped unless `--force` is supplied.

Overrides become part of the output configuration name, which the
command reports. Use `--class-alias person=pedestrian` when matching that label
convention. Batch selection uses repeated `--sub`, `--from-file`, or `--all`.

Build the resulting predictions with:

```sh
PREDICTED_CONFIG=yolo11l-bytetrack-traffic@myolo11x-tbotsort-sz960-cls2.5.7
engine/target/release/trql-cli --data-dir "$DATA_ROOT" --dataset predicted \
  --sub-dataset "$FAMILY/$CLIP/$PREDICTED_CONFIG" \
  --max-gap 1 --max-error-px 0 dataset build
```

The worker is also called by the TUI's `d` action. The VLM comparison runs no
detector: it ingests the YTStream video's own tracks and reports the recorded
ingestion cost (see its [README](suite/vlm/README.md#ingestion-cost)).

Launch the TUI from the repository root:

```sh
engine/target/release/trql-tui --data-dir "$PWD/datasets/processed" \
  --python "$PWD/.venv-detect/bin/python" \
  --detect-script "$PWD/scripts/detect_track.py"
```

Select the family with the arrow keys, press Right to focus its sub-datasets,
select a clip, then press `d`. Choose a detector preset with Up/Down and press
Enter to run; `L` opens the log and `q` exits. The TUI's
`YOLO11x · BoT-SORT · traffic` preset uses classes 0/1/2/3/5/7. It verifies the
interactive detection path, but reproducing DETRAC's narrower 2/5/7 class filter
requires the CLI command above. The TUI does not expose those overrides.

## Coordinate units and fitting error

Coordinate and time units are dataset-specific; consult each converter's
metadata and documentation. Fitting error uses stored coordinate units, even
though the CLI option is named `max_error_px`. Budgets therefore cannot be
compared directly across datasets.

Zero error means exactness to stored coordinates. Choose fitting budgets with
the source data's precision in mind; demanding finer precision can greatly
increase the number of segments.
