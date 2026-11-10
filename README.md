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
| `suite/vlm/` | the Gemini comparison: harness, prompts, raw responses, scoring |
| `plotter/` | plot renderers, recorded auxiliary measurements, and the preserved MIRIS PDF |
| `scripts/detect_track.py` | the ingest worker: detector + tracker -> the `objects`/`bbox` schema |
| `converters/` | 18 dataset converters, shared helpers, and projection dependencies; see its README for coverage |
| `lean-proof/` | interval-set algebra proofs and pinned Lean/mathlib dependencies |
| `bench.db` | the measurements behind the paper's figures (SQLite), filtered to the queries and error budgets the figures actually read |

## Building and preparing inputs

The engine requires Rust/Cargo 1.92+, a C/C++ compiler toolchain, and the
**DuckDB 1.5.5 shared library** (`libduckdb.so` on Linux). DuckDB is not bundled;
install the matching library before building. For a library outside the system
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
`duckdb.db`, `metadata.json`, and a video when available. The SoccerTrack v2
converter is not included; SoccerNet is distinct from SoccerTrack. The merged
sub-datasets the experiments use are built with `converters/paper_merges.sh`.

## Reproducing the figures from recorded measurements

The root `bench.db` and `plotter/data/` contain the measurements needed by the
figures. No datasets or API keys are needed to render them:

```sh
cd plotter
uv run --locked benchplot render
uv run --locked python scripts/degree_ablation.py
uv run --locked python scripts/interval_ablation.py
```

Outputs go to `plotter/plots/`. See the [plotter README](plotter/README.md) for
individual figures and new measurements. The MIRIS figure is a preserved PDF;
its generator and execution repository are not included.

## Re-running the measurements

### One command

`reproduce_paper.sh` re-measures every point that Figures 4a, 5 and 8 draw from
`bench.db`, then redraws those figures from the new measurements:

```sh
./reproduce_paper.sh --dry-run          # the plan and the exact commands; runs nothing
./reproduce_paper.sh                    # compression (4a), p15 (5), epsilon (8)
./reproduce_paper.sh --plots compression --data-dir /path/to/processed
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
listed. A figure is drawn only when every point it needs has been measured.
The engine is built if `engine/target/release/trql-cli` is absent.

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

The worker is also called by the TUI's `d` action and
`suite/vlm/ingest_cost.py`. The VLM timer has its own documented estimated-cost
methodology; it does not claim that its detector generated the annotated tracks.

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
