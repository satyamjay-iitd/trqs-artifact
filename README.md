# TrQS artifact

Source and evaluation harness for the TrQS spatio-temporal query engine.
**Datasets are not included** -- they are third-party, large, and in one case
(openDD) under a NoDerivatives licence that forbids redistributing a converted
copy. Converters are provided instead.

## Layout

| path | what |
|---|---|
| `engine/` | the Rust workspace: algebra, DSL, polynomial fitting, selector, storage, CLI |
| `suite/` | the evaluation workload: TrQL queries, hand-written reference SQL, drivers |
| `suite/sql-ref/` | reference SQL per query, written against the DSL's denotation, not generated from the engine -- this is what the exactness gate compares against |
| `suite/vlm/` | the Gemini comparison: harness, prompts, raw responses, scoring |
| `plotter/` | `benchplot`, which renders every figure in the paper from `bench.db` |
| `scripts/detect_track.py` | the ingest worker: detector + tracker -> the `objects`/`bbox` schema |
| `converters/` | dataset converters and their shared helpers |
| `bench.db` | the measurements behind the paper's figures (SQLite), filtered to the queries and error budgets the figures actually read |

## What is in `bench.db`

Only what the figures need: the eleven queries the manifest names, and for the
six workload queries only the error budgets some figure asks for. Exploratory
runs -- wider sweeps, candidate queries that were measured and rejected -- are
not here. Every figure renders pixel-identically from this file and from the
full database it was filtered out of.

## Reproducing the figures without re-running anything

`bench.db` holds the measurements, so the figures can be redrawn directly:

```sh
cd plotter && uv run benchplot render          # all figures
cd plotter && uv run benchplot render --only main
```

## Re-running the measurements

Needs the datasets. Each is converted into a common schema --
`objects(oid, class)` and `bbox(oid, fid, x1, y1, x2, y2)` -- then fitted into a
`<max_gap>_<max_error_px>.trql` artifact:

```sh
cargo build --release
trql-cli --dataset <family> --sub-dataset <name> \
         --max-gap 1 --max-error-px 0 dataset build
trql-cli --dataset <family> --sub-dataset <name> \
         --max-gap 1 --max-error-px 0 bench <id> "<TrQL query>" 1 3 \
         --sql suite/sql-ref/<id>.sql --bench-db bench.db
```

`--sql` is what makes a run trustworthy: it runs the hand-written reference in
DuckDB on the same data and refuses the result unless the two answers agree
exactly on object tuples and interval endpoints.

## Two things a reader should know

**Coordinate units differ by dataset.** i24-Motion and openDD are world
coordinates (centimetres); the rest are pixels. The error budget is in those
same units, so budgets are not comparable across datasets -- each figure labels
its own.

**`max_error_px = 0` is not always the right reference point.** It means
*exact to the stored coordinate*. For openDD, whose positions are accurate to
about 10 cm, demanding centimetre-exactness forces the fitter to split down to
~3 points per segment and reports 3.2x compression; at a budget justified by
the data's own accuracy it is ~47x. Pick the floor from the data, not from 0.
