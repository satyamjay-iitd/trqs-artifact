# benchplot

Render the paper's recorded measurements from the repository's `bench.db` and
`plotter/data/`. No datasets, detector environment, or API key are needed to
redraw these figures. Python 3.13+ and uv are required; `uv.lock` fixes the
plotting dependencies.

## Commands

Run from `plotter/`:

```sh
uv run --locked benchplot render
```

This produces the quantitative paper figures under `plots/`, including Figure 9
(`degree_ablation`, `interval_ablation`). The
manifest command also renders older and supplementary plots. MIRIS is the
preserved-PDF exception described below.

| Paper figure | Output | Command |
|---|---|---|
| 4a: motion complexity and compression | `compression.pdf` | `uv run --locked benchplot render --only compression` |
| 5: six-query latency and memory comparison | `p15.pdf` | `uv run --locked benchplot render --only p15` |
| 6: VLM comparison | `vlmbars.pdf` | `uv run --locked benchplot render --only vlmbars` |
| 7: MIRIS comparison | `miris_timeline_plot.pdf` | `uv run --locked benchplot render --only miris_timeline_plot` |
| 8: fitting-budget ablation | `epsilon.pdf` | `uv run --locked benchplot render --only epsilon` |
| 9a: polynomial-degree ablation | `degree_ablation.pdf` | `uv run --locked benchplot render --only degree_ablation` |
| 9b: interval-representation ablation | `interval_ablation.pdf` | `uv run --locked benchplot render --only interval_ablation` |

Figure 7 is copied unchanged from `data/miris_timeline_plot.pdf`. Its original
measurements, generator, and MIRIS execution repository are not included.
Figures 1–3 and Figure 4b are manuscript illustrations,
video stills, or TikZ drawings, not benchmark plots produced by this package.

`main.pdf` is the broader budget sweep; it is not the paper's main Figure 5.
The degree script defaults to the narrow, three-budget layout used in Figure 9a;
`--drop-widest 0 --width 3.4` draws the broader version.

## Inspect or select plots

```sh
uv run --locked benchplot render --dry-run
uv run --locked benchplot plots
uv run --locked benchplot list
uv run --locked benchplot kinds
uv run --locked benchplot render --only p15 --only epsilon
uv run --locked benchplot render --out-dir /tmp/trqs-figures
```

The manifest is an optional positional argument to `render` and `plots`.
`--db` and `--out-dir` override its database and output directory for one run.
Paths inside the manifest are relative to the manifest, not the shell.
The default database is `../bench.db`.

## Recorded inputs and new measurements

Inputs are listed in [data/README.md](data/README.md). Suite baseline reruns
write under `results/suite/`, leaving recorded plot inputs unchanged.

`scripts/repro_plan.py` lists the `trql-cli bench` runs behind chosen plots,
read from this manifest and the recorded `bench.db` opened read-only; the root
`reproduce_paper.sh` runs that plan. For new paired benchmarks, select the
database explicitly:

```sh
uv run --locked benchplot render --only p15 --db ../results/suite/bench.db \
  --out-dir ../results/suite/plots
```

It must contain the queries, scales, and budgets selected by the plot. The
[experiment suite](../suite/README.md#run-queries) explains the database schemas.
For separate inputs, update these manifest keys:

| Plot | Input setting |
|---|---|
| `compression` | `baselines`: deduplicated DuckDB results JSON |
| `rekall` | `results`: Rekall benchmark JSON |
| `vlm`, `vlmbars` | `source`: directory with `results-vlm*.json`; ingestion is the recorded `INGEST_S` in `plots/vlm.py` |
| `cooccurrence` | `stats`: measured co-occurrence JSON |
| `degree_ablation` | `source`: `<variant>.db` per build; `stages`: `<variant>-inst.db` |
| `interval_ablation` | `source`: `<query>-<variant>.json` per build |

For the plots that read their own directory (`vlm`, `vlmbars` and the two
ablations), `--source-dir DIR` overrides `source` (and `stages`) for one run
without editing the manifest; `reproduce_paper.sh` draws Figure 9 that way.
The ablation entries also carry `[plots.measure]`, the matrix behind them:
query, sub-dataset, budgets, and one `[[plots.measure.variants]]` per engine
build (`name`, cargo `features`, `default_features`, `output`, and `fit`:
variants with the same `fit` share fitted files).
`scripts/degree_ablation.py` and `scripts/interval_ablation.py` do the drawing
and still run directly (`--data-dir`, `--stage-dir`, `--out`).

A `[[plots]]` table may set `mark_missing = true`: a named query with no runs is
then drawn as a marked "not measured" gap instead of stopping the figure, and
the render summary names it. Figure 5 (`p15`) sets it.

## Manifest

A `[defaults]` table sets the database, output directory, format, and aggregation.
Each `[[plots]]` table names a plot and its kind. For example:

```toml
[defaults]
db = "../bench.db"
out_dir = "plots"
format = "pdf"
aggregate = "latest"

[[plots]]
name = "example"
kind = "speedup"
queries = ["CL-overtake"]
errs = [0]
scales = [1]
```

Common selection keys are `queries`, `exclude`, `errs`, `scales`, and
`aggregate`. `latest`, `median`, `min`, and `best` choose how repeated runs are
collapsed; `best` is a deliberately optimistic accuracy selection, used by the
historical DETRAC plots. `enabled = false` disables rendering while retaining
validation. Use `benchplot kinds` for the options supported by each kind.
Unknown keys, absent queries, and selections with no measurements are errors.

## Dataset-derived exploratory plots

`scripts/motion_cdf.py`, `scripts/displacement_cdf.py`,
`scripts/cooccurrence.py`, and `tools/cooccurrence_stats.py` read processed
datasets rather than the recorded plot inputs. They are not needed to redraw
the current paper. `TRQS_DATA_DIR` selects the processed tree; these scripts
use DuckDB in addition to the plotting environment.
`motion_cdf.py` and `displacement_cdf.py` declare their dependencies for `uv run`.
`tools/cooccurrence_stats.py` uses the Python package and can be run with
`uv run --with duckdb --with numpy tools/cooccurrence_stats.py`.
`scripts/cooccurrence.py` uses the DuckDB CLI executable on `PATH`; install it
as described in [the suite README](../suite/README.md#setup) and check
`duckdb --version` before running `uv run python scripts/cooccurrence.py`.
`scripts/gap_sweep.py` uses the recorded `bench.db` and produces the older,
currently commented-out DETRAC experiment.
