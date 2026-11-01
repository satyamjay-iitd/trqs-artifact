# benchplot

Plots for the VLDB'27 evaluation, generated from a `trql-cli bench.db`.

`bench.db` is written by `trql-cli … bench --sql <ref.sql> --bench-db bench.db`.
`Dataset` and `Query` are upserted by natural key, so one query's whole history —
every error budget, every repeat — is a set of `Run` rows sharing a `query_id`,
and a sweep needs no special naming to stay together.

Every figure in the paper is described in **`manifest.toml`**, which is checked in
beside it. The CLI reads the manifest and draws what it asks for; it has no flags
for how a plot looks, because that belongs in the file the paper ships with.

## Use

```bash
uv run benchplot render                   # every figure in manifest.toml
uv run benchplot render -n speedup        # just one, by name
uv run benchplot render --dry-run         # what would be written, where
uv run benchplot plots                    # what the manifest asks for
uv run benchplot list                     # what the database holds
uv run benchplot kinds                    # the plot kinds and their keys
```

`render` takes the manifest as an optional argument (`uv run benchplot render
other.toml`), and `--db` / `--out-dir` override the manifest for one run — the
"same figures, other database" case, which otherwise means editing a checked-in
file.

## The manifest

```toml
[defaults]
db = "../../../bench.db"      # where the benchmark scripts write
out_dir = "plots"
format = "png"                # png | pdf | svg | eps
aggregate = "latest"          # latest | median | min
exclude = ["I24-lane-any"]

[[plots]]
name = "speedup"              # picks the file name and `--only`; must be unique
kind = "speedup"
log = true
```

Paths are relative to the manifest, not to the shell — `render` writes the same
files from anywhere in the tree. A `[[plots]]` table inherits `[defaults]` and
overrides whatever it needs; a key that is neither common nor the kind's own is
an error, so a misspelled option fails loudly instead of silently drawing the
default chart.

### Keys every kind takes

| Key | Meaning |
| --- | --- |
| `name` | Required, unique. The output file name and the handle for `--only`. |
| `kind` | Required. One of `benchplot kinds`. |
| `enabled` | `false` keeps a figure in the manifest but out of the run. Still validated. |
| `title` | Title above the figure. |
| `db` | This figure's `bench.db`, if not the shared one. |
| `out` | Explicit output path, overriding `out_dir`/`format`. |
| `out_dir`, `format` | Together they make `<out_dir>/<name>.<format>`. |
| `aggregate` | How repeats of one `(query, err)` collapse: `latest`, `median`, `min`. |
| `queries` | Restrict to these queries, **in this order**; overrides `exclude`. |
| `exclude` | Drop these queries. Ignored when `queries` is given. |
| `errs` | Restrict to these error budgets. Default: all. |
| `scales` | Restrict to these `--db-repeat` tiling factors. Default: all. |

An unknown name in `queries`, or an `errs`/`scales` that matches no run, is an error
rather than an empty figure: the manifest is checked in, and a renamed query
should say so rather than leave a blank space in the paper.

**`I24-lane-any` is excluded in `[defaults]`.** It stays in `bench.db` and in
`final_queries.md`, where it earns its place as the unselective control, but it
is off the charts. A plot that names it in `queries` plots it anyway.

### Repeated runs

A `(query, err)` point measured more than once is collapsed by `aggregate`.
The default `latest` suits repeats that straddle a code change, where only the
newest describes the current engine; `median` and `min` suit several runs of one
build, where the spread is measurement noise.

## Plot kinds

### `speedup`

Bar chart of speedup over single-threaded DuckDB: **one bar group per query, one
bar per `max_error_px`**, `sql_time / trql_time` on the y-axis.

| Key | Default | Meaning |
| --- | --- | --- |
| `log` | `true` | Log y-axis. |
| `accuracy` | `true` | Precision/recall panel sharing the x-axis. |
| `memory` | `true` | Peak-memory panel; dropped when the database records none. |

```toml
[[plots]]
name = "speedup-i24"
kind = "speedup"
title = "i24, medians"
queries = ["I24-journey4", "I24-lane-truck"]
aggregate = "median"
errs = [0, 1, 2, 4]
memory = false
```

The figure has **three panels sharing the x-axis** — speedup, peak memory,
precision/recall — so a bar and its footprint and its accuracy sit in one column.
`accuracy = false` and `memory = false` drop the lower two; the memory panel also
drops itself when the database records none (runs made before it was measured
carry NULL, not 0).

Four things the chart encodes deliberately:

- **Log y by default.** Speedups span 2.8× to 852×; linearly, every query but the
  fastest collapses onto the baseline. `log = false` overrides.
- **A red parity line at 1×.** Below it the engine is slower than DuckDB.
- **Nothing in the bars marks an inexact answer.** In practice every budget above
  `max_error_px = 0` fails to match the reference SQL on objects *and* intervals,
  so the honest reading — every speedup above the first is bought with some
  accuracy — comes from the accuracy panel, which says by how much. Read the two
  panels together; a bar alone does not say what it cost.
- **Colour, not position, identifies the budget.** Groups are ragged — i24
  artifacts stop at 25 centi-feet while nuScenes runs to 50 cm — so bars are
  packed contiguously and a query measured at fewer budgets gets wider bars
  rather than gaps.

In the memory panel the bars are the **engine's whole-process RSS**, which includes
its eagerly-loaded `.trql`, and the red rule is **DuckDB's own `duckdb_memory()`**
for that query — one rule per group, because DuckDB never sees the error budget and
its figure does not vary with it. The two are different quantities and the gap
between them is not a like-for-like subtraction; they share an axis because the
question being asked is which system needs more to answer the query.

In the accuracy panel, precision is a filled circle and recall an open triangle,
offset slightly to either side of the bar centre — both sit at 1.0 for most points
and would otherwise be drawn on top of each other. The y-range is clipped to the
data rather than spanning [0, 1], which would show a flat line at the top.

Budgets are in **stored units**, which differ per dataset (i24 centi-feet,
nuScenes and SoccerTrackv2 centimetres, CLEVRER pixels). They are not comparable
*across* groups, only within one.

### `compression`

What each encoding *stores*, and what that buys: **one line per query, the error
budget on the x-axis**. A dashed rule per query marks its `bbox` row count — what
the relational encoding stores, flat, because it never sees the budget — and the
solid line is the polynomial pieces the engine stores instead. The lower panel is
speedup over single-threaded DuckDB on the same axis.

| Key | Default | Meaning |
| --- | --- | --- |
| `log` | `true` | Log y on both panels; the counts span three decades. |
| `speedup` | `true` | Speedup panel sharing the x-axis. |
| `ratio_labels` | `true` | Label bbox-rows-per-segment at each series' end points. |

```toml
[[plots]]
name = "compression"
kind = "compression"
queries = ["VIRAT-static", "VIRAT-dynamic"]
```

The figure exists to make one claim visible: a relational store writes one row per
object per frame however the object behaves, while a stationary object's
trajectory is a constant and fits in **one** piece however long it lasts. So the
gap between the dashed rule and the solid line is the redundancy the relational
encoding carries, and it widens with how still the scene is — 99,499 rows against
99 pieces (1,005×) for a scene where 94 % of tracked objects barely move, against
261× for one where only 13 % do.

Two things to read carefully:

- **The x-axis is `symlog`**, so `err = 0` has a place on an axis that is
  otherwise multiplicative. Ticks are exactly the budgets measured, so no reader
  infers a point between them.
- **`num_box` and `segments` must be recorded.** They are written by
  `trql-cli bench` from `TrqsStats`; a database predating those columns makes the
  plot fail with a message saying so rather than drawing an empty panel.
- **Pick a scale.** On a small clip the engine's fixed per-query cost (~5 ms)
  swamps the algebra and the speedup panel becomes a ratio of two overheads, flat
  across the whole budget. `scales = [100]` selects the `--db-repeat 100` runs,
  which multiply needle and haystack together — the ratio stays honest, but both
  sides clear their floors. Mixing scales in one figure draws a line per
  (query, scale), since runs at different tilings read different databases.

### `cooccurrence`

Objects, all pairs, and the pairs that actually co-occur, over a video's length:
**one line each, video time on the x-axis, log y**. A binary relationship ("A is
left of B") is only defined while both objects are visible in the same frame, so
an engine only ever considers pairs that co-occur — not all C(n, 2) of them. The
gap between the orange and blue lines is the work the quadratic bound predicts
and the video never asks for; the arrows label each against the object count.

| Key | Default | Meaning |
| --- | --- | --- |
| `stats` | `data/cooccurrence_stats.json` | The measured cache to read. |
| `label` | `YTStream` | Which dataset in the cache to draw. |
| `ratio_labels` | `true` | The two labelled arrows against the object count. |
| `legend` | `true` | In-axes legend. |
| `width`, `height` | `3.3`, `2.0` | Figure size in inches — one paper column. |

```toml
[[plots]]
name = "cooccurrence"
kind = "cooccurrence"
format = "pdf"
label = "YTStream"
```

**This kind reads no `bench.db`.** The quantity is a property of the video, not
of a query run over it, so it comes from a JSON cache measured straight off the
datasets by `tools/cooccurrence_stats.py` — which is not run by `benchplot
render`: it needs `duckdb`, reads `datasets/processed/`, and takes minutes.

```bash
uv run --with duckdb --with numpy tools/cooccurrence_stats.py
```

Because there are no runs, none of the selection keys (`queries`, `errs`,
`scales`, `aggregate`, …) mean anything here; the cache holds `DETRAC`, `VIRAT`,
`MEVA`, `Stanford Drone`, `YTStream` and `I-24 MOTION`, and a second `[[plots]]`
table naming another `label` draws that one.

Two things to read carefully:

- **The 52-minute stream is the interesting one.** What the figure is about —
  the separation between the quadratic bound and the pairs that co-occur — is
  what grows with video length. A 2-minute clip barely shows it.
- **`≤` in front of a ratio means it is an upper bound.** Where the exact
  per-frame self join was too expensive (I-24 MOTION, 830 k objects), each object
  is closed to `[min fid, max fid]` and any overlap counts as co-occurrence, which
  track gaps make an over-count. `exact` in the cache records which was used.

## Adding a plot kind

The paper wants more figures than one. A kind is a module in
`src/benchplot/plots/` exporting a `KIND`:

```python
@dataclass(frozen=True)
class RooflineOptions:
    annotate: bool = True

def _parse(table: Table) -> RooflineOptions:
    return RooflineOptions(annotate=table.boolean("annotate", True))

def _render(points: pd.DataFrame, plot: Plot) -> Path:
    ...  # points is already selected and collapsed; write plot.out, return it

KIND = PlotKind(name="roofline", summary="…", parse=_parse, render=_render,
                option_help=(("annotate", "Label each point (default true)."),))
```

Then add it to `KINDS` in `plots/__init__.py`. Nothing else names a kind — the
manifest reader, the CLI and `benchplot kinds` all work off that dict, so the new
kind is immediately writable in `manifest.toml`. `_render` never sees the common
keys' machinery: `queries`, `exclude`, `errs`, `scales` and `aggregate` are applied for it
in `render.py`, and `plot.order` is the query order the manifest asked for.

A kind whose numbers are not in `bench.db` at all sets `needs_runs=False`. The
renderer then opens no database, applies no selection, and hands `_render` an
empty frame; the kind reads whatever file its own options name, resolved against
the manifest's directory with `table.path("stats", default=table.base / …)`.
Give it a `describe` too, since "44 points, latest of 3067 runs" is not what such
a plot drew — see `plots/cooccurrence.py`.

## Layout

- `manifest.toml` — the figures.
- `spec.py` — what a plot is (`Plot`, `Selection`, `PlotKind`) and the TOML
  reader that gives manifest errors their `file plots[2] "name".key:` prefix.
- `manifest.py` — `manifest.toml` → `list[Plot]`.
- `render.py` — `Plot` → an image, reading each `bench.db` at most once.
- `db.py` — the SQLite read and `collapse_repeats`.
- `plots/` — one module per kind.
- `cli.py` — argument parsing and printing, nothing else.
- `data/` — measurements that are not `bench.db`, checked in because what
  produced them is expensive and rarely rerun.
- `tools/` — what produced them. Not part of the package; run by hand.
