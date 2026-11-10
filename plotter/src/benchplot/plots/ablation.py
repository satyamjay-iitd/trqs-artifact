"""Figure 9: the polynomial-degree (9a) and interval-representation (9b) ablations.

Both compare *builds* of the engine, not budgets of one build, so their
measurements are not rows of the main `bench.db`. Each manifest entry carries
two things:

* where the measurements live (`source`, and `stages` for the degree
  ablation's instrumented runs), which the drawing reads; and
* the matrix that produced them (`[plots.measure]`): one query on one
  sub-dataset at the listed budgets, once per `[[plots.measure.variants]]`
  build. `reproduce_paper.sh` builds each variant with its cargo features and
  re-measures the matrix; `benchplot render --source-dir` then draws from the
  new directory instead of the recorded one.

A variant's `fit` names the fitted files it uses. Variants with the same `fit`
share one data directory, so the `.trql` artifacts are fitted once: declare it
only for builds whose features leave fitting untouched (`instrument` and
`naive-intervals` change query time only; a different degree changes the fit).
It defaults to the variant's own name.

A variant's `output` says what the drawing reads from it: `bench-db` is the
`--bench-db` file `<variant>.db`, and `report` is the per-run JSON `bench` writes,
kept as `<query>-<variant>.json`.

The drawing itself is `scripts/degree_ablation.py` and
`scripts/interval_ablation.py`, run with this interpreter, so the figure is the
same whether it is drawn from here or by hand.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from benchplot.spec import Plot, PlotKind, Table

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
OUTPUTS = ("bench-db", "report")


@dataclass(frozen=True)
class Variant:
    """One engine build of the ablation."""

    name: str
    #: Cargo features of `trql-cli` for this build.
    features: tuple[str, ...]
    #: Whether the default features (the quadratic degree) stay on.
    default_features: bool
    #: What the drawing reads from it; see the module docstring.
    output: str
    #: Fitted-file group; variants sharing it share `.trql` artifacts.
    fit: str


@dataclass(frozen=True)
class Measure:
    """The runs behind the figure, for `reproduce_paper.sh`."""

    query: str
    #: `family/sub-dataset`, as in `bench.db`.
    dataset: str
    max_gap: int
    budgets: tuple[float, ...]
    runs: int
    #: Gate every run against `suite/sql-ref/<query>.sql` (and time DuckDB).
    sql: bool
    variants: tuple[Variant, ...]


@dataclass(frozen=True)
class AblationOptions:
    """The `[[plots]]` keys only these kinds understand."""

    #: Directory the drawing reads.
    source: Path
    #: Directory of the instrumented runs; the degree ablation only. Defaults
    #: to `source`, which is how a re-measurement lays them out.
    stages: Path | None
    measure: Measure
    #: Degree ablation: drop this many of the widest budgets from the drawing.
    drop_widest: int | None
    width: float | None


def _variant(t: Table) -> Variant:
    name = t.required_string("name")
    out = t.choice("output", OUTPUTS, "bench-db")
    v = Variant(
        name=name,
        features=t.strings("features", ()) or (),
        default_features=t.boolean("default_features", True),
        output=out,
        fit=t.string("fit", name) or name,
    )
    t.reject_unread()
    return v


def _measure(table: Table) -> Measure:
    t = table.table("measure")
    query = t.required_string("query")
    dataset = t.required_string("dataset")
    if "/" not in dataset:
        t.fail("dataset", f"expected family/sub-dataset, got {dataset!r}")
    budgets = t.numbers("budgets")
    if not budgets:
        t.fail("budgets", "is required: the error budgets to measure")
    variants = tuple(_variant(v) for v in t.tables("variants"))
    if not variants:
        t.fail("variants", "is required: at least one [[plots.measure.variants]]")
    names = [v.name for v in variants]
    if len(set(names)) != len(names):
        t.fail("variants", "variant names must be distinct; they name output files")
    m = Measure(
        query=query,
        dataset=dataset,
        max_gap=int(t.number("max_gap", 1) or 1),
        budgets=budgets,
        runs=int(t.number("runs", 3) or 3),
        sql=t.boolean("sql", False),
        variants=variants,
    )
    t.reject_unread()
    return m


def _parse(table: Table) -> AblationOptions:
    source = table.path("source")
    if source is None:
        table.fail("source", "is required: the directory the drawing reads")
    drop = table.number("drop_widest")
    return AblationOptions(
        source=source,
        stages=table.path("stages"),
        measure=_measure(table),
        drop_widest=None if drop is None else int(drop),
        width=table.number("width"),
    )


def _run(script: str, args: list[str]) -> None:
    proc = subprocess.run(
        [sys.executable, str(_SCRIPTS / script), *args],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        raise ValueError(f"{script} failed: {' | '.join(tail)}")


def _render_degree(points: pd.DataFrame, plot: Plot) -> Path:
    o: AblationOptions = plot.options
    args = ["--data-dir", str(o.source), "--stage-dir", str(o.stages or o.source),
            "--query", o.measure.query, "--out", str(plot.out)]
    if o.drop_widest is not None:
        args += ["--drop-widest", str(o.drop_widest)]
    if o.width is not None:
        args += ["--width", f"{o.width:g}"]
    plot.out.parent.mkdir(parents=True, exist_ok=True)
    _run("degree_ablation.py", args)
    return plot.out


def _render_interval(points: pd.DataFrame, plot: Plot) -> Path:
    o: AblationOptions = plot.options
    plot.out.parent.mkdir(parents=True, exist_ok=True)
    _run("interval_ablation.py",
         ["--data-dir", str(o.source), "--query", o.measure.query, "--out", str(plot.out)])
    return plot.out


def _describe(plot: Plot) -> str:
    o: AblationOptions = plot.options
    m = o.measure
    return (f"{m.query} on {m.dataset}, {len(m.variants)} builds x "
            f"{len(m.budgets)} budget{'s' if len(m.budgets) > 1 else ''}, from {o.source}")


_HELP = (
    ("source", "Directory of the measurements the drawing reads (required)."),
    ("stages", "Degree ablation: directory of the instrumented runs (default: source)."),
    ("drop_widest", "Degree ablation: leave out this many of the widest budgets."),
    ("width", "Degree ablation: figure width in inches."),
    ("measure", "Table: query, dataset, max_gap, budgets, runs, sql, [[variants]] "
                "(name, features, default_features, output, fit)."),
)

DEGREE_KIND = PlotKind(
    name="degree_ablation",
    summary="Fig 9a: linear vs quadratic vs cubic builds against the budget.",
    parse=_parse,
    render=_render_degree,
    option_help=_HELP,
    needs_runs=False,
    describe=_describe,
)

INTERVAL_KIND = PlotKind(
    name="interval_ablation",
    summary="Fig 9b: 5-tuple vs naive interval sets, where the query time goes.",
    parse=_parse,
    render=_render_interval,
    option_help=_HELP,
    needs_runs=False,
    describe=_describe,
)
