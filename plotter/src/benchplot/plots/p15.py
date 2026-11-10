"""Speedup and peak memory at the lowest budget, one row of two panels.

The headline figure draws five budgets per query; this is the slice of it at the
first one (`p15`), for a reader who wants the two numbers that matter -- how much
faster, and how much memory -- without the budget sweep. Queries run down the
left edge and are shared by both panels, so a row reads across: the bar in the
left panel is that query's speedup over DuckDB, the bar in the right is what the
engine held to get it, with DuckDB's own figure as a dashed rule.

Takes the same `budgets` table as `kind = "speedup"` and uses the first budget
of each query, so the two figures cannot drift apart on what "p15" means.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

from benchplot.plots.speedup import _PALETTE, _bar_label, _layout
from benchplot.spec import Plot, PlotKind, Table

_BAR = 0.62

#: Dash pattern for DuckDB's memory rule. Each rule only spans one bar's slot,
#: and in a short figure the default "--" fits a single dash and a stub of the
#: next, which reads as "!" rather than dashed; shorter dashes keep three.
_SHORT_DASH = (0, (1.6, 1.0))


def plot_p15(
    df: pd.DataFrame,
    out_path: Path,
    *,
    order: list[str],
    budgets: dict[str, list[float]],
    labels: dict[str, str] | None = None,
    log: bool = True,
    width: float = 3.4,
    height: float = 2.3,
) -> Path:
    """Writes the chart to `out_path` and returns it."""
    if df.empty:
        raise ValueError("nothing to plot")
    missing = [q for q in order if q not in set(df["query"])]
    if missing:
        raise ValueError(f"query not in the database: {', '.join(missing)}")

    # Only the first budget of each query, and only it is required to have been
    # measured: the later budgets belong to the headline figure, not this one.
    first = {q: list(v[:1]) for q, v in budgets.items()}
    rows = {b.row["query"]: b.row for b in _layout(df, order, first)}
    type_scale = 0.95 if width <= 4.5 else 1.0

    fig, (ax_speed, ax_mem) = plt.subplots(
        1,
        2,
        figsize=(width, height),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    ys = range(len(order))

    for y, query in zip(ys, order):
        row = rows[query]
        # One colour for every query: the rows are named on the axis, and the
        # colour is the p15 colour of the headline figure.
        colour = _PALETTE(0)
        ax_speed.barh(
            y,
            row["speedup"],
            height=_BAR,
            color=colour,
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
        ax_speed.annotate(
            _bar_label(row["speedup"]),
            (row["speedup"], y),
            xytext=(1, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=6.5 * type_scale,
            zorder=4,
        )
        mem = row.get("trql_mem_mb")
        if pd.notna(mem):
            ax_mem.barh(
                y,
                mem,
                height=_BAR,
                color=colour,
                edgecolor="black",
                linewidth=0.4,
                zorder=3,
            )
        # DuckDB never sees the error budget, so its figure is one value per
        # query: a rule across the bar's slot, not a bar beside it.
        sql_mem = row.get("sql_mem_mb")
        if pd.notna(sql_mem):
            ax_mem.vlines(
                sql_mem,
                y - _BAR / 2 - 0.08,
                y + _BAR / 2 + 0.08,
                color="crimson",
                linewidth=1.2,
                linestyle=_SHORT_DASH,
                zorder=4,
            )

    # Parity: left of this line the engine is slower than single-threaded DuckDB.
    ax_speed.axvline(1.0, color="crimson", linewidth=1.0, linestyle="--", zorder=2)

    if log:
        ax_speed.set_xscale("log")
        ax_mem.set_xscale("log")
        # Room on the right of the longest bar for the number standing at its end.
        top = max(rows[q]["speedup"] for q in order)
        ax_speed.set_xlim(0.7, top * 8)
        # Decades only: the panels are narrow, and a label on every minor tick
        # runs into its neighbour.
        for ax in (ax_speed, ax_mem):
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
            ax.xaxis.set_minor_formatter(NullFormatter())
        # Whole decades on the memory axis, so it has more than one tick to read.
        mems = [
            v
            for q in order
            for v in (rows[q].get("trql_mem_mb"), rows[q].get("sql_mem_mb"))
            if pd.notna(v) and v > 0
        ]
        if mems:
            ax_mem.set_xlim(10 ** math.floor(math.log10(min(mems))), 10 ** math.ceil(math.log10(max(mems))))

    ax_speed.set_yticks(list(ys))
    ax_speed.set_yticklabels([(labels or {}).get(q, q) for q in order], fontsize=6.5 * type_scale)
    ax_speed.set_ylim(len(order) - 0.5, -0.5)  # first query on top
    ax_speed.set_xlabel("Speedup over DuckDB", fontsize=7.4 * type_scale, labelpad=2)
    ax_mem.set_xlabel("Peak memory (MB)", fontsize=7.4 * type_scale, labelpad=2)
    for ax in (ax_speed, ax_mem):
        ax.tick_params(axis="x", labelsize=6 * type_scale, pad=1.5)
        ax.grid(axis="x", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
    ax_mem.tick_params(axis="y", left=False)

    # Inside the speedup panel, bottom right, kept small and tight to the corner so
    # it stays clear of the last bar's number.
    ax_speed.legend(
        handles=[Line2D([], [], color="crimson", linewidth=1.4, linestyle=_SHORT_DASH, label="DuckDB")],
        loc="lower right",
        fontsize=6.5 * type_scale,
        handlelength=1.2,
        handletextpad=0.4,
        borderpad=0.2,
        borderaxespad=0.2,
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out_path


@dataclass(frozen=True)
class P15Options:
    """The `[[plots]]` keys only this kind understands."""

    log: bool = True
    labels: dict[str, str] | None = None
    #: Query id -> its budgets, as for `kind = "speedup"`; the first of each is drawn.
    budgets: dict[str, list[float]] | None = None
    width: float | None = None
    height: float | None = None


def _parse(table: Table) -> P15Options:
    budgets = table.number_list_map("budgets") or None
    if not budgets:
        table.fail("budgets", "is required: the first budget of each query is the one drawn")
    return P15Options(
        log=table.boolean("log", True),
        labels=table.string_map("labels") or None,
        budgets=budgets,
        width=table.number("width"),
        height=table.number("height"),
    )


def _render(points: pd.DataFrame, plot: Plot) -> Path:
    options: P15Options = plot.options
    if not plot.order:
        raise ValueError("p15 needs `queries`, which also fixes the order")
    return plot_p15(
        points,
        plot.out,
        order=plot.order,
        budgets=options.budgets,
        labels=options.labels,
        log=options.log,
        width=options.width or 3.4,
        height=options.height or 2.3,
    )


KIND = PlotKind(
    name="p15",
    summary="Speedup and peak memory at each query's first budget: one row, two panels, queries down the side.",
    parse=_parse,
    render=_render,
    option_help=(
        ("log", "Log x-axes (default true)."),
        ("budgets", "Query id -> budgets, as for `speedup`; the first of each is drawn (required)."),
    ),
)
