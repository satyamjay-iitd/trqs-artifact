"""Error-budget ablation: what each budget buys in speed and costs in accuracy.

One panel per query, laid out in a grid. In a panel, each error budget drawn is
a point: x is the engine's runtime in milliseconds, y is the F1 of its answer
against the reference answer. The points are joined in order of budget, so the
line reads left to right as "wider budget, faster, and -- if it bends down --
less exact". A query whose points all sit on 1.0 shows a flat line, which is
the finding: the budget is free there.

Every budget measured is drawn unless something narrows that. `eps` gives an
explicit per-query list, which is the blunt instrument: a sweep is often
measured wider than it is worth showing, and the panels do not have to agree on
a range because the budgets are in each dataset's own coordinate units anyway.
`budgets` with `show` is the other way, naming percentile budgets of the
headline figure instead. Either may be given for some queries and not others;
a query named by neither keeps all of its points.

F1 combines the tuple-level `precision` and `recall` the other speedup figures
draw as two markers. Budgets are in whatever units the dataset stores
coordinates in, so they are not comparable across panels; each panel labels its
own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

from benchplot.plots.speedup import _PALETTE
from benchplot.spec import Plot, PlotKind, Table


#: One marker per rung of a named ladder, in `rank_labels` order. Colour
#: already encodes the panel, so shape is free to encode the budget.
_MARKERS = ("o", "s", "^", "D", "v", "X", "P", "*")


def _f1(precision: float, recall: float) -> float:
    if pd.isna(precision) or pd.isna(recall) or precision + recall == 0:
        return float("nan")
    return 2 * precision * recall / (precision + recall)


def _annotate(ax, xs: list[float], ys: list[float], errs: list[float], fs: float) -> None:
    """Label budgets, skipping any whose point sits too near one already labelled.

    The first and last are always labelled, so the ends of the sweep are named
    even where the middle is crowded. Distance is measured in the panel's own
    coordinates, after the limits are set, so it is what the eye sees.
    """
    fig = ax.get_figure()
    fig.canvas.draw()
    to_axes = ax.transAxes.inverted().transform
    pts = [to_axes(ax.transData.transform((x, y))) for x, y in zip(xs, ys)]
    shown: list[int] = []
    order = [0, len(xs) - 1] + list(range(1, len(xs) - 1))
    for i in order:
        if i in shown:
            continue
        if all(math.dist(pts[i], pts[j]) > 0.16 for j in shown):
            shown.append(i)
    for i in shown:
        ax.annotate(
            f"{errs[i]:g}",
            (xs[i], ys[i]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fs,
            color="0.35",
            zorder=5,
        )


def _rung(name: str, rank_labels: tuple[str, ...] | None) -> int:
    """Index of a point's name in the ladder, for picking its marker.

    A point may carry two names where two percentiles resolve to one budget
    ("p15/p25"); the first one decides the shape.
    """
    first = name.split("/")[0]
    order = list(rank_labels or ())
    return order.index(first) % len(_MARKERS) if first in order else 0


def plot_epsilon(
    df: pd.DataFrame,
    out_path: Path,
    *,
    order: list[str],
    labels: dict[str, str] | None = None,
    rows: int = 2,
    cols: int = 3,
    log: bool = True,
    width: float = 7.0,
    height: float = 3.6,
    budgets: dict[str, list[float]] | None = None,
    rank_labels: tuple[str, ...] | None = None,
    show: tuple[str, ...] | None = None,
    eps: dict[str, list[float]] | None = None,
    legend_names: bool = False,
    ylim: tuple[float, float] | None = None,
    sharey: bool = False,
    align_one: bool = False,
) -> Path:
    """Writes the chart to `out_path` and returns it."""
    if df.empty:
        raise ValueError("nothing to plot")
    missing = [q for q in order if q not in set(df["query"])]
    if missing:
        raise ValueError(f"query not in the database: {', '.join(missing)}")
    if len(order) > rows * cols:
        raise ValueError(f"{len(order)} queries do not fit a {rows}x{cols} grid")

    fig, axes = plt.subplots(rows, cols, figsize=(width, height), sharey=sharey, squeeze=False)
    flat = [ax for row in axes for ax in row]
    fs = 7.0 if width <= 4.5 else 10.0

    for i, (ax, query) in enumerate(zip(flat, order)):
        d = df[df["query"] == query].sort_values("err")
        if eps and query in eps:
            # An explicit per-query budget list. Unlike `budgets`, which names
            # percentiles, this just says which budgets to draw -- a sweep is
            # often measured wider than it is worth showing. A budget asked for
            # but never measured is an error rather than a silently short line.
            want = eps[query]
            d = d[d["err"].isin(want)]
            absent = set(want) - set(d["err"])
            if absent:
                raise ValueError(
                    f"{query}: no run at max_error_px "
                    f"{', '.join(format(e, 'g') for e in sorted(absent))}")
            if d.empty:
                raise ValueError(f"{query}: `eps` left no points to draw")
        names: list[str] | None = None
        if budgets:
            # The named budgets only. Where two names resolve to one budget (p15
            # and p25 are both 0 on half the datasets) the point is drawn once
            # and carries both names.
            wanted: dict[float, list[str]] = {}
            for name, err in zip(rank_labels or (), budgets[query]):
                if show is None or name in show:
                    wanted.setdefault(err, []).append(name)
            d = d[d["err"].isin(wanted)]
            absent = set(wanted) - set(d["err"])
            if absent:
                raise ValueError(f"{query}: no run at max_error_px {sorted(absent)}")
            names = ["/".join(wanted[e]) for e in d["err"]]
        xs = list(d["trql_ms"])
        ys = [_f1(p, r) for p, r in zip(d["precision"], d["recall"])]
        errs = list(d["err"])
        # One colour throughout: the panels are named in their titles, and shape,
        # not colour, carries the budget.
        colour = _PALETTE(0)
        ax.plot(xs, ys, color=colour, linewidth=0.9, alpha=0.7, zorder=2)
        if names and legend_names:
            # A named ladder bunches in x wherever the budget has stopped
            # buying speed -- five of Q2's six land inside 5 ms -- so no
            # per-point text can fit. Shape carries the name instead, and one
            # legend serves every panel because the ladder is the same in each.
            for x, y, name in zip(xs, ys, names):
                ax.scatter([x], [y], s=16, color=colour, edgecolor="black",
                           linewidth=0.4, zorder=3,
                           marker=_MARKERS[_rung(name, rank_labels)])
        else:
            ax.scatter(xs, ys, s=14, color=colour, edgecolor="black",
                       linewidth=0.4, zorder=3)

        if log:
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
            ax.xaxis.set_minor_formatter(NullFormatter())
        # Zoom to the data: F1 here runs from 1.0 down to a few percent below it,
        # and a [0, 1] axis would draw every panel as a flat line.
        if ylim:
            ax.set_ylim(*ylim)
        else:
            lo, hi = min(ys), max(ys)
            span = hi - lo
            # Room above for the markers and the budget labels, which sit on
            # top of each point: with a dense sweep several points land on the
            # panel's maximum, so the old 12% headroom clipped their labels
            # against the top spine. Below, room for the labels that hang under
            # each point.
            below = 1.0 if (names and not legend_names) else 0.6
            ax.set_ylim(lo - max(0.002, below * span), hi + max(0.0025, 0.26 * span))
            if align_one:
                # Fit each panel independently, with F1=1 at 90% of its height.
                bottom = ax.get_ylim()[0]
                ax.set_ylim(bottom, 1.0 + (1.0 - bottom) / 9.0)
        # Likewise the runtime axis: fitted to the points, with a margin so the
        # outermost labels are not cut by the panel edge.
        if len(xs) > 0:
            if log:
                a, b = math.log10(min(xs)), math.log10(max(xs))
                m = max(0.25, 0.2 * (b - a))
                ax.set_xlim(10 ** (a - m), 10 ** (b + m))
                if b - a + 2 * m < 1.6:
                    # Under two decades there may be no power of ten to label,
                    # so label 1-2-5 steps instead.
                    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=12))
                    ax.xaxis.set_major_formatter(ScalarFormatter())
                    ax.xaxis.set_minor_formatter(NullFormatter())
            else:
                a, b = min(xs), max(xs)
                m = max(0.05 * a, 0.18 * (b - a))
                ax.set_xlim(a - m, b + m)
        if names and not legend_names:
            lo_y = ax.get_ylim()[0]
            drawn = 0
            # Two lines, name over budget: a percentile ladder bunches in x
            # wherever the budget has stopped buying speed -- five of Q2's six
            # land within 5 ms -- and a one-line "p99.9 (540)" is far too wide
            # to sit under points that close together.
            for x, y, name, err in zip(xs, ys, names, errs):
                text = f"{name}\n{err:g}"
                if y < lo_y:
                    # Below the fixed range: say so at the bottom edge, with the
                    # value, rather than dropping the point silently.
                    ax.annotate(
                        f"{text}\nF1 {y:.3f} \u2193",
                        (x, 0.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=fs * 0.75,
                        color=colour,
                        zorder=5,
                    )
                    continue
                # Under the point, cycling three heights: with two, a run of
                # three near-coincident points still collides.
                ax.annotate(
                    text,
                    (x, y),
                    xytext=(0, -5 - 13 * (drawn % 3)),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=fs * 0.75,
                    color="0.35",
                    zorder=5,
                )
                drawn += 1
        elif not names:
            _annotate(ax, xs, ys, errs, fs * 0.8)

        ax.set_title((labels or {}).get(query, query), fontsize=fs + 0.5, pad=2)
        ax.tick_params(labelsize=fs * 0.9, pad=1.5)
        ax.grid(alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        if i % cols == 0:
            ax.set_ylabel("F1", fontsize=fs, labelpad=2)
        if i // cols == rows - 1 or i + cols >= len(order):
            ax.set_xlabel("Runtime (ms)", fontsize=fs, labelpad=2)
    for ax in flat[len(order):]:
        ax.set_visible(False)

    if bool(rank_labels) and legend_names:
        from matplotlib.lines import Line2D

        handles = [
            Line2D([], [], linestyle="none", marker=_MARKERS[i % len(_MARKERS)],
                   markersize=fs * 0.7, markerfacecolor=_PALETTE(0),
                   markeredgecolor="black", markeredgewidth=0.4, label=n)
            for i, n in enumerate(rank_labels or ())
            if show is None or n in show
        ]
        # Inside the last panel, placed where it covers the fewest points.
        flat[len(order) - 1].legend(
            handles=handles, loc="best", ncol=2, frameon=True, framealpha=0.9,
            edgecolor="none", handlelength=1.0, borderpad=0.3, labelspacing=0.25,
            columnspacing=0.8, handletextpad=0.3, fontsize=fs * 0.85,
        )

    fig.subplots_adjust(wspace=0.32, hspace=0.55)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out_path


@dataclass(frozen=True)
class EpsilonOptions:
    """The `[[plots]]` keys only this kind understands."""

    log: bool = True
    labels: dict[str, str] | None = None
    rows: int = 2
    cols: int = 3
    width: float | None = None
    height: float | None = None
    #: Query id -> budgets named by `rank_labels`, as for `kind = "speedup"`;
    #: `show` picks which names are drawn. Without them every budget is drawn.
    budgets: dict[str, list[float]] | None = None
    rank_labels: tuple[str, ...] | None = None
    show: tuple[str, ...] | None = None
    #: Query id -> the budgets to draw for it. Queries left out keep every
    #: budget measured. Independent of `budgets`/`show`, which name percentiles.
    eps: dict[str, list[float]] | None = None
    #: Draw a named ladder as marker shapes with one shared legend, instead of
    #: text under each point. The ladder is the same in every panel, and where
    #: the budget has stopped buying speed its points bunch too tightly in x
    #: for any per-point label to fit.
    legend_names: bool = False
    #: Fixed y range (low, high); by default fitted to each panel's data.
    ylim: tuple[float, float] | None = None
    #: One y axis for every panel, so F1 reads on the same scale across queries.
    sharey: bool = False
    #: Independent automatic ranges with F1=1 at the same relative height.
    align_one: bool = False


def _parse(table: Table) -> EpsilonOptions:
    ylim = table.numbers("ylim")
    if ylim is not None and len(ylim) != 2:
        table.fail("ylim", "expected two numbers: low, high")
    return EpsilonOptions(
        budgets=table.number_list_map("budgets") or None,
        rank_labels=table.strings("rank_labels"),
        show=table.strings("show"),
        eps=table.number_list_map("eps") or None,
        legend_names=table.boolean("legend_names", False),
        ylim=tuple(ylim) if ylim else None,
        sharey=table.boolean("sharey", False),
        align_one=table.boolean("align_one", False),
        log=table.boolean("log", True),
        labels=table.string_map("labels") or None,
        rows=int(table.number("rows", 2)),
        cols=int(table.number("cols", 3)),
        width=table.number("width"),
        height=table.number("height"),
    )


def _render(points: pd.DataFrame, plot: Plot) -> Path:
    options: EpsilonOptions = plot.options
    if not plot.order:
        raise ValueError("epsilon needs `queries`, which also fixes the order")
    return plot_epsilon(
        points,
        plot.out,
        order=plot.order,
        labels=options.labels,
        rows=options.rows,
        cols=options.cols,
        log=options.log,
        width=options.width or 7.0,
        height=options.height or 3.6,
        budgets=options.budgets,
        rank_labels=options.rank_labels,
        show=options.show,
        eps=options.eps,
        legend_names=options.legend_names,
        ylim=options.ylim,
        sharey=options.sharey,
        align_one=options.align_one,
    )


KIND = PlotKind(
    name="epsilon",
    summary="Error-budget ablation: one panel per query, speedup against F1, each budget a point.",
    parse=_parse,
    render=_render,
    option_help=(
        ("log", "Log x-axes (default true)."),
        ("rows", "Grid rows (default 2)."),
        ("cols", "Grid columns (default 3)."),
        ("budgets", "Query id -> budgets named by `rank_labels`; draw only those named in `show`."),
        ("eps", "Query id -> the budgets to draw for it; queries left out keep all of theirs."),
        ("legend_names", "Draw a named ladder as marker shapes with one shared legend (default false)."),
        ("sharey", "One y axis for every panel (default false)."),
        ("align_one", "Align F1=1 with independent automatic y ranges (default false; ylim takes precedence)."),
        ("ylim", "Fixed y range as [low, high] (default: fitted to each panel)."),
    ),
)
