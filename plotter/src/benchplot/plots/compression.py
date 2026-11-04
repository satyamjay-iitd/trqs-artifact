"""What each encoding stores, and what that buys — static scene against dynamic.

A relational store writes one bounding-box row per object per frame, however the
object behaves. Dropping the rows that repeat their predecessor removes that
redundancy without changing the data. The engine does something else again: one
polynomial piece per fitted segment, so a stationary object's whole appearance is
one piece no matter how many frames it lasts, and the piece count falls further
as the fitting budget grows.

One group per clip, one bar per encoding: the two relational baselines first —
neither of which has an error budget — then the engine at each budget measured.
The top panel is what each stores, the bottom what the same query costs over it,
so the two can be read against each other. Storing less is only interesting if
the query gets cheaper, and on a clip with little redundancy to remove it does
not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import LogLocator
from matplotlib.patches import Patch

from benchplot.spec import Plot, PlotKind, Table

# The engine's bars all carry one colour and name their own budget above them,
# which is what a ramp across five near-identical greens could only gesture at.
# The two relational baselines sit outside that scale — neither sees a budget —
# so they are drawn apart from it.
_TRQS_COLOUR = plt.get_cmap("tab20")(0)
_FRAME_COLOUR = "#c1440e"
_DEDUP_COLOUR = "#f0b08a"

_GROUP_WIDTH = 0.84


@dataclass(frozen=True)
class CompressionOptions:
    """The `[[plots]]` keys only this kind understands."""

    log: bool = True
    time: bool = True
    ratio_labels: bool = True
    #: Figure width in inches (default 7.2, the full text width). Set it
    #: smaller to sit beside another figure; the type comes down with it so the
    #: labels stay the size they would have been on the page.
    width: float | None = None
    #: Figure height in inches. Defaults to a shape that suits the width.
    height: float | None = None
    #: Print each engine bar's error budget above it, on the upper panel. The
    #: engine's bars share one colour, so without this nothing tells them apart.
    eps_labels: bool = True
    #: Print the recall of any bar that does not reproduce the frame-expanded
    #: answer exactly. The first bar of a group *is* that answer; bars that
    #: match it carry no label either.
    recall_labels: bool = True
    #: A second relational baseline to draw as a bar per query: the same query
    #: over the frame-deduplicated copy of the clip, measured by
    #: `suite/bench_dedup.py` into the JSON this names. Keyed by
    #: query id, with `rows_deduped`, `ms_deduped` and the `scale` they were
    #: measured at.
    baselines: Path | None = None


def load_baselines(path: Path, scale: float | None) -> dict:
    """The dedup baseline measurements, checked against the scale being drawn."""
    if not path.is_file():
        raise FileNotFoundError(
            f"no baseline measurements at {path} — run suite/bench_dedup.py"
        )
    data = json.loads(path.read_text())
    if scale is not None:
        # Drawn beside runs at one tiling, so a baseline measured at another is
        # not a comparison — it is a different database.
        wrong = {q: v["scale"] for q, v in data.items() if v.get("scale") != scale}
        if wrong:
            raise ValueError(
                f"{path.name} was measured at scale "
                + ", ".join(f"{q}={s:g}" for q, s in wrong.items())
                + f", but this figure draws scale {scale:g}"
            )
    return data


@dataclass(frozen=True)
class _Bar:
    """One bar: where it sits, what it stores, what the query cost over it."""

    x: float
    width: float
    rows: float
    ms: float
    #: Recall against the frame-expanded answer, which is what the first bar of
    #: each group computes; `None` for that bar, and for a run that records none.
    recall: float | None
    colour: object
    #: Bars with the same key share a colour and one legend entry across groups.
    key: str


def _layout(
    df: pd.DataFrame, queries: list[str], errs: list[float], baselines: dict
) -> list[_Bar]:
    """Bar positions, computed once so both panels land on the same x.

    A group is the two relational baselines followed by the engine at each
    budget, in that order: the thing being improved on, then the improvement.
    """
    width = _GROUP_WIDTH / (len(errs) + (2 if baselines else 1))
    bars: list[_Bar] = []
    for gi, query in enumerate(queries):
        rows = df[df["query"] == query].sort_values("err")
        if rows.empty:
            raise ValueError(f"no runs for {query}")
        left = gi - _GROUP_WIDTH / 2

        # `num_box` is a property of the clip; `sql_ms` is recorded per budget
        # although DuckDB never sees one, so the spread across budgets is
        # measurement noise and the bar is their mean.
        slots: list[tuple[str, float, float, float | None, object]] = [
            (
                "every frame",
                float(rows["num_box"].dropna().iloc[0]),
                float(rows["sql_ms"].dropna().mean()),
                None,
                _FRAME_COLOUR,
            )
        ]
        if query in baselines:
            b = baselines[query]
            slots.append(
                (
                    "deduplicated",
                    float(b["rows_deduped"]),
                    float(b["ms_deduped"]),
                    b.get("recall_deduped"),
                    _DEDUP_COLOUR,
                )
            )
        for _, row in rows.iterrows():
            recall = row.get("recall")
            slots.append(
                (
                    f"{row['err']:g}",
                    float(row["segments"]),
                    float(row["trql_ms"]),
                    None if pd.isna(recall) else float(recall),
                    _TRQS_COLOUR,
                )
            )

        for bi, (key, n_rows, ms, recall, colour) in enumerate(slots):
            bars.append(
                _Bar(left + width * (bi + 0.5), width, n_rows, ms, recall, colour, key)
            )
    return bars


def _draw(ax, bars: list[_Bar], value, log: bool, label: str, scale: float = 1.0) -> None:
    for bar in bars:
        ax.bar(
            bar.x,
            value(bar),
            width=bar.width * 0.88,
            color=bar.colour,
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
    if log:
        # 10 k pieces against 17 M rows, 15 ms against 5 s: three decades either
        # way, so a linear axis would show the baselines and nothing else.
        ax.set_yscale("log")
        # Shallow panels otherwise come out with two labelled decades, which
        # reads as an unlabelled axis; ask for one per decade and let matplotlib
        # thin them if they collide.
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.set_ylabel(label, fontsize=9 * scale, labelpad=2 if scale < 1 else 4)


def _annotate(ax, bars, queries, value, fmt, lift: float = 0, scale: float = 1.0) -> None:
    """Label the engine's widest budget in each group, against the first bar.

    The figure's point is the distance between a group's first bar and its last,
    which a log axis makes easy to see and hard to state. One number per group
    states it.
    """
    per_group = len(bars) // max(len(queries), 1)
    for gi in range(len(queries)):
        group = bars[gi * per_group : (gi + 1) * per_group]
        if not group:
            continue
        base = next((b for b in group if b.key == "every frame"), None)
        best = group[-1]
        if base is None or not value(best):
            continue
        ax.annotate(
            fmt(value(base) / value(best)),
            (best.x, value(best)),
            textcoords="offset points",
            xytext=(0, 4 + lift),
            ha="center",
            va="bottom",
            fontsize=7 * scale,
            fontweight="bold",
            color="0.25",
            zorder=4,
        )
    # Room for the label standing on the tallest annotated bar, and for
    # whatever was lifted above it.
    lo, hi = ax.get_ylim()
    grow = 1.6 + lift / 30
    ax.set_ylim(lo, hi * grow if ax.get_yscale() == "log" else hi * (1 + 0.12 * grow))


def _annotate_eps(ax, bars: list[_Bar], value, scale: float = 1.0) -> None:
    """Print each engine bar's error budget above it.

    The engine's bars share a colour, so the budget has to be written rather
    than keyed, and it goes on one panel only: the groups are the same bars in
    both, and saying it twice would be noise. Rotated, because at column width a
    group's bars are narrower than the characters a budget needs.
    """
    labelled = [b for b in bars if b.key not in ("every frame", "deduplicated")]
    for bar in labelled:
        ax.annotate(
            rf"$\varepsilon = {bar.key}$",
            (bar.x, value(bar)),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            va="bottom",
            fontsize=7 * scale,
            rotation=90,
            color="0.3",
            zorder=4,
        )
    if labelled:
        # Room for the tallest of them to stand clear of the axis.
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 2.6 if ax.get_yscale() == "log" else hi * 1.25)


def _annotate_recall(ax, bars: list[_Bar], value, scale: float = 1.0) -> None:
    """The recall of any bar that does not return the whole answer.

    The first bar of a group is the reference the others are scored against, so
    it carries nothing: a baseline cannot miss its own answers. Nor does a bar
    that matches it exactly -- a column of `1.000` is a lot of ink to say that
    one bar in eight is different. What is left is the exceptions, which is the
    only part a reader has to weigh against the bar's height.
    """
    for bar in bars:
        # 0.9995 rounds to 1.000 at the precision printed, so a label there
        # would read as "exact" while marking the bar as an exception.
        if bar.recall is None or int(bar.recall) == 1:
            continue
        ax.annotate(
            f"F1={bar.recall:.2f}",
            (bar.x, value(bar)),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            va="bottom",
            fontsize=7.0 * scale,
            rotation=90,
            color="0.3",
            zorder=4,
        )


def plot_compression(
    df: pd.DataFrame,
    out_path: Path,
    *,
    order: list[str] | None = None,
    options: CompressionOptions,
    baselines: dict | None = None,
    title: str | None = None,
) -> Path:
    """Writes the chart to `out_path` and returns it."""
    baselines = baselines or {}
    if df.empty:
        raise ValueError("nothing to plot")
    if "segments" not in df.columns or df["segments"].isna().all():
        raise ValueError(
            "no run records a segment count — re-run `trql-cli bench` with a build "
            "that writes trql_segments"
        )

    df = df[df["segments"].notna()]
    queries = order or sorted(df["query"].unique())
    missing = [q for q in queries if q not in set(df["query"])]
    if missing:
        raise ValueError(f"no segment counts for: {', '.join(missing)}")

    errs = sorted(df["err"].unique())
    bars = _layout(df, queries, errs, baselines)

    # Both panels are shallow on purpose -- the figure is one of three in a row
    # across the page -- and the lower one is shorter still: its bars span two
    # decades against the upper panel's four, and the legend sits in the space
    # the short bars leave rather than in space taken from them.
    ratios = [0.7, 0.8] if options.time else [1.0]
    width = options.width or 7.2
    # Narrow, the figure is published at close to the size it is drawn, so the
    # type comes down with the canvas -- but only so far. Below ~0.85 the labels
    # stop being readable in print, and a figure sharing a float with another
    # has to carry type of the same size as its neighbour, not type scaled to
    # its own width. 7.2 in is the full text width, where nothing is scaled.
    scale = max(min(width / 7.2, 1.0), 0.85)
    height = options.height or (2.0 + 1.6 * (len(ratios) - 1)) * (0.5 + 0.5 * scale)
    fig, axes = plt.subplots(
        len(ratios),
        1,
        figsize=(width, height),
        sharex=True,
        height_ratios=ratios,
        gridspec_kw={"hspace": 0.1},
        squeeze=False,
    )
    panels = [ax for (ax,) in axes]
    ax_rows = panels.pop(0)
    ax_time = panels.pop(0) if options.time else None

    _draw(ax_rows, bars, lambda b: b.rows, options.log, "#Rows", scale)
    if options.eps_labels:
        _annotate_eps(ax_rows, bars, lambda b: b.rows, scale)
    if ax_time is not None:
        _draw(ax_time, bars, lambda b: b.ms, options.log, "Time (ms)", scale)
    # Recall rides the time panel: "how much faster" and "at what cost to the
    # answer" are the pair a reader has to weigh together, and the panel above
    # is about storage, which has no recall of its own.
    lift = 0.0
    if options.recall_labels and ax_time is not None:
        _annotate_recall(ax_time, bars, lambda b: b.ms, scale)
        lift = 20.0 * scale
    if options.ratio_labels:
        _annotate(ax_rows, bars, queries, lambda b: b.rows,
                  lambda r: f"{r:,.0f}× fewer", 0, scale)
        if ax_time is not None:
            _annotate(ax_time, bars, queries, lambda b: b.ms,
                      lambda r: f"{r:,.0f}× faster", lift, scale)

    bottom = ax_time or ax_rows
    bottom.set_xticks(range(len(queries)))
    bottom.set_xticklabels(["VIRAT", "DETRAC"], fontsize=8 * scale)
    # Keep a little breathing room beside the outer bars and their labels.
    outer_margin = 0.07
    left = min(b.x - b.width * 0.88 / 2 for b in bars) - outer_margin
    right = max(b.x + b.width * 0.88 / 2 for b in bars) + outer_margin
    bottom.set_xlim(left, right)
    for ax in filter(None, (ax_rows, ax_time)):
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7.5 * scale, pad=1.5)
    if title:
        ax_rows.set_title(title)

    # One entry per *encoding* rather than per bar: the engine's budgets now
    # share a colour and name themselves above their bars, so five swatches of
    # one colour would say nothing the bars do not already say.
    seen: dict[str, object] = {}
    for bar in bars:
        key = bar.key if bar.key in ("every frame", "deduplicated") else "TrQS"
        seen.setdefault(key, bar.colour)
    handles = [
        Patch(
            facecolor=colour,
            edgecolor="black",
            linewidth=0.4,
            label=f"DuckDB, {key}" if key != "TrQS" else "TrQS, at different ε",
        )
        for key, colour in seen.items()
    ]
    # Keyed on the width in inches, not on the type scale: what decides how
    # many swatches fit on a line is how wide the axes are, and the type has a
    # floor. Five columns need the full text width; a single column takes three.
    rows_of_legend = 2 if width >= 6.5 else 3 if width >= 4.0 else 4
    # Shallow, the legend goes *inside* the top panel rather than above it: three
    # rows of swatches over a 2 in figure is a sixth of its height, and the bars
    # fall away to the right of each group, which leaves that corner clear. The
    # panel is grown to make the room rather than taking it from the bars.
    inside = height is not None and height <= 2.5
    # In the query-time panel: its bars fall away to the right of each group, so
    # the top of it is the clearest stretch in the figure, and the panel above
    # keeps its full height for the counts.
    key_ax = ax_time if (inside and ax_time is not None) else ax_rows
    if inside:
        lo, hi = key_ax.get_ylim()
        if key_ax.get_yscale() == "log":
            import math

            key_ax.set_ylim(lo, 10 ** (math.log10(hi) + (math.log10(hi) - math.log10(lo)) * 0.3))
        else:
            key_ax.set_ylim(lo, hi + (hi - lo) * 0.3)
    key_ax.legend(
        handles=handles,
        fontsize=6.5 * scale,
        ncol=-(-len(handles) // (2 if inside else rows_of_legend)),
        loc="upper right" if inside else "lower center",
        bbox_to_anchor=(1.0, 1.0) if inside else (0.5, 1.0),
        borderaxespad=0.2 if inside else 0.5,
        frameon=inside,
        framealpha=0.85,
        edgecolor="none",
        facecolor="white",
        borderpad=0.05 if inside else 0.2,
        columnspacing=(0.7 if inside else 1.2) * scale,
        handlelength=1.1 if inside else 1.4,
        handletextpad=0.35,
        handleheight=0.9,
        labelspacing=0.2 if inside else 0.3,
    )

    # Remove the trimmed space from the canvas too, preserving the physical
    # bar widths, inter-group gap, text size, and margins outside the axes.
    retained = (right - left) / (len(queries) + 0.2)
    axes_width = ax_rows.get_position().width
    cropped_width = width * (1 - axes_width * (1 - retained))
    for ax in filter(None, (ax_rows, ax_time)):
        pos = ax.get_position()
        ax.set_position((pos.x0 * width / cropped_width, pos.y0,
                         pos.width * width * retained / cropped_width, pos.height))
    fig.set_size_inches(cropped_width, height)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out_path


def _parse(table: Table) -> CompressionOptions:
    return CompressionOptions(
        log=table.boolean("log", True),
        time=table.boolean("time", True),
        ratio_labels=table.boolean("ratio_labels", True),
        eps_labels=table.boolean("eps_labels", True),
        recall_labels=table.boolean("recall_labels", True),
        width=table.number("width"),
        height=table.number("height"),
        baselines=table.path("baselines"),
    )


def _render(points: pd.DataFrame, plot: Plot) -> Path:
    options: CompressionOptions = plot.options
    scales = plot.selection.scales
    baselines = (
        load_baselines(options.baselines, scales[0] if scales and len(scales) == 1 else None)
        if options.baselines
        else {}
    )
    return plot_compression(
        points,
        plot.out,
        order=plot.order,
        options=options,
        baselines=baselines,
        title=plot.title,
    )


KIND = PlotKind(
    name="compression",
    summary=(
        "Bars of what each encoding stores and what the query costs over it: "
        "the relational baselines, then the engine at each error budget."
    ),
    parse=_parse,
    render=_render,
    option_help=(
        ("log", "Log y-axis on both panels (default true; the counts span three decades)."),
        ("width", "Figure width in inches (default 7.2, the full text width)."),
        ("height", "Figure height in inches (default follows the width)."),
        ("time", "Add a query-time panel sharing the x-axis (default true)."),
        (
            "baselines",
            "JSON of a second relational baseline to draw as a bar per query "
            "(suite/bench_dedup.py writes it).",
        ),
        (
            "eps_labels",
            "Print each engine bar's error budget above it, in the rows panel "
            "(default true); the engine's bars share one colour.",
        ),
        (
            "recall_labels",
            "Print the recall of any bar that is not exact, on top of it, in "
            "the time panel (default true).",
        ),
        (
            "ratio_labels",
            "Label the engine's widest budget against the frame-expanded "
            "baseline, once per group and panel (default true).",
        ),
    ),
)
