"""Three encodings of one query, against input size.

Rekall holds a materialised label per detection and composes them in Python; a
relational store holds one row per object per frame and joins them; the engine
holds fitted polynomial segments and evaluates the predicate analytically. All
three answer the *same* query — Rekall's own cyclist-detection tutorial — on the
same detections, and `bench_rekall.py` requires them to return the same answer
before any of these times is recorded.

Log-log, because the finding is the slopes rather than any one point. Rekall
pays per candidate pair, so its cost grows with the boxes; the engine pays per
segment and per root, so its cost grows with what the boxes fit into. That is
why the lines diverge rather than sit at a constant offset.

Rekall is drawn in its faster configuration, with its `IntervalSetMapping` keyed
by frame so each join sees one frame's boxes. `IntervalSet.join` otherwise copies
a list slice of the other set per outer interval, a cost quadratic in set size
and unrelated to the query, and timing that would report an implementation
artifact rather than the query. `whole = true` adds the tutorial's own
whole-video shape as a dashed line for comparison; it is off by default, so the
figure reports the baseline at its best.

Reads `suite/results-rekall.json`, not `bench.db`: the Rekall
column comes from `bench_rekall.py`, which `trql-cli bench` knows nothing about.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator

from benchplot.spec import Plot, PlotKind, Table

DEFAULT_RESULTS = Path("data/results-rekall.json")

# Categorical slots of the validated default palette: the two Python-label
# encodings warm, the engine cool, so the figure reads as "materialised rows
# against fitted segments" before any label is read.
_C_REKALL = "#c1440e"
_C_DUCKDB = "#eb6834"
_C_TRQS = "#2a78d6"
_INK = "#0b0b0b"
_INK_MUTED = "#52514e"

_RC: Any = {
    "font.family": "serif",
    "font.serif": ["Linux Libertine", "DejaVu Serif"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.edgecolor": "#c9c8c3",
    "axes.linewidth": 0.6,
    "xtick.color": _INK_MUTED,
    "ytick.color": _INK_MUTED,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "grid.color": "#e6e5e0",
    "grid.linewidth": 0.5,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    # Type 42 embeds the fonts as TrueType rather than Type 3, which is what
    # camera-ready checkers insist on.
    "pdf.fonttype": 42,
}


@dataclass(frozen=True)
class RekallOptions:
    """The `[[plots]]` keys only this kind understands."""

    #: The measurements, written by `suite/bench_rekall.py`.
    results: Path = DEFAULT_RESULTS
    #: Which input variant to draw. `tracked` is the like-for-like comparison;
    #: `raw` is Rekall's release untouched, where the absence of track ids
    #: leaves the engine one segment per detection and nothing to compress.
    variant: str = "tracked"
    #: The engine's budget to draw. Only 0 is gated on reproducing the answer
    #: exactly, and on this query the wider budgets cost the same anyway.
    err: float = 0.0
    #: Also draw the tutorial's own whole-video decomposition, as a dashed
    #: line. Off by default: the frame-keyed configuration is the faster of the
    #: two, so drawing it alone reports Rekall at its best.
    whole: bool = False
    #: Label the gap between the engine and the frame-keyed Rekall line at the
    #: largest size measured.
    ratio_label: bool = True
    legend: bool = True
    width: float = 3.3
    height: float = 2.0


def load_results(path: Path, variant: str, err: float) -> list[dict]:
    """The measured points for one variant and budget, ascending in size."""
    if not path.is_file():
        raise FileNotFoundError(
            f"no Rekall measurements at {path} — run "
            "suite/bench_rekall.py"
        )
    rows = json.loads(path.read_text())
    variants = sorted({r["variant"] for r in rows})
    if variant not in variants:
        raise ValueError(
            f"no variant {variant!r} in {path}; it holds {', '.join(variants)}"
        )
    points = [r for r in rows if r["variant"] == variant and r["max_error_px"] == err]
    if not points:
        budgets = sorted({r["max_error_px"] for r in rows if r["variant"] == variant})
        raise ValueError(
            f"{path}: no {variant} runs at err={err:g}; measured budgets are "
            f"{', '.join(format(b, 'g') for b in budgets)}"
        )
    # At the lossless budget every point must have passed the exactness gate, or
    # the figure would be putting times for different answers on one axis.
    if err == 0:
        unchecked = [r for r in points if not r.get("trqs_exact", False)]
        if unchecked:
            raise ValueError(
                f"{path}: {len(unchecked)} of {len(points)} {variant} points at "
                "err=0 did not reproduce the reference answer exactly"
            )
    return sorted(points, key=lambda r: r["boxes"])


def panel(ax, points: list[dict], options: RekallOptions, scale: float = 1.0) -> None:
    boxes = [r["boxes"] for r in points]
    rk_frame = [r["rekall"]["frame"]["ms_query"] for r in points]
    rk_whole = [r["rekall"]["whole"]["ms_query"] for r in points]
    duckdb = [r["ms_duckdb"] for r in points]
    trqs = [r["ms_trqs"] for r in points]

    if options.whole:
        ax.plot(
            boxes, rk_whole,
            color=_C_REKALL, ls="--", lw=1.0, marker="^", ms=2.8 * scale,
            label="Rekall, as written",
        )
    ax.plot(
        boxes, rk_frame,
        color=_C_REKALL, lw=1.3, marker="o", ms=2.8 * scale,
        label="Rekall, frame-keyed" if options.whole else "Rekall",
    )
    # At a narrow width the thread count moves to the caption: it is what makes
    # the legend a row taller, and a row of legend costs a decade of the panel.
    ax.plot(
        boxes, duckdb,
        color=_C_DUCKDB, lw=1.3, marker="s", ms=2.8 * scale,
        label="DuckDB" if options.width < 2.4 else "DuckDB (1 thread)",
    )
    ax.plot(
        boxes, trqs,
        color=_C_TRQS, lw=1.6, marker="D", ms=2.8 * scale, label="TrQS",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Bounding boxes", fontsize=7 * scale, labelpad=2)
    ax.set_ylabel("Query time (ms)", fontsize=7 * scale, labelpad=2)
    ax.tick_params(labelsize=6.5 * scale, pad=1.5)
    ax.grid(True, which="major", axis="both")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.yaxis.set_major_locator(LogLocator(numticks=6))

    top = max(rk_whole) if options.whole else max(rk_frame)
    # Headroom for the in-axes legend, which sits above the lines on the left:
    # every series rises to the right, which leaves that corner clear. Wide
    # enough and the entries sit two to a row and cost about one decade;
    # narrower they stack, and the headroom has to grow with them.
    rows = 1 if options.width >= 2.4 else (2 if options.whole else 1.35)
    headroom = (12 if rows == 1 else 12 ** rows) if options.legend else 2
    ax.set_ylim(min(trqs) / 2.5, top * headroom)

    if not options.ratio_label:
        return

    # The gap at the largest size measured, which is the one the text quotes.
    # On a log axis the geometric mean is the visual midpoint of the arrow.
    x, lo, hi = boxes[-1], trqs[-1], rk_frame[-1]
    ax.annotate(
        "",
        xy=(x, hi), xytext=(x, lo),
        arrowprops=dict(
            arrowstyle="<->", color=_INK_MUTED, lw=0.7, shrinkA=1, shrinkB=1
        ),
    )
    ax.text(
        x, float(np.sqrt(lo * hi)), f"{hi / lo:.0f}$\\times$ ",
        ha="right", va="center", color=_INK, fontsize=6.5 * scale,
    )


def plot_rekall(
    out_path: Path, *, options: RekallOptions, title: str | None = None
) -> Path:
    """Writes the chart to `out_path` and returns it."""
    points = load_results(options.results, options.variant, options.err)
    # The type was sized for a 3.3 in column. Drawn narrower it comes down with
    # the axes, with a floor so a half-column panel is still legible; drawn
    # wider it stays put, since type that grows with the figure reads as a
    # screenshot rather than as part of the page.
    scale = max(min(options.width / 3.3, 1.0), 0.85)

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(options.width, options.height))
        panel(ax, points, options, scale)
        if title:
            ax.set_title(title, fontsize=7.5 * scale)
        if options.legend:
            ax.legend(
                loc="upper left", ncol=2, frameon=False,
                handlelength=1.5, borderpad=0.1, labelspacing=0.2,
                columnspacing=1.0, handletextpad=0.4,
                fontsize=6.5 * scale,
            )
        fig.tight_layout(pad=0.2)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _parse(table: Table) -> RekallOptions:
    return RekallOptions(
        results=table.path("results", default=table.base / DEFAULT_RESULTS)
        or DEFAULT_RESULTS,
        variant=table.choice("variant", ("raw", "tracked"), "tracked"),
        err=table.number("err", 0.0) or 0.0,
        whole=table.boolean("whole", False),
        ratio_label=table.boolean("ratio_label", True),
        legend=table.boolean("legend", True),
        width=table.number("width", 3.3) or 3.3,
        height=table.number("height", 2.0) or 2.0,
    )


def _render(points: pd.DataFrame, plot: Plot) -> Path:
    """`points` is empty by design — see `needs_runs` on the kind below."""
    return plot_rekall(plot.out, options=plot.options, title=plot.title)


def _describe(plot: Plot) -> str:
    options: RekallOptions = plot.options
    return (
        f"{options.variant} at eps={options.err:g}, from {options.results.name}"
    )


KIND = PlotKind(
    name="rekall",
    summary=(
        "Rekall, DuckDB and the engine on Rekall's own cyclist query against "
        "input size, log-log. Reads results-rekall.json, not bench.db."
    ),
    parse=_parse,
    render=_render,
    needs_runs=False,
    describe=_describe,
    option_help=(
        ("results", f"The measurements to read (default {DEFAULT_RESULTS})."),
        ("variant", "`tracked` (default) or `raw` — see bench_rekall.py."),
        ("err", "The engine's budget to draw (default 0, the exact point)."),
        ("whole", "Also draw the tutorial's whole-video shape, dashed (default false)."),
        ("ratio_label", "Label the engine-to-Rekall gap at the largest size (default true)."),
        ("legend", "In-axes legend (default true)."),
        ("width", "Figure width in inches (default 3.3, one column)."),
        ("height", "Figure height in inches (default 2.0)."),
    ),
)
