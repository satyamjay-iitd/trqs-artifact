"""Object count against all pairs against the pairs that actually co-occur.

A binary relationship ("A is left of B") is only defined while both objects are
visible in the same frame, so an engine only ever has to consider pairs that
co-occur — not all C(n, 2) of them. As a video plays the distinct-object count n
grows, C(n, 2) grows quadratically, and the co-occurring set grows close to
linearly: the gap between the two lines is the work the quadratic bound predicts
and the video never asks for.

Unlike every other kind here, this one reads no `bench.db`. Its numbers come
from the datasets themselves, measured by `tools/cooccurrence_stats.py` into a
JSON cache, because the quantity is a property of the video rather than of a
query run over it.
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

from benchplot.spec import Plot, PlotKind, Table

DEFAULT_STATS = Path("data/cooccurrence_stats.json")

# Categorical slots 2 / 1 / 3 of the validated default palette.
_C_ALL = "#eb6834"
_C_COOC = "#2a78d6"
_C_OBJ = "#1baf7a"
_INK = "#0b0b0b"
_INK_MUTED = "#52514e"

# A column-width figure for the paper body, so it is typeset in the document's
# own serif at the size it will actually be read at, rather than shrunk down
# from a screen-sized chart with everything on it becoming unreadable.
# `Any` because rcParams' keys are typed as a Literal union of every setting
# matplotlib has, which a plain dict literal cannot satisfy.
_RC: Any = {
    "font.family": "serif",
    "font.serif": ["Linux Libertine", "DejaVu Serif"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 7,
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
class CooccurrenceOptions:
    """The `[[plots]]` keys only this kind understands."""

    stats: Path = DEFAULT_STATS
    label: str = "YTStream"
    ratio_labels: bool = True
    legend: bool = True
    width: float = 3.3
    height: float = 2.0


def load_stats(path: Path, label: str) -> dict:
    """The one measured dataset `label` names, from the cache at `path`."""
    if not path.is_file():
        raise FileNotFoundError(
            f"no co-occurrence cache at {path} — run tools/cooccurrence_stats.py"
        )
    stats = json.loads(path.read_text())
    matches = [s for s in stats if s["label"] == label]
    if not matches:
        raise ValueError(
            f"no dataset labelled {label!r} in {path}; it holds "
            f"{', '.join(sorted(s['label'] for s in stats))}"
        )
    return matches[0]


def _gap_arrow(ax, x: float, lo: float, hi: float, text: str) -> None:
    """A double-headed arrow spanning two series, labelled with their ratio.

    On a log axis the geometric mean is the visual midpoint, which is where the
    label goes so it sits centred in the arrow rather than near one end.
    """
    ax.annotate(
        "",
        xy=(x, hi),
        xytext=(x, lo),
        arrowprops=dict(arrowstyle="<->", color=_INK_MUTED, lw=0.7, shrinkA=1, shrinkB=1),
    )
    ax.text(
        x,
        np.sqrt(lo * hi),
        text,
        ha="right",
        va="center",
        color=_INK,
        fontsize=7,
        bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.0),
    )


def panel(ax, s: dict, options: CooccurrenceOptions) -> None:
    t = np.asarray(s["time_s"]) / 60.0
    n = np.asarray(s["objects"])
    cooc = np.asarray(s["cooc"])
    all_pairs = n * (n - 1) / 2

    ax.plot(t, all_pairs, color=_C_ALL, label=r"All pairs $\binom{n}{2}$")
    ax.plot(t, cooc, color=_C_COOC, label="Co-occurring pairs")
    ax.plot(t, n, color=_C_OBJ, lw=2.0, ls=":", label="Num Objects")

    ax.set_yscale("log")
    ax.set_xlim(0, t[-1])
    ax.set_ylim(1, all_pairs[-1] * 10)  # headroom for the in-axes legend
    ax.grid(True, which="major", axis="y")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    if not options.ratio_labels:
        return

    # Where the interval closure was used instead of the exact self join, the
    # co-occurring count is an upper bound, and so is every ratio drawn from it.
    bound = "" if s["exact"] else r"$\leq$"

    # Both arrows measure against the object count, near the end of the video
    # where the curves have separated: how much bigger the quadratic bound is,
    # and how much bigger the set that actually co-occurs is.
    for frac, series in ((0.86, all_pairs), (0.95, cooc)):
        x = t[-1] * frac
        lo, hi = np.interp(x, t, n), np.interp(x, t, series)
        _gap_arrow(ax, x, float(lo), float(hi), f"{bound}{series[-1] / n[-1]:,.0f}" + r"$\times$")


def plot_cooccurrence(
    out_path: Path,
    *,
    options: CooccurrenceOptions,
    title: str | None = None,
) -> Path:
    """Writes the chart to `out_path` and returns it."""
    s = load_stats(options.stats, options.label)

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(options.width, options.height))
        panel(ax, s, options)
        ax.set_ylabel("Count")
        ax.set_xlabel("Video time (min)")
        if title:
            ax.set_title(title)
        if options.legend:
            ax.legend(
                loc="lower right",
                ncol=1,
                frameon=False,
                handlelength=1.8,
                borderpad=0.1,
                labelspacing=0.25,
            )
        fig.tight_layout(pad=0.2)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _parse(table: Table) -> CooccurrenceOptions:
    return CooccurrenceOptions(
        stats=table.path("stats", default=table.base / DEFAULT_STATS) or DEFAULT_STATS,
        label=table.string("label", "YTStream") or "YTStream",
        ratio_labels=table.boolean("ratio_labels", True),
        legend=table.boolean("legend", True),
        width=table.number("width", 3.3) or 3.3,
        height=table.number("height", 2.0) or 2.0,
    )


def _render(points: pd.DataFrame, plot: Plot) -> Path:
    """`points` is empty by design — see `needs_runs` on the kind below."""
    return plot_cooccurrence(plot.out, options=plot.options, title=plot.title)


def _describe(plot: Plot) -> str:
    options: CooccurrenceOptions = plot.options
    return f"{options.label}, from {options.stats.name}"


KIND = PlotKind(
    name="cooccurrence",
    summary=(
        "Objects, all pairs and actually-co-occurring pairs over a video's "
        "length, for one measured dataset. Reads a JSON cache, not bench.db."
    ),
    parse=_parse,
    render=_render,
    needs_runs=False,
    describe=_describe,
    option_help=(
        ("stats", f"The measured cache to read (default {DEFAULT_STATS})."),
        ("label", "Which dataset in the cache to draw (default YTStream)."),
        ("ratio_labels", "Arrows labelling each series against the object count (default true)."),
        ("legend", "In-axes legend (default true)."),
        ("width", "Figure width in inches (default 3.3, one column)."),
        ("height", "Figure height in inches (default 2.0)."),
    ),
)
