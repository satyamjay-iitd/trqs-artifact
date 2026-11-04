"""Cost of a first and a second query, at two video lengths.

One panel per window. Within a panel, one group per system and two bars per
group: what the **first** query over that video cost, and what the **second**
one cost. The engine's two bars are the whole argument -- the first carries
ingestion, the second is the query alone, four orders of magnitude smaller --
while each model's two bars are the same height, because a prompt has nothing
to re-use and re-reads the video every time.

The y axis is logarithmic and has to be: the engine's second query is
milliseconds against the models' tens of seconds, and on a linear axis that bar
would be invisible, which is the one thing the figure must show.

F1, computed from precision and recall, is drawn on a linear right axis.
The engine's F1 is `1.00` by construction: the TrQL region query defines the
event being scored, so it reports expressibility rather than an accuracy win.

Reads `results-vlm*.json` and `ingest.json` from `suite/vlm/`,
not `bench.db`.
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

DEFAULT_DIR = Path("data/vlm")

# System colours from tab20c, repeated for LQ and RQ.
_C_STATIC = ("#e6550d", "#e6550d")
_C_AGENTIC = ("#fd8d3c", "#fd8d3c")
_C_TRQS = (plt.get_cmap("tab20")(0), ) * 2 
_C_F1 = "#303030"
_INK = "#000000"
_INK_MUTED = "#000000"

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
    "pdf.fonttype": 42,
    "hatch.linewidth": 0.5,
}

#: `(panel label, ingest cell, Q1 results, Q2 results)`, shortest window first.
_CELLS = (("30 s", "30s", "results-vlm.json", "results-vlm-top.json"),
          ("10 min", "600s", "results-vlm-600s.json", "results-vlm-top-600s.json"),
          # ("15 min", "900s", "results-vlm-900s.json", "results-vlm-top-900s.json"),
          )

#: `(key, colour pair, system label)` in the order the groups are drawn.
_SERIES = (("trqs", _C_TRQS, "TrQS"),
           ("static", _C_STATIC, "Gemini, static"),
           ("agentic", _C_AGENTIC, "Gemini, agentic"))


@dataclass(frozen=True)
class VlmBarsOptions:
    """The `[[plots]]` keys only this kind understands."""

    #: Directory holding results-vlm*.json and ingest.json.
    source: Path = DEFAULT_DIR
    #: Charge the engine its ingestion on the first query. On by default: with
    #: it off the first bar is milliseconds too and the figure says nothing.
    with_ingest: bool = True
    #: Print time above each bar.
    annotate: bool = True
    legend: bool = True
    width: float = 7.0
    height: float = 2.1


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"no VLM measurements at {path} — run "
            "suite/vlm/bench_gemini.py"
        )
    return json.loads(path.read_text())


def load_cells(source: Path, with_ingest: bool) -> list[dict]:
    """Per window: time, precision and recall of query 1 and query 2."""
    ingest_path = source / "ingest.json"
    if not ingest_path.is_file():
        raise FileNotFoundError(
            f"no ingest measurements at {ingest_path} — run "
            ".venv-detect/bin/python suite/vlm/ingest_cost.py"
        )
    cells = {c["name"]: c for c in json.loads(ingest_path.read_text())["cells"]}

    out = []
    for label, cell, f1, f2 in _CELLS:
        if cell not in cells:
            raise ValueError(f"{ingest_path}: no cell named {cell!r}")
        ingest = cells[cell]["ingest_s"] if with_ingest else 0.0
        panel: dict[str, Any] = {"label": label, "ingest": ingest}
        for key in ("trqs", "static", "agentic"):
            panel[key] = []
        for i, fname in enumerate((f1, f2)):
            res = _read(source / fname)
            seen = set()
            for run in res["runs"]:
                s = run.get("score")
                if s is None:
                    raise ValueError(
                        f"{fname}: the {run['mode']} run has no score "
                        f"({run.get('parse_error') or run.get('error', 'failed')})"
                    )
                panel[run["mode"]].append(
                    {"t": run["latency_ms"] / 1000.0,
                     "p": s["precision"], "r": s["recall"]})
                seen.add(run["mode"])
            missing = {"static", "agentic"} - seen
            if missing:
                raise ValueError(f"{fname}: no run for {', '.join(sorted(missing))}")
            # Exact by construction, and only the first query pays ingestion.
            panel["trqs"].append({"t": res["trqs_ms"] / 1000.0 + (ingest if i == 0 else 0.0),
                                  "p": 1.0, "r": 1.0})
        out.append(panel)
    return out


def _fmt(v: float) -> str:
    """Seconds, at a width that fits over a bar."""
    if v < 0.01:
        return f"{v * 1000:.0f}ms"
    if v < 10:
        return f"{v:.1f}s"
    return f"{v:.0f}s"


def panel(ax, cell: dict, options: VlmBarsOptions, scale: float, top: float,
          *, f1_label: bool = True) -> None:
    w = 0.34
    group_spacing = 0.82
    for gi, (key, colour, _) in enumerate(_SERIES):
        recs = cell[key]
        for qi, rec in enumerate(recs):
            x = gi * group_spacing + (qi - 0.5) * w
            ax.bar(x, rec["t"], w * 0.9, color=colour[qi], zorder=3,
                   linewidth=0, edgecolor="none")
            if not options.annotate:
                continue
            ax.annotate(_fmt(rec["t"]),
                        (x, rec["t"]), textcoords="offset points", xytext=(0, 2),
                        ha="center", va="bottom", fontsize=6.5,
                        color=_INK, linespacing=0.95)

    ax.set_yscale("log")
    ax.set_ylim(0.0015, top)
    bar_xs = [gi * group_spacing + (qi - 0.5) * w
              for gi in range(len(_SERIES)) for qi in range(2)]
    ax.set_xticks(bar_xs)
    ax.set_xticklabels(["LQ", "RQ"] * len(_SERIES))
    for gi, name in enumerate(("TrQS", "Static", "Agentic")):
        ax.annotate(name, (gi * group_spacing, 0),
                    xycoords=ax.get_xaxis_transform(), xytext=(0, -13),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=10 * scale, color=_INK_MUTED)
    ax.tick_params(labelsize=10 * scale, pad=1.5)
    ax.grid(True, which="major", axis="y")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    f1_ax = ax.twinx()
    xs, scores = [], []
    for gi, (key, _, _) in enumerate(_SERIES):
        for qi, rec in enumerate(cell[key]):
            p, r = rec["p"], rec["r"]
            scores.append(2 * p * r / (p + r) if p + r else 0.0)
            xs.append(gi * group_spacing + (qi - 0.5) * w)
    f1_ax.plot(xs, scores, color=_C_F1, marker="o",
               markersize=3, linewidth=0.9, markerfacecolor="white")
    f1_ax.set_ylim(-0.05, 1.1)
    f1_ax.set_yticks([0, 0.5, 1])
    if f1_label:
        f1_ax.set_ylabel("F1", fontsize=10 * scale, rotation=90, labelpad=6,
                         va="center")
    f1_ax.tick_params(labelsize=10 * scale, pad=1.5)
    f1_ax.grid(False)
    for side in ("top", "left", "bottom"):
        f1_ax.spines[side].set_visible(False)


def plot_vlmbars(out_path: Path, *, options: VlmBarsOptions,
                 title: str | None = None) -> Path:
    """Writes the chart to `out_path` and returns it."""
    cells = load_cells(options.source, options.with_ingest)
    scale = max(min(options.width / 7.0, 1.0), 0.85)
    # One shared ceiling, with room for the time label over the tallest
    # bar: panels at different scales would invite comparing bar heights across
    # windows, which is not what the figure is for.
    hi = max(r["t"] for c in cells for k in ("trqs", "static", "agentic") for r in c[k])
    top = hi * 14

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, len(cells), figsize=(options.width, options.height),
                                 layout="constrained", sharey=True)
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.0)
        for i, (ax, cell) in enumerate(zip(axes, cells)):
            panel(ax, cell, options, scale, top, f1_label=i == len(cells) - 1)
            ax.text(0.5, 0.98, cell["label"], transform=ax.transAxes,
                    fontsize=11, ha="center", va="top", color=_INK)
        time_label = axes[0].set_ylabel("Time (s, log)", fontsize=10 * scale, labelpad=2)
        # Copy the resolved font, including family, size, style and weight.
        axes[-1].figure.axes[-1].yaxis.label.set_fontproperties(
            time_label.get_fontproperties().copy())
        axes[-1].figure.axes[-1].yaxis.label.set_fontsize(10 * scale)
        if title:
            fig.suptitle(title, fontsize=7.5 * scale)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Text bounding boxes include font bearings beyond the visible glyphs.
        # Trim that extra space at the right of F1 using the rendered ink;
        # only the crop is measured from pixels, the saved PDF stays vector.
        fig.canvas.draw()
        crop = fig.get_tightbbox(fig.canvas.get_renderer()).frozen()
        pixels = np.asarray(fig.canvas.buffer_rgba())
        ink_columns = np.flatnonzero(np.any(pixels[:, :, :3] < 250, axis=(0, 2)))
        if ink_columns.size:
            crop.x1 = min(crop.x1, (ink_columns[-1] + 1) / fig.dpi)
        fig.savefig(out_path, bbox_inches=crop, pad_inches=0)
    plt.close(fig)
    return out_path


def _parse(table: Table) -> VlmBarsOptions:
    return VlmBarsOptions(
        source=table.path("source", default=table.base / DEFAULT_DIR) or DEFAULT_DIR,
        with_ingest=table.boolean("with_ingest", True),
        annotate=table.boolean("annotate", True),
        legend=table.boolean("legend", True),
        width=table.number("width", 7.0) or 7.0,
        height=table.number("height", 2.1) or 2.1,
    )


def _render(points: pd.DataFrame, plot: Plot) -> Path:
    """`points` is empty by design — see `needs_runs` on the kind below."""
    return plot_vlmbars(plot.out, options=plot.options, title=plot.title)


def _describe(plot: Plot) -> str:
    options: VlmBarsOptions = plot.options
    return (f"{len(_CELLS)} windows x 3 systems x 2 queries, "
            f"{'with' if options.with_ingest else 'without'} ingest")


KIND = PlotKind(
    name="vlmbars",
    summary=(
        "First- and second-query time for the engine and the two Gemini modes, "
        "at two video lengths, with F1 lines on the right axis. "
        "Reads results-vlm*.json, not bench.db."
    ),
    parse=_parse,
    render=_render,
    needs_runs=False,
    describe=_describe,
    option_help=(
        ("source", f"Directory of the VLM measurements (default {DEFAULT_DIR})."),
        ("with_ingest", "Charge the engine ingestion on query 1 (default true)."),
        ("annotate", "Print the time over each bar (default true)."),
        ("legend", "Figure legend below the panels (default true)."),
        ("width", "Figure width in inches (default 7.0, two columns)."),
        ("height", "Figure height in inches (default 2.1)."),
    ),
)
