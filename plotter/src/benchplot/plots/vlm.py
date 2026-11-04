"""What a second query costs each system, and how well each answers.

Two queries over one aerial video, both asking for *(object, interval)* pairs:
`Q1` is every car turning right out of the queue lane, `Q2` every car that
enters from the top of the frame and turns right. They select disjoint sets of
cars, so they are genuinely different questions of the same footage.

**(a) Cumulative cost against number of queries.** The engine pays detection
and tracking once -- $49.9$ s with the largest YOLO11 checkpoint -- and then answers
in milliseconds, so its line is flat. The model has no persistent state to
re-use: it must re-read the video for every question, so its line is linear in
the number of queries. One query cannot tell those two shapes apart, which is
why there are two. `k = 1, 2` are measured and drawn solid; beyond that the line
is the mean measured per-query cost extrapolated, and is drawn dashed because an
extrapolation is not a measurement.

**(b) F1 per query.** The accuracy side, kept so the cost panel is not read on
its own. The engine's F1 is 1.0 **by construction** -- the TrQL region query defines
the event being scored -- so its bars report expressibility, not an accuracy win.
Q2's manoeuvre is shorter and in a corner of the frame, so its F1 is not
comparable with Q1's; neither panel's claim rests on comparing them.

Reads `results-vlm-*.json` and `ingest.json` from `suite/vlm/`,
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
import pandas as pd

from benchplot.spec import Plot, PlotKind, Table

DEFAULT_DIR = Path("data/vlm")

# Baselines warm, the engine cool, matching the other figures: the reader sees
# "model against engine" before reading a label.
_C_STATIC = "#c1440e"
_C_AGENTIC = "#eb6834"
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
    "pdf.fonttype": 42,
}

#: Per window: the `ingest.json` cell, and `(results file, label)` for each
#: query in the order it was asked. The 30 s window has only one query and only
#: five ground-truth turns -- recall quantises into steps of 0.2 -- so it cannot
#: carry the figure; it stays in the README and in the length sweep below.
_WINDOWS = {
    "600s": ("600s", (("results-vlm-600s.json", "Q1"),
                      ("results-vlm-top-600s.json", "Q2"))),
    "900s": ("900s", (("results-vlm-900s.json", "Q1"),
                      ("results-vlm-top-900s.json", "Q2"))),
}
#: `(ingest cell, Q1 results file, label)` for the length sweep, ascending.
_LENGTHS = (("30s", "results-vlm.json", "30 s"),
            ("600s", "results-vlm-600s.json", "10 min"),
            ("900s", "results-vlm-900s.json", "15 min"))
#: How far to project the cumulative-cost panel.
_KMAX = 10


@dataclass(frozen=True)
class VlmOptions:
    """The `[[plots]]` keys only this kind understands."""

    #: Directory holding results-vlm*.json and ingest.json.
    source: Path = DEFAULT_DIR
    #: Which window the left panel is drawn for.
    window: str = "900s"
    #: What the right panel shows: `accuracy` is F1 per query on that window,
    #: `length` is the cost of one query against how much video it covers.
    right_panel: str = "accuracy"
    #: Charge the engine its ingest cost (detection + tracking) as well as the
    #: queries. On by default: with it off the engine's line starts at zero and
    #: the figure stops being a comparison of what each system actually spends.
    with_ingest: bool = True
    #: Largest query count on the cost panel.
    kmax: int = _KMAX
    legend: bool = True
    width: float = 3.3
    height: float = 1.7


def _f1(tp: int, fp: int, fn: int) -> float:
    """F1 from counts, so it does not inherit the rounding in the stored rates."""
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 0.0


def ingest_cells(source: Path) -> dict:
    """`ingest.json` keyed by cell name."""
    path = source / "ingest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no ingest measurements at {path} — run "
            ".venv-detect/bin/python suite/vlm/ingest_cost.py"
        )
    return {c["name"]: c for c in json.loads(path.read_text())["cells"]}


def load_queries(source: Path, with_ingest: bool, window: str = "900s") -> dict:
    """Per-query cost and F1 for each system, plus the one-off ingest cost."""
    if window not in _WINDOWS:
        raise ValueError(
            f"unknown window {window!r}; known: {', '.join(_WINDOWS)}")
    cell, queries = _WINDOWS[window]
    ingest = 0.0
    if with_ingest:
        cells = ingest_cells(source)
        if cell not in cells:
            raise ValueError(f"ingest.json has no cell named {cell!r}")
        ingest = cells[cell]["ingest_s"]

    labels: list[str] = []
    cost: dict[str, list[float]] = {"static": [], "agentic": [], "trqs": []}
    f1: dict[str, list[float]] = {"static": [], "agentic": [], "trqs": []}
    for fname, label in queries:
        path = source / fname
        if not path.is_file():
            raise FileNotFoundError(
                f"no VLM measurements at {path} — run "
                "suite/vlm/bench_gemini.py"
            )
        res = json.loads(path.read_text())
        n_gt = len(res["ground_truth"]["turns"])
        labels.append(label)
        seen = set()
        for run in res["runs"]:
            s = run.get("score")
            if s is None:
                raise ValueError(
                    f"{path}: the {run['mode']} run has no score "
                    f"({run.get('parse_error') or run.get('error', 'failed')})"
                )
            tp, fp = s["true_positives"], s["false_positives"]
            cost[run["mode"]].append(run["latency_ms"] / 1000.0)
            f1[run["mode"]].append(_f1(tp, fp, n_gt - tp))
            seen.add(run["mode"])
        missing = {"static", "agentic"} - seen
        if missing:
            raise ValueError(f"{path}: no run for {', '.join(sorted(missing))}")
        # Exact by construction: the query defines the event, so tp = n_gt.
        cost["trqs"].append(res["trqs_ms"] / 1000.0)
        f1["trqs"].append(1.0)
    return {"labels": labels, "cost": cost, "f1": f1, "ingest": ingest}


def load_lengths(source: Path, with_ingest: bool) -> dict:
    """Cost of answering Q1 once, against how much video it is asked about."""
    cells = ingest_cells(source) if with_ingest else {}
    out: dict[str, list] = {"labels": [], "seconds": [],
                            "static": [], "agentic": [], "trqs": []}
    for cell, fname, label in _LENGTHS:
        path = source / fname
        if not path.is_file():
            raise FileNotFoundError(f"no VLM measurements at {path}")
        res = json.loads(path.read_text())
        w = res["ground_truth"]["window_s"]
        out["labels"].append(label)
        out["seconds"].append(w[1] - w[0])
        for run in res["runs"]:
            out[run["mode"]].append(run["latency_ms"] / 1000.0)
        # The engine's first query over a video costs ingestion plus the query.
        out["trqs"].append(res["trqs_ms"] / 1000.0
                           + (cells[cell]["ingest_s"] if with_ingest else 0.0))
    return out


def cumulative(per_query: list[float], ingest: float, kmax: int) -> list[float]:
    """Total spend after k queries, extrapolating past what was measured.

    Beyond the measured queries the mean per-query cost is used. The caller
    draws that stretch dashed: it is an extrapolation, not a measurement.
    """
    mean = sum(per_query) / len(per_query)
    total, out = ingest, []
    for k in range(1, kmax + 1):
        total += per_query[k - 1] if k <= len(per_query) else mean
        out.append(total)
    return out


_SERIES = (("static", _C_STATIC, "o", "Gemini, static"),
           ("agentic", _C_AGENTIC, "s", "Gemini, agentic"),
           ("trqs", _C_TRQS, "D", "TrQS"))


def panel_cost(ax, data: dict, options: VlmOptions, scale: float) -> None:
    """(a) Cumulative seconds against number of queries."""
    n_meas = len(data["labels"])
    ks = list(range(1, options.kmax + 1))
    for system, colour, marker, _ in _SERIES:
        ing = data["ingest"] if system == "trqs" else 0.0
        ys = cumulative(data["cost"][system], ing, options.kmax)
        lw = 1.6 if system == "trqs" else 1.3
        # Measured stretch solid, extrapolation dashed, joined at k = n_meas so
        # the line reads as one series rather than two.
        ax.plot(ks[:n_meas], ys[:n_meas], color=colour, lw=lw,
                marker=marker, ms=3.2 * scale, zorder=3)
        ax.plot(ks[n_meas - 1:], ys[n_meas - 1:], color=colour, lw=lw,
                ls=(0, (3, 2)), zorder=2)

    ax.set_xlabel("Queries asked", fontsize=7 * scale, labelpad=2)
    ax.set_ylabel("Cumulative time (s)", fontsize=7 * scale, labelpad=2)
    ax.set_xlim(0.6, options.kmax + 0.4)
    ax.set_xticks([1, 2] + list(range(4, options.kmax + 1, 2)))
    ax.set_ylim(0, None)
    ax.tick_params(labelsize=6.5 * scale, pad=1.5)
    ax.grid(True, which="major", axis="both")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # Name the shape rather than make the reader infer it from a flat line.
    eng = cumulative(data["cost"]["trqs"], data["ingest"], options.kmax)
    # Below the engine's line, not above it: everything above is occupied by the
    # models' lines, and the band under a flat line is the one clear space here.
    ax.annotate("ingest once, then ms/query",
                xy=(options.kmax * 0.56, eng[-1]),
                xytext=(0, -4), textcoords="offset points",
                ha="center", va="top", fontsize=5.6 * scale, color=_C_TRQS)


def panel_f1(ax, data: dict, options: VlmOptions, scale: float) -> None:
    """(b) F1 per query, grouped by system."""
    labels = data["labels"]
    x = range(len(labels))
    w = 0.26
    for i, (system, colour, _, label) in enumerate(_SERIES):
        xs = [v + (i - 1) * w for v in x]
        ys = data["f1"][system]
        ax.bar(xs, ys, w, color=colour, label=label, zorder=3, linewidth=0)
        # A zero bar has no height, so without this it is indistinguishable
        # from a missing measurement -- and on Q2 both models really do score 0.
        for xi, yi in zip(xs, ys):
            if yi == 0.0:
                ax.annotate("0", (xi, 0), textcoords="offset points",
                            xytext=(0, 2), ha="center", va="bottom",
                            fontsize=5.8 * scale, color=colour, zorder=4)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("$F_1$", fontsize=7 * scale, labelpad=2)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(labelsize=6.5 * scale, pad=1.5)
    ax.grid(True, which="major", axis="y")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def panel_length(ax, data: dict, options: VlmOptions, scale: float) -> None:
    """(b, alternative) Cost of answering once, against how much video it covers."""
    xs = data["seconds"]
    for system, colour, marker, _ in _SERIES:
        ax.plot(xs, data[system], color=colour,
                lw=1.6 if system == "trqs" else 1.3,
                marker=marker, ms=3.2 * scale, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Video length (s)", fontsize=7 * scale, labelpad=2)
    ax.set_ylabel("Time, 1 query (s)", fontsize=7 * scale, labelpad=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(data["labels"])
    ax.tick_params(labelsize=6.5 * scale, pad=1.5)
    ax.grid(True, which="major", axis="both")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_vlm(out_path: Path, *, options: VlmOptions, title: str | None = None) -> Path:
    """Writes the chart to `out_path` and returns it."""
    data = load_queries(options.source, options.with_ingest, options.window)
    scale = max(min(options.width / 3.3, 1.0), 0.85)

    with plt.rc_context(_RC):
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(options.width, options.height), layout="constrained",
            gridspec_kw={"width_ratios": [1.5, 1.0]})
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.06, hspace=0.0)
        panel_cost(ax1, data, options, scale)
        ax1.set_title("(a) cost of $k$ queries", fontsize=6.8 * scale, pad=3)
        if options.right_panel == "length":
            panel_length(ax2, load_lengths(options.source, options.with_ingest),
                         options, scale)
            ax2.set_title("(b) one query vs length", fontsize=6.8 * scale, pad=3)
        else:
            panel_f1(ax2, data, options, scale)
            ax2.set_title("(b) accuracy", fontsize=6.8 * scale, pad=3)
        if title:
            fig.suptitle(title, fontsize=7.5 * scale)
        if options.legend:
            from matplotlib.lines import Line2D
            handles = [Line2D([], [], color=c, marker=m, ms=3.2 * scale,
                              lw=1.4, label=lab)
                       for _, c, m, lab in _SERIES]
            fig.legend(handles, [h.get_label() for h in handles], loc="lower center", ncol=3, frameon=False,
                       handlelength=1.2, borderpad=0.1, labelspacing=0.2,
                       columnspacing=1.2, handletextpad=0.4,
                       bbox_to_anchor=(0.5, -0.09), fontsize=6.5 * scale)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _parse(table: Table) -> VlmOptions:
    return VlmOptions(
        source=table.path("source", default=table.base / DEFAULT_DIR) or DEFAULT_DIR,
        with_ingest=table.boolean("with_ingest", True),
        window=table.choice("window", tuple(_WINDOWS), "900s"),
        right_panel=table.choice("right_panel", ("accuracy", "length"), "accuracy"),
        kmax=int(table.number("kmax", _KMAX) or _KMAX),
        legend=table.boolean("legend", True),
        width=table.number("width", 3.3) or 3.3,
        height=table.number("height", 1.7) or 1.7,
    )


def _render(points: pd.DataFrame, plot: Plot) -> Path:
    """`points` is empty by design — see `needs_runs` on the kind below."""
    return plot_vlm(plot.out, options=plot.options, title=plot.title)


def _describe(plot: Plot) -> str:
    options: VlmOptions = plot.options
    return (f"{options.window} window, k={options.kmax}, "
            f"{'with' if options.with_ingest else 'without'} ingest, "
            f"from {options.source.name}/")


KIND = PlotKind(
    name="vlm",
    summary=(
        "Cumulative cost against query count, and F1 per query, for Gemini "
        "(static and agentic) and the engine. Reads results-vlm-*.json, not "
        "bench.db."
    ),
    parse=_parse,
    render=_render,
    needs_runs=False,
    describe=_describe,
    option_help=(
        ("source", f"Directory of the VLM measurements (default {DEFAULT_DIR})."),
        ("with_ingest", "Charge the engine detection + tracking too (default true)."),
        ("window", f"Which window to draw: {', '.join(_WINDOWS)} (default 900s)."),
        ("right_panel", "`accuracy` (default) or `length` — see the module docstring."),
        ("kmax", f"Largest query count on the cost panel (default {_KMAX})."),
        ("legend", "Figure legend below the panels (default true)."),
        ("width", "Figure width in inches (default 3.3, one column)."),
        ("height", "Figure height in inches (default 1.7)."),
    ),
)
