"""Recall against ground truth on detector output, as gap smoothing varies.

A tracker fragments: one vehicle becomes several tracks with holes between them,
and a hole breaks the co-appearance a multi-object query needs to see a match.
`max_gap` bridges those holes, so the engine can recover events fragmentation
would otherwise hide.

SQL has no equivalent notion — it reads the rows as they are — so its score is a
fixed baseline whatever `max_gap` the engine was given, and the same query over
human annotation is the ceiling. One bar group per query: the engine at each gap,
then those two references.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from benchplot.spec import Plot, PlotKind, Table

_PALETTE = plt.get_cmap("Blues")
_SQL_COLOUR = "#c1440e"
_GT_COLOUR = "#2e7d32"
_GROUP_WIDTH = 0.82


@dataclass(frozen=True)
class RecallOptions:
    """The `[[plots]]` keys only this kind understands."""

    metric: str = "recall"
    baseline: bool = True
    ceiling: bool = True
    value_labels: bool = True


def _mean(df: pd.DataFrame, col: str) -> float | None:
    vals = df[col].dropna()
    return float(vals.mean()) if len(vals) else None


def plot_recall(
    df: pd.DataFrame,
    out_path: Path,
    *,
    order: list[str] | None = None,
    options: RecallOptions,
    title: str | None = None,
) -> Path:
    """Writes the chart to `out_path` and returns it.

    `df` holds one row per `(query, err, scale)` after the usual selection; this
    kind additionally splits on `gap` and on whether the run read predictions or
    ground truth, so it is given the *un*collapsed rows and averages them itself.
    """
    metric = options.metric
    gt_col, sql_col = f"gt_{metric}", f"sql_gt_{metric}"
    for col in (gt_col, sql_col, "gap", "detector"):
        if col not in df.columns:
            raise ValueError(f"the database records no `{col}` — re-run `trql-cli bench --gt`")
    df = df[df[gt_col].notna()]
    if df.empty:
        raise ValueError(
            f"no run carries a ground-truth {metric}; `bench` needs `--gt` for this plot"
        )

    pred = df[df["detector"] != "GT"]
    truth = df[df["detector"] == "GT"]
    if pred.empty:
        raise ValueError("no run over detector output; every row is ground truth")

    queries = order or sorted(pred["query"].unique())
    gaps = sorted(pred["gap"].unique())
    shades = {g: _PALETTE(0.35 + 0.5 * i / max(len(gaps) - 1, 1)) for i, g in enumerate(gaps)}

    fig, ax = plt.subplots(figsize=(max(7.0, 2.1 * len(queries)), 4.6))
    drawn: list[float] = []

    for gi, query in enumerate(queries):
        qp = pred[pred["query"] == query]
        # The engine at each gap, then the two references, packed contiguously so
        # a query measured at fewer gaps gets wider bars rather than holes.
        bars: list[tuple[str, float | None, object, str | None]] = [
            (f"{g:g}", _mean(qp[qp["gap"] == g], gt_col), shades[g], None) for g in gaps
        ]
        if options.baseline:
            bars.append(("SQL", _mean(qp, sql_col), _SQL_COLOUR, "SQL on the same predictions"))
        if options.ceiling:
            tq = truth[truth["query"] == query]
            bars.append(("GT", _mean(tq, gt_col), _GT_COLOUR, "ground truth"))

        width = _GROUP_WIDTH / len(bars)
        left = gi - _GROUP_WIDTH / 2
        for bi, (_, value, colour, _) in enumerate(bars):
            if value is None:
                continue
            x = left + width * (bi + 0.5)
            drawn.append(value)
            ax.bar(x, value, width=width * 0.9, color=colour, edgecolor="black",
                   linewidth=0.4, zorder=3)
            if options.value_labels:
                ax.annotate(f"{value:.2f}", (x, value), textcoords="offset points",
                            xytext=(0, 2), ha="center", va="bottom", fontsize=6.5,
                            rotation=90, zorder=4)

    ax.set_xticks(range(len(queries)))
    ax.set_xticklabels(queries)
    ax.set_xlim(-0.6, len(queries) - 0.4)
    # Clipped to the bars actually drawn, not to the per-sequence values behind
    # them: these are means, and some individual sequences score 0, which would
    # drag the axis to the floor and flatten the differences the figure is about.
    lo = min(drawn) if drawn else 0.0
    ax.set_ylim(max(0.0, lo - (1.0 - lo) * 0.45), 1.04)
    ax.set_ylabel(f"{metric.capitalize()} vs ground-truth answer")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    if title:
        # Padded clear of the value labels, which stand upright above each bar and
        # otherwise run into it.
        ax.set_title(title, pad=16)

    handles = [Patch(facecolor=shades[g], edgecolor="black", linewidth=0.4,
                     label=f"TrQS, max_gap={g:g}") for g in gaps]
    if options.baseline:
        handles.append(Patch(facecolor=_SQL_COLOUR, edgecolor="black", linewidth=0.4,
                             label="DuckDB, same predictions"))
    if options.ceiling:
        handles.append(Patch(facecolor=_GT_COLOUR, edgecolor="black", linewidth=0.4,
                             label="DuckDB, ground truth"))
    ax.legend(handles=handles, fontsize=7, ncol=min(len(handles), 4),
              loc="upper center", bbox_to_anchor=(0.5, -0.09), frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _parse(table: Table) -> RecallOptions:
    return RecallOptions(
        metric=table.choice("metric", ("recall", "precision", "iou"), "recall"),
        baseline=table.boolean("baseline", True),
        ceiling=table.boolean("ceiling", True),
        value_labels=table.boolean("value_labels", True),
    )


def _render(points: pd.DataFrame, plot: Plot) -> Path:
    return plot_recall(points, plot.out, order=plot.order, options=plot.options,
                       title=plot.title)


KIND = PlotKind(
    name="recall",
    summary=(
        "Accuracy against ground truth on detector output, one bar per max_gap, "
        "against SQL on the same predictions and on ground truth."
    ),
    parse=_parse,
    render=_render,
    option_help=(
        ("metric", "recall (default), precision, or iou."),
        ("baseline", "Bar for SQL over the same predictions (default true)."),
        ("ceiling", "Bar for the same query over ground truth (default true)."),
        ("value_labels", "Print each bar's value above it (default true)."),
    ),
)
