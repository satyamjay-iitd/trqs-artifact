"""5-tuple vs naive interval sets: where Q3's time goes under each.

The interval-set representation is a compile-time choice: the default 5-tuple
`(S, E, c, L, R)`, or `--features naive-intervals`, which materialises every
`(start, end)` pair the query admits. Both return the same answer, so the
comparison is time alone. Each bar is one representation, stacked by the stages
of the instrumented run: root-finding (with the segment sweep around it, as in
`degree_ablation.py`), temporal (the THEN/CONTAINS/OVERLAP/DURATION algebra plus lifting
solved frames into interval sets -- the two the representation changes), and
everything else (candidate selection, planning, residual).

The stacks are each stage's share of its own query, with the total time printed
on top. The two totals are four orders of magnitude apart, so stacking absolute
times fails both ways: on a linear axis the 5-tuple bar vanishes, and on a log
axis a segment's height depends on what sits under it.

The inputs are `trql-cli bench --format json` reports in `data/interval-ablation/`,
from `--features instrument` builds, so absolute times carry the timing scopes'
overhead; the proportions are what is meaningful.

    uv run python scripts/interval_ablation.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

_HERE = Path(__file__).resolve().parent.parent

_APPROACHES = (("5tuple", "5-tuple"), ("naive", "Naive"))

# Names and colours follow `degree_ablation.py`'s stages: the two figures share a
# row in the paper, so they read alike.
_STAGES = (
    ("root-finding", "#dd8452"),
    ("temporal", "#55a868"),
    ("other", "#b0b0b0"),
)


def stages(path: Path) -> tuple[float, list[float]]:
    """Mean query time, and root-finding, temporal, and other time (ms), from one
    bench report."""
    report = json.loads(path.read_text())
    s = report["stages"]
    return report["avg_ms"], [
        s["poly_ms"],
        s["temporal_ms"] + s["lift_ms"],
        s["selection_ms"] + s["other_ms"],
    ]


def fmt_ms(v: float) -> str:
    if v >= 1000:
        return f"{v / 1000:.1f}s"
    return f"{v:.1f}ms" if v < 10 else f"{v:.0f}ms"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", type=Path, default=_HERE / "data" / "interval-ablation")
    ap.add_argument("--query", default="CL-overtake")
    ap.add_argument("--out", type=Path, default=_HERE / "plots" / "interval_ablation.pdf")
    args = ap.parse_args()

    # Sits beside the degree figure (`degree_ablation.py`), and the paper includes
    # both at one height. Its frame is placed to match that figure's stack of
    # three: the same total height (2.217 in, tight), 1.843 in of axes, and
    # 0.267 in below them for tick and axis labels -- so the frames line up
    # top and bottom, and the type is the same size. Measured from that
    # figure; re-measure if its layout changes.
    # Width is what is left of the column beside it.
    width, height = 1.42, 2.217
    left, below, axes_h, axes_w = 0.44, 0.267, 1.843, 0.95
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((left / width, below / height, axes_w / width, axes_h / height))
    for g, (key, _) in enumerate(_APPROACHES):
        avg_ms, vals = stages(args.data_dir / f"{args.query}-{key}.json")
        total = sum(vals)
        bottom = 0.0
        for (label, colour), v in zip(_STAGES, vals):
            pct = 100 * v / total
            ax.bar(g, pct, bottom=bottom, width=0.6, color=colour, edgecolor="black",
                   linewidth=0.2, label=label if g == 0 else None, zorder=3)
            # Only segments tall enough to hold their number get one.
            if pct >= 8:
                ax.text(g, bottom + pct / 2, f"{pct:.0f}%", ha="center", va="center",
                        fontsize=5.5, color="black", zorder=4)
            bottom += pct
        # The measured query time, not the stage sum: the stages give the split.
        ax.annotate(fmt_ms(avg_ms), (g, 100), xytext=(0, 1.5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6, fontweight="bold", zorder=4)

    ax.set_xticks(range(len(_APPROACHES)), [name for _, name in _APPROACHES])
    ax.tick_params(axis="x", length=0, labelsize=6.5, pad=2)
    ax.tick_params(axis="y", labelsize=5.5, pad=1.5)
    # Headroom above 100% for the totals, and above them the legend.
    ax.set_ylim(0, 138)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylabel("Share of query time", fontsize=6.5, labelpad=2)
    ax.set_xlim(-0.6, len(_APPROACHES) - 0.4)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    # Inside the frame, in the headroom above the 5-tuple's total: one column,
    # as the panel is too narrow for the three entries in a row.
    ax.legend(fontsize=5.5, ncol=1, loc="upper left", frameon=False, handlelength=0.8,
              handletextpad=0.3, labelspacing=0.15, borderaxespad=0.25)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Not "tight": the fixed canvas is what keeps the frame aligned.
    fig.savefig(args.out)
    fig.savefig(args.out.with_suffix(".png"), dpi=300)
    print(args.out)


if __name__ == "__main__":
    main()
