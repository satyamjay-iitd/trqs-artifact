"""What the gap parameter buys on real detector and tracker output.

`delta_gap` is the largest hole in a trajectory that ingestion merges into one
segment. It is inert on annotated data, where an object is labelled in every
frame it is visible, so this reads the runs over predicted tracks (YOLO11 with
ByteTrack on DETRAC) scored against the human annotations -- the `gt_oid_*`
columns -- rather than against the reference SQL.

The frame-based baseline over the same tracks has no trajectory to bridge, so
its figures are the `delta_gap = 1` values whatever the parameter is set to;
they are drawn as the dotted rules the recall curves climb away from.

    uv run python scripts/gap_sweep.py
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

# Query id -> (label, colour). Two queries, so recall and precision can share a
# panel: the two quantities sit 15 points apart and never cross.
_QUERIES = {
    "DETRAC-approach": ("approach", "#1f77b4"),
    "DETRAC-dwell": ("dwell", "#d62728"),
}

_SQL = """
SELECT q.name AS query, r.trql_gap AS gap,
       avg(r.gt_oid_recall) * 100 AS recall,
       avg(r.gt_oid_precision) * 100 AS precision,
       count(*) AS runs
FROM Run r
JOIN Query q ON q.id = r.query_id
JOIN Dataset d ON d.id = q.dataset_id
WHERE d.name LIKE 'predicted/detrac%'
  AND r.trql_error_px = 0
  AND r.gt_oid_recall IS NOT NULL
  AND q.name IN ({placeholders})
GROUP BY q.name, r.trql_gap
ORDER BY q.name, r.trql_gap
"""


def load(db: Path) -> pd.DataFrame:
    names = list(_QUERIES)
    sql = _SQL.format(placeholders=",".join("?" * len(names)))
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        return pd.read_sql_query(sql, conn, params=names)


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=here.parent / "bench.db")
    ap.add_argument("--out", type=Path, default=here / "plots" / "gap_sweep.pdf")
    args = ap.parse_args()

    df = load(args.db)
    if df.empty:
        raise SystemExit(f"no gap sweep in {args.db}")

    gaps = sorted(df["gap"].unique())
    base, rest = gaps[0], gaps[1:]

    # Plotted as the change from the tightest setting, in points, rather than as
    # the levels: the whole effect is a couple of points on an 80--90 % recall,
    # which a bar from zero would hide and a truncated axis would exaggerate.
    # Zero is then also exactly where the frame-based baseline sits, since it has
    # no trajectory to bridge and does not move with the parameter.
    series = []
    for name, (label, colour) in _QUERIES.items():
        d = df[df["query"] == name].set_index("gap").reindex(gaps)
        for metric, hatch in (("recall", None), ("precision", "///")):
            delta = (d[metric] - d[metric].loc[base]).loc[rest]
            series.append((f"{label} {metric}", colour, hatch, delta.tolist()))

    fig, ax = plt.subplots(figsize=(3.4, 1.9), constrained_layout=True)
    width = 0.8 / len(series)
    for i, (label, colour, hatch, vals) in enumerate(series):
        xs = [g + (i - (len(series) - 1) / 2) * width for g in range(len(rest))]
        ax.bar(xs, vals, width=width * 0.92, label=label, color=colour,
               edgecolor="black", linewidth=0.4, hatch=hatch, zorder=3)

    ax.axhline(0, color="black", linewidth=0.8, zorder=4)
    # Headroom for the legend, which sits inside: the bars reach 2.4 points, so
    # a band above them costs nothing and keeps the panel one rectangle.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.30)
    ax.set_xticks(range(len(rest)))
    ax.set_xticklabels([f"{g:g}" for g in rest])
    ax.set_xlabel(r"$\delta_{\mathrm{gap}}$", fontsize=7, labelpad=2)
    ax.set_ylabel("change vs $\\delta_{\\mathrm{gap}}{=}1$ (pts)", fontsize=7, labelpad=2)
    ax.tick_params(labelsize=6, pad=1.5)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=5.5, ncol=2, loc="upper left", frameon=True,
              framealpha=0.9, edgecolor="none", handlelength=1.4,
              borderpad=0.3, labelspacing=0.2, columnspacing=1.0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(args.out)
    for name, (label, _) in _QUERIES.items():
        d = df[df["query"] == name].set_index("gap").reindex(gaps)
        print(f"  {label:<9} recall {d['recall'].iloc[0]:.1f} -> {d['recall'].max():.1f}"
              f"   precision {d['precision'].iloc[0]:.1f} -> {d['precision'].iloc[-1]:.1f}"
              f"   ({int(d['runs'].iloc[0])} clips)")


if __name__ == "__main__":
    main()
