# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "duckdb==1.5.6",
#     "matplotlib>=3.10",
#     "numpy>=2.1",
# ]
# ///
"""How still each clip is: the distribution of per-timestep motion.

The two clips in `plots/compression.png` sit at opposite ends of this, and
everything in that figure follows from the difference it measures. A step is
one object's movement between two consecutive frames, and the mass at zero is
the share of frames that repeat their predecessor exactly -- which is what the
deduplicated baseline drops and what a constant polynomial piece absorbs for
free. The intercept *is* the compression: a clip whose steps are zero a fraction
z of the time deduplicates by 1/(1-z).

    uv run scripts/motion_cdf.py

Boxes rather than centroids for the zero mass: deduplication compares all four
coordinates, and a box can resize around a stationary centre.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os

import duckdb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(os.environ.get("TRQS_DATA_DIR", Path(__file__).resolve().parents[2] / "datasets" / "processed"))

# label -> sub-dataset under the processed tree. DETRAC is read from the
# integer-pixel copy the figure measures, not the float original.
_CLIPS = {
    "VIRAT": "virat/VIRAT_S_010109_04_000487_000538",
    "DETRAC": "detrac_px/MVI_40191",
}

# Consecutive frames only: a detection gap is an absence, not a large step.
_SQL = """
WITH s AS (
    SELECT oid, fid, x1, y1, x2, y2,
           (x1 + x2) / 2.0 AS cx, (y1 + y2) / 2.0 AS cy
    FROM bbox
),
d AS (
    SELECT fid - lag(fid) OVER w AS df,
           cx - lag(cx) OVER w AS dcx,
           cy - lag(cy) OVER w AS dcy,
           (x1 = lag(x1) OVER w AND y1 = lag(y1) OVER w
            AND x2 = lag(x2) OVER w AND y2 = lag(y2) OVER w) AS same_box
    FROM s WINDOW w AS (PARTITION BY oid ORDER BY fid)
)
SELECT sqrt(dcx * dcx + dcy * dcy) AS step, same_box FROM d WHERE df = 1
"""


def steps(db: Path):
    """`(displacement, box-unchanged)` for every consecutive-frame step."""
    with duckdb.connect(str(db), read_only=True) as conn:
        rows = conn.execute(_SQL).fetchall()
    step = np.fromiter((r[0] for r in rows), dtype=float, count=len(rows))
    same = np.fromiter((bool(r[1]) for r in rows), dtype=bool, count=len(rows))
    return step, same


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=here / "plots" / "motion_cdf.pdf")
    # One panel of a three-panel row spanning both columns, beside
    # plots/compression.png; published at the size it is drawn.
    ap.add_argument("--width", type=float, default=1.95)
    ap.add_argument("--height", type=float, default=1.9)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(args.width, args.height), constrained_layout=True)
    colours = {"VIRAT": "#1b6ca8", "DETRAC": "#c1440e"}

    print(f"{'clip':<16}{'steps':>10}{'box held':>10}{'centroid 0':>12}"
          + "".join(f"{q:>8}" for q in ("p50", "p90", "p99", "max")) + f"{'dedup':>9}")
    for label, rel in _CLIPS.items():
        x, same = steps(_ROOT / rel / "duckdb.db")
        ok = np.isfinite(x)
        x, same = x[ok], same[ok]
        held = same.mean()
        qs = np.percentile(x, [50, 90, 99])
        print(f"{label:<16}{x.size:>10,}{100 * held:>9.1f}%{100 * (x == 0).mean():>11.1f}%"
              + "".join(f"{q:>8.2f}" for q in qs)
              + f"{x.max():>8.1f}{1 / (1 - held):>8.1f}x")

        # Symlog, so the mass at zero -- the whole point -- is on the axis rather
        # than dropped by a log scale. The curve therefore starts at its own
        # intercept, which is that mass.
        xs = np.sort(x)
        ys = np.arange(1, xs.size + 1) / xs.size
        ax.step(xs, ys, where="post", color=colours[label], linewidth=1.5, label=label)
        ax.plot([0], [(x == 0).mean()], marker="o", markersize=3.5,
                color=colours[label], zorder=4)
        ax.annotate(
            # One decimal below 1 %, where "0 %" would read as none at all
            # rather than as the three in a thousand that there are.
            f"{100 * (x == 0).mean():{'.1f' if (x == 0).mean() < 0.01 else '.0f'}}% unmoved",
            (0, (x == 0).mean()),
            textcoords="offset points",
            xytext=(8, -11 if label.startswith("VIRAT") else 7),
            fontsize=5.8,
            color=colours[label],
        )

    ax.set_xscale("symlog", linthresh=1.0)
    # A hair left of zero, so the marker on the intercept is not halved by the
    # spine. Symlog is linear through this region, so it costs no distortion.
    ax.set_xlim(-0.12, None)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("per-timestep centroid displacement (px)", fontsize=7, labelpad=2)
    ax.set_ylabel("fraction of steps", fontsize=7, labelpad=2)
    ax.tick_params(labelsize=6, pad=1.5)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=5.5, loc="lower right", frameon=True, framealpha=0.9,
              edgecolor="none", handlelength=1.2, borderpad=0.3, labelspacing=0.25)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"\n{args.out}")


if __name__ == "__main__":
    main()
