# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "duckdb==1.5.6",
# ]
# ///
"""How far an object moves between one frame and the next, per dataset.

The fitting budget `max_error_px` is in whatever units a dataset stores its
coordinates in, which makes a budget incomparable across datasets. This measures
the natural scale to read it against: the distribution of per-timestep centroid
displacement, in those same units, over every object.

Objects are sampled per dataset -- I24-Motion alone holds 280 M boxes -- and only
consecutive frames are used, so a detection gap never shows up as one large step.

    uv run python scripts/displacement_cdf.py
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

# label -> (path under the processed tree, units, objects to sample)
_DATASETS = {
    "I24-Motion": ("i24_motion/post1", "centi-feet", 2000),
    "nuScenes": ("nuscenes_maps/boston-seaport", "cm of map", 4000),
    "SoccerTrack": ("merged/SoccerTrackPano-117092-1st+19more", "px", 522),
    "Stanford Drone": ("merged/bookstore_video0+59more", "px", 4000),
    "CLEVRER": ("clevrer_merged/video_00000+9999more", "px", 4000),
    "VIRAT": ("virat/VIRAT_S_000001", "px", 4000),
    "DETRAC": ("detrac/MVI_20011", "px", 4000),
    "YTStream": ("ytstream/shibuya_0", "px", 4000),
}

# Sampling the objects, not the rows: a step is only defined between two
# consecutive frames of one object, so thinning rows would destroy the quantity.
_SQL = """
WITH s AS (
    SELECT oid, fid, (x1 + x2) / 2.0 AS cx, (y1 + y2) / 2.0 AS cy
    FROM bbox
    WHERE oid IN (SELECT oid FROM (SELECT DISTINCT oid FROM bbox) USING SAMPLE {n} ROWS)
),
d AS (
    SELECT fid - lag(fid) OVER w AS df,
           cx - lag(cx) OVER w AS dx,
           cy - lag(cy) OVER w AS dy
    FROM s WINDOW w AS (PARTITION BY oid ORDER BY fid)
)
SELECT sqrt(dx * dx + dy * dy) AS step FROM d WHERE df = 1
"""


def steps(db: Path, n: int) -> np.ndarray:
    with duckdb.connect(f"{db}", read_only=True) as conn:
        return conn.execute(_SQL.format(n=n)).fetch_df()["step"].to_numpy()


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=here / "plots" / "displacement_cdf.pdf")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(3.4, 2.3), constrained_layout=True)
    cmap = plt.get_cmap("tab10")

    print(f"{'dataset':<16}{'units':<11}{'steps':>11}{'zero %':>8}"
          + "".join(f"{q:>9}" for q in ("p50", "p90", "p99", "max")))
    for i, (label, (rel, units, n)) in enumerate(_DATASETS.items()):
        db = _ROOT / rel / "duckdb.db"
        if not db.is_file():
            print(f"{label:<16} missing {db}")
            continue
        x = steps(db, n)
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        # A zero step is a real observation -- the object did not move between
        # those two frames -- but it cannot be drawn on a log axis, so it is
        # reported here and shows on the plot as the curve's intercept.
        zero = 100.0 * (x == 0).sum() / x.size
        qs = np.percentile(x, [50, 90, 99])
        print(f"{label:<16}{units:<11}{x.size:>11,}{zero:>8.1f}"
              + "".join(f"{q:>9.2f}" for q in qs) + f"{x.max():>9.1f}")

        # Zero steps are real (a stationary object) but cannot be drawn on a log
        # axis; they are reported as the intercept instead of being dropped.
        xs = np.sort(x)
        ys = np.arange(1, xs.size + 1) / xs.size
        keep = xs > 0
        ax.plot(xs[keep], ys[keep], label=f"{label} ({units})", color=cmap(i), linewidth=1.4)

    ax.set_xscale("log")
    ax.set_xlabel("per-timestep displacement (dataset units)", fontsize=7, labelpad=2)
    ax.set_ylabel("fraction of steps", fontsize=7, labelpad=2)
    ax.tick_params(labelsize=6, pad=1.5)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=5, loc="upper left", frameon=True, framealpha=0.9,
              edgecolor="none", handlelength=1.4, borderpad=0.3, labelspacing=0.25)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"\n{args.out}")


if __name__ == "__main__":
    main()
