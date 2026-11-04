"""Measure how many object pairs actually co-appear in a frame.

A binary relationship ("A is left of B") is only defined when both objects are
visible in the same frame, so a query engine only ever has to look at pairs that
co-occur -- not all C(n, 2) pairs.  This script measures, for a set of video
subsets, how the two quantities grow over the length of the video.

For every sample point t (in video seconds) we record

    objects(t)   distinct objects that have appeared in frames <= t
    all_pairs(t) C(objects(t), 2)                     -- the O(n^2) bound
    cooc(t)      distinct unordered pairs {a, b} that have shared
                 at least one frame <= t

``cooc`` is computed two ways:

  exact     per-frame self join, taking the first frame each pair shares.
            Cost is sum_f k_f^2, so it is only run when that stays sane.
  interval  each object is closed to [min fid, max fid] and two objects are
            assumed to co-occur iff their intervals overlap.  A pair then first
            co-occurs at max(start_a, start_b), so the count follows from a
            sweep over start/end events -- O(n log n) regardless of density.
            Track gaps make this an *upper* bound on the exact count.

Results are cached to a JSON file consumed by the ``cooccurrence`` plot kind.
This script is not part of the ``benchplot`` package and ``benchplot render``
never runs it: it reads the datasets themselves, needs ``duckdb``, and takes
minutes. Run it by hand when a dataset changes::

    uv run --with duckdb --with numpy tools/cooccurrence_stats.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import os

import duckdb
import numpy as np

# plotter/tools/this.py -> the repository root.
REPO_ROOT = Path(__file__).absolute().parents[2]
DATA_ROOT = Path(os.environ.get("TRQS_DATA_DIR", REPO_ROOT / "datasets/processed"))
CACHE = Path(__file__).absolute().parents[1] / "data/cooccurrence_stats.json"

# (dataset, subset, label, domain) -- one long continuous video per dataset.
# Concatenated clips are deliberately avoided: objects from different clips can
# never co-occur, which would inflate the result for free.
SUBSETS = [
    ("detrac", "MVI_40191", "DETRAC", "traffic camera"),
    ("virat", "VIRAT_S_050300_01_000148_000396", "VIRAT", "surveillance"),
    ("meva", "2018-03-09.10-15-00.10-20-00.school.G299", "MEVA", "surveillance"),
    ("stanford_drone", "deathCircle_video1", "Stanford Drone", "drone"),
    ("ytstream", "beach_4", "YTStream", "webcam stream"),
    ("i24_motion", "post1", "I-24 MOTION", "highway"),
]

# Above this much per-frame pair work the exact self join is skipped.
EXACT_PAIRWORK_LIMIT = 5e8
N_SAMPLES = 120


def _bin_edges(max_fid: int, n: int = N_SAMPLES) -> np.ndarray:
    """Right edges of ``n`` equal-width frame bins covering [0, max_fid]."""
    return np.linspace(max_fid / n, max_fid, n)


def _cumulative(events_fid: np.ndarray, weights: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Cumulative sum of ``weights`` for events at or before each bin edge."""
    order = np.argsort(events_fid, kind="stable")
    fids, w = events_fid[order], weights[order]
    idx = np.searchsorted(fids, edges, side="right")
    return np.concatenate([[0.0], np.cumsum(w, dtype=np.float64)])[idx]


def first_seen(con: duckdb.DuckDBPyConnection) -> np.ndarray:
    """Frame in which each object first appears."""
    rel = con.sql("SELECT min(fid) AS s FROM bbox GROUP BY oid").fetchnumpy()
    return rel["s"].astype(np.int64)


def cooc_exact(con: duckdb.DuckDBPyConnection, edges: np.ndarray) -> np.ndarray:
    """Cumulative count of distinct co-occurring pairs, by per-frame self join."""
    rel = con.sql(
        """
        WITH d AS (SELECT DISTINCT fid, oid FROM bbox),
             p AS (
                 SELECT a.oid AS o1, b.oid AS o2, min(a.fid) AS f
                 FROM d a JOIN d b ON a.fid = b.fid AND a.oid < b.oid
                 GROUP BY 1, 2
             )
        SELECT f, count(*) AS n FROM p GROUP BY f
        """
    ).fetchnumpy()
    return _cumulative(rel["f"].astype(np.int64), rel["n"].astype(np.float64), edges)


def cooc_interval(con: duckdb.DuckDBPyConnection, edges: np.ndarray) -> np.ndarray:
    """Cumulative co-occurring pairs under interval closure (an upper bound).

    Pair {a, b} first co-occurs at max(start_a, start_b), so the pairs created
    when object b starts are exactly the objects already live at that instant.
    """
    iv = con.sql("SELECT min(fid) AS s, max(fid) AS e FROM bbox GROUP BY oid").fetchnumpy()
    starts = np.sort(iv["s"].astype(np.int64))
    ends = np.sort(iv["e"].astype(np.int64))

    # Objects live when b starts: those earlier in start order (position i,
    # which breaks ties so a pair of co-starting objects is counted once) minus
    # those already finished.  `left` on ends keeps objects whose last frame is
    # b's first frame -- they do share that frame.
    finished = np.searchsorted(ends, starts, side="left")
    new_pairs = (np.arange(len(starts)) - finished).astype(np.float64)
    return _cumulative(starts, new_pairs, edges)


def measure(dataset: str, subset: str, label: str, domain: str) -> dict:
    path = DATA_ROOT / dataset / subset
    con = duckdb.connect(str(path / "duckdb.db"), read_only=True)
    meta = json.loads((path / "metadata.json").read_text())

    max_fid, n_obj = con.sql("SELECT max(fid), count(DISTINCT oid) FROM bbox").fetchone()
    pairwork = con.sql(
        "SELECT sum(k * k) / 2 FROM (SELECT count(DISTINCT oid) AS k FROM bbox GROUP BY fid)"
    ).fetchone()[0]

    edges = _bin_edges(max_fid)
    objects = _cumulative(first_seen(con), np.ones(n_obj), edges)

    interval = cooc_interval(con, edges)
    exact = cooc_exact(con, edges) if pairwork <= EXACT_PAIRWORK_LIMIT else None
    print(
        f"{label:16} objects={n_obj:>7d} pairwork={pairwork:.2e} "
        f"cooc_interval={interval[-1]:.3e}"
        + (f" cooc_exact={exact[-1]:.3e} (bound {interval[-1] / exact[-1]:.2f}x)" if exact is not None else " [exact skipped]")
    )

    fps = meta.get("fps") or 30.0
    return {
        "dataset": dataset,
        "subset": subset,
        "label": label,
        "domain": domain,
        "fps": fps,
        "n_objects": int(n_obj),
        "max_fid": int(max_fid),
        "exact": exact is not None,
        "time_s": (edges / fps).tolist(),
        "objects": objects.tolist(),
        "cooc": (exact if exact is not None else interval).tolist(),
        "cooc_interval": interval.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=CACHE)
    args = ap.parse_args()

    stats = [measure(*s) for s in SUBSETS]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
