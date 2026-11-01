"""Distinct objects, all pairs, and actually co-occurring pairs, over video time.

The introduction's claim is that a binary relationship never has to be evaluated
on the quadratic candidate set, because most pairs of objects are never on screen
together. This draws that: as a video plays, the number of objects seen so far
grows, the pairs among them grow quadratically, and the pairs that ever share a
frame grow far more slowly.

Unlike the rest of the plotter this reads a *dataset*, not `bench.db`, so it is a
script rather than a `[[plots]]` kind:

    uv run python scripts/cooccurrence.py \
        --db /data/trql_data/processed/ytstream/shibuya_0/duckdb.db

Needs the `duckdb` CLI on PATH; the rows come out through it as CSV so the
plotter keeps its dependency on matplotlib alone.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from itertools import combinations
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

_DEFAULT_DB = Path("/data/trql_data/processed/ytstream/shibuya_0/duckdb.db")


def read_frames(db: Path) -> list[tuple[int, list[int]]]:
    """`(frame, objects present)` for every frame that has any, in frame order."""
    out = subprocess.run(
        ["duckdb", str(db), "-csv", "-c", "SELECT fid, oid FROM bbox ORDER BY fid"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    rows = csv.reader(out.splitlines())
    next(rows, None)  # header

    frames: list[tuple[int, list[int]]] = []
    current_fid: int | None = None
    current: list[int] = []
    for fid_s, oid_s in rows:
        fid, oid = int(fid_s), int(oid_s)
        if fid != current_fid:
            if current_fid is not None:
                frames.append((current_fid, current))
            current_fid, current = fid, []
        current.append(oid)
    if current_fid is not None:
        frames.append((current_fid, current))
    return frames


def curves(frames: list[tuple[int, list[int]]], points: int):
    """Cumulative object and pair counts, sampled at `points` instants."""
    last_fid = frames[-1][0]
    marks = [last_fid * (i + 1) / points for i in range(points)]

    seen: set[int] = set()
    together: set[tuple[int, int]] = set()
    xs, objs, all_pairs, co_pairs = [], [], [], []

    mark = 0
    for fid, present in frames:
        seen.update(present)
        # Every pair on screen in this frame has now co-occurred. Re-adding a
        # known pair is what keeps this cheap: the set holds one entry per
        # co-occurring pair, not one per frame they share.
        together.update(combinations(sorted(set(present)), 2))
        while mark < points and fid >= marks[mark]:
            xs.append(fid)
            objs.append(len(seen))
            all_pairs.append(comb(len(seen), 2))
            co_pairs.append(len(together))
            mark += 1
    return xs, objs, all_pairs, co_pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=_DEFAULT_DB)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--points", type=int, default=240)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "plots" / "cooccurrence_plot.pdf")
    args = ap.parse_args()

    frames = read_frames(args.db)
    xs, objs, all_pairs, co_pairs = curves(frames, args.points)
    minutes = [f / args.fps / 60 for f in xs]

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.plot(minutes, all_pairs, color="#c1440e", linewidth=1.8,
            label=r"all pairs $\binom{n}{2}$")
    ax.plot(minutes, co_pairs, color="#1b6ca8", linewidth=1.8,
            label="pairs that co-occur")
    ax.plot(minutes, objs, color="#2e7d32", linewidth=1.8, linestyle="--",
            label="objects $n$")

    ax.set_yscale("log")
    ax.set_xlabel("Video time (minutes)")
    ax.set_ylabel("Count (log scale)")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="lower right")

    # The gap at the end is the number the introduction quotes.
    ax.annotate(
        f"{all_pairs[-1] / co_pairs[-1]:.0f}$\\times$ fewer",
        xy=(minutes[-1], (all_pairs[-1] * co_pairs[-1]) ** 0.5),
        xytext=(-6, 0), textcoords="offset points",
        ha="right", va="center", fontsize=8,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(
        f"{args.out}: {objs[-1]} objects, {all_pairs[-1]:,} pairs, "
        f"{co_pairs[-1]:,} co-occurring ({all_pairs[-1] / co_pairs[-1]:.1f}x fewer)"
    )


if __name__ == "__main__":
    main()
