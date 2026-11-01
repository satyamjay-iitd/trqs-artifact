#!/usr/bin/env python3
"""Per-sub-dataset facts read from the source `duckdb.db`: which classes it
carries, and where in its coordinate space its objects actually are.

Both exist because nuScenes is 850 scenes rather than one dataset. A scene
occupies its own patch of a city-sized map raster, so a region written in
absolute map coordinates would land outside almost every scene; and a scene
carries only the handful of classes that happened to be in view, so a query
constraining on `vehicle.truck` is a parse error -- not an empty result -- on
the 151 scenes without one.

Everything here is cached in-process: the suite driver asks the same questions
of the same 850 scenes once per query.
"""
from __future__ import annotations

import functools

import duckdb

from frames import PROCESSED


def _db(dataset: str, sub: str) -> str:
    return str(PROCESSED / dataset / sub / "duckdb.db")


@functools.lru_cache(maxsize=None)
def scene_classes(dataset: str, sub: str) -> frozenset[str]:
    con = duckdb.connect(_db(dataset, sub), read_only=True)
    try:
        return frozenset(r[0] for r in con.execute("SELECT DISTINCT class FROM objects").fetchall())
    finally:
        con.close()


@functools.lru_cache(maxsize=None)
def scene_extent(dataset: str, sub: str) -> tuple[float, float, float, float]:
    """`(x1, y1, x2, y2)` covering every box in the sub-dataset, stored units."""
    con = duckdb.connect(_db(dataset, sub), read_only=True)
    try:
        return con.execute("SELECT min(x1), min(y1), max(x2), max(y2) FROM bbox").fetchone()
    finally:
        con.close()


@functools.lru_cache(maxsize=None)
def scene_regions(dataset: str, sub: str) -> dict[str, str]:
    """Three bands across the scene's dominant axis of motion, plus the whole.

    The dominant axis is whichever of the scene's extents is longer -- for a
    driving scene that is the direction the ego vehicle travelled, so `near`,
    `mid` and `far` are ordered along the road rather than across it. Splitting
    the *short* axis instead would put the bands across the lanes, where every
    object crosses all three in a second and the ordering means nothing.

    Returned as ready-to-paste `REGIONS` rectangles in stored units.
    """
    x1, y1, x2, y2 = scene_extent(dataset, sub)
    if (x2 - x1) >= (y2 - y1):
        cuts = [x1 + (x2 - x1) * k / 3 for k in range(4)]
        bands = [(cuts[i], y1, cuts[i + 1], y2) for i in range(3)]
    else:
        cuts = [y1 + (y2 - y1) * k / 3 for k in range(4)]
        bands = [(x1, cuts[i], x2, cuts[i + 1]) for i in range(3)]

    def rect(b) -> str:
        return "[%d, %d, %d, %d]" % tuple(round(v) for v in b)

    names = ["near", "mid", "far"]
    out = {n: rect(b) for n, b in zip(names, bands)}
    out["whole"] = rect((x1, y1, x2, y2))
    return out
