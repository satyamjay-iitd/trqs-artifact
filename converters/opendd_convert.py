#!/usr/bin/env python
"""Convert the raw openDD roundabout dataset into the TrQL common format.

openDD ships one SQLite database per *recording* (one coherent drone flight),
each holding a single table named after the file. Unlike every other dataset
here it carries **no video and no pixels**: road users are given as UTM
positions with a heading and physical dimensions, already tracked.

Raw layout expected::

    raw/openDD/rdb<N>/trajectories_rdb<N>_v3.sqlite   (all its recordings)
    raw/openDD/rdb<N>/map_rdb<N>/...                  (maps, unused here)

One SQLite file per roundabout holds *every* recording of it as a separate
table named ``rdb<N>_<k>`` -- 501 tables across the seven files. Each table
becomes one subset.

Source columns::

    ID, TIMESTAMP, OBJID, UTM_X, UTM_Y, UTM_ANGLE, V, ACC, ACC_LAT, ACC_TAN,
    CLASS, WIDTH, LENGTH, TRAILER_ID

Three conversions are needed, and each loses something; all are recorded in
``metadata.json`` so a reader can tell what was done.

**Coordinates.** UTM metres become centimetres relative to a per-roundabout
origin, so every value is a small non-negative integer (the common format
stores ``UINTEGER``). The origin is shared by every recording of a roundabout
-- taken as the floor of the minimum over all of them -- so a named region
means the same thing across recordings. It is written to the metadata as
``utm_origin_m``. This mirrors I24-Motion, whose coordinates are likewise world
units rather than pixels.

**Boxes.** The common format wants an axis-aligned box, while openDD gives an
*oriented* one: centre, heading, width and length. ``--box aabb`` (the default)
stores the axis-aligned bounding box of the rotated rectangle, which is exact
at 0 and 90 degrees and inflates by up to sqrt(2) in between -- a 4.4 m car at
45 degrees becomes a 3.7 m square. ``--box centre`` instead stores a degenerate
one-centimetre box at the centre, which keeps positions exact and discards
extent entirely. Pick by what the query needs: predicates over centres are
unaffected by the choice, predicates over box edges are not.

**Time.** ``fid`` is the index of the row's ``TIMESTAMP`` among the
recording's *distinct* timestamps, so it is exactly the frame the sample came
from. Scaling by an estimated rate instead -- ``round(TIMESTAMP * fps)`` --
collides: openDD's sampling jitters either side of 1/30 s, so a pair of
consecutive frames periodically rounds to one id, which silently duplicates
``(oid, fid)`` and splits any run that touches it. That cost 0.2% of rows and a
quarter of the intervals in a trial query before it was caught. The assumption
here is instead that a timestamp absent from the whole recording is not a
dropped frame; openDD's tracks are dense enough for that to hold.

Run with the project venv::

    .trqs-pyenv/bin/python converters/opendd_convert.py --help
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_ROOT = SCRIPT_DIR.parent / "datasets"
sys.path.insert(0, str(SCRIPT_DIR))

from common import Bbox, Obj, build_duckdb  # noqa: E402

DATASET = "opendd"
SOURCE = "openDD: A Large-Scale Roundabout Drone Dataset (CC BY-ND 4.0)"

#: openDD's class names, lowercased and underscored to match the other datasets.
CLASS_MAP = {
    "Car": "car",
    "Pedestrian": "pedestrian",
    "Bicycle": "bicycle",
    "Motorcycle": "motorcycle",
    "Bus": "bus",
    "Trailer": "trailer",
    "Medium Vehicle": "medium_vehicle",
    "Heavy Vehicle": "heavy_vehicle",
}


def trajectory_db(rdb_dir: Path) -> Path | None:
    """The one trajectory database of a roundabout, excluding its map."""
    found = [
        p for p in rdb_dir.rglob("*.sqlite") if not p.name.startswith("map_")
    ]
    if len(found) > 1:
        raise SystemExit(f"{rdb_dir}: expected 1 trajectory db, found {found}")
    return found[0] if found else None


def tables_of(db: Path) -> list[str]:
    """Every recording in a roundabout's database, in numeric order."""
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        names = [
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]

    def key(n: str) -> tuple:
        tail = n.rsplit("_", 1)[-1]
        return (0, int(tail)) if tail.isdigit() else (1, n)

    return sorted(names, key=key)


def origin_of(db: Path, tables: list[str]) -> tuple[float, float]:
    """Shared UTM origin for a roundabout: the floor of the min over recordings.

    Per-recording origins would shift by a metre or two between flights of the
    same roundabout, which would silently move any named region.
    """
    xs, ys = [], []
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        for t in tables:
            x, y = con.execute(
                f'SELECT min(UTM_X), min(UTM_Y) FROM "{t}"'
            ).fetchone()
            if x is not None:
                xs.append(x)
                ys.append(y)
    if not xs:
        raise SystemExit(f"{db}: no rows in any recording")
    return math.floor(min(xs)), math.floor(min(ys))


def frame_index(con: sqlite3.Connection, table: str) -> tuple[dict[float, int], float]:
    """Map each distinct timestamp to its frame number, and the sampling rate.

    The map is what makes ``fid`` exact; the rate is for metadata only.
    """
    ts = [
        r[0]
        for r in con.execute(
            f'SELECT DISTINCT TIMESTAMP FROM "{table}" ORDER BY TIMESTAMP'
        ).fetchall()
    ]
    if len(ts) < 2 or ts[-1] == ts[0]:
        return ({t: i for i, t in enumerate(ts)}, 30.0)
    return ({t: i for i, t in enumerate(ts)}, (len(ts) - 1) / (ts[-1] - ts[0]))


def half_extents(angle: float, width: float, length: float) -> tuple[float, float]:
    """Half-width and half-height of the AABB of a rotated rectangle."""
    ca, sa = abs(math.cos(angle)), abs(math.sin(angle))
    return (length / 2) * ca + (width / 2) * sa, (length / 2) * sa + (width / 2) * ca


def convert_one(
    db: Path, table: str, out_dir: Path, origin: tuple[float, float], mode: str
) -> dict:
    ox, oy = origin
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        fid_of, fps = frame_index(con, table)
        rows = con.execute(
            f'SELECT TIMESTAMP, OBJID, UTM_X, UTM_Y, UTM_ANGLE, CLASS, WIDTH, LENGTH '
            f'FROM "{table}" ORDER BY OBJID, TIMESTAMP'
        )

        # One row per track; openDD gives a single class per object, but take
        # the majority anyway so a mislabelled frame cannot rename a track.
        seen: dict[int, dict[str, int]] = {}
        bboxes: list[Bbox] = []
        unknown: set[str] = set()
        max_fid = 0
        for ts, oid, x, y, ang, cls, w, ln in rows:
            if x is None or y is None:
                continue
            name = CLASS_MAP.get(cls)
            if name is None:
                unknown.add(cls)
                name = str(cls).strip().lower().replace(" ", "_")
            seen.setdefault(oid, {})
            seen[oid][name] = seen[oid].get(name, 0) + 1

            fid = fid_of[ts]
            max_fid = max(max_fid, fid)
            cx, cy = (x - ox) * 100.0, (y - oy) * 100.0  # metres -> centimetres
            if mode == "centre":
                dx = dy = 0.5
            else:
                hw, hh = half_extents(ang or 0.0, w or 0.0, ln or 0.0)
                dx, dy = hw * 100.0, hh * 100.0
            # Round to whole centimetres. Every other dataset in the tree
            # stores integer coordinates, and `build_duckdb` infers the column
            # type from what it is handed: floats here would make `bbox` DOUBLE
            # and leave a centre like 2350.0 representable only to within a
            # float ulp, which is enough to flip a region-boundary test against
            # the fitted polynomial.
            bboxes.append(
                Bbox(
                    fid=fid,
                    oid=oid,
                    x1=max(0, round(cx - dx)),
                    y1=max(0, round(cy - dy)),
                    x2=max(0, round(cx + dx)),
                    y2=max(0, round(cy + dy)),
                )
            )

    objects = [
        Obj(oid=oid, cls=max(counts.items(), key=lambda kv: kv[1])[0])
        for oid, counts in sorted(seen.items())
    ]
    if unknown:
        print(f"    note: unmapped classes passed through: {sorted(unknown)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    counts = build_duckdb(out_dir / "duckdb.db", objects, bboxes)

    classes = sorted({o.cls for o in objects})
    meta = {
        "dataset": DATASET,
        "source": SOURCE,
        "scene": db.parent.name,
        "video_id": table,
        "fps": round(fps, 4),
        "num_frames": max_fid + 1,
        "duration_s": round((max_fid + 1) / fps, 2),
        # No pixels: the "frame" is a patch of ground, sized from the data.
        "width": None,
        "height": None,
        "obj_classes": classes,
        "activity_classes": [],
        "has_raw_detections": False,
        "row_counts": counts,
        "coordinate_units": "centimetres, UTM relative to utm_origin_m",
        "utm_origin_m": [ox, oy],
        "box_mode": mode,
        "notes": (
            "openDD carries trajectories only -- no video. Positions are UTM "
            "metres converted to centimetres relative to a per-roundabout "
            "origin shared by all its recordings. "
            + (
                "Boxes are the axis-aligned bounding box of openDD's oriented "
                "box, which inflates by up to sqrt(2) at 45 degrees."
                if mode == "aabb"
                else "Boxes are degenerate 1 cm squares at the object centre; "
                "extent is discarded."
            )
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=DATASETS_ROOT / "raw" / "openDD",
                    help="directory holding rdb*/ (default: datasets/raw/openDD)")
    ap.add_argument("--out", type=Path,
                    default=DATASETS_ROOT / "processed" / DATASET,
                    help="processed output root")
    ap.add_argument("--box", choices=("aabb", "centre"), default="aabb",
                    help="how to turn openDD's oriented box into an "
                         "axis-aligned one (default: aabb)")
    ap.add_argument("--rdb", nargs="*",
                    help="only these roundabout directories, e.g. rdb1 rdb2")
    ap.add_argument("--limit", type=int,
                    help="at most N recordings per roundabout (for a smoke test)")
    args = ap.parse_args()

    dirs = sorted(
        d for d in args.raw.iterdir()
        if d.is_dir() and (args.rdb is None or d.name in args.rdb)
    )
    if not dirs:
        raise SystemExit(f"no roundabout directories under {args.raw}")

    for rdb in dirs:
        db = trajectory_db(rdb)
        if db is None:
            continue
        tables = tables_of(db)
        origin = origin_of(db, tables)
        print(f"{rdb.name}: {len(tables)} recordings, origin {origin} m")
        for table in tables[: args.limit]:
            out = args.out / table
            meta = convert_one(db, table, out, origin, args.box)
            print(f"  {table:18} {meta['row_counts']['objects']:>6} objects "
                  f"{meta['row_counts']['bbox']:>9,} boxes  "
                  f"{meta['num_frames']:>7,} frames @ {meta['fps']:.2f} fps")


if __name__ == "__main__":
    main()
