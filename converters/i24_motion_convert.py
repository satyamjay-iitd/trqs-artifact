#!/usr/bin/env python
"""Convert the I-24 MOTION INCEPTION trajectory dataset into the TrQL format.

I-24 MOTION INCEPTION ships *trajectory-only* data: per-vehicle tracks in roadway
coordinates (feet) at 25 Hz, with NO video or imagery. There is therefore no
image to draw boxes on, so (per project decision) this converter stores the
tracks with **road-coordinate** boxes and writes NO ``video.mp4``.

Raw layout expected (see README.md)::

    raw/I24_Motion/<hash>__post1.json      # ~19 GB JSON array of trajectories
    raw/I24_Motion/<hash>__post2.json

Each JSON file (a recording session) becomes one subset::

    <out>/i24_motion/<postN>/duckdb.db
                            /metadata.json
    (no video.mp4 -- trajectory-only dataset)

Each trajectory document has parallel arrays ``timestamp`` / ``x_position`` /
``y_position`` plus scalar ``length`` / ``width`` / ``height`` / ``direction``
and ``coarse_vehicle_class``.

  * objects.oid   = trajectory order index; class = coarse_vehicle_class name.
  * bbox          = vehicle road-coordinate footprint per timestamp:
                    x in [x, x + direction*length], y in [y - W/2, y + W/2].
                    Coordinates are FEET (roadway/state-plane), not image pixels.
  * activities    = empty (pure tracking dataset).

Requires ``ijson`` (the 19 GB files are streamed, never fully loaded).

Run with the project venv::

    .trqs-pyenv/bin/python converters/i24_motion_convert.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import duckdb
import ijson

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_ROOT = SCRIPT_DIR.parent / "datasets"
sys.path.insert(0, str(SCRIPT_DIR))

from common import SCHEMA  # noqa: E402

DATASET = "i24_motion"
HZ = 25.0  # trajectories are sampled at 25 Hz

# --- stored coordinate frame -------------------------------------------------
# Source positions are roadway/state-plane FEET: x spans ~309,502..332,545 ft
# (7.0 km of freeway) and y spans ~-126..126 ft (the 77 m cross-section, negative
# on the westbound carriageway).
#
# The engine stores bounding boxes as u32 (`trql_types::BoundingBox`), so raw feet
# survive neither end: negative y saturates to 0 -- silently flattening 22% of all
# rows onto the centreline -- and the fraction is truncated, which at 12 ft lanes
# is 8.3% of a lane width. Both are cured by moving into a non-negative, scaled
# integer frame here, so what the database holds is exactly what the engine holds.
#
#     stored = round((feet - origin) * COORD_SCALE)
#
# The origins are fixed round numbers rather than each subset's own minimum, so one
# query means the same thing across post1/post2/post3. x100 gives 0.01 ft (~3 mm)
# -- far finer than the trajectories' real accuracy -- and peaks at ~2.35e6, which
# leaves three orders of magnitude of headroom under u32.
COORD_SCALE = 100.0
X_ORIGIN_FT = 309000.0
Y_ORIGIN_FT = -200.0
U32_MAX = 4294967295

# What `metadata.json` records. Every other scaled-coordinate dataset in the tree
# (nuscenes, SoccerTrackv2, alfheim) writes its frame as
#
#     stored = natural * coordinate_scale + delta
#
# so i24 records its origins in that form too, and one formula --
# `natural = (stored - delta) / coordinate_scale` -- reads all four. The origins
# above stay in feet because that is the readable way to write the shift; these
# are the same two numbers expressed the way a consumer needs them.
X_DELTA = -X_ORIGIN_FT * COORD_SCALE  # -30_900_000.0
Y_DELTA = -Y_ORIGIN_FT * COORD_SCALE  #     20_000.0


def to_stored(value: float, origin: float) -> int:
    """Map one coordinate in feet into the stored u32 frame."""
    v = round((value - origin) * COORD_SCALE)
    if not 0 <= v <= U32_MAX:
        raise ValueError(
            f"coordinate {value} ft maps to {v}, outside u32 -- "
            f"COORD_SCALE/X_ORIGIN_FT/Y_ORIGIN_FT no longer bracket this data"
        )
    return v


# The common schema types the box columns as DOUBLE; i24 stores the scaled
# integers above instead, so the engine's u32 read is lossless.
SCHEMA_U32 = """
CREATE TABLE objects (
    oid   BIGINT PRIMARY KEY,
    class VARCHAR
);

CREATE TABLE activities (
    aid  BIGINT PRIMARY KEY,
    name VARCHAR
);

CREATE TABLE bbox (
    fid BIGINT,
    oid BIGINT REFERENCES objects(oid),
    x1  UINTEGER,
    y1  UINTEGER,
    x2  UINTEGER,
    y2  UINTEGER,
    aid BIGINT REFERENCES activities(aid)
);
"""

# I-24 MOTION coarse_vehicle_class ids (assumed mapping; see README).
COARSE_CLASS = {
    0: "sedan",
    1: "midsize",
    2: "van",
    3: "pickup",
    4: "semi",
    5: "truck",
    6: "motorcycle",
}


def subset_name(path: Path) -> str:
    stem = path.stem  # e.g. "637b0234..._post1"  or  "..__post1"
    return stem.split("__")[-1] if "__" in stem else stem


def process_file(json_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "duckdb.db"
    if db_path.exists():
        db_path.unlink()

    obj_fd, obj_csv = tempfile.mkstemp(suffix=".objects.csv", dir=str(out_dir))
    bbox_fd, bbox_csv = tempfile.mkstemp(suffix=".bbox.csv", dir=str(out_dir))

    classes: dict[str, int] = {}
    n_obj = 0
    n_bbox = 0
    t0 = None
    min_fid = None

    try:
        with os.fdopen(obj_fd, "w", newline="") as of, os.fdopen(bbox_fd, "w", newline="") as bf:
            ow = csv.writer(of)
            bw = csv.writer(bf)
            with json_path.open("rb") as fh:
                for doc in ijson.items(fh, "item", use_float=True):
                    ts = doc.get("timestamp") or []
                    xs = doc.get("x_position") or []
                    ys = doc.get("y_position") or []
                    if not ts or not (len(ts) == len(xs) == len(ys)):
                        continue
                    if t0 is None:
                        t0 = math.floor(doc.get("first_timestamp", ts[0]))

                    oid = n_obj
                    cls = COARSE_CLASS.get(doc.get("coarse_vehicle_class"),
                                           f"class{doc.get('coarse_vehicle_class')}")
                    classes[cls] = classes.get(cls, 0) + 1
                    ow.writerow((oid, cls))
                    n_obj += 1

                    length = float(doc.get("length") or 0.0)
                    half_w = float(doc.get("width") or 0.0) / 2.0
                    direction = 1.0 if (doc.get("direction") or 1) >= 0 else -1.0
                    dx = direction * length
                    for t, x, y in zip(ts, xs, ys):
                        # Quantize onto the frame grid before rounding. The tracks are an
                        # exact 25 Hz grid, but half of them are offset half a frame from
                        # the integer second `t0` is floored to, so `(t - t0) * HZ` lands
                        # on `k + 0.5`. Rounding that directly loses samples two ways:
                        # `round()` is round-half-to-even, so 0.5, 1.5, 2.5, 3.5 collapse
                        # to 0, 2, 2, 4; and float error around the .5 makes which way it
                        # goes depend on the bit pattern. Snapping to 6 decimals first and
                        # then rounding half up is stable and collision-free -- it is what
                        # keeps one frame per sample instead of silently dropping 1.9% of
                        # them onto a neighbour's fid.
                        fid = math.floor(round((t - t0) * HZ, 6) + 0.5)
                        if min_fid is None or fid < min_fid:
                            min_fid = fid
                        xa, xb = x, x + dx
                        x1, x2 = (xa, xb) if xa <= xb else (xb, xa)
                        bw.writerow((
                            fid, oid,
                            to_stored(x1, X_ORIGIN_FT), to_stored(y - half_w, Y_ORIGIN_FT),
                            to_stored(x2, X_ORIGIN_FT), to_stored(y + half_w, Y_ORIGIN_FT),
                            "",
                        ))
                        n_bbox += 1

        con = duckdb.connect(str(db_path))
        try:
            con.execute(SCHEMA_U32)
            if n_obj:
                con.execute(
                    "INSERT INTO objects SELECT * FROM read_csv(?, header=false, "
                    "columns={'oid':'BIGINT','class':'VARCHAR'})",
                    [obj_csv],
                )
            if n_bbox:
                # offset so fid is 0-based from the earliest trajectory point
                # (docs aren't perfectly time-ordered, so min_fid may be < 0)
                con.execute(
                    "INSERT INTO bbox SELECT fid - ?, oid, x1, y1, x2, y2, aid FROM "
                    "read_csv(?, header=false, "
                    "columns={'fid':'BIGINT','oid':'BIGINT','x1':'UINTEGER','y1':'UINTEGER',"
                    "'x2':'UINTEGER','y2':'UINTEGER','aid':'BIGINT'})",
                    [min_fid or 0, bbox_csv],
                )

        finally:
            con.close()
    finally:
        os.unlink(obj_csv)
        os.unlink(bbox_csv)

    # epoch time corresponding to fid 0 after the offset
    frame0_epoch = (t0 + (min_fid or 0) / HZ) if t0 is not None else None
    return {
        "objects": n_obj,
        "bbox": n_bbox,
        "obj_classes": sorted(classes),
        "frame0_epoch": frame0_epoch,
    }


def convert(raw_root: Path, out_root: Path) -> None:
    out_dataset = out_root / DATASET
    files = sorted(raw_root.glob("*.json"))
    if not files:
        print(f"!! no .json trajectory files under {raw_root}")
        return
    for json_path in files:
        sub = subset_name(json_path)
        out_dir = out_dataset / sub
        print(f"processing {json_path.name} -> {sub} (streaming ~{json_path.stat().st_size/1e9:.0f}GB) ...")
        stats = process_file(json_path, out_dir)

        metadata = {
            "dataset": DATASET,
            "source": "I-24 MOTION INCEPTION (trajectory-only)",
            "subset": sub,
            "no_video": True,
            "coordinate_space": (
                "roadway/state-plane feet x 100, shifted non-negative "
                "(NOT image pixels): feet = (stored - delta) / coordinate_scale"
            ),
            "coordinate_scale": COORD_SCALE,
            "x_delta": X_DELTA,
            "y_delta": Y_DELTA,
            "fps": HZ,
            "frame0_epoch": stats["frame0_epoch"],
            "obj_classes": stats["obj_classes"],
            "activity_classes": [],
            "coarse_class_map": {str(k): v for k, v in COARSE_CLASS.items()},
            "row_counts": {"objects": stats["objects"], "activities": 0, "bbox": stats["bbox"]},
            "notes": (
                "Trajectory-only dataset: there is NO video, so no video.mp4 is "
                "written and bbox coordinates are vehicle road-coordinate "
                "footprints stored as centi-feet shifted non-negative -- recover "
                "feet with (stored - delta) / coordinate_scale. They are NOT image "
                "pixels. The shift exists because the engine stores boxes as u32, "
                "which would otherwise clamp the westbound carriageway's negative y "
                "to zero and truncate every coordinate to whole feet. fid is a "
                "0-based 25Hz frame index; fid 0 corresponds to frame0_epoch. "
                "Class is the coarse_vehicle_class; activities are absent (aid=NULL)."
            ),
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"[ok] {sub}: {stats['objects']} objects, {stats['bbox']} bboxes, "
              f"classes={stats['obj_classes']} -> {out_dir}")
    print(f"\nDone: wrote {len(files)} subset(s) under {out_dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw", type=Path, default=DATASETS_ROOT / "raw" / "I24_Motion")
    parser.add_argument("--out", type=Path, default=DATASETS_ROOT / "processed")
    args = parser.parse_args()
    convert(args.raw.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
