#!/usr/bin/env python
"""Convert Argoverse 2 Sensor world-frame trajectories into the TrQL format.

AV2 Sensor ships 3D cuboid tracks with ring-camera video, but this copy of the
dataset deliberately holds **no imagery** -- only annotations, ego poses and HD
maps (~2 MB/log instead of ~1-2 GB/log). It is therefore trajectory-only, and
like ``I24_Motion/convert.py`` it stores **world-coordinate** boxes and writes
NO ``video.mp4``.

**Each city becomes one dataset.** All logs of a city share that city's frame --
verified, not assumed: overlapping logs carry the same ``lane_segment`` ids with
bit-identical geometry (max coordinate difference 0.00e+00 m). So a city's logs
are genuinely co-registered and belong in one database::

    <out>/argoverse2_pittsburgh/all/duckdb.db
                                   /metadata.json
                                   /map.json      # merged vector HD map
    <out>/argoverse2_miami/all/...
    ... six cities

Input is the city-merged tables written by ``converters/argoverse2/merge_by_city.py``::

    derived/by_city/<city>/trajectories.parquet
                          /map.json

Mapping into the common schema:

  * objects.oid   = dense index over ``global_track_id`` (= "<log_id>:<track_uuid>").
                    Namespacing matters: object uuids are globally unique, but the
                    ego's is the literal string "EGO_VEHICLE" in all 850 logs, so
                    keying on ``track_uuid`` would fuse every ego into one track.
                    class = the AV2 category, lowercased (29 classes + ego_vehicle).
  * bbox          = the cuboid's bird's-eye footprint, as the axis-aligned extent
                    of the heading-rotated rectangle, in **city-frame METRES**
                    (not image pixels). See the heading caveat in the notes.
  * activities    = empty (pure tracking dataset).

fid encodes the log as well as the frame. A city's logs are separate drives, not
one continuous recording -- AV2 timestamps are anonymized and the whole 850-log
corpus spans only 5.9 h of timestamp values, so logs months apart can share a
range. Laying them end-to-end would invent temporal adjacency. Instead each log
gets its own disjoint block::

    fid = log_index * FID_BLOCK + frame_index_within_log
    log_index = fid // FID_BLOCK      frame_in_log = fid % FID_BLOCK

With FID_BLOCK = 10000 and ~157 frames per log, consecutive drives are ~1000 s
apart on the fid axis, so no realistic query window can span two of them. The
per-log offsets are recorded in ``metadata.json``.

Run with the project venv::

    .trqs-pyenv/bin/python converters/argoverse2_convert.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import duckdb

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_ROOT = SCRIPT_DIR.parent / "datasets"
sys.path.insert(0, str(SCRIPT_DIR))

from common import SCHEMA  # noqa: E402

DATASET_PREFIX = "argoverse2"
SUBSET = "all"
HZ = 10.0          # annotations sit on lidar sweeps: 10 Hz (cameras are 20 Hz)
FID_BLOCK = 10_000  # fid block per log; logs are ~157 frames, so drives stay disjoint


def city_slug(city: str) -> str:
    return city.lower().replace(" ", "_")


def build_city(parquet: Path, map_src: Path, out_dir: Path) -> dict:
    """Build one city's duckdb.db straight from the merged parquet.

    Everything runs inside DuckDB: it reads parquet natively (no pyarrow needed)
    and 3.5 M boxes never materialize in Python.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "duckdb.db"
    if db_path.exists():
        db_path.unlink()

    con = duckdb.connect(str(db_path))
    try:
        con.execute(SCHEMA)
        # CREATE VIEW cannot take a prepared parameter, so the path is inlined.
        pq = str(parquet).replace("'", "''")
        con.execute(f"CREATE TEMP VIEW t AS SELECT * FROM read_parquet('{pq}')")

        # Per-log fid blocks, ordered by log_id so the assignment is reproducible.
        con.execute("""
            CREATE TEMP TABLE log_idx AS
            SELECT log_id, split,
                   (ROW_NUMBER() OVER (ORDER BY log_id) - 1) AS log_index
            FROM (SELECT DISTINCT log_id, split FROM t)
        """)
        # Frame index within a log: dense rank over the log's own sweep timestamps.
        con.execute("""
            CREATE TEMP TABLE frame_idx AS
            SELECT log_id, timestamp_ns,
                   (DENSE_RANK() OVER (PARTITION BY log_id ORDER BY timestamp_ns) - 1) AS frame_in_log
            FROM (SELECT DISTINCT log_id, timestamp_ns FROM t)
        """)
        con.execute("""
            CREATE TEMP TABLE obj_idx AS
            SELECT global_track_id, category,
                   (ROW_NUMBER() OVER (ORDER BY global_track_id) - 1) AS oid
            FROM (SELECT DISTINCT global_track_id, category FROM t)
        """)

        con.execute("INSERT INTO objects SELECT oid, lower(category) FROM obj_idx ORDER BY oid")

        # Axis-aligned extent of the heading-rotated footprint rectangle:
        #   dx = |cos h|*L/2 + |sin h|*W/2 ,  dy = |sin h|*L/2 + |cos h|*W/2
        con.execute(f"""
            INSERT INTO bbox
            SELECT  l.log_index * {FID_BLOCK} + f.frame_in_log        AS fid,
                    o.oid                                            AS oid,
                    t.x - (abs(cos(t.heading_rad)) * t.length_m / 2
                         + abs(sin(t.heading_rad)) * t.width_m  / 2)  AS x1,
                    t.y - (abs(sin(t.heading_rad)) * t.length_m / 2
                         + abs(cos(t.heading_rad)) * t.width_m  / 2)  AS y1,
                    t.x + (abs(cos(t.heading_rad)) * t.length_m / 2
                         + abs(sin(t.heading_rad)) * t.width_m  / 2)  AS x2,
                    t.y + (abs(sin(t.heading_rad)) * t.length_m / 2
                         + abs(cos(t.heading_rad)) * t.width_m  / 2)  AS y2,
                    NULL                                             AS aid
            FROM t
            JOIN log_idx   l USING (log_id)
            JOIN frame_idx f ON f.log_id = t.log_id AND f.timestamp_ns = t.timestamp_ns
            JOIN obj_idx   o ON o.global_track_id = t.global_track_id
        """)

        logs = con.execute(f"""
            SELECT l.log_id, l.split, l.log_index,
                   l.log_index * {FID_BLOCK} AS fid_offset,
                   count(DISTINCT f.frame_in_log) AS n_frames,
                   min(t.timestamp_ns) AS t0_ns
            FROM log_idx l
            JOIN t ON t.log_id = l.log_id
            JOIN frame_idx f ON f.log_id = t.log_id AND f.timestamp_ns = t.timestamp_ns
            GROUP BY 1,2,3 ORDER BY l.log_index
        """).fetchall()
        classes = [r[0] for r in con.execute(
            "SELECT DISTINCT class FROM objects ORDER BY class").fetchall()]
        counts = {
            "objects": con.execute("SELECT count(*) FROM objects").fetchone()[0],
            "activities": 0,
            "bbox": con.execute("SELECT count(*) FROM bbox").fetchone()[0],
        }
        extent = con.execute(
            "SELECT min(x1), max(x2), min(y1), max(y2) FROM bbox").fetchone()
    finally:
        con.close()

    shutil.copyfile(map_src, out_dir / "map.json")
    mp = json.loads((out_dir / "map.json").read_text())

    return {
        "logs": [{"log_id": r[0], "split": r[1], "log_index": r[2],
                  "fid_offset": r[3], "n_frames": r[4], "t0_ns": r[5]} for r in logs],
        "obj_classes": classes,
        "row_counts": counts,
        "extent_m": {"x_min": extent[0], "x_max": extent[1],
                     "y_min": extent[2], "y_max": extent[3]},
        "map_elements": {k: len(v) for k, v in mp.items()},
    }


def convert(raw_root: Path, out_root: Path, only_city: str | None = None) -> None:
    by_city = raw_root / "derived" / "by_city"
    if not by_city.is_dir():
        print(f"!! {by_city} missing -- run converters/argoverse2/merge_by_city.py first")
        return

    cities = sorted(d for d in by_city.iterdir() if d.is_dir())
    if only_city:
        cities = [d for d in cities if d.name == city_slug(only_city)]
    if not cities:
        print("!! no cities to convert")
        return

    for d in cities:
        parquet, map_src = d / "trajectories.parquet", d / "map.json"
        if not parquet.exists():
            print(f"!! {parquet} missing, skipping")
            continue
        dataset = f"{DATASET_PREFIX}_{d.name}"
        out_dir = out_root / dataset / SUBSET
        print(f"processing {d.name} ...")
        st = build_city(parquet, map_src, out_dir)

        metadata = {
            "dataset": dataset,
            "source": "Argoverse 2 Sensor (trajectory-only; no imagery downloaded)",
            "subset": SUBSET,
            "city": d.name,
            "no_video": True,
            "coordinate_space": "Argoverse 2 city frame, METRES (NOT image pixels)",
            "fps": HZ,
            "fid_block": FID_BLOCK,
            "n_logs": len(st["logs"]),
            "extent_m": st["extent_m"],
            "obj_classes": st["obj_classes"],
            "activity_classes": [],
            "row_counts": st["row_counts"],
            "map": {"file": "map.json", "frame": "city", "elements": st["map_elements"]},
            "logs": st["logs"],
            "notes": (
                "Trajectory-only: no video.mp4, and bbox x1,y1,x2,y2 are the "
                "bird's-eye footprint of the 3D cuboid in city-frame METRES, not "
                "image pixels. The box is the axis-aligned extent of the "
                "heading-rotated rectangle, so a diagonally-oriented vehicle gets a "
                "box larger than its footprint; exact centroid, heading and extent "
                "live in derived/by_city/<city>/trajectories.parquet. "
                f"fid = log_index * {FID_BLOCK} + frame_in_log at {HZ} Hz; each log "
                "occupies a disjoint block because a city's logs are separate drives, "
                "not one recording (AV2 timestamps are anonymized and the whole "
                "corpus spans 5.9 h of values, so logs months apart can share a "
                "range). Never treat a fid window spanning two blocks as continuous. "
                "oid keys global_track_id ('<log_id>:<track_uuid>'): object uuids are "
                "globally unique but the ego's is the literal 'EGO_VEHICLE' in every "
                "log. Cross-log registration is ~0.39 m median (measured on 193 "
                "static objects seen by two logs), so sub-metre comparisons between "
                "logs are not supported. map.json is the union of the per-log HD map "
                "crops, in the same city frame as the boxes; coverage is patchy and "
                "6.6-11.2% of lane-graph edges dangle outside it. Annotations are "
                "amodal and cover objects near the drivable area only -- objects "
                "further out are unlabeled, not absent."
            ),
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        c = st["row_counts"]
        print(f"[ok] {dataset}: {c['objects']:,} objects, {c['bbox']:,} bboxes, "
              f"{len(st['logs'])} logs, {len(st['obj_classes'])} classes -> {out_dir}")

    print(f"\nDone: wrote {len(cities)} dataset(s) under {out_root}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw", type=Path, default=DATASETS_ROOT / "raw" / "argoverse2")
    parser.add_argument("--out", type=Path, default=DATASETS_ROOT / "processed")
    parser.add_argument("--city", help="convert only this city (e.g. 'Palo Alto')")
    args = parser.parse_args()
    convert(args.raw.resolve(), args.out.resolve(), args.city)


if __name__ == "__main__":
    main()
