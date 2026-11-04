#!/usr/bin/env python
"""Convert the raw VIRAT Ground 2.0 dataset into the TrQL common format.

VIRAT is the first dataset here with *events*, so it populates the
``activities`` table and links object boxes to activities via ``bbox.aid``.

Raw layout expected (see README.md)::

    raw/virat/videos/<clip>.mp4
    raw/virat/annotations/<clip>.viratdata.objects.txt
    raw/virat/annotations/<clip>.viratdata.mapping.txt   # optional
    raw/virat/annotations/<clip>.viratdata.events.txt    # not needed here

Each clip with an objects file *and* a video becomes one subset::

    <out>/virat/<clip>/video.mp4   (symlink)
                      /duckdb.db
                      /metadata.json

Annotation formats (whitespace separated, see docs/README_format_release2.txt):

  objects:  oid, duration, frame, x, y, w, h, object_type
  mapping:  eid, etype, duration, start, end, num_obj, <binary vector over oid>

So:
  * objects.oid    -> objects table; class from object_type.
  * mapping events -> activities table (aid = eid, name = event type).
  * a detection (oid, frame) gets one bbox row per event whose span contains the
    frame and whose object set includes oid; detections in no event get a single
    row with aid = NULL.

Run with the project venv::

    .trqs-pyenv/bin/python converters/virat_convert.py
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# Make the shared helpers importable regardless of CWD.
SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_ROOT = SCRIPT_DIR.parent / "datasets"
sys.path.insert(0, str(SCRIPT_DIR))

from common import Obj, Activity, Bbox, build_subset  # noqa: E402

DATASET = "virat"

OBJECT_TYPE = {
    1: "person",
    2: "car",
    3: "vehicle",
    4: "object",
    5: "bike",
}

EVENT_TYPE = {
    1: "loading_vehicle",
    2: "unloading_vehicle",
    3: "opening_trunk",
    4: "closing_trunk",
    5: "getting_into_vehicle",
    6: "getting_out_of_vehicle",
    7: "gesturing",
    8: "digging",
    9: "carrying",
    10: "running",
    11: "entering_facility",
    12: "exiting_facility",
}

OBJECTS_SUFFIX = ".viratdata.objects.txt"
MAPPING_SUFFIX = ".viratdata.mapping.txt"


def parse_objects(path: Path) -> tuple[dict[int, Obj], list[tuple[int, int, float, float, float, float]]]:
    """Return ``(objects, detections)``.

    ``detections`` are ``(oid, frame, x1, y1, x2, y2)`` tuples. Zero-area boxes
    (object not visible in that frame) are skipped. An object's class is taken
    from the first row seen for that oid.
    """
    objects: dict[int, Obj] = {}
    detections: list[tuple[int, int, float, float, float, float]] = []
    for line in path.read_text().split("\n"):
        p = line.split()
        if not p:
            continue
        oid = int(p[0])
        frame = int(p[2])
        x, y, w, h = float(p[3]), float(p[4]), float(p[5]), float(p[6])
        otype = int(p[7])
        if oid not in objects:
            objects[oid] = Obj(oid=oid, cls=OBJECT_TYPE.get(otype, f"type{otype}"))
        if w <= 0 or h <= 0:
            continue
        detections.append((oid, frame, x, y, x + w, y + h))
    return objects, detections


def parse_mapping(path: Path) -> tuple[dict[int, Activity], dict[int, list[tuple[int, int, int]]]]:
    """Return ``(activities, obj_events)``.

    ``obj_events`` maps oid -> list of ``(eid, start_frame, end_frame)`` the
    object participates in.
    """
    activities: dict[int, Activity] = {}
    obj_events: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for line in path.read_text().split("\n"):
        p = line.split()
        if not p:
            continue
        eid = int(p[0])
        etype = int(p[1])
        start, end = int(p[3]), int(p[4])
        vector = p[6:]
        activities[eid] = Activity(aid=eid, name=EVENT_TYPE.get(etype, f"event{etype}"))
        for oid, v in enumerate(vector):
            if v == "1":
                obj_events[oid].append((eid, start, end))
    return activities, obj_events


def build_bboxes(
    detections: list[tuple[int, int, float, float, float, float]],
    obj_events: dict[int, list[tuple[int, int, int]]],
):
    """Yield Bbox records, emitting one row per (detection, active event) and a
    single aid=NULL row for detections that belong to no event."""
    for oid, frame, x1, y1, x2, y2 in detections:
        active = [eid for eid, s, e in obj_events.get(oid, ()) if s <= frame <= e]
        if active:
            for eid in active:
                yield Bbox(fid=frame, oid=oid, x1=x1, y1=y1, x2=x2, y2=y2, aid=eid)
        else:
            yield Bbox(fid=frame, oid=oid, x1=x1, y1=y1, x2=x2, y2=y2, aid=None)


def iter_clips(raw_root: Path):
    """Yield ``(clip, objects_path, mapping_path_or_None, video_path)``.

    Only clips with a readable objects *file* and an existing video are yielded.
    Some annotation entries are empty directories (download artifacts); those
    are treated as missing.
    """
    anno_dir = raw_root / "annotations"
    video_dir = raw_root / "videos"
    for obj_path in sorted(anno_dir.glob(f"*{OBJECTS_SUFFIX}")):
        if not obj_path.is_file():
            continue
        clip = obj_path.name[: -len(OBJECTS_SUFFIX)]
        video = video_dir / f"{clip}.mp4"
        if not video.is_file():
            continue
        mapping = anno_dir / f"{clip}{MAPPING_SUFFIX}"
        yield clip, obj_path, (mapping if mapping.is_file() else None), video


def convert(raw_root: Path, out_root: Path) -> None:
    out_dataset = out_root / DATASET
    n = 0
    for clip, obj_path, mapping_path, video in iter_clips(raw_root):
        objects, detections = parse_objects(obj_path)
        if mapping_path is not None:
            activities, obj_events = parse_mapping(mapping_path)
        else:
            activities, obj_events = {}, {}

        bboxes = list(build_bboxes(detections, obj_events))
        obj_classes = sorted({o.cls for o in objects.values()})
        act_classes = sorted({a.name for a in activities.values()})

        out_dir = out_dataset / clip
        metadata = {
            "dataset": DATASET,
            "source": "VIRAT Ground Video Dataset Release 2.0",
            "clip": clip,
            "obj_classes": obj_classes,
            "activity_classes": act_classes,
            "object_type_map": {str(k): v for k, v in OBJECT_TYPE.items()},
            "event_type_map": {str(k): v for k, v in EVENT_TYPE.items()},
            "has_events": mapping_path is not None,
            "notes": (
                "fid is the VIRAT frame number (0-based, matches the video). A "
                "detection participating in N concurrent events yields N bbox "
                "rows (one per aid); detections in no event have aid=NULL. "
                "Zero-area boxes (object absent in a frame) are dropped."
            ),
        }
        meta = build_subset(
            out_dir,
            video_src=video,
            objects=list(objects.values()),
            bboxes=bboxes,
            activities=list(activities.values()),
            metadata=metadata,
        )
        rc = meta["row_counts"]
        print(
            f"[ok] {clip}: {rc['objects']} objects, {rc['activities']} activities, "
            f"{rc['bbox']} bboxes, fps={meta['fps']:.3f} -> {out_dir}"
        )
        n += 1
    print(f"\nDone: wrote {n} subset(s) under {out_dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=DATASETS_ROOT / "raw" / "virat",
        help="Raw virat root (default: datasets/raw for this dataset).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATASETS_ROOT / "processed",
        help="Output root; a 'virat/' subdir is created under it.",
    )
    args = parser.parse_args()
    convert(args.raw.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
