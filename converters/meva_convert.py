#!/usr/bin/env python
"""Convert the raw MEVA dataset into the TrQL common format.

MEVA (DIVA phase 2) is surveillance video with object tracks *and* activities,
annotated in KPF YAML. Like VIRAT it populates the ``activities`` table and
links object boxes to activities via ``bbox.aid``.

Raw layout expected (see README.md)::

    raw/meva/video/<clip>.r13.mp4
    raw/meva/annotation/DIVA-phase-2/MEVA/<source>/<date>/<hour>/<clip>.geom.yml
                                                                  <clip>.types.yml
                                                                  <clip>.activities.yml

where ``<source>`` is ``kitware`` and/or ``kitware-meva-training`` (disjoint
clip sets that together cover all videos). Each clip becomes one subset::

    <out>/meva/<clip>/video.mp4   (symlink)
                     /duckdb.db
                     /metadata.json

KPF records (one flow-mapping per line):
  geom:  {'geom': {'g0': 'x1 y1 x2 y2', 'id1': <track>, 'ts0': <frame>, ...}}
  types: {'types': {'cset3': {<type>: conf, ...}, 'id1': <track>}}
  act:   {'act': {'act2': {<name>: conf}, 'id2': <act id>,
                  'actors': [{'id1': <track>, 'timespan': [{'tsr0': [s, e]}]}], ...}}

So:
  * objects.oid    = track id (id1); class from types.yml.
  * activities.aid = activity id (id2); name from act2.
  * a detection (track, frame) gets one bbox row per activity in which it is an
    actor and whose actor-timespan contains the frame; detections in no activity
    get a single row with aid = NULL.

Run with the project venv::

    .trqs-pyenv/bin/python converters/meva_convert.py
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

# Make the shared helpers importable regardless of CWD.
SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_ROOT = SCRIPT_DIR.parent / "datasets"
sys.path.insert(0, str(SCRIPT_DIR))

from common import Obj, Activity, Bbox, build_subset  # noqa: E402

DATASET = "meva"
DEFAULT_SOURCES = ("kitware", "kitware-meva-training")
ANNO_SUBDIR = "annotation/DIVA-phase-2/MEVA"

# KPF ships two flow styles: Python-dict-ish with quoted keys
# (``'g0': '1575 21 1601 73'``) and unquoted YAML (``g0: 810 541 975 862``),
# with differing key order. Extract each field independently, optional-quote
# tolerant. g0 must be exactly four numbers (this also rejects ``meta`` lines).
G0_RE = re.compile(r"['\"]?g0['\"]?:\s*['\"]?(-?[\d.]+(?:\s+-?[\d.]+){3})")
ID1_RE = re.compile(r"['\"]?id1['\"]?:\s*(-?\d+)")
TS0_RE = re.compile(r"['\"]?ts0['\"]?:\s*(-?\d+)")


def parse_types(path: Path) -> dict[int, str]:
    """Map track id (id1) -> class (highest-confidence type in cset3)."""
    out: dict[int, str] = {}
    if not path.is_file():
        return out
    for entry in yaml.safe_load(path.read_text()) or []:
        rec = entry.get("types") if isinstance(entry, dict) else None
        if not rec:
            continue
        tid = rec.get("id1")
        cset = rec.get("cset3") or {}
        if tid is None or not cset:
            continue
        out[int(tid)] = max(cset.items(), key=lambda kv: kv[1])[0]
    return out


def parse_activities(path: Path):
    """Return (activities, obj_activities).

    activities: aid -> Activity. obj_activities: track id -> list of
    (aid, start_frame, end_frame) from each actor's timespan.
    """
    activities: dict[int, Activity] = {}
    obj_activities: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    if not path.is_file():
        return activities, obj_activities
    for entry in yaml.safe_load(path.read_text()) or []:
        act = entry.get("act") if isinstance(entry, dict) else None
        if not act:
            continue
        aid = act.get("id2")
        name_map = act.get("act2") or {}
        if aid is None or not name_map:
            continue
        aid = int(aid)
        name = max(name_map.items(), key=lambda kv: kv[1])[0]
        activities[aid] = Activity(aid=aid, name=name)
        for actor in act.get("actors", []):
            tid = actor.get("id1")
            if tid is None:
                continue
            for span in actor.get("timespan", []):
                tsr0 = span.get("tsr0")
                if tsr0 and len(tsr0) == 2:
                    obj_activities[int(tid)].append((aid, int(tsr0[0]), int(tsr0[1])))
    return activities, obj_activities


def parse_geom(path: Path):
    """Yield (track_id, frame, x1, y1, x2, y2) tuples from a geom.yml.

    Uses a per-field regex (not YAML) because geom files hold millions of rows.
    """
    with path.open() as fh:
        for line in fh:
            g = G0_RE.search(line)
            if not g:
                continue
            i = ID1_RE.search(line)
            t = TS0_RE.search(line)
            if not i or not t:
                continue
            x1, y1, x2, y2 = (float(c) for c in g.group(1).split())
            yield int(i.group(1)), int(t.group(1)), x1, y1, x2, y2


def build_records(types_map, geom_path, obj_activities):
    """Build (objects, bboxes). One bbox row per (detection, active activity);
    detections in no activity get a single aid=NULL row."""
    objects: dict[int, Obj] = {}
    bboxes: list[Bbox] = []
    for tid, fid, x1, y1, x2, y2 in parse_geom(geom_path):
        if tid not in objects:
            objects[tid] = Obj(oid=tid, cls=types_map.get(tid, "unknown"))
        active = [aid for aid, s, e in obj_activities.get(tid, ()) if s <= fid <= e]
        if active:
            for aid in active:
                bboxes.append(Bbox(fid=fid, oid=tid, x1=x1, y1=y1, x2=x2, y2=y2, aid=aid))
        else:
            bboxes.append(Bbox(fid=fid, oid=tid, x1=x1, y1=y1, x2=x2, y2=y2, aid=None))
    return objects, bboxes


def iter_clips(raw_root: Path, sources):
    """Yield (clip, source, geom, types, activities_path, video)."""
    anno_root = raw_root / ANNO_SUBDIR
    video_root = raw_root / "video"
    for source in sources:
        src_dir = anno_root / source
        if not src_dir.is_dir():
            print(f"!! annotation source '{source}' not found under {anno_root}; skipping")
            continue
        for geom in sorted(src_dir.rglob("*.geom.yml")):
            clip = geom.name[: -len(".geom.yml")]
            types = geom.with_name(f"{clip}.types.yml")
            acts = geom.with_name(f"{clip}.activities.yml")
            videos = sorted(video_root.glob(f"{clip}*.mp4"))
            yield clip, source, geom, types, acts, (videos[0] if videos else None)


def convert(raw_root: Path, out_root: Path, sources) -> None:
    out_dataset = out_root / DATASET
    n = 0
    for clip, source, geom, types_path, acts_path, video in iter_clips(raw_root, sources):
        if video is None:
            print(f"!! {clip}: no matching video; skipping")
            continue

        types_map = parse_types(types_path)
        activities, obj_activities = parse_activities(acts_path)
        objects, bboxes = build_records(types_map, geom, obj_activities)

        obj_classes = sorted({o.cls for o in objects.values()})
        act_classes = sorted({a.name for a in activities.values()})

        out_dir = out_dataset / clip
        metadata = {
            "dataset": DATASET,
            "source": f"MEVA / DIVA phase-2 ({source})",
            "clip": clip,
            "obj_classes": obj_classes,
            "activity_classes": act_classes,
            "notes": (
                "KPF YAML annotations. fid is the KPF frame number ts0 (0-based, "
                "matches the video). A detection that is an actor in N concurrent "
                "activities yields N bbox rows (one per aid); detections in no "
                "activity have aid=NULL."
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
        n += 1
        if n % 50 == 0 or n <= 3:
            print(
                f"[ok] {clip}: {rc['objects']} objects, {rc['activities']} activities, "
                f"{rc['bbox']} bboxes -> {out_dir}"
            )
    print(f"\nDone: wrote {n} subset(s) under {out_dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=DATASETS_ROOT / "raw" / "meva",
        help="Raw meva root (default: datasets/raw for this dataset).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATASETS_ROOT / "processed",
        help="Output root; a 'meva/' subdir is created under it.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(DEFAULT_SOURCES),
        help="Annotation sources (default: kitware kitware-meva-training).",
    )
    args = parser.parse_args()
    convert(args.raw.resolve(), args.out.resolve(), args.sources)


if __name__ == "__main__":
    main()
