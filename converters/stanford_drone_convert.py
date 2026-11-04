#!/usr/bin/env python
"""Convert the raw Stanford Drone Dataset (SDD) into the TrQL common format.

SDD is aerial multi-class tracking (bikers, pedestrians, cars, ...) over 60
scene/video clips. It is pure tracking (no activity labels).

Raw layout expected (see README.md)::

    raw/StanfordDroneDataset/video/<scene>/<videoN>/video.mp4
    raw/StanfordDroneDataset/annotations/<scene>/<videoN>/annotations.txt

Each scene/video becomes one subset::

    <out>/stanford_drone/<scene>_<videoN>/video.mp4   (symlink)
                                          /duckdb.db
                                          /metadata.json

Annotation columns (whitespace separated):

    track_id  xmin  ymin  xmax  ymax  frame  lost  occluded  generated  "label"

So:
  * objects.oid   = track_id; objects.class = lowercased label.
  * bbox          = one row per non-lost annotation; aid is always NULL.
  * activities    = empty (pure tracking dataset).

Run with the project venv::

    .trqs-pyenv/bin/python converters/stanford_drone_convert.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the shared helpers importable regardless of CWD.
SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_ROOT = SCRIPT_DIR.parent / "datasets"
sys.path.insert(0, str(SCRIPT_DIR))

from common import Obj, Bbox, build_subset  # noqa: E402

DATASET = "stanford_drone"


def parse_annotations(path: Path) -> tuple[list[Obj], list[Bbox], int]:
    """Parse an SDD annotations.txt into objects/bboxes.

    Rows with ``lost == 1`` (object outside the view) are dropped. ``fid`` is
    the SDD frame number (already 0-based and aligned to the video). An object's
    class is taken from the first row seen for that track.
    Returns ``(objects, bboxes, n_lost_dropped)``.
    """
    objects: dict[int, Obj] = {}
    bboxes: list[Bbox] = []
    lost_dropped = 0
    for line in path.read_text().splitlines():
        p = line.split()
        if not p:
            continue
        tid = int(p[0])
        xmin, ymin, xmax, ymax = int(p[1]), int(p[2]), int(p[3]), int(p[4])
        frame = int(p[5])
        lost = int(p[6])
        label = p[9].strip('"').lower()
        if tid not in objects:
            objects[tid] = Obj(oid=tid, cls=label)
        if lost == 1:
            lost_dropped += 1
            continue
        bboxes.append(
            Bbox(fid=frame, oid=tid, x1=xmin, y1=ymin, x2=xmax, y2=ymax, aid=None)
        )
    return list(objects.values()), bboxes, lost_dropped


def iter_clips(raw_root: Path):
    """Yield ``(scene, video_name, annotations_path, video_path)``."""
    anno_root = raw_root / "annotations"
    video_root = raw_root / "video"
    for scene_dir in sorted(p for p in anno_root.iterdir() if p.is_dir()):
        scene = scene_dir.name
        for vid_dir in sorted(p for p in scene_dir.iterdir() if p.is_dir()):
            vname = vid_dir.name
            anno = vid_dir / "annotations.txt"
            video = video_root / scene / vname / "video.mp4"
            if not anno.is_file():
                continue
            yield scene, vname, anno, video


def convert(raw_root: Path, out_root: Path) -> None:
    out_dataset = out_root / DATASET
    n = 0
    for scene, vname, anno, video in iter_clips(raw_root):
        if not video.is_file():
            print(f"!! {scene}/{vname}: video missing ({video}); skipping")
            continue
        objects, bboxes, lost = parse_annotations(anno)
        obj_classes = sorted({o.cls for o in objects})

        subset = f"{scene}_{vname}"
        out_dir = out_dataset / subset
        metadata = {
            "dataset": DATASET,
            "source": "Stanford Drone Dataset (Robicquet et al., ECCV 2016)",
            "scene": scene,
            "video": vname,
            "obj_classes": obj_classes,
            "activity_classes": [],
            "notes": (
                "fid is the SDD frame number (0-based, matches the video). Rows "
                "with lost=1 (object outside view) are dropped; labels are "
                "lowercased; activities are absent (pure tracking)."
            ),
        }
        meta = build_subset(
            out_dir,
            video_src=video,
            objects=objects,
            bboxes=bboxes,
            activities=(),
            metadata=metadata,
        )
        rc = meta["row_counts"]
        print(
            f"[ok] {subset}: {rc['objects']} objects, {rc['bbox']} bboxes "
            f"({lost} lost dropped), classes={obj_classes} -> {out_dir}"
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
        default=DATASETS_ROOT / "raw" / "StanfordDroneDataset",
        help="Raw StanfordDroneDataset root (default: datasets/raw for this dataset).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATASETS_ROOT / "processed",
        help="Output root; a 'stanford_drone/' subdir is created under it.",
    )
    args = parser.parse_args()
    convert(args.raw.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
