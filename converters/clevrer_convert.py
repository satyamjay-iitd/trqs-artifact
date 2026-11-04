#!/usr/bin/env python
"""Convert the CLEVRER dataset into the TrQL common format.

CLEVRER is synthetic videos of moving objects (spheres/cubes/cylinders in 8
colors, 2 materials). Object detections (with attributes and image boxes) are
provided for the 10000 *train* videos in ``obj_clevrer.csv``; the validation and
test splits have no object annotations and are not converted.

Raw layout expected (see README.md)::

    raw/clevrer/obj_clevrer.csv                 # oid,video,fid,shape,color,material,x1,y1,x2,y2
    raw/clevrer/video_train/**/video_<id>.mp4   # extracted from video_train.zip

Each video becomes one subset::

    <out>/clevrer/video_<id>/video.mp4  (symlink)
                            /duckdb.db
                            /metadata.json

  * objects.oid   = per-video object index; class = "<color> <material> <shape>"
                    (CLEVRER objects are uniquely identified by this combination).
  * bbox          = one row per detection; aid is always NULL.
  * activities    = empty (collision events are not in this annotation source).

Run with the project venv::

    .trqs-pyenv/bin/python converters/clevrer_convert.py --raw datasets/raw/clevrer --out datasets/processed
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_ROOT = SCRIPT_DIR.parent / "datasets"
sys.path.insert(0, str(SCRIPT_DIR))

from common import Obj, Bbox, build_subset  # noqa: E402

DATASET = "clevrer"


def video_index(video_root: Path) -> dict[str, Path]:
    """Map ``video_<id>`` -> mp4 path (the zip nests them in subfolders)."""
    return {p.stem: p for p in video_root.rglob("video_*.mp4")}


def _shape_cost(a: Bbox, b: Bbox) -> float:
    """Motion discontinuity between two boxes: centre displacement plus size change."""
    return (
        abs((a.x1 + a.x2) - (b.x1 + b.x2)) / 2
        + abs((a.y1 + a.y2) - (b.y1 + b.y2)) / 2
        + abs((a.x2 - a.x1) - (b.x2 - b.x1))
        + abs((a.y2 - a.y1) - (b.y2 - b.y1))
    )


def resolve_duplicate_boxes(bboxes: list[Bbox]) -> tuple[list[Bbox], int]:
    """Keep one box per (oid, fid), choosing the one that follows the object's motion.

    ``obj_clevrer.csv`` gives some objects two or three boxes in a single frame: 2,687
    such frames across 1,466 tracks, of which only 11 are exact copies. One box continues
    the track (median 1.5 px from the interpolation of its neighbours) and the rest are
    spurious, typically far away (median 204 px) and usually touching the frame edge.

    Neither file order nor "it is another object's box" identifies the good one, so the
    choice is made from the trajectory itself: a Viterbi pass over the track picks the
    chain of boxes with the least total motion discontinuity. Doing it per track rather
    than per frame is what handles the runs of up to 20 consecutive duplicated frames,
    where a frame's neighbours are themselves ambiguous and there is nothing clean to
    compare against locally.

    Returns the filtered boxes and how many rows were dropped.
    """
    by_oid: dict[int, dict[int, list[Bbox]]] = {}
    for b in bboxes:
        by_oid.setdefault(b.oid, {}).setdefault(b.fid, []).append(b)

    kept: list[Bbox] = []
    dropped = 0
    for frames in by_oid.values():
        if all(len(c) == 1 for c in frames.values()):
            kept.extend(c[0] for c in frames.values())
            continue

        # Viterbi: cost[j] is the best total discontinuity of a chain ending in the j-th
        # candidate of the current frame. Transition cost is divided by the frame gap so a
        # detection gap is not mistaken for a jump.
        fids = sorted(frames)
        cost = [0.0] * len(frames[fids[0]])
        back: list[list[int]] = []
        for prev_fid, fid in zip(fids, fids[1:]):
            span = fid - prev_fid
            prev_cands, cands = frames[prev_fid], frames[fid]
            new_cost, choice = [], []
            for cand in cands:
                # min() over (cost, index) breaks ties toward the earlier candidate, so the
                # result does not depend on how the rows happened to be ordered in the file.
                best, arg = min(
                    (cost[i] + _shape_cost(prev, cand) / span, i)
                    for i, prev in enumerate(prev_cands)
                )
                new_cost.append(best)
                choice.append(arg)
            back.append(choice)
            cost = new_cost

        idx = min(range(len(cost)), key=lambda i: (cost[i], i))
        chain = [idx]
        for choice in reversed(back):
            idx = choice[idx]
            chain.append(idx)
        chain.reverse()

        for fid, pick in zip(fids, chain):
            cands = frames[fid]
            kept.append(cands[pick])
            dropped += len(cands) - 1

    kept.sort(key=lambda b: (b.fid, b.oid))
    return kept, dropped


def grouped_videos(csv_path: Path):
    """Yield (video_id, objects, bboxes, dropped) per video (csv is grouped by video).

    obj_clevrer.csv columns: oid, video, fid, shape, color, material, x1,y1,x2,y2.
    Boxes pass through :func:`resolve_duplicate_boxes`, so each (oid, fid) appears once.
    """
    cur_vid = None
    objects: dict[int, Obj] = {}
    bboxes: list[Bbox] = []
    with csv_path.open(newline="") as f:
        for row in csv.reader(f):
            oid = int(row[0])
            vid = int(row[1])
            fid = int(row[2])
            shape, color, material = row[3], row[4], row[5]
            x1, y1, x2, y2 = float(row[6]), float(row[7]), float(row[8]), float(row[9])
            if vid != cur_vid:
                if cur_vid is not None:
                    yield (cur_vid, list(objects.values()), *resolve_duplicate_boxes(bboxes))
                cur_vid, objects, bboxes = vid, {}, []
            if oid not in objects:
                objects[oid] = Obj(oid=oid, cls=f"{color} {material} {shape}")
            bboxes.append(Bbox(fid=fid, oid=oid, x1=x1, y1=y1, x2=x2, y2=y2, aid=None))
    if cur_vid is not None:
        yield (cur_vid, list(objects.values()), *resolve_duplicate_boxes(bboxes))


def convert(raw_root: Path, out_root: Path) -> None:
    out_dataset = out_root / DATASET
    videos = video_index(raw_root / "video_train")
    if not videos:
        print(f"!! no extracted videos under {raw_root/'video_train'} "
              f"(unzip video_train.zip there first)")
        return

    n = 0
    missing = 0
    total_dropped = 0
    for vid, objects, bboxes, dropped in grouped_videos(raw_root / "obj_clevrer.csv"):
        total_dropped += dropped
        name = f"video_{vid:05d}"
        video = videos.get(name)
        if video is None:
            missing += 1
            continue
        out_dir = out_dataset / name
        metadata = {
            "dataset": DATASET,
            "source": "CLEVRER",
            "split": "train",
            "video_id": vid,
            "obj_classes": sorted({o.cls for o in objects}),
            "activity_classes": [],
            "notes": (
                "Class is '<color> <material> <shape>' (CLEVRER objects are "
                "uniquely identified by this combination). fid is the 0-based "
                "frame index. Collision events are not present in this annotation "
                "source, so activities are absent (aid=NULL)."
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
        n += 1
        if n % 1000 == 0 or n <= 2:
            rc = meta["row_counts"]
            print(f"[{n}] {name}: {rc['objects']} objects, {rc['bbox']} bboxes -> {out_dir}")
    print(f"\nDone: wrote {n} subset(s) under {out_dataset}"
          + (f" ({missing} videos missing from video_train/)" if missing else ""))
    print(f"Resolved {total_dropped} duplicate box(es) to one per (oid, fid).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw", type=Path, default=DATASETS_ROOT / "raw" / "clevrer")
    parser.add_argument("--out", type=Path, default=DATASETS_ROOT / "processed")
    args = parser.parse_args()
    convert(args.raw.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
