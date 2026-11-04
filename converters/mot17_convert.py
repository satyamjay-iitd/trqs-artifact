#!/usr/bin/env python
"""Convert the raw MOT17 dataset into the TrQL common format.

MOT17 is MOTChallenge pedestrian tracking. Each sequence ships frames (not a
video), a MOT ``gt/gt.txt`` and a ``seqinfo.ini``. Every sequence appears three
times -- once per public detector (``-DPM``, ``-FRCNN``, ``-SDP``) -- but the
images and ground truth are identical across the three, so they are deduplicated
to one subset per base sequence (``MOT17-02``, ``MOT17-04``, ...).

Raw layout expected (see README.md)::

    raw/mot17/<split>/<SEQ>-<DET>/img1/000001.jpg ...
    raw/mot17/<split>/<SEQ>-<DET>/gt/gt.txt
    raw/mot17/<split>/<SEQ>-<DET>/seqinfo.ini

Only the ``train`` split has ground truth (``test`` has none and is skipped).

Each base sequence becomes one subset::

    <out>/mot17/<SEQ>/video.mp4   (encoded from img1 at seqinfo frameRate)
                     /duckdb.db
                     /metadata.json

gt.txt columns (MOT17):
    frame, id, bb_left, bb_top, bb_width, bb_height, flag, class, visibility

  * objects.oid   = track id; objects.class = MOT17 class name (kept rows are
    the pedestrian evaluation targets, flag == 1).
  * bbox          = one row per kept gt detection; aid is always NULL.
  * activities    = empty (pure tracking dataset).

Run with the project venv::

    .trqs-pyenv/bin/python converters/mot17_convert.py
"""

from __future__ import annotations

import argparse
import configparser
import re
import sys
from collections import defaultdict
from pathlib import Path

# Make the shared helpers importable regardless of CWD.
SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_ROOT = SCRIPT_DIR.parent / "datasets"
sys.path.insert(0, str(SCRIPT_DIR))

from common import Obj, Bbox, build_subset  # noqa: E402

DATASET = "mot17"
SPLITS = ("train",)  # "test" has no gt -> skipped

# MOT17 class ids. Only flag==1 rows are kept (the pedestrian eval targets);
# the rest (static persons, distractors, occluders, ...) are ignore regions.
MOT17_CLASS = {
    1: "pedestrian",
    2: "person_on_vehicle",
    3: "car",
    4: "bicycle",
    5: "motorbike",
    6: "non_mot_vehicle",
    7: "static_person",
    8: "distractor",
    9: "occluder",
    10: "occluder_on_ground",
    11: "occluder_full",
    12: "reflection",
    13: "crowd",
}

# strip the trailing detector tag to get the base sequence id
DET_SUFFIX_RE = re.compile(r"-(DPM|FRCNN|SDP)$")


def parse_gt(path: Path) -> tuple[list[Obj], list[Bbox], int]:
    """Parse gt.txt, keeping only flag==1 rows. Returns (objects, bboxes, skipped)."""
    objects: dict[int, Obj] = {}
    bboxes: list[Bbox] = []
    skipped = 0
    for line in path.read_text().splitlines():
        p = line.strip().split(",")
        if not p or not p[0]:
            continue
        frame = int(p[0])
        tid = int(p[1])
        left, top, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
        flag = int(p[6])
        cls_id = int(p[7])
        if flag != 1 or w <= 0 or h <= 0:
            skipped += 1
            continue
        if tid not in objects:
            objects[tid] = Obj(oid=tid, cls=MOT17_CLASS.get(cls_id, f"class{cls_id}"))
        bboxes.append(
            Bbox(fid=frame - 1, oid=tid, x1=left, y1=top, x2=left + w, y2=top + h, aid=None)
        )
    return list(objects.values()), bboxes, skipped


def read_ini(path: Path) -> dict:
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return {k: v for section in cfg.sections() for k, v in cfg.items(section)}


def iter_sequences(raw_root: Path, splits):
    """Yield (split, base_seq, chosen_dir, [variant tags]) deduplicated by base seq."""
    for split in splits:
        split_dir = raw_root / split
        if not split_dir.is_dir():
            print(f"!! split '{split}' not found under {raw_root}; skipping")
            continue
        groups: dict[str, list[Path]] = defaultdict(list)
        for d in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            base = DET_SUFFIX_RE.sub("", d.name)
            groups[base].append(d)
        for base, dirs in sorted(groups.items()):
            variants = [DET_SUFFIX_RE.search(d.name).group(1) for d in dirs if DET_SUFFIX_RE.search(d.name)]
            # all variants share identical img1/gt -> use the first
            yield split, base, dirs[0], sorted(variants)


def convert(raw_root: Path, out_root: Path, splits) -> None:
    out_dataset = out_root / DATASET
    n = 0
    for split, base, seq_dir, variants in iter_sequences(raw_root, splits):
        gt = seq_dir / "gt" / "gt.txt"
        img_dir = seq_dir / "img1"
        if not gt.is_file():
            print(f"!! {base}: no gt.txt in {seq_dir}; skipping")
            continue
        if not img_dir.is_dir():
            print(f"!! {base}: no img1/ in {seq_dir}; skipping")
            continue

        seqinfo = read_ini(seq_dir / "seqinfo.ini")
        fps = float(seqinfo.get("framerate", 30))
        objects, bboxes, skipped = parse_gt(gt)
        obj_classes = sorted({o.cls for o in objects})

        out_dir = out_dataset / base
        metadata = {
            "dataset": DATASET,
            "source": "MOT17 (MOTChallenge)",
            "split": split,
            "sequence": base,
            "detector_variants": variants,
            "obj_classes": obj_classes,
            "activity_classes": [],
            "notes": (
                "Frames encoded to H.264 at the seqinfo.ini frameRate. fid is "
                "0-based (gt frame - 1). Only flag==1 gt rows (pedestrian eval "
                "targets) are kept; ignore regions (static persons, distractors, "
                "occluders, ...) are dropped. The -DPM/-FRCNN/-SDP detector "
                "variants share identical frames/gt and are deduplicated to one "
                "subset. No activities (aid=NULL)."
            ),
        }
        meta = build_subset(
            out_dir,
            frames_dir=img_dir,
            fps=fps,
            frames_pattern="*.jpg",
            objects=objects,
            bboxes=bboxes,
            activities=(),
            metadata=metadata,
        )
        rc = meta["row_counts"]
        print(
            f"[ok] {base}: {rc['objects']} objects, {rc['bbox']} bboxes "
            f"({skipped} ignore dropped), classes={obj_classes}, fps={meta['fps']:.0f}, "
            f"frames={meta['num_frames']}, variants={variants} -> {out_dir}"
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
        default=DATASETS_ROOT / "raw" / "mot17",
        help="Raw mot17 root (default: datasets/raw for this dataset).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATASETS_ROOT / "processed",
        help="Output root; a 'mot17/' subdir is created under it.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLITS),
        help="Splits to convert (default: train). 'test' has no gt.",
    )
    args = parser.parse_args()
    convert(args.raw.resolve(), args.out.resolve(), args.splits)


if __name__ == "__main__":
    main()
