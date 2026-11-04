#!/usr/bin/env python
"""Convert the raw VisDrone2019-MOT dataset into the TrQL common format.

VisDrone ships each sequence as a directory of JPEG *frames* (no video) plus a
single MOT-style annotation ``.txt``. Each sequence becomes one subset whose
``video.mp4`` is encoded from the frames (see README.md).

Raw layout expected::

    raw/VisDrone/<split>/sequences/<seq>/0000001.jpg ...
    raw/VisDrone/<split>/annotations/<seq>.txt

where ``<split>`` is one of ``VisDrone2019-MOT-{train,val,test-dev}``.

Annotation columns (comma separated), per the VisDrone toolkit::

    frame_index, target_id, bbox_left, bbox_top, bbox_width, bbox_height,
    score, object_category, truncation, occlusion

  * objects.oid   = target_id remapped to a continuous 0-based range;
                    objects.class = category name (table below).
  * bbox          = one row per annotation; aid is always NULL.
  * activities    = empty (pure tracking dataset).

Run with the project venv::

    .trqs-pyenv/bin/python converters/visdrone_convert.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the shared helpers importable regardless of CWD.
SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_ROOT = SCRIPT_DIR.parent / "datasets"
sys.path.insert(0, str(SCRIPT_DIR))

from common import Obj, Bbox, build_subset, probe_video  # noqa: E402

DATASET = "visdrone"
DEFAULT_FPS = 30.0

# split directory name -> short label used in subset names
SPLITS = {
    "VisDrone2019-MOT-train": "train",
    "VisDrone2019-MOT-val": "val",
    "VisDrone2019-MOT-test-dev": "test-dev",
}

# VisDrone object_category id -> class name. Category 0 ("ignored regions") and
# any row with score 0 are dropped (not real, evaluated objects).
CATEGORY = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}


def parse_annotation(path: Path) -> tuple[list[Obj], list[Bbox], int]:
    """Parse one VisDrone MOT annotation file.

    Returns ``(objects, bboxes, n_skipped)``. ``fid`` is made 0-based
    (``frame_index - 1``) so it aligns with the encoded video's first frame.
    Objects take the class of their first kept detection; rows whose target_id
    later appears with a different category are still attributed to the first.

    VisDrone ``target_id``s are sparse (dropping score-0 / ignored rows leaves
    gaps), so they are remapped to a **continuous** 0-based ``oid`` range,
    ordered by original id.
    """
    first_class: dict[int, str] = {}  # target_id -> class of first kept box
    raw: list[tuple[int, int, float, float, float, float]] = []
    skipped = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        f = line.split(",")
        frame_index = int(f[0])
        target_id = int(f[1])
        left, top, w, h = (float(f[2]), float(f[3]), float(f[4]), float(f[5]))
        score = int(f[6])
        category = int(f[7])
        if score == 0 or category not in CATEGORY:
            skipped += 1
            continue
        if target_id not in first_class:
            first_class[target_id] = CATEGORY[category]
        raw.append((frame_index - 1, target_id, left, top, left + w, top + h))

    # Remap sparse target_ids to a continuous 0-based oid range.
    remap = {tid: i for i, tid in enumerate(sorted(first_class))}
    objects = [Obj(oid=remap[tid], cls=first_class[tid]) for tid in sorted(first_class)]
    bboxes = [
        Bbox(fid=fid, oid=remap[tid], x1=x1, y1=y1, x2=x2, y2=y2, aid=None)
        for (fid, tid, x1, y1, x2, y2) in raw
    ]
    return objects, bboxes, skipped


def iter_sequences(raw_root: Path, splits: list[str]):
    """Yield ``(split_label, seq_name, frames_dir, annotation_path)``."""
    for split_dir, label in SPLITS.items():
        if label not in splits:
            continue
        base = raw_root / split_dir
        anno_dir = base / "annotations"
        seq_root = base / "sequences"
        if not anno_dir.is_dir() or not seq_root.is_dir():
            print(f"!! split '{label}' not found under {base}; skipping")
            continue
        for anno in sorted(anno_dir.glob("*.txt")):
            seq = anno.stem
            yield label, seq, seq_root / seq, anno


def convert(
    raw_root: Path, out_root: Path, splits: list[str], fps: float, reuse_video: bool = False
) -> None:
    out_dataset = out_root / DATASET
    n = 0
    for label, seq, frames_dir, anno in iter_sequences(raw_root, splits):
        if not frames_dir.is_dir():
            print(f"!! {label}/{seq}: frames dir missing ({frames_dir}); skipping")
            continue

        objects, bboxes, skipped = parse_annotation(anno)
        used_classes = sorted({o.cls for o in objects})

        # Record the original (pre-encode) frame size for reference.
        first_frame = sorted(frames_dir.glob("*.jpg"))[0]
        img = probe_video(first_frame)

        subset = f"{label}_{seq}"
        out_dir = out_dataset / subset
        metadata = {
            "dataset": DATASET,
            "source": "VisDrone2019-MOT (http://aiskyeye.com/)",
            "split": label,
            "sequence": seq,
            "obj_classes": used_classes,
            "activity_classes": [],
            "category_map": {str(k): v for k, v in CATEGORY.items()},
            "assumed_fps": fps,
            "image_width": img["width"],
            "image_height": img["height"],
            "notes": (
                "Frames encoded to H.264 at an ASSUMED fps (VisDrone ships no "
                "video/fps). fid is 0-based (frame_index - 1). Odd frame sizes "
                "are padded to even on the right/bottom; bbox coords (top-left "
                "origin) are unaffected. Rows with score 0 or category 0 "
                "(ignored regions) are dropped; activities are absent. "
                "target_ids are remapped to a continuous 0-based oid range."
            ),
        }
        meta = build_subset(
            out_dir,
            frames_dir=frames_dir,
            fps=fps,
            frames_pattern="*.jpg",
            objects=objects,
            bboxes=bboxes,
            activities=(),
            metadata=metadata,
            reuse_video=reuse_video,
        )
        rc = meta["row_counts"]
        note = f" ({skipped} skipped)" if skipped else ""
        print(
            f"[ok] {subset}: {rc['objects']} objects, {rc['bbox']} bboxes{note}, "
            f"classes={used_classes}, frames={meta['num_frames']} -> {out_dir}"
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
        default=DATASETS_ROOT / "raw" / "VisDrone",
        help="Raw VisDrone root (default: datasets/raw for this dataset).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATASETS_ROOT / "processed",
        help="Output root; a 'visdrone/' subdir is created under it.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help="Assumed frame rate for encoding (default: %(default)s).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(set(SPLITS.values())),
        default=sorted(set(SPLITS.values())),
        help="Splits to convert (default: all).",
    )
    parser.add_argument(
        "--reuse-video",
        action="store_true",
        help="Reuse an existing video.mp4 instead of re-encoding frames "
        "(only rewrites duckdb.db/metadata.json).",
    )
    args = parser.parse_args()
    convert(args.raw.resolve(), args.out.resolve(), args.splits, args.fps, args.reuse_video)


if __name__ == "__main__":
    main()
