#!/usr/bin/env python
"""Convert the raw TrackID3x3 dataset into the TrQL common format.

TrackID3x3 is 3x3 basketball player tracking with three camera categories
(Indoor, Outdoor, Drone). Ground truth is MOT-style; all tracked objects are
players (single class), and there are no activity labels.

Raw layout expected (see README.md)::

    raw/TrackID3x3/videos/Indoor/raw/<name>.mp4
    raw/TrackID3x3/videos/Outdoor/top/<name>.MOV
    raw/TrackID3x3/videos/Drone/split/<vid>/<clip>.mp4
    raw/TrackID3x3/ground_truth/Indoor/MOT/<name>.txt
    raw/TrackID3x3/ground_truth/Outdoor/MOT/<name>.txt
    raw/TrackID3x3/ground_truth/Drone/MOT/<vid>/<segment>.txt

Each MOT file becomes one subset (a per-video file for Indoor/Outdoor, a
per-segment file for Drone)::

    <out>/trackid3x3/<Category>_<name>/video.mp4   (symlink)
                                      /duckdb.db
                                      /metadata.json

MOT columns: frame, track_id, bb_left, bb_top, bb_width, bb_height, conf, ...

Frame numbering differs by source (see README.md). Each Drone segment has its
own pre-split clip under videos/Drone/split/<vid>/, so every subset aligns to
its own video:
  * Indoor/Outdoor: 1-based, aligned to the full video   -> fid = frame - 1
  * Drone "<start>_<end>_gt.txt": 1-based within the clip -> fid = frame - 1
  * Drone "<vid>_NN.txt": 0-based within the clip         -> fid = frame

Run with the project venv::

    .trqs-pyenv/bin/python converters/trackid3x3_convert.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Make the shared helpers importable regardless of CWD.
SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_ROOT = SCRIPT_DIR.parent / "datasets"
sys.path.insert(0, str(SCRIPT_DIR))

from common import Obj, Bbox, build_subset  # noqa: E402

DATASET = "trackid3x3"
CLASS = "player"

# Drone segment files named like "40_1215_gt.txt": <start>_<end>_gt, frames are
# 1-based relative to <start>.
DRONE_RANGE_RE = re.compile(r"^(\d+)_(\d+)_gt$")


class SubsetSpec:
    """One MOT file to convert into a subset."""

    def __init__(self, subset, category, video, gt, sep, frame_offset, alignment, segment=None):
        self.subset = subset
        self.category = category
        self.video = video
        self.gt = gt
        self.sep = sep  # "," or None (whitespace)
        self.frame_offset = frame_offset  # fid = frame + frame_offset
        self.alignment = alignment  # "absolute" | "segment_local"
        self.segment = segment


def iter_subsets(raw_root: Path):
    gt_root = raw_root / "ground_truth"
    vid_root = raw_root / "videos"

    # Indoor / Outdoor: one MOT .txt per video, comma separated, 1-based.
    for category, vid_subdir, ext in (("Indoor", "raw", ".mp4"), ("Outdoor", "top", ".MOV")):
        mot_dir = gt_root / category / "MOT"
        if not mot_dir.is_dir():
            continue
        for gt in sorted(mot_dir.glob("*.txt")):
            video = vid_root / category / vid_subdir / f"{gt.stem}{ext}"
            yield SubsetSpec(
                subset=f"{category}_{gt.stem}",
                category=category,
                video=video,
                gt=gt,
                sep=",",
                frame_offset=-1,
                alignment="absolute",
            )

    # Drone: one MOT file per segment, under a per-video subdirectory. Each
    # segment has its own pre-split video clip under videos/Drone/split/<vid>/,
    # so every segment aligns to its own clip (segment-local frames -> fid).
    drone_mot = gt_root / "Drone" / "MOT"
    split_root = vid_root / "Drone" / "split"
    if drone_mot.is_dir():
        for vid_dir in sorted(p for p in drone_mot.iterdir() if p.is_dir()):
            vid = vid_dir.name
            clip_dir = split_root / vid
            for gt in sorted(vid_dir.glob("*.txt")):
                m = DRONE_RANGE_RE.match(gt.stem)
                if m:  # "<start>_<end>_gt": whitespace, 1-based within the clip
                    seg = f"{m.group(1)}_{m.group(2)}"
                    clips = sorted(clip_dir.glob(f"*_{seg}.mp4"))
                    sep, frame_offset = None, -1  # fid = frame - 1
                else:  # "<vid>_NN": comma, already 0-based within the clip
                    seg = gt.stem[len(vid) + 1 :] if gt.stem.startswith(vid + "_") else gt.stem
                    clips = sorted(clip_dir.glob(f"{gt.stem}.mp4"))
                    sep, frame_offset = ",", 0  # fid = frame
                if not clips:
                    print(f"!! Drone_{vid}_{seg}: no split clip in {clip_dir}; skipping")
                    continue
                yield SubsetSpec(
                    subset=f"Drone_{vid}_{seg}",
                    category="Drone",
                    video=clips[0],
                    gt=gt,
                    sep=sep,
                    frame_offset=frame_offset,
                    alignment="absolute",
                    segment=seg,
                )


def parse_mot(spec: SubsetSpec) -> tuple[list[Obj], list[Bbox], int]:
    """Parse a MOT file into objects/bboxes. Returns (objects, bboxes, skipped)."""
    objects: dict[int, Obj] = {}
    bboxes: list[Bbox] = []
    skipped = 0
    for line in spec.gt.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        p = line.split(spec.sep) if spec.sep else line.split()
        frame = int(float(p[0]))
        tid = int(float(p[1]))
        left, top, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
        if w <= 0 or h <= 0:
            skipped += 1
            continue
        if tid not in objects:
            objects[tid] = Obj(oid=tid, cls=CLASS)
        bboxes.append(
            Bbox(
                fid=frame + spec.frame_offset,
                oid=tid,
                x1=left,
                y1=top,
                x2=left + w,
                y2=top + h,
                aid=None,
            )
        )
    return list(objects.values()), bboxes, skipped


def convert(raw_root: Path, out_root: Path) -> None:
    out_dataset = out_root / DATASET
    n = 0
    for spec in iter_subsets(raw_root):
        if not spec.video.is_file():
            print(f"!! {spec.subset}: video missing ({spec.video}); skipping")
            continue
        objects, bboxes, skipped = parse_mot(spec)
        fids = [b.fid for b in bboxes]
        frame_range = [min(fids), max(fids)] if fids else None

        out_dir = out_dataset / spec.subset
        metadata = {
            "dataset": DATASET,
            "source": "TrackID3x3 (MMSports'25, arXiv:2503.18282)",
            "category": spec.category,
            "segment": spec.segment,
            "obj_classes": [CLASS],
            "activity_classes": [],
            "num_players": len(objects),
            "frame_alignment": spec.alignment,
            "annotated_fid_range": frame_range,
            "notes": (
                "Players tracked via MOT ground truth; no activities (aid=NULL). "
                "fid is 0-based and aligns with the subset's video.mp4 (Drone "
                "subsets use the per-segment split clip under videos/Drone/split)."
            ),
        }
        meta = build_subset(
            out_dir,
            video_src=spec.video,
            objects=objects,
            bboxes=bboxes,
            activities=(),
            metadata=metadata,
        )
        rc = meta["row_counts"]
        note = f" ({skipped} zero-box skipped)" if skipped else ""
        flag = "" if spec.alignment == "absolute" else " [segment_local]"
        print(
            f"[ok] {spec.subset}: {rc['objects']} players, {rc['bbox']} bboxes{note}, "
            f"fps={meta['fps']:.3f}{flag} -> {out_dir}"
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
        default=DATASETS_ROOT / "raw" / "TrackID3x3",
        help="Raw TrackID3x3 root (default: datasets/raw for this dataset).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATASETS_ROOT / "processed",
        help="Output root; a 'trackid3x3/' subdir is created under it.",
    )
    args = parser.parse_args()
    convert(args.raw.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
