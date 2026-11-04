#!/usr/bin/env python
"""Convert the CityFlow-NL dataset into the TrQL common format.

CityFlow-NL (AI City 2023, NL-based vehicle retrieval) provides vehicle *tracks*
-- each a single vehicle followed through one camera, with per-frame boxes and
(for the train split) natural-language descriptions.

Raw layout expected (see README.md)::

    raw/cityflow/data/train-tracks.json   # 2155 tracks (with nl)
    raw/cityflow/data/test-tracks.json    # 184 tracks (no nl)
    raw/cityflow/data/<split>/<scene>/<camera>/vdo.avi   # camera videos

Tracks are grouped by camera; each camera becomes one subset::

    <out>/cityflow/<split>_<scene>_<camera>/video.mp4  (symlink to vdo.avi/mp4)
                                            /duckdb.db
                                            /metadata.json

A track entry has ``frames`` (image paths like ``./validation/S02/c006/img1/
000007.jpg``) and ``boxes`` ([x, y, w, h] per frame).

  * objects.oid   = per-camera track index; class = "vehicle" (no structured
                    class is provided -- only NL text). The track UUID and NL
                    descriptions are preserved per object in metadata.json.
  * bbox          = one row per track frame; aid is always NULL.
  * activities    = empty (pure tracking dataset).

Run with the project venv::

    .trqs-pyenv/bin/python converters/cityflow_convert.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_ROOT = SCRIPT_DIR.parent / "datasets"
sys.path.insert(0, str(SCRIPT_DIR))

from common import Obj, Bbox, build_subset  # noqa: E402

DATASET = "cityflow"
DATA_SUBDIR = "data"
TRACK_FILES = {"train": "train-tracks.json", "test": "test-tracks.json"}


def camera_of(frame_path: str) -> str:
    """'./validation/S02/c006/img1/000007.jpg' -> 'validation/S02/c006'."""
    return "/".join(frame_path.strip("./").split("/")[:3])


def frame_num(frame_path: str) -> int:
    return int(Path(frame_path).stem)


def load_tracks(data_dir: Path):
    """Group all tracks by camera. Returns camera -> list of track dicts."""
    by_cam: dict[str, list[dict]] = defaultdict(list)
    for split, fname in TRACK_FILES.items():
        path = data_dir / fname
        if not path.is_file():
            print(f"!! {fname} not found; skipping {split} tracks")
            continue
        for uuid, t in json.loads(path.read_text()).items():
            cam = camera_of(t["frames"][0])
            by_cam[cam].append({"uuid": uuid, "split": split, **t})
    return by_cam


def find_video(data_dir: Path, cam: str) -> Path | None:
    base = data_dir / cam
    for name in ("vdo.mp4", "vdo.avi"):
        if (base / name).is_file():
            return base / name
    return None


def convert(raw_root: Path, out_root: Path) -> None:
    data_dir = raw_root / DATA_SUBDIR
    by_cam = load_tracks(data_dir)
    out_dataset = out_root / DATASET
    n = 0
    for cam in sorted(by_cam):
        video = find_video(data_dir, cam)
        if video is None:
            print(f"!! {cam}: no vdo.avi/mp4; skipping")
            continue

        objects: list[Obj] = []
        bboxes: list[Bbox] = []
        track_meta: dict[str, dict] = {}
        for oid, t in enumerate(by_cam[cam]):
            objects.append(Obj(oid=oid, cls="vehicle"))
            track_meta[str(oid)] = {
                "uuid": t["uuid"],
                "split": t["split"],
                "nl": t.get("nl", []),
                "nl_other_views": t.get("nl_other_views", []),
            }
            for frame, box in zip(t["frames"], t["boxes"]):
                x, y, w, h = box
                bboxes.append(
                    Bbox(fid=frame_num(frame) - 1, oid=oid,
                         x1=float(x), y1=float(y), x2=float(x + w), y2=float(y + h), aid=None)
                )

        subset = cam.replace("/", "_")
        out_dir = out_dataset / subset
        metadata = {
            "dataset": DATASET,
            "source": "CityFlow-NL (AI City 2023, NL vehicle retrieval)",
            "camera": cam,
            "obj_classes": ["vehicle"],
            "activity_classes": [],
            "tracks": track_meta,
            "notes": (
                "Each object is a CityFlow-NL track (one vehicle in one camera). "
                "Class is generic 'vehicle' (only natural-language descriptions "
                "are provided, kept per object under 'tracks'). fid = frame number "
                "from the image path minus 1 (0-based, aligned to vdo). video.mp4 "
                "symlinks the source vdo.avi. No activities (aid=NULL)."
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
        print(f"[ok] {subset}: {rc['objects']} tracks, {rc['bbox']} bboxes, "
              f"fps={meta['fps']:.0f}, frames={meta['num_frames']} -> {out_dir}")
        n += 1
    print(f"\nDone: wrote {n} subset(s) under {out_dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw", type=Path, default=DATASETS_ROOT / "raw" / "cityflow")
    parser.add_argument("--out", type=Path, default=DATASETS_ROOT / "processed")
    args = parser.parse_args()
    convert(args.raw.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
