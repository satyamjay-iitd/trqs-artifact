#!/usr/bin/env python
"""Convert the raw YTStream (MIRIS) dataset into the TrQL common format.

Raw layout expected (see README.md)::

    raw/ytstream/<scene>/videos/<id>.mp4
    raw/ytstream/<scene>/json/<id>-baseline.json     # tracked  (track_id set)
    raw/ytstream/<scene>/json/<id>-detections.json   # optional (track_id == -1)

Each ``<scene>/<id>`` video becomes one subset::

    <out>/ytstream/<scene>_<id>/video.mp4
                               /duckdb.db
                               /metadata.json

YTStream provides only bounding boxes + track ids (no class label, no activity
labels), so:
  * ``objects.oid``  = track id; ``objects.class`` = the per-scene class below.
  * ``bbox``         = one row per detection; ``aid`` is always NULL.
  * ``activities``   = empty (pure tracking dataset).

The JSON boxes are in the original 1920x1080 stream's coordinates while the
distributed videos are downscaled, so boxes are rescaled to each video's probed
dimensions (see ``ANNOTATION_SIZE`` / ``--ann-size``).

Run with the project venv::

    .trqs-pyenv/bin/python converters/ytstream_convert.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the shared helpers importable regardless of CWD.
SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_ROOT = SCRIPT_DIR.parent / "datasets"
sys.path.insert(0, str(SCRIPT_DIR))

from common import Obj, Bbox, build_subset, probe_video  # noqa: E402

DATASET = "ytstream"

# The YTStream JSONs were produced on the original 1920x1080 streams, but the
# distributed .mp4s are downscaled (960x540 in the reference copy), so the raw
# box coordinates do not line up with the video pixels. Boxes are rescaled by
# (video_width / ANNOTATION_W, video_height / ANNOTATION_H) per video; override
# with --ann-size if your copy annotates a different resolution.
ANNOTATION_SIZE = (1920, 1080)

# YTStream JSONs carry no class label. These are the dominant object class per
# scene in the MIRIS YTStream videos; see README.md ("Assumptions"). Edit here
# if you disagree with a scene's labeling.
SCENE_CLASS = {
    "beach": "person",
    "shibuya": "person",
    "warsaw": "car",
    "uav": "car",
}

BASELINE_SUFFIX = "-baseline.json"
DETECTIONS_SUFFIX = "-detections.json"


def parse_baseline(
    path: Path,
    cls: str,
    *,
    size: tuple[int, int],
    ann_size: tuple[int, int],
) -> tuple[list[Obj], list[Bbox]]:
    """Parse a ``*-baseline.json`` file into objects and bboxes.

    The file is a JSON list indexed by frame; each element is a list of
    detections (or ``null`` for a frame with none). Detections with a negative
    track id (unmatched) carry no object identity and are skipped.

    Coordinates are annotated against ``ann_size`` and rescaled to the actual
    video ``size``, then clipped to the frame (rounding on the source side can
    push a box a fraction of a pixel past the edge).
    """
    width, height = size
    ann_w, ann_h = ann_size
    sx, sy = width / ann_w, height / ann_h

    frames = json.loads(path.read_text())
    objects: dict[int, Obj] = {}
    bboxes: list[Bbox] = []
    skipped = 0
    max_x = max_y = 0
    for frame in frames:
        if not frame:  # None or empty list
            continue
        for det in frame:
            tid = det["track_id"]
            if tid < 0:
                skipped += 1
                continue
            if tid not in objects:
                objects[tid] = Obj(oid=tid, cls=cls)
            max_x = max(max_x, det["right"])
            max_y = max(max_y, det["bottom"])
            bboxes.append(
                Bbox(
                    fid=det["frame_idx"],
                    oid=tid,
                    x1=min(max(det["left"] * sx, 0.0), width),
                    y1=min(max(det["top"] * sy, 0.0), height),
                    x2=min(max(det["right"] * sx, 0.0), width),
                    y2=min(max(det["bottom"] * sy, 0.0), height),
                    aid=None,
                )
            )
    if skipped:
        print(f"    note: skipped {skipped} detections with track_id < 0")
    if max_x > ann_w or max_y > ann_h:
        print(
            f"    !! boxes reach ({max_x}, {max_y}), past the assumed annotation "
            f"size {ann_w}x{ann_h} -- check --ann-size"
        )
    return list(objects.values()), bboxes


def iter_videos(raw_root: Path):
    """Yield ``(scene, video_id, baseline_path, video_path)`` tuples."""
    for scene in sorted(p.name for p in raw_root.iterdir() if p.is_dir()):
        json_dir = raw_root / scene / "json"
        video_dir = raw_root / scene / "videos"
        if not json_dir.is_dir() or not video_dir.is_dir():
            continue
        for baseline in sorted(json_dir.glob(f"*{BASELINE_SUFFIX}")):
            vid = baseline.name[: -len(BASELINE_SUFFIX)]
            video = video_dir / f"{vid}.mp4"
            yield scene, vid, baseline, video


def convert(raw_root: Path, out_root: Path, ann_size: tuple[int, int]) -> None:
    out_dataset = out_root / DATASET
    n = 0
    for scene, vid, baseline, video in iter_videos(raw_root):
        if scene not in SCENE_CLASS:
            print(f"!! skipping scene '{scene}': no class mapping in SCENE_CLASS")
            continue
        if not video.exists():
            print(f"!! skipping {scene}/{vid}: video not found at {video}")
            continue

        # The video is the ground truth for the coordinate space we emit, so it
        # has to be probed before parsing rather than after (as build_subset
        # does for the rest of the metadata).
        info = probe_video(video)
        size = (info["width"], info["height"])
        if not all(size):
            print(f"!! skipping {scene}/{vid}: ffprobe reported no dimensions")
            continue

        cls = SCENE_CLASS[scene]
        objects, bboxes = parse_baseline(baseline, cls, size=size, ann_size=ann_size)

        subset = f"{scene}_{vid}"
        out_dir = out_dataset / subset
        detections = baseline.with_name(f"{vid}{DETECTIONS_SUFFIX}")
        metadata = {
            "dataset": DATASET,
            "source": "MIRIS: Fast Object Track Queries in Video (YTStream)",
            "scene": scene,
            "video_id": vid,
            "obj_classes": [cls],
            "activity_classes": [],
            "has_raw_detections": detections.exists(),
            "annotation_size": list(ann_size),
            "bbox_scale": [size[0] / ann_size[0], size[1] / ann_size[1]],
            "notes": (
                "Class is assumed per-scene (no label in source JSON); "
                "activities are absent (pure tracking dataset). Bboxes were "
                f"rescaled from the {ann_size[0]}x{ann_size[1]} annotation "
                f"space to the {size[0]}x{size[1]} video."
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
        sx, sy = meta["bbox_scale"]
        print(
            f"[ok] {subset}: {rc['objects']} objects, {rc['bbox']} bboxes, "
            f"fps={meta['fps']}, frames={meta['num_frames']}, "
            f"bbox scale={sx:g}x{sy:g} -> {out_dir}"
        )
        n += 1
    print(f"\nDone: wrote {n} subset(s) under {out_dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--raw",
        type=Path,
        default=DATASETS_ROOT / "raw" / "ytstream",
        help="Raw ytstream root (default: datasets/raw for this dataset).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATASETS_ROOT / "processed",
        help="Output root; a '%s/' subdir is created under it (default: %s)."
        % (DATASET, DATASETS_ROOT / "processed"),
    )
    parser.add_argument(
        "--ann-size",
        default="%dx%d" % ANNOTATION_SIZE,
        metavar="WxH",
        help="Resolution the JSON boxes are annotated against; they are "
        "rescaled to each video's actual size (default: %(default)s).",
    )
    args = parser.parse_args()
    try:
        ann_w, ann_h = (int(v) for v in args.ann_size.lower().split("x"))
        if ann_w <= 0 or ann_h <= 0:
            raise ValueError
    except ValueError:
        parser.error(f"--ann-size must look like 1920x1080, got {args.ann_size!r}")
    convert(args.raw.resolve(), args.out.resolve(), (ann_w, ann_h))


if __name__ == "__main__":
    main()
