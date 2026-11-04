#!/usr/bin/env python
"""Convert the I24-3D dataset into the TrQL common format.

I24-3D is multi-camera highway tracking with 3D vehicle annotations in roadway
coordinates. Each camera has a real 4K video and a homography; vehicle road
boxes are projected into the image with the dataset's own homography code.

Raw layout expected (see README.txt):

    raw/I24-3D/video/<sceneN>/<cam>.mp4
    raw/I24-3D/data/obj/<sceneN>_annotations.csv
    raw/I24-3D/data/hg/<sceneN>_hg.json

annotations.csv columns: frame, camera, id, x, y, l, w, h, direction, class, gen
(x, y, l, w, h are roadway coordinates / dimensions in feet).

Each scene/camera becomes one subset::

    <out>/i24_3d/<sceneN>_<cam>/video.mp4  (symlink)
                               /duckdb.db
                               /metadata.json

  * objects.oid    = annotation id; class = annotation class.
  * bbox           = the road box projected to the image (2D box of 8 corners),
                     for every annotated frame in that camera.
  * activities     = empty (pure tracking dataset).

Projection uses the vendored ``i24_3d_homography.py`` (from the I24-3D dataset
repo). The homography is calibrated on 1080p imagery, so projected coordinates
are scaled by 2 to the 4K video (verified visually).

Run with the project venv::

    .trqs-pyenv/bin/python converters/i24_3d_convert.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_ROOT = SCRIPT_DIR.parent / "datasets"
sys.path.insert(0, str(SCRIPT_DIR))

from common import Obj, Bbox, build_subset, probe_video  # noqa: E402
from i24_3d_homography import Homography_Wrapper  # noqa: E402

DATASET = "i24_3d"
HG_SCALE = 2.0  # homography is on 1080p imagery; video is 4K


def load_camera_rows(csv_path: Path) -> dict[str, list[dict]]:
    """Group annotation rows by camera."""
    by_cam: dict[str, list[dict]] = defaultdict(list)
    with csv_path.open(newline="") as f:
        for r in csv.DictReader(f):
            by_cam[r["camera"].strip()].append(r)
    return by_cam


def project_rows(hg: Homography_Wrapper, cam: str, rows: list[dict], vw: int, vh: int):
    """Project a camera's annotation rows to image bboxes.

    Returns (objects, bboxes). Boxes are projected with the dataset homography
    and scaled to the video; only boxes overlapping the frame are kept.
    """
    state = torch.tensor(
        [[float(r["x"]), float(r["y"]), float(r["l"]), float(r["w"]),
          float(r["h"]), float(r["direction"] or 1)] for r in rows],
        dtype=torch.double,
    )
    im = hg.state_to_im(state, name=cam) * HG_SCALE  # [d,8,2] in video pixels
    x1 = im[:, :, 0].min(1).values
    x2 = im[:, :, 0].max(1).values
    y1 = im[:, :, 1].min(1).values
    y2 = im[:, :, 1].max(1).values
    overlap = (x2 > 0) & (x1 < vw) & (y2 > 0) & (y1 < vh) & (x2 > x1) & (y2 > y1)

    objects: dict[int, Obj] = {}
    bboxes: list[Bbox] = []
    for i in overlap.nonzero().flatten().tolist():
        r = rows[i]
        oid = int(r["id"])
        if oid not in objects:
            objects[oid] = Obj(oid=oid, cls=r["class"].strip() or "vehicle")
        bboxes.append(
            Bbox(fid=int(r["frame"]), oid=oid,
                 x1=float(x1[i]), y1=float(y1[i]), x2=float(x2[i]), y2=float(y2[i]), aid=None)
        )
    return list(objects.values()), bboxes


def convert(raw_root: Path, out_root: Path) -> None:
    out_dataset = out_root / DATASET
    n = 0
    for scene_dir in sorted((raw_root / "video").glob("scene*")):
        scene = scene_dir.name
        hg_path = raw_root / "data" / "hg" / f"{scene}_hg.json"
        anno_path = raw_root / "data" / "obj" / f"{scene}_annotations.csv"
        if not hg_path.is_file() or not anno_path.is_file():
            print(f"!! {scene}: missing hg or annotations; skipping")
            continue

        hg = Homography_Wrapper(save_file=str(hg_path))
        by_cam = load_camera_rows(anno_path)

        for video in sorted(scene_dir.glob("*.mp4")):
            cam = video.stem
            if cam not in hg.hg1.correspondence or cam not in by_cam:
                print(f"!! {scene}/{cam}: no homography or annotations; skipping")
                continue

            info = probe_video(video)
            vw, vh = info["width"], info["height"]
            objects, bboxes = project_rows(hg, cam, by_cam[cam], vw, vh)
            obj_classes = sorted({o.cls for o in objects})

            subset = f"{scene}_{cam}"
            out_dir = out_dataset / subset
            metadata = {
                "dataset": DATASET,
                "source": "I24-3D dataset (Gloudemans et al.)",
                "scene": scene,
                "camera": cam,
                "obj_classes": obj_classes,
                "activity_classes": [],
                "notes": (
                    "3D roadway-coordinate vehicle annotations projected to the "
                    "image via the I24-3D homography and scaled x2 from 1080p "
                    "calibration to the 4K video. fid = annotation frame index "
                    "(0-based). Only boxes overlapping the frame are kept. No "
                    "activities (aid=NULL)."
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
            print(f"[ok] {subset}: {rc['objects']} objects, {rc['bbox']} bboxes, "
                  f"classes={obj_classes}, fps={meta['fps']:.0f} -> {out_dir}")
            n += 1
    print(f"\nDone: wrote {n} subset(s) under {out_dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw", type=Path, default=DATASETS_ROOT / "raw" / "I24-3D")
    parser.add_argument("--out", type=Path, default=DATASETS_ROOT / "processed")
    args = parser.parse_args()
    convert(args.raw.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
