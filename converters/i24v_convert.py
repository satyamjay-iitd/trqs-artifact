#!/usr/bin/env python
"""Convert the I-24V (WACV 2024) dataset into the TrQL common format.

I-24V ("So you think you can track?") is highway traffic captured by many fixed
cameras. Ground truth is a set of vehicle trajectories in *3D roadway
coordinates* (feet), not image boxes. To fit the TrQL image-bbox schema, each
trajectory point is projected into a camera's image using the I-24 roadway
coordinate system (the ``i24_rcs`` homography), then scaled from the 4K
reference resolution the homography was calibrated on down to the video
resolution.

Raw layout expected (see README.md)::

    raw/I24V_WACV2024/video/<CAM>_<id>.mkv
    raw/I24V_WACV2024/annotations/final_gps.csv          # dense 10Hz GT tracks
    raw/I24V_WACV2024/annotations/WACV2024_hg_save.cpkl  # i24_rcs homography

Each provided camera video becomes one subset::

    <out>/i24v/<CAM>/video.mp4 (symlink to the .mkv)
                    /duckdb.db
                    /metadata.json

final_gps.csv columns: id, State Plane X/Y, Roadway X/Y, Timestamp (s),
Length/Width/Height (ft).

  * objects.oid   = integer index of the vehicle id; class = "vehicle".
  * bbox          = one row per GT point that projects inside the camera frame.
  * activities    = empty (pure tracking dataset).

Requires the installed ``i24_rcs`` 1.0.1 package and dependencies from
``converters/requirements-projection.txt``. See ``converters/README.md`` for setup.

Run with the project venv::

    .trqs-pyenv/bin/python converters/i24v_convert.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_ROOT = SCRIPT_DIR.parent / "datasets"
sys.path.insert(0, str(SCRIPT_DIR))

from common import Obj, Bbox, build_subset, probe_video  # noqa: E402
from i24_rcs import I24_RCS  # noqa: E402

DATASET = "i24v"
CLASS = "vehicle"
# The i24_rcs correspondences were calibrated on 4K reference imagery; projected
# image coordinates are scaled by (video_w/REF_W, video_h/REF_H) to the video.
REF_W, REF_H = 3840, 2160
FOV_BUFFER_FT = 75.0  # padding around a camera's roadway-x window when prefiltering


def camera_x_window(hg: I24_RCS, key: str) -> tuple[float, float]:
    """Roadway-x extent visible to a camera, from its correspondence points."""
    sp = torch.tensor(hg.correspondence[key]["space_pts"], dtype=torch.double)  # [n,2]
    pts = torch.cat([sp, torch.zeros((sp.shape[0], 1), dtype=torch.double)], dim=1)
    state = hg.space_to_state(pts.unsqueeze(1))  # [n,6] -> x,y,...
    xs = state[:, 0]
    return float(xs.min()) - FOV_BUFFER_FT, float(xs.max()) + FOV_BUFFER_FT


def load_gps(path: Path):
    """Yield (vid, x, y, L, W, H, t) for every GT trajectory point."""
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            yield (
                row["id"],
                float(row["Roadway X (ft)"]),
                float(row["Roadway Y (ft)"]),
                float(row["Length (ft)"]),
                float(row["Width (ft)"]),
                float(row["Height (ft)"]),
                float(row["Timestamp (s)"]),
            )


def project_camera(hg: I24_RCS, key: str, gps_rows, fps: float, num_frames: int,
                   scale_x: float, scale_y: float, vw: int, vh: int):
    """Project the GT points that fall in this camera's window into image bboxes.

    Returns (objects, bboxes, oid_map). A point is kept only if its projected
    box overlaps the frame *and* its frame index lies within the video
    ([0, num_frames)); the per-vehicle string id is mapped to an int oid.
    """
    xmin_w, xmax_w = camera_x_window(hg, key)
    # prefilter to the camera's roadway-x window (cheap) before projecting
    sel = [r for r in gps_rows if xmin_w <= r[1] <= xmax_w]
    if not sel:
        return [], [], {}

    vid = [r[0] for r in sel]
    state = torch.tensor(
        [[r[1], r[2], r[3], r[4], r[5], (1.0 if r[2] >= 0 else -1.0)] for r in sel],
        dtype=torch.double,
    )
    times = torch.tensor([r[6] for r in sel], dtype=torch.double)

    im = hg.state_to_im(state, name=[key] * len(sel))  # [n,8,2] in 4K ref coords
    im[..., 0] *= scale_x
    im[..., 1] *= scale_y
    x1 = im[:, :, 0].min(1).values
    x2 = im[:, :, 0].max(1).values
    y1 = im[:, :, 1].min(1).values
    y2 = im[:, :, 1].max(1).values

    overlap = (x2 > 0) & (x1 < vw) & (y2 > 0) & (y1 < vh) & (x2 > x1) & (y2 > y1)

    oid_map: dict[str, int] = {}
    objects: dict[int, Obj] = {}
    bboxes: list[Bbox] = []
    idx = overlap.nonzero().flatten().tolist()
    for i in idx:
        fid = int(round(float(times[i]) * fps))
        if fid < 0 or (num_frames and fid >= num_frames):
            continue  # GT point outside this video's time window
        v = vid[i]
        oid = oid_map.setdefault(v, len(oid_map))
        if oid not in objects:
            objects[oid] = Obj(oid=oid, cls=CLASS)
        bboxes.append(
            Bbox(fid=fid, oid=oid, x1=float(x1[i]), y1=float(y1[i]),
                 x2=float(x2[i]), y2=float(y2[i]), aid=None)
        )
    return list(objects.values()), bboxes, oid_map


def find_camera_key(hg: I24_RCS, cam: str) -> str | None:
    keys = [k for k in hg.correspondence if k.split("_")[0] == cam]
    return keys[0] if keys else None


def convert(raw_root: Path, out_root: Path) -> None:
    anno = raw_root / "annotations"
    hg = I24_RCS(str(anno / "WACV2024_hg_save.cpkl"))
    print("loading GT trajectories ...")
    gps_rows = list(load_gps(anno / "final_gps.csv"))
    print(f"  {len(gps_rows)} GT points")

    out_dataset = out_root / DATASET
    n = 0
    videos = sorted(
        v for v in (raw_root / "video").iterdir()
        if v.suffix.lower() in {".mp4", ".mkv"}
    )
    for video in videos:
        cam = video.stem.split("_")[0]
        key = find_camera_key(hg, cam)
        if key is None:
            print(f"!! {cam}: no homography correspondence; skipping")
            continue

        info = probe_video(video)
        fps = info["fps"]
        vw, vh = info["width"], info["height"]
        num_frames = info["num_frames"] or 0
        scale_x, scale_y = vw / REF_W, vh / REF_H
        objects, bboxes, oid_map = project_camera(
            hg, key, gps_rows, fps, num_frames, scale_x, scale_y, vw, vh
        )

        out_dir = out_dataset / cam
        metadata = {
            "dataset": DATASET,
            "source": "I-24V / 'So you think you can track?' (WACV 2024)",
            "camera": cam,
            "direction": key.split("_")[-1],
            "homography_key": key,
            "obj_classes": [CLASS],
            "activity_classes": [],
            "ref_resolution": [REF_W, REF_H],
            "video_resolution": [vw, vh],
            "id_map": {str(o): k for k, o in oid_map.items()},
            "notes": (
                "Vehicle GT is 3D roadway-coordinate trajectories (final_gps.csv, "
                "10Hz) projected to image via i24_rcs and scaled from the 4K "
                "reference resolution to the video. Only points whose box overlaps "
                "the frame are kept. fid = round(timestamp * fps), assuming the GT "
                "clock is synchronized to video frame 0. No activities (aid=NULL)."
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
            f"[ok] {cam} ({key}): {rc['objects']} vehicles, {rc['bbox']} bboxes, "
            f"fps={meta['fps']:.0f} -> {out_dir}"
        )
        n += 1
    print(f"\nDone: wrote {n} subset(s) under {out_dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw", type=Path, default=DATASETS_ROOT / "raw" / "I24V_WACV2024")
    parser.add_argument("--out", type=Path, default=DATASETS_ROOT / "processed")
    args = parser.parse_args()
    convert(args.raw.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
