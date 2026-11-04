#!/usr/bin/env python
"""Convert the AI City 2022 Track 1 (CityFlowV2 MTMC) dataset to the TrQL format.

AIC22 Track 1 is multi-target multi-camera (MTMC) vehicle tracking: 46 cameras
arranged in 6 scenarios (S01..S06), each camera a fixed traffic-surveillance
view. Ground truth is image-space bounding boxes in MOTChallenge format, with a
*global per-scenario* vehicle ID -- the same ``ID`` denotes the same physical
vehicle across every camera in that scenario. We preserve that ID directly as
``oid`` so cross-camera identity survives into the per-camera subsets.

Raw layout (see ReadMe.txt)::

    <split>/<scenario>/<cam>/vdo.avi              # the video (10 fps; c015@8fps)
    <split>/<scenario>/<cam>/gt/gt.txt            # MOTChallenge GT (train/val only)
    <split>/<scenario>/<cam>/calibration.txt      # 3x3 ground-plane homography
    <split>/<scenario>/<cam>/roi.jpg              # region of interest mask
    cam_timestamp/<scenario>.txt                  # per-cam start time (seconds)

  split in {train, validation, test}; only train/validation carry gt.txt.

gt.txt columns: ``frame, ID, left, top, width, height, 1, -1, -1, -1`` -- frame
is 1-based; box is pixel (left, top) + (width, height).

Each *scenario-camera* recording becomes one subset (camera IDs are reused
across scenarios, e.g. S04 and S05 both have c016, so the subset name carries
the scenario)::

    <out>/aicity22/<scenario>_<cam>/video.mp4   (symlink to vdo.avi)
                                   /duckdb.db
                                   /metadata.json

  * objects.oid = the MOTChallenge vehicle ID; class = "vehicle".
  * bbox        = one row per GT box; fid = frame - 1 (0-based); aid = NULL.
  * activities  = empty (pure tracking dataset).

Cameras without gt.txt (the test scenario S06) are skipped: no GT to convert.

Run with the project venv::

    .trqs-pyenv/bin/python converters/aicity22_convert.py
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

DATASET = "aicity22"
CLASS = "vehicle"
SPLITS = ("train", "validation", "test")

# Approximate scenario GPS (centre of each scenario; from ReadMe.txt). S03/S04/S05
# share the S0345 map location.
SCENARIO_GPS = {
    "S01": [42.525678, -90.723601],
    "S02": [42.491916, -90.723723],
    "S03": [42.498780, -90.686393],
    "S04": [42.498780, -90.686393],
    "S05": [42.498780, -90.686393],
    "S06": [42.492448, -90.723343],
}


def load_homography(path: Path) -> list[list[float]] | None:
    """Parse the 3x3 ground-plane homography from a calibration.txt, if present."""
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith("Homography matrix:"):
            body = line.split(":", 1)[1].strip()
            return [[float(v) for v in row.split()] for row in body.split(";")]
    return None


def load_start_times(scenario_dir: Path, scenario: str) -> dict[str, float]:
    """Map cam -> start timestamp (s) from cam_timestamp/<scenario>.txt, if present."""
    ts_file = scenario_dir.parent.parent / "cam_timestamp" / f"{scenario}.txt"
    times: dict[str, float] = {}
    if ts_file.exists():
        for line in ts_file.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                times[parts[0]] = float(parts[1])
    return times


def load_gt(gt_path: Path):
    """Yield (fid, oid, x1, y1, x2, y2) for every MOTChallenge GT box.

    frame is 1-based in the file; we emit 0-based fid. The box is stored as
    (left, top)-(left+w, top+h) and kept as-authored (not clamped to frame).
    """
    with gt_path.open(newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            frame, oid, left, top, w, h = (float(row[i]) for i in range(6))
            yield (
                int(frame) - 1,
                int(oid),
                left,
                top,
                left + w,
                top + h,
            )


def convert_camera(cam_dir: Path, split: str, scenario: str, out_dir: Path,
                   start_times: dict[str, float]) -> dict | None:
    cam = cam_dir.name
    video = cam_dir / "vdo.avi"
    gt_path = cam_dir / "gt" / "gt.txt"
    if not video.exists():
        print(f"!! {scenario}/{cam}: no vdo.avi; skipping")
        return None
    if not gt_path.exists():
        print(f"!! {scenario}/{cam}: no gt.txt ({split}); skipping")
        return None

    objects: dict[int, Obj] = {}
    bboxes: list[Bbox] = []
    for fid, oid, x1, y1, x2, y2 in load_gt(gt_path):
        if oid not in objects:
            objects[oid] = Obj(oid=oid, cls=CLASS)
        bboxes.append(Bbox(fid=fid, oid=oid, x1=x1, y1=y1, x2=x2, y2=y2, aid=None))

    metadata = {
        "dataset": DATASET,
        "source": "AI City Challenge 2022, Track 1 (CityFlowV2 MTMC)",
        "split": split,
        "scenario": scenario,
        "camera": cam,
        "obj_classes": [CLASS],
        "activity_classes": [],
        "scenario_gps": SCENARIO_GPS.get(scenario),
        "start_time_s": start_times.get(cam),
        "homography": load_homography(cam_dir / "calibration.txt"),
        "notes": (
            "MOTChallenge GT boxes (gt.txt); oid is the per-scenario MTMC vehicle "
            "ID (same vehicle across cameras in a scenario). fid = frame - 1 "
            "(GT frames are 1-based). Only vehicles passing through >=2 cameras "
            "are annotated. Single class 'vehicle'; no activities (aid=NULL)."
        ),
    }
    meta = build_subset(
        out_dir,
        video_src=video,
        objects=list(objects.values()),
        bboxes=bboxes,
        activities=(),
        metadata=metadata,
    )
    rc = meta["row_counts"]
    print(
        f"[ok] {scenario}_{cam} ({split}): {rc['objects']} vehicles, "
        f"{rc['bbox']} bboxes, fps={meta['fps']:.0f} -> {out_dir}"
    )
    return meta


def convert(raw_root: Path, out_root: Path) -> None:
    out_dataset = out_root / DATASET
    n = 0
    for split in SPLITS:
        split_dir = raw_root / split
        if not split_dir.is_dir():
            continue
        for scenario_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            scenario = scenario_dir.name
            start_times = load_start_times(scenario_dir, scenario)
            for cam_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
                out_dir = out_dataset / f"{scenario}_{cam_dir.name}"
                if convert_camera(cam_dir, split, scenario, out_dir, start_times):
                    n += 1
    print(f"\nDone: wrote {n} subset(s) under {out_dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw", type=Path, default=DATASETS_ROOT / "raw" / "AICity22_Track1_MTMC_Tracking")
    parser.add_argument("--out", type=Path, default=DATASETS_ROOT / "processed")
    args = parser.parse_args()
    convert(args.raw.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
