#!/usr/bin/env python3
"""What it costs TrQS to turn pixels into the trajectories its queries run on.

The VLM reads the video directly, so a comparison that charges TrQS only its
query time is not a comparison at all. This measures the other side: detection
and tracking over the same clips, with the largest YOLO available, through the
same `scripts/detect_track.py` the ingest pipeline actually uses.

Two terms, measured separately because one of them cannot be measured
end-to-end here:

* **Detector + decode + tracker overhead** -- wall clock of the real pipeline.
* **Association** -- the tracker's per-frame cost at the video's true object
  density. COCO-pretrained YOLO detects essentially nothing in a top-down
  aerial view (it has no overhead-vehicle imagery), so the end-to-end run does
  no association work. Replaying the dataset's own boxes through the same
  tracker recovers that term instead of silently omitting it.

The sum is an estimate, and the script says so: it is the honest number to put
on a cost axis next to a VLM's end-to-end latency, not a measured pipeline that
also produced these particular tracks.

Run with the GPU environment:

    .venv-detect/bin/python suite/vlm/ingest_cost.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import duckdb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import PROCESSED, ROOT, SCRATCH, VLM

_REPO = ROOT
_HERE = VLM
_DB = PROCESSED / "ytstream/uav_0006/duckdb.db"
_SCRATCH = SCRATCH
MODEL = "yolo11x.pt"          # the largest YOLO11 checkpoint in the tree
TRACKER = "botsort.yaml"      # GMC on, which is the right choice for a drone
FPS = 5

CELLS = [
    {"name": "30s", "clip": "clip/uav0006_180_210.mp4", "frames": [900, 1050]},
    {"name": "600s", "clip": "clip/uav0006_180_780.mp4", "frames": [900, 3900]},
    {"name": "900s", "clip": "clip/uav0006_180_1080.mp4", "frames": [900, 5400]},
]


class _Dets:
    """Minimal Results-like view the ultralytics trackers accept."""

    def __init__(self, xywh, conf, cls):
        self.xywh, self.conf, self.cls = xywh, conf, cls

    def __getitem__(self, m):
        return _Dets(self.xywh[m], self.conf[m], self.cls[m])

    def __len__(self):
        return len(self.conf)

    @property
    def xyxy(self):
        """Corner boxes used by BoT-SORT's camera-motion compensation."""
        half = self.xywh[:, 2:4] / 2
        return np.concatenate((self.xywh[:, :2] - half, self.xywh[:, :2] + half), axis=1)


def pipeline_seconds(clip: Path, out: Path) -> float:
    """Wall clock of the real ingest worker over one clip."""
    t0 = time.perf_counter()
    r = subprocess.run(
        [sys.executable, str(_REPO / "scripts/detect_track.py"),
         "--video", str(clip), "--out", str(out),
         "--model", MODEL, "--tracker", TRACKER, "--classes", "2"],
        capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        sys.exit(f"detect_track failed:\n{r.stderr[-2000:]}")
    return dt


def association_ms_per_frame(clip: Path, f0: int, f1: int) -> tuple[float, float]:
    """Tracker cost per frame when fed the dataset's own boxes."""
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.utils import IterableSimpleNamespace, YAML
    import ultralytics

    cfg = Path(ultralytics.__file__).parent / "cfg/trackers" / TRACKER
    trk = BOTSORT(IterableSimpleNamespace(**YAML.load(cfg)))

    con = duckdb.connect(str(_DB), read_only=True)
    rows = con.execute(
        "SELECT fid,x1,y1,x2,y2 FROM bbox WHERE fid >= ? AND fid < ? ORDER BY fid",
        [f0, f1]).fetchall()
    con.close()
    per: dict[int, list] = {}
    for fid, x1, y1, x2, y2 in rows:
        per.setdefault(fid, []).append(
            ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))

    cap = cv2.VideoCapture(str(clip))
    n, tot, nbox = 0, 0.0, 0
    for i in range(f1 - f0):
        ok, img = cap.read()
        if not ok:
            break
        b = per.get(f0 + i, [])
        d = _Dets(np.array(b, dtype=np.float32).reshape(-1, 4),
                  np.full(len(b), 0.9, dtype=np.float32),
                  np.zeros(len(b), dtype=np.float32))
        t0 = time.perf_counter()
        trk.update(d, img)
        tot += time.perf_counter() - t0
        n += 1
        nbox += len(b)
    cap.release()
    return tot / max(n, 1) * 1000.0, nbox / max(n, 1)


def main() -> None:
    _SCRATCH.mkdir(parents=True, exist_ok=True)
    out = {"model": MODEL, "tracker": TRACKER, "cells": []}
    for c in CELLS:
        clip = _HERE / c["clip"]
        f0, f1 = c["frames"]
        n = f1 - f0
        wall = pipeline_seconds(clip, _SCRATCH / f"ingest_{c['name']}.db")
        assoc_ms, density = association_ms_per_frame(clip, f0, f1)
        total = wall + assoc_ms / 1000.0 * n
        out["cells"].append({
            "name": c["name"], "frames": n, "seconds_of_video": n / FPS,
            "pipeline_s": round(wall, 2),
            "assoc_ms_per_frame": round(assoc_ms, 3),
            "boxes_per_frame": round(density, 1),
            "ingest_s": round(total, 2),
            "ms_per_frame": round(total / n * 1000, 1),
            "realtime_factor": round((n / FPS) / total, 2),
        })
        print(f"{c['name']:>5}  {n:5} frames  pipeline {wall:6.2f} s  "
              f"+ assoc {assoc_ms:.2f} ms/frame at {density:.1f} boxes  "
              f"= {total:6.2f} s  ({total / n * 1000:.1f} ms/frame)")

    p = _HERE / "ingest.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {p.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
