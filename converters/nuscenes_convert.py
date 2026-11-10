#!/usr/bin/env python
"""Convert nuScenes (v1.0-trainval) annotations into the TrQL common format, in map space.

One sub-dataset per scene (``<out>/nuscenes/<scene>``), from the annotation tables
alone -- no camera images, lidar or map rasters are read, so only the metadata
download (``v1.0-trainval_meta.tgz``) is needed.

Raw layout expected::

    <raw>/v1.0-trainval/{scene,sample,sample_data,sample_annotation,instance,
                         category,log,sensor,calibrated_sensor}.json

Rules
-----
* **Frames.** ``fid`` indexes the scene's CAM_FRONT ``sample_data`` (keyframes and
  sweeps) in timestamp order, 0-based and consecutive -- the ~12 Hz camera clock.
* **Objects.** Every annotated instance in the scene; ``class`` is its nuScenes
  category name (``vehicle.car``, ``human.pedestrian.adult``, ...).
* **Interpolation.** nuScenes annotates at 2 Hz. An instance spans the frames from
  the CAM_FRONT keyframe of its first annotated sample to that of its last; on
  every frame in between its box is interpolated at the frame's timestamp,
  measured against the samples' own timestamps (clamped to the annotated
  interval): centre linearly, heading by the shortest arc, extent fixed to the
  earlier annotation's size.
* **Boxes are ground footprints in map space.** The length x width rectangle,
  turned by the heading, then its axis-aligned bounds. Each side is at least
  ``MIN_SIDE_M`` (0.30 m) so traffic cones and similar do not collapse to a few
  centimetres. Coordinates are map metres x 100 from the map raster's corner, with
  y flipped (``stored_y = y_delta - y_m * 100``), the devkit's MapMask
  convention; ``x_delta``/``y_delta`` in ``metadata.json`` undo it. Bounds are
  rounded outward (floor on the low edge, ceil on the high one) to integers.

These are the rules recorded in the original processed data's metadata. That
converter is not available; this one re-implements the rules, so its output agrees
with the recorded data closely (same frames and objects, boxes within a few
centimetres) rather than byte for byte.

No video is produced (camera images are not read); the engine does not need one.

Run with the project venv::

    .trqs-pyenv/bin/python converters/nuscenes_convert.py \\
        --raw datasets/raw/nuscenes --out datasets/processed --location boston-seaport
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_ROOT = SCRIPT_DIR.parent / "datasets"
sys.path.insert(0, str(SCRIPT_DIR))

from common import Bbox, Obj, build_duckdb  # noqa: E402

DATASET = "nuscenes"
COORD_SCALE = 100.0
MAP_M_PER_PX = 0.1
MIN_SIDE_M = 0.30
# Map raster size in pixels (width, height) at MAP_M_PER_PX, per location. These
# are the nuScenes semantic-prior map rasters; only their extents are used here.
MAP_PX = {
    "boston-seaport": (29795, 21181),
    "singapore-onenorth": (15856, 20250),
    "singapore-queenstown": (32286, 36871),
    "singapore-hollandvillage": (28083, 29229),
}


def _load(root: Path, name: str) -> list[dict]:
    with open(root / f"{name}.json") as fh:
        return json.load(fh)


def _yaw(q: list[float]) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _footprint(cx: float, cy: float, heading: float, size: list[float], y_delta: float):
    w, l = max(size[0], MIN_SIDE_M), max(size[1], MIN_SIDE_M)
    c, s = math.cos(heading), math.sin(heading)
    xs, ys = [], []
    for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        px, py = sx * l / 2, sy * w / 2
        xs.append(cx + px * c - py * s)
        ys.append(cy + px * s + py * c)
    return (
        math.floor(min(xs) * COORD_SCALE),
        math.floor(y_delta - max(ys) * COORD_SCALE),
        math.ceil(max(xs) * COORD_SCALE),
        math.ceil(y_delta - min(ys) * COORD_SCALE),
    )


class Tables:
    """The annotation tables, indexed once for all scenes."""

    def __init__(self, meta: Path) -> None:
        print(f"loading {meta} ...", flush=True)
        self.scenes = _load(meta, "scene")
        self.logs = {l["token"]: l for l in _load(meta, "log")}
        self.samples = {s["token"]: s for s in _load(meta, "sample")}
        sensors = {s["token"]: s["channel"] for s in _load(meta, "sensor")}
        cam_front = {
            c["token"] for c in _load(meta, "calibrated_sensor")
            if sensors[c["sensor_token"]] == "CAM_FRONT"
        }
        self.cam_by_sample: dict[str, list[dict]] = defaultdict(list)
        for sd in _load(meta, "sample_data"):
            if sd["calibrated_sensor_token"] in cam_front:
                self.cam_by_sample[sd["sample_token"]].append(sd)
        cats = {c["token"]: c["name"] for c in _load(meta, "category")}
        self.inst_class = {i["token"]: cats[i["category_token"]] for i in _load(meta, "instance")}
        self.anns_by_sample: dict[str, list[dict]] = defaultdict(list)
        for a in _load(meta, "sample_annotation"):
            self.anns_by_sample[a["sample_token"]].append(a)

    def location(self, scene: dict) -> str:
        return self.logs[scene["log_token"]]["location"]


def convert_scene(t: Tables, scene: dict, out_root: Path) -> dict:
    location = t.location(scene)
    map_w, map_h = MAP_PX[location]
    y_delta = map_h * MAP_M_PER_PX * COORD_SCALE

    sample_tokens = []
    tok = scene["first_sample_token"]
    while tok:
        sample_tokens.append(tok)
        tok = t.samples[tok]["next"]

    frames = sorted(
        (sd for s in sample_tokens for sd in t.cam_by_sample[s]), key=lambda d: d["timestamp"]
    )
    fid_of = {sd["token"]: f for f, sd in enumerate(frames)}
    key_fid = {sd["sample_token"]: fid_of[sd["token"]] for sd in frames if sd["is_key_frame"]}
    frame_ts = [sd["timestamp"] for sd in frames]

    # instance -> its annotations as (sample time, keyframe fid, annotation), in time order
    tracks: dict[str, list[tuple[int, int, dict]]] = defaultdict(list)
    for s in sample_tokens:
        for a in t.anns_by_sample[s]:
            tracks[a["instance_token"]].append((t.samples[s]["timestamp"], key_fid[s], a))
    for seq in tracks.values():
        seq.sort(key=lambda x: x[0])

    order = sorted(tracks, key=lambda it: (tracks[it][0][1], t.inst_class[it], it))
    objects = [Obj(oid, t.inst_class[it]) for oid, it in enumerate(order)]
    bboxes: list[Bbox] = []
    for oid, it in enumerate(order):
        seq = tracks[it]
        times = [x[0] for x in seq]
        for f in range(seq[0][1], seq[-1][1] + 1):
            tc = min(max(frame_ts[f], times[0]), times[-1])
            j = bisect.bisect_right(times, tc) - 1
            a = seq[j][2]
            if j + 1 < len(seq) and times[j + 1] > times[j]:
                b = seq[j + 1][2]
                r = (tc - times[j]) / (times[j + 1] - times[j])
            else:
                b, r = a, 0.0
            cx = a["translation"][0] + r * (b["translation"][0] - a["translation"][0])
            cy = a["translation"][1] + r * (b["translation"][1] - a["translation"][1])
            ya, yb = _yaw(a["rotation"]), _yaw(b["rotation"])
            heading = ya + r * ((yb - ya + math.pi) % (2 * math.pi) - math.pi)
            x1, y1, x2, y2 = _footprint(cx, cy, heading, a["size"], y_delta)
            bboxes.append(Bbox(f, oid, x1, y1, x2, y2))

    out_dir = out_root / DATASET / scene["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = build_duckdb(out_dir / "duckdb.db", objects, bboxes)
    with_boxes = {b.fid for b in bboxes}
    fps = 12.0
    meta = {
        "dataset": DATASET,
        "world": scene["name"],
        "scene_token": scene["token"],
        "scene_description": scene["description"],
        "log": t.logs[scene["log_token"]]["logfile"],
        "location": location,
        "source": "nuScenes v1.0-trainval",
        "converter": "converters/nuscenes_convert.py",
        "coordinate_scale": COORD_SCALE,
        "x_delta": 0.0,
        "y_delta": y_delta,
        "coordinate_space": "map metres x 100 (NOT image pixels)",
        "map_m_per_px": MAP_M_PER_PX,
        "map_px": [map_w, map_h],
        "map_extent_m": {"x0": 0.0, "x1": map_w * MAP_M_PER_PX, "y0": 0.0, "y1": map_h * MAP_M_PER_PX},
        "fps": fps,
        "num_frames": len(frames),
        "duration_s": len(frames) / fps,
        # The engine reads these as unsigned integers: the stored-coordinate extent.
        "width": round(map_w * MAP_M_PER_PX * COORD_SCALE),
        "height": round(y_delta),
        "obj_classes": sorted({o.cls for o in objects}),
        "activity_classes": [],
        "row_counts": counts,
        "fid_range": [0, len(frames) - 1],
        "frames_without_boxes": len(frames) - len(with_boxes),
        "notes": (
            "One sub-dataset per nuScenes scene, in map space. See the module docstring of "
            "converters/nuscenes_convert.py for the frame, interpolation and footprint rules. "
            "No video: camera images are not read."
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return {"scene": scene["name"], "frames": len(frames), **counts}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", type=Path, default=DATASETS_ROOT / "raw" / "nuscenes",
                    help="directory holding <version>/ (default: datasets/raw/nuscenes)")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--out", type=Path, default=DATASETS_ROOT / "processed",
                    help="processed root; scenes go to <out>/nuscenes/<scene>")
    ap.add_argument("--scene", action="append", default=[], help="scene name, repeatable (default: all)")
    ap.add_argument("--location", action="append", default=[], help="only scenes on this map, repeatable")
    args = ap.parse_args()

    meta = args.raw / args.version
    if not (meta / "scene.json").is_file():
        ap.error(f"no {args.version} tables under {args.raw} (expected {meta}/scene.json)")
    t = Tables(meta)
    scenes = [
        s for s in sorted(t.scenes, key=lambda s: s["name"])
        if (not args.scene or s["name"] in args.scene)
        and (not args.location or t.location(s) in args.location)
    ]
    missing = set(args.scene) - {s["name"] for s in scenes}
    if missing:
        ap.error(f"unknown scene(s): {', '.join(sorted(missing))}")
    for i, s in enumerate(scenes, 1):
        r = convert_scene(t, s, args.out)
        print(f"[{i}/{len(scenes)}] {r['scene']}: {r['frames']} frames, "
              f"{r.get('objects', '?')} objects, {r.get('bbox', '?')} boxes", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
