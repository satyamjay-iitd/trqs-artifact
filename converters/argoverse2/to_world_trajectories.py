#!/usr/bin/env python3
"""Lift AV2 Sensor cuboid annotations from the ego frame into the city frame.

    p_city = city_SE3_egovehicle(t) o p_ego

Reads each downloaded log's annotations.feather + city_SE3_egovehicle.feather,
composes the poses, and writes one tidy trajectory table per log plus a
concatenated table for the whole split. The ego vehicle is emitted as its own
track so it can be queried like any other object.

Output columns
  log_id, city, timestamp_ns, t_s        t_s = seconds since first sweep of log
  track_uuid, category, is_ego
  x, y, z                                cuboid centroid, CITY frame, metres
  qw, qx, qy, qz                         orientation, CITY frame
  heading_rad, heading_deg               yaw about city +z, atan2 convention
  length_m, width_m, height_m            amodal extent, constant per track
  vx, vy, speed_mps                      central-difference velocity, city frame
  num_interior_pts                       lidar returns inside the cuboid
  ego_x, ego_y, ego_z                    ego pose at the same timestamp
  range_m                                object distance from ego (BEV)

Usage:
  to_world_trajectories.py --split val                 # all downloaded logs
  to_world_trajectories.py --split val --logs <log_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import os

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("TRQS_RAW_DIR", Path(__file__).resolve().parents[2] / "datasets" / "raw")) / "argoverse2"

CITY_CODES = {
    "ATX": "Austin",
    "DTW": "Detroit",
    "MIA": "Miami",
    "PAO": "Palo Alto",
    "PIT": "Pittsburgh",
    "WDC": "Washington DC",
}


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(N,4) [w,x,y,z] -> (N,3,3) rotation matrices. AV2 quats are unit-norm."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(-1, 3, 3)


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """(N,3,3) -> (N,4) [w,x,y,z], via the numerically stable branch-free trace form."""
    m = R
    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = np.empty((len(m), 4))
    # Four candidate formulations; pick the one with the largest denominator.
    cands = np.stack([1 + t, 1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2],
                      1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2],
                      1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2]], axis=1)
    best = cands.argmax(axis=1)
    for k in range(4):
        i = best == k
        if not i.any():
            continue
        s = np.sqrt(np.maximum(cands[i, k], 1e-12)) * 2
        mi = m[i]
        if k == 0:
            q[i] = np.stack([0.25 * s, (mi[:, 2, 1] - mi[:, 1, 2]) / s,
                             (mi[:, 0, 2] - mi[:, 2, 0]) / s,
                             (mi[:, 1, 0] - mi[:, 0, 1]) / s], axis=1)
        elif k == 1:
            q[i] = np.stack([(mi[:, 2, 1] - mi[:, 1, 2]) / s, 0.25 * s,
                             (mi[:, 0, 1] + mi[:, 1, 0]) / s,
                             (mi[:, 0, 2] + mi[:, 2, 0]) / s], axis=1)
        elif k == 2:
            q[i] = np.stack([(mi[:, 0, 2] - mi[:, 2, 0]) / s,
                             (mi[:, 0, 1] + mi[:, 1, 0]) / s, 0.25 * s,
                             (mi[:, 1, 2] + mi[:, 2, 1]) / s], axis=1)
        else:
            q[i] = np.stack([(mi[:, 1, 0] - mi[:, 0, 1]) / s,
                             (mi[:, 0, 2] + mi[:, 2, 0]) / s,
                             (mi[:, 1, 2] + mi[:, 2, 1]) / s, 0.25 * s], axis=1)
    q[q[:, 0] < 0] *= -1  # canonical hemisphere
    return q


def map_meta(log_dir: Path) -> tuple[str, Path | None]:
    """City code and vector-map path, both read off the map archive filename."""
    hits = list((log_dir / "map").glob("log_map_archive_*.json"))
    if not hits:
        return "UNKNOWN", None
    # log_map_archive_<log_id>____<CITY>_city_<n>.json
    code = hits[0].stem.split("____")[-1].split("_city_")[0]
    return code, hits[0]


def process_log(log_dir: Path) -> pd.DataFrame:
    log_id = log_dir.name
    ann = pd.read_feather(log_dir / "annotations.feather")
    pose = pd.read_feather(log_dir / "city_SE3_egovehicle.feather")
    code, _ = map_meta(log_dir)

    missing = set(ann.timestamp_ns) - set(pose.timestamp_ns)
    if missing:
        # Annotations are on lidar sweeps, which always carry an exact pose row;
        # a gap here means a corrupt/partial download rather than a real edge case.
        raise RuntimeError(f"{log_id}: {len(missing)} annotation timestamps lack a pose")

    df = ann.merge(pose, on="timestamp_ns", suffixes=("", "_ego"), validate="m:1")

    R_ego = quat_to_mat(df[["qw_ego", "qx_ego", "qy_ego", "qz_ego"]].to_numpy())
    t_ego = df[["tx_m_ego", "ty_m_ego", "tz_m_ego"]].to_numpy()
    R_obj = quat_to_mat(df[["qw", "qx", "qy", "qz"]].to_numpy())
    t_obj = df[["tx_m", "ty_m", "tz_m"]].to_numpy()

    xyz_city = np.einsum("nij,nj->ni", R_ego, t_obj) + t_ego
    q_city = mat_to_quat(R_ego @ R_obj)

    # Add the ego vehicle as a track of its own, at the annotated timestamps.
    ego_ts = np.sort(ann.timestamp_ns.unique())
    ep = pose.set_index("timestamp_ns").loc[ego_ts]
    ego = pd.DataFrame({
        "timestamp_ns": ego_ts,
        "track_uuid": "EGO_VEHICLE",
        "category": "EGO_VEHICLE",
        "length_m": 4.877, "width_m": 2.032, "height_m": 1.473,  # AV2 Ford Fusion
        "num_interior_pts": 0,
        "qw": ep.qw.to_numpy(), "qx": ep.qx.to_numpy(),
        "qy": ep.qy.to_numpy(), "qz": ep.qz.to_numpy(),
        "x": ep.tx_m.to_numpy(), "y": ep.ty_m.to_numpy(), "z": ep.tz_m.to_numpy(),
        "tx_m_ego": ep.tx_m.to_numpy(), "ty_m_ego": ep.ty_m.to_numpy(),
        "tz_m_ego": ep.tz_m.to_numpy(),
    })

    obj = pd.DataFrame({
        "timestamp_ns": df.timestamp_ns,
        "track_uuid": df.track_uuid, "category": df.category,
        "length_m": df.length_m, "width_m": df.width_m, "height_m": df.height_m,
        "num_interior_pts": df.num_interior_pts,
        "qw": q_city[:, 0], "qx": q_city[:, 1], "qy": q_city[:, 2], "qz": q_city[:, 3],
        "x": xyz_city[:, 0], "y": xyz_city[:, 1], "z": xyz_city[:, 2],
        "tx_m_ego": df.tx_m_ego, "ty_m_ego": df.ty_m_ego, "tz_m_ego": df.tz_m_ego,
    })

    out = pd.concat([obj, ego], ignore_index=True)
    out["is_ego"] = out.category == "EGO_VEHICLE"
    out["log_id"] = log_id
    out["city"] = CITY_CODES.get(code, code)
    out["t_s"] = (out.timestamp_ns - out.timestamp_ns.min()) / 1e9

    # Yaw about city +z from the composed quaternion.
    qw, qx, qy, qz = (out[c].to_numpy() for c in ("qw", "qx", "qy", "qz"))
    out["heading_rad"] = np.arctan2(2 * (qw * qz + qx * qy),
                                    1 - 2 * (qy * qy + qz * qz))
    out["heading_deg"] = np.degrees(out.heading_rad)

    out = out.rename(columns={"tx_m_ego": "ego_x", "ty_m_ego": "ego_y",
                              "tz_m_ego": "ego_z"})
    out["range_m"] = np.hypot(out.x - out.ego_x, out.y - out.ego_y)

    out = out.sort_values(["track_uuid", "timestamp_ns"], ignore_index=True)
    out[["vx", "vy", "speed_mps"]] = _velocity(out)

    cols = ["log_id", "city", "timestamp_ns", "t_s", "track_uuid", "category", "is_ego",
            "x", "y", "z", "qw", "qx", "qy", "qz", "heading_rad", "heading_deg",
            "length_m", "width_m", "height_m", "vx", "vy", "speed_mps",
            "num_interior_pts", "ego_x", "ego_y", "ego_z", "range_m"]
    return out[cols]


def _velocity(out: pd.DataFrame) -> pd.DataFrame:
    """Central-difference city-frame velocity, per track. Endpoints one-sided."""
    g = out.groupby("track_uuid", sort=False)
    dt = g.timestamp_ns.diff(2).to_numpy() / 1e9
    vx = g.x.diff(2).to_numpy() / dt
    vy = g.y.diff(2).to_numpy() / dt
    # diff(2) leaves the first two rows of each track NaN; backfill from diff(1).
    dt1 = g.timestamp_ns.diff().to_numpy() / 1e9
    vx1, vy1 = g.x.diff().to_numpy() / dt1, g.y.diff().to_numpy() / dt1
    vx, vy = np.where(np.isnan(vx), vx1, vx), np.where(np.isnan(vy), vy1, vy)
    # ...and the very first row of each track has no predecessor: use its successor.
    v = pd.DataFrame({"vx": vx, "vy": vy}, index=out.index)
    v = v.groupby(out.track_uuid.to_numpy(), sort=False).bfill().fillna(0.0)
    v["speed_mps"] = np.hypot(v.vx, v.vy)
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="sensor")
    ap.add_argument("--split", default="val")
    ap.add_argument("--logs", nargs="*")
    a = ap.parse_args()

    split_dir = ROOT / a.dataset / a.split
    logs = ([split_dir / lg for lg in a.logs] if a.logs
            else sorted(d for d in split_dir.iterdir() if d.is_dir()))
    out_dir = ROOT / "derived" / a.dataset / a.split
    (out_dir / "per_log").mkdir(parents=True, exist_ok=True)

    frames, manifest = [], []
    for i, log_dir in enumerate(logs, 1):
        try:
            t = process_log(log_dir)
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(logs)}] FAIL {log_dir.name}: {e}", file=sys.stderr)
            continue
        t.to_parquet(out_dir / "per_log" / f"{log_dir.name}.parquet", index=False)
        frames.append(t)
        code, map_path = map_meta(log_dir)
        manifest.append({
            "log_id": log_dir.name, "city": t.city.iloc[0], "city_code": code,
            "n_frames": int(t.timestamp_ns.nunique()),
            "n_tracks": int(t.track_uuid.nunique()),
            "n_boxes": len(t),
            "duration_s": float(t.t_s.max()),
            "x_min": float(t.x.min()), "x_max": float(t.x.max()),
            "y_min": float(t.y.min()), "y_max": float(t.y.max()),
            "vector_map": str(map_path.relative_to(ROOT)) if map_path else None,
        })
        print(f"[{i}/{len(logs)}] {log_dir.name}  {t.city.iloc[0]}  "
              f"{manifest[-1]['n_tracks']} tracks  {len(t)} boxes", file=sys.stderr)

    if not frames:
        print("nothing processed", file=sys.stderr)
        return 1
    allt = pd.concat(frames, ignore_index=True)
    allt.to_parquet(out_dir / "trajectories.parquet", index=False)
    pd.DataFrame(manifest).to_csv(out_dir / "manifest.csv", index=False)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(allt)} rows, {allt.track_uuid.nunique()} tracks, "
          f"{len(manifest)} logs -> {out_dir}/trajectories.parquet", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
