#!/usr/bin/env python3
"""Merge per-log trajectories and HD maps into one set of files per city.

Every log of a city shares that city's frame -- verified directly: overlapping
logs carry the same lane_segment ids with bit-identical geometry (max coordinate
difference 0.00e+00 m). So merging is a concatenation plus an id-keyed union of
the map crops; no registration or fuzzy matching is involved.

Two things the merge has to fix up, both of which would silently corrupt a
merged table:

  * `track_uuid` is globally unique for objects but is the literal string
    "EGO_VEHICLE" in all 850 logs, so ego tracks from different logs collide.
    A `global_track_id` of "<log_id>:<track_uuid>" is added and is unique.
  * AV2 timestamps are anonymized, not wall-clock -- the whole corpus spans
    5.9 h of timestamp values. Logs recorded months apart can share a timestamp
    range, so a merged table is a spatial union of separate drives, NOT one
    continuous scene. Always group or filter by `log_id` for temporal queries.

Usage:
  merge_by_city.py                 # every city
  merge_by_city.py --city Miami
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import os

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("TRQS_RAW_DIR", Path(__file__).resolve().parents[2] / "datasets" / "raw")) / "argoverse2"
SPLITS = ("train", "val")
SECTIONS = ("lane_segments", "drivable_areas", "pedestrian_crossings")


def slug(city: str) -> str:
    return city.lower().replace(" ", "_")


# Connectivity is stated relative to a crop: a lane's predecessor that falls
# outside this log's map window is simply absent from its list. Taking the first
# crop's list would drop real edges, so these fields are unioned across crops.
TOPOLOGY = ("successors", "predecessors")


def merge_maps(logs: pd.DataFrame) -> tuple[dict, dict]:
    """Union the per-log map crops, keyed by element id.

    Geometry is bit-identical where crops overlap (verified: max coordinate
    difference 0.00e+00 m), so first-wins is correct for it. Connectivity is not
    -- it is crop-relative and must be unioned or the merged lane graph loses
    edges. Mark-type disagreements are counted and left at first-wins.
    """
    out = {s: {} for s in SECTIONS}
    stats = {"geometry_conflicts": 0, "edges_recovered": 0, "attr_disagreements": 0}
    for r in logs.itertuples():
        path = next((ROOT / "sensor" / r.split / r.log_id / "map").glob("log_map_archive_*.json"))
        mp = json.load(open(path))
        for sec in SECTIONS:
            for eid, el in mp.get(sec, {}).items():
                prev = out[sec].get(eid)
                if prev is None:
                    out[sec][eid] = dict(el)
                    continue
                for k, v in el.items():
                    if k in TOPOLOGY:
                        before = len(prev[k])
                        prev[k] = sorted(set(prev[k]) | set(v))
                        stats["edges_recovered"] += len(prev[k]) - before
                    elif json.dumps(prev.get(k), sort_keys=True) != json.dumps(v, sort_keys=True):
                        if k.endswith("_lane_boundary") or k == "area_boundary" or k.startswith("edge"):
                            stats["geometry_conflicts"] += 1
                        else:
                            stats["attr_disagreements"] += 1
    return out, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="sensor")
    ap.add_argument("--city")
    ap.add_argument("--out", default=str(ROOT / "derived" / "by_city"))
    a = ap.parse_args()

    frames = []
    for s in SPLITS:
        f = ROOT / "derived" / a.dataset / s / "trajectories.parquet"
        if not f.exists():
            continue
        t = pd.read_parquet(f)
        t["split"] = s
        frames.append(t)
    if not frames:
        print("no per-split trajectories found; run to_world_trajectories.py first")
        return 1
    allt = pd.concat(frames, ignore_index=True)

    out_root = Path(a.out)
    cities = [a.city] if a.city else sorted(allt.city.unique())
    summary = []

    for city in cities:
        t = allt[allt.city == city].copy()
        # Namespace track ids: object uuids are globally unique, EGO_VEHICLE is not.
        t["global_track_id"] = t.log_id.astype(str) + ":" + t.track_uuid.astype(str)
        t = t.sort_values(["log_id", "global_track_id", "timestamp_ns"], ignore_index=True)

        d = out_root / slug(city)
        d.mkdir(parents=True, exist_ok=True)

        logs = (t.groupby(["log_id", "split"], as_index=False)
                 .agg(n_frames=("timestamp_ns", "nunique"), n_tracks=("global_track_id", "nunique"),
                      n_boxes=("x", "size"), t0=("timestamp_ns", "min"), t1=("timestamp_ns", "max"),
                      x0=("x", "min"), x1=("x", "max"), y0=("y", "min"), y1=("y", "max")))

        mp, mstats = merge_maps(logs)
        json.dump(mp, open(d / "map.json", "w"))

        # Flag log pairs that overlap in both space and anonymized timestamp -- a
        # naive spatiotemporal query would read them as one scene.
        arr = logs[["x0", "x1", "y0", "y1", "t0", "t1"]].to_numpy()
        collide = [(logs.log_id[i], logs.log_id[j])
                   for i, j in itertools.combinations(range(len(logs)), 2)
                   if arr[i, 0] < arr[j, 1] and arr[j, 0] < arr[i, 1]
                   and arr[i, 2] < arr[j, 3] and arr[j, 2] < arr[i, 3]
                   and arr[i, 4] < arr[j, 5] and arr[j, 4] < arr[i, 5]]

        t.to_parquet(d / "trajectories.parquet", index=False)
        logs.to_csv(d / "manifest.csv", index=False)
        json.dump({"city": city, "n_logs": len(logs), "n_boxes": int(len(t)),
                   "n_tracks": int(t.global_track_id.nunique()),
                   "map_elements": {s: len(mp[s]) for s in SECTIONS},
                   "map_merge": mstats,
                   "space_time_colliding_log_pairs": [list(p) for p in collide]},
                  open(d / "meta.json", "w"), indent=2)

        summary.append(dict(city=city, logs=len(logs), boxes=len(t),
                            tracks=t.global_track_id.nunique(),
                            lanes=len(mp["lane_segments"]),
                            areas=len(mp["drivable_areas"]),
                            xwalks=len(mp["pedestrian_crossings"]),
                            geom_conflicts=mstats["geometry_conflicts"],
                            edges_recovered=mstats["edges_recovered"],
                            attr_disagree=mstats["attr_disagreements"],
                            st_collisions=len(collide)))
        print(f"{city:15s} {len(logs):4d} logs  {len(t):9,d} boxes  "
              f"{t.global_track_id.nunique():7,d} tracks  "
              f"{len(mp['lane_segments']):6,d} lanes  "
              f"{mstats['geometry_conflicts']} geom conflicts  "
              f"+{mstats['edges_recovered']} lane edges recovered  "
              f"{len(collide)} space-time collisions")

    s = pd.DataFrame(summary)
    s.to_csv(out_root / "summary.csv", index=False)
    print(f"\n{s.logs.sum()} logs, {s.boxes.sum():,} boxes -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
