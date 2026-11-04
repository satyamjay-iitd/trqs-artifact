#!/usr/bin/env python3
"""Render a log's HD map, optionally with its world-frame trajectories on top.

Bird's-eye plan view in city coordinates -- the same frame the trajectory tables
use, so map and tracks overlay with no transform.

The map is the recessive base layer (grays); trajectories carry the only real
colour. The 29 annotation categories collapse to three groups plus a muted
"other", because a plan view puts every pair of colours side by side at once and
only three categorical hues survive that all-pairs test. The ego track is drawn
in ink rather than a category hue -- it is the reference entity, not a series.

Usage:
  plot_map.py --split val --log <log_id>                 # map + trajectories
  plot_map.py --split val --log <log_id> --map-only
  plot_map.py --split val --log <log_id> --frame 60      # cuboid footprints too
  plot_map.py --split val --log <log_id> --dark
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon

ROOT = Path(os.environ.get("TRQS_RAW_DIR", Path(__file__).resolve().parents[2] / "datasets" / "raw")) / "argoverse2"

# Validated categorical slots 1-3 (all-pairs, both modes) + ink for the ego and
# muted gray for everything else. See references/palette.md.
THEME = {
    "light": dict(surface="#fcfcfb", area="#eeece6", lane="#c3c2b7", xwalk="#a9a79f",
                  ink="#0b0b0b", muted="#898781", grid="#e1e0d9",
                  veh="#2a78d6", ped="#eb6834", cyc="#1baf7a", other="#898781"),
    "dark": dict(surface="#1a1a19", area="#26262a", lane="#45443f", xwalk="#5c5a53",
                 ink="#ffffff", muted="#898781", grid="#2c2c2a",
                 veh="#3987e5", ped="#d95926", cyc="#199e70", other="#898781"),
}

GROUP = {
    "veh": {"REGULAR_VEHICLE", "LARGE_VEHICLE", "BUS", "ARTICULATED_BUS", "SCHOOL_BUS",
            "BOX_TRUCK", "TRUCK", "TRUCK_CAB", "VEHICULAR_TRAILER", "RAILED_VEHICLE",
            "MESSAGE_BOARD_TRAILER", "TRAFFIC_LIGHT_TRAILER"},
    "ped": {"PEDESTRIAN", "WHEELCHAIR", "STROLLER", "OFFICIAL_SIGNALER", "DOG", "ANIMAL"},
    "cyc": {"BICYCLE", "BICYCLIST", "MOTORCYCLE", "MOTORCYCLIST",
            "WHEELED_RIDER", "WHEELED_DEVICE"},
}
GROUP_LABEL = {"veh": "Vehicle", "ped": "Pedestrian", "cyc": "Cyclist / rider",
               "other": "Static object"}


def group_of(cat: str) -> str:
    for g, members in GROUP.items():
        if cat in members:
            return g
    return "other"


def load_map(log_dir: Path) -> dict:
    return json.load(open(next((log_dir / "map").glob("log_map_archive_*.json"))))


def draw_map(ax, mp: dict, c: dict) -> None:
    """Drivable areas as fills, lane boundaries as hairlines, crossings dashed."""
    for da in mp["drivable_areas"].values():
        xy = np.array([[p["x"], p["y"]] for p in da["area_boundary"]])
        ax.add_patch(MplPolygon(xy, closed=True, facecolor=c["area"],
                                edgecolor="none", zorder=1))
    for ls in mp["lane_segments"].values():
        for side in ("left_lane_boundary", "right_lane_boundary"):
            xy = np.array([[p["x"], p["y"]] for p in ls[side]])
            # Intersection lanes get a lighter weight so junctions stay legible.
            ax.plot(xy[:, 0], xy[:, 1], color=c["lane"],
                    lw=0.6 if ls["is_intersection"] else 0.9, zorder=2)
    for pc in mp["pedestrian_crossings"].values():
        for side in ("edge1", "edge2"):
            xy = np.array([[p["x"], p["y"]] for p in pc[side]])
            ax.plot(xy[:, 0], xy[:, 1], color=c["xwalk"], lw=1.0, ls=(0, (3, 2)), zorder=3)


def draw_cuboids(ax, frame: pd.DataFrame, c: dict) -> None:
    """Oriented cuboid footprints (length x width, rotated by heading) at one sweep."""
    for r in frame.itertuples():
        l, w, h = r.length_m / 2, r.width_m / 2, r.heading_rad
        corners = np.array([[-l, -w], [l, -w], [l, w], [-l, w]])
        R = np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])
        xy = corners @ R.T + np.array([r.x, r.y])
        col = c["ink"] if r.is_ego else c[group_of(r.category)]
        ax.add_patch(MplPolygon(xy, closed=True, facecolor=col, alpha=0.85,
                                edgecolor=c["surface"], lw=0.8, zorder=7))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="sensor")
    ap.add_argument("--split", default="val")
    ap.add_argument("--log", required=True)
    ap.add_argument("--map-only", action="store_true")
    ap.add_argument("--frame", type=int, help="also draw cuboid footprints at this sweep index")
    ap.add_argument("--dark", action="store_true")
    ap.add_argument("--zoom", nargs="?", type=float, const=30.0, metavar="MARGIN_M",
                    help="crop to the trajectories' extent plus a margin (default 30 m). "
                         "The map crop is far larger than the ego route, so cuboid "
                         "footprints are invisible without this.")
    ap.add_argument("--out", help="output path (default: derived/figures/<log>.png)")
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()

    c = THEME["dark" if a.dark else "light"]
    log_dir = ROOT / a.dataset / a.split / a.log
    mp = load_map(log_dir)

    fig, ax = plt.subplots(figsize=(11, 9), facecolor=c["surface"])
    ax.set_facecolor(c["surface"])
    draw_map(ax, mp, c)

    traj = None
    if not a.map_only:
        traj = pd.read_parquet(ROOT / "derived" / a.dataset / a.split / "per_log" / f"{a.log}.parquet")
        drawn = set()
        for uuid, tr in traj[~traj.is_ego].groupby("track_uuid", sort=False):
            g = group_of(tr.category.iloc[0])
            ax.plot(tr.x, tr.y, color=c[g], lw=2.0, alpha=0.75, solid_capstyle="round",
                    zorder=5, label=GROUP_LABEL[g] if g not in drawn else None)
            drawn.add(g)
        ego = traj[traj.is_ego].sort_values("timestamp_ns")
        ax.plot(ego.x, ego.y, color=c["ink"], lw=2.6, solid_capstyle="round",
                zorder=6, label="Ego vehicle")
        ax.plot(ego.x.iloc[0], ego.y.iloc[0], "o", ms=9, color=c["ink"], zorder=6)

        ts = None
        if a.frame is not None:
            ts = np.sort(traj.timestamp_ns.unique())[a.frame]
            draw_cuboids(ax, traj[traj.timestamp_ns == ts], c)

    if a.zoom is not None and traj is not None:
        m = a.zoom
        if a.frame is not None:
            # Centre on the ego at that sweep -- over a whole 15 s route a 5 m cuboid
            # is a few pixels, so a frame view has to be local to be worth drawing.
            e = ego[ego.timestamp_ns == ts].iloc[0]
            ax.set_xlim(e.x - m, e.x + m)
            ax.set_ylim(e.y - m, e.y + m)
        else:
            ax.set_xlim(traj.x.min() - m, traj.x.max() + m)
            ax.set_ylim(traj.y.min() - m, traj.y.max() + m)

    ax.set_aspect("equal")
    ax.tick_params(colors=c["muted"], labelsize=8, length=3)
    for s in ax.spines.values():
        s.set_color(c["grid"])
    ax.grid(True, color=c["grid"], lw=0.5, zorder=0)
    ax.set_xlabel("city x (m)", color=c["muted"], fontsize=9)
    ax.set_ylabel("city y (m)", color=c["muted"], fontsize=9)

    city = traj.city.iloc[0] if traj is not None else ""
    n = f"  ·  {traj.track_uuid.nunique()} tracks" if traj is not None else ""
    ax.set_title(f"{a.log[:8]}  ·  {city}{n}", color=c["ink"], fontsize=12,
                 loc="left", pad=12)

    # Legend always present: identity is never carried by colour alone. Map layers
    # are spelled out too, since the grays are otherwise unlabelled.
    handles = ax.get_legend_handles_labels()[0]
    handles += [MplPolygon([(0, 0)], facecolor=c["area"], edgecolor="none", label="Drivable area"),
                Line2D([], [], color=c["lane"], lw=0.9, label="Lane boundary"),
                Line2D([], [], color=c["xwalk"], lw=1.0, ls=(0, (3, 2)), label="Crosswalk")]
    leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                    frameon=False, fontsize=9, labelcolor=c["ink"])
    leg.set_title("Layers", prop={"size": 9})
    leg.get_title().set_color(c["muted"])

    out = Path(a.out) if a.out else ROOT / "derived" / "figures" / f"{a.log}{'_dark' if a.dark else ''}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=a.dpi, facecolor=c["surface"], bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
