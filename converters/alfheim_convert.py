#!/usr/bin/env python
"""Convert the raw Alfheim (Tromsø IL - Tottenham Hotspurs) dataset into the TrQL format.

Alfheim is a soccer dataset from Simula (https://datasets.simula.no/alfheim/),
captured at Alfheim Stadium (Tromsø IL, Norway). This directory holds the
"camera setting 2" game -- 2013-11-28 Tromsø IL vs Tottenham Hotspurs -- which
ships two very different kinds of annotation:

  * a stitched **panorama video** (3-second raw H.264 segments) plus a manually
    annotated **ball pixel position** for every one of its frames, and
  * **ZXY Sport Tracking** player logs: on-field positions in *metres* at 20 Hz
    for 14 anonymized sensor tags.

There is no usable field->panorama homography (the site states the panorama warp
is non-linear and the matrices were lost), so pitch metres and panorama pixels
cannot be expressed in one coordinate space. The output is therefore split by
coordinate space, not by modality:

    <out>/alfheim/                              WORLD: players on a pitch map
      world-metadata.json                       the world's coordinate frame
      pitch.svg                                 the map, drawn in that frame
      duckdb.db                                 players, world-space
      panorama/video.mp4                        the footage, as a viewpoint

    <out>/alfheim_ball/<game>_pano/             STANDALONE: ball in pixels
      video.mp4 + duckdb.db                     same footage, pixel-space boxes

The panorama appears in both because it is the one thing both spaces share. In
the world it is a *viewpoint only*: with no homography its camera has no model,
so world positions cannot be projected into it and no boxes are drawn over the
footage -- the players are shown on the map instead. The ball is annotated
directly in panorama pixels, so it stays standalone, where its boxes do land on
the video.

Raw layout expected (see README.md)::

    raw/alfheim_Totten/pano.tgz                        # 767 panorama .h264 segments
    raw/alfheim_Totten/ball/track/<seg>.h264_track.txt # 75 lines "<frame> <x> <y>"
    raw/alfheim_Totten/zxy/2013-11-28_tromso_tottenham.csv       # interpolated 20 Hz
    raw/alfheim_Totten/zxy/2013-11-28_tromso_tottenham_raw.csv   # raw 20 Hz
    raw/alfheim_Totten/zxy/2013-11-28_tromso_tottenham_agg.csv   # 1 Hz statistics

ball (standalone)
  * The 767 segments (0000..0766, 75 frames each, 4450x2000 @ 25 fps) are
    concatenated -- pure H.264 elementary streams, so a byte concat is enough --
    into one mp4 cached next to the raw data and symlinked as ``video.mp4``.
  * objects   = a single ``ball``.
  * bbox      = a fixed-size box centred on the annotated ball pixel, one row
                per video frame; ``fid`` = segment_offset + (track line - 1).
  * activities= empty.

world
  * objects   = one ``player`` per ZXY tag id (identities are randomized upstream).
  * bbox      = a square footprint of ``--player-radius`` m around (x_pos, y_pos),
                in the world frame declared by ``world-metadata.json``:
                ``stored = round(metres * coordinate_scale) + delta``. Y is
                stored mirrored so that the server's own Y negation cancels and
                map/query coordinates are pitch metres (0..105 by 0..68).
  * ``fid``   = **panorama frame index**. The 20 Hz ZXY samples are resampled
                onto the video's 25 fps grid, so one timeline serves both.
  * activities= empty.

ZXY 20 Hz columns (both raw and interpolated), no header, per the dataset paper
(``paper.pdf`` in this directory, section 2.1.2)::

    timestamp, tag_id, x_pos, y_pos, heading, direction, energy, speed, total_distance

Run with the project venv::

    .trqs-pyenv/bin/python converters/alfheim_convert.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

# Make the shared helpers importable regardless of CWD.
SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_ROOT = SCRIPT_DIR.parent / "datasets"
sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    Obj, Bbox, build_subset, build_duckdb, probe_video, symlink_video,
)

DATASET = "alfheim"            # the world: pitch map + player trajectories
DATASET_BALL = "alfheim_ball"  # standalone: panorama video + ball in pixels
GAME = "2013-11-28_tromso_tottenham"
PANO_CAMERA = "panorama"       # the world's single camera (viewpoint only)

PANO_FPS = 25.0  # panorama segments are 25 fps (75 frames per 3-second segment)
ZXY_HZ = 20.0  # ZXY Sport Tracking sampling rate

BALL_CLASS = "ball"
PLAYER_CLASS = "player"

PITCH_M = (105.0, 68.0)  # nominal pitch size; ZXY coordinates run slightly outside it
MOVING_MS = 0.3  # above this speed a tag counts as "moving" rather than parked
PLAYER_MIN_MAX_SPEED_MS = 4.0  # a worn tag reaches running speed at some point

# "0000_2013-11-28 19:03:54.469509000.h264"
SEGMENT_RE = re.compile(r"^(\d+)_(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\.h264$")

ZXY_SOURCES = {
    "interpolated": f"{GAME}.csv",
    "raw": f"{GAME}_raw.csv",
}
ZXY_AGG = f"{GAME}_agg.csv"

# Margin in metres kept between the pitch/samples and the edge of the drawing
# area the zxy subset declares. Players legitimately leave the pitch (throw-ins,
# warm-ups), so the area is the union of the pitch and every box, plus this.
PITCH_MARGIN_M = 6.0

SCENE_VIEW_FILE = "pitch.svg"  # the world map, drawn in the world frame

# Regulation markings in metres (pitch is PITCH_M = 105 x 68).
GOAL_WIDTH_M = 7.32
GOAL_DEPTH_M = 2.0
PENALTY_AREA_DEPTH_M = 16.5
PENALTY_AREA_WIDTH_M = 40.32
GOAL_AREA_DEPTH_M = 5.5
GOAL_AREA_WIDTH_M = 18.32
PENALTY_SPOT_M = 11.0
CENTRE_CIRCLE_R_M = 9.15
CORNER_ARC_R_M = 1.0


# --------------------------------------------------------------------------- #
# Panorama segments
# --------------------------------------------------------------------------- #
def parse_segment_name(name: str) -> tuple[int, datetime]:
    """Return (segment index, capture start time) for a panorama segment name."""
    m = SEGMENT_RE.match(name)
    if not m:
        raise ValueError(f"unrecognized panorama segment name: {name!r}")
    return int(m.group(1)), datetime.fromisoformat(m.group(2))


def extract_pano(tgz: Path, pano_dir: Path) -> None:
    """Extract the top-level ``*.h264`` segments from ``pano.tgz`` into ``pano_dir``.

    The archive also carries copies of the ball tracks (``track/``, ``trackTmp/``)
    and a ``manifest.txt``; only the video segments are needed here. It is read
    in streaming mode -- a 6.3 GB gzip is scanned once, not twice.
    """
    pano_dir.mkdir(parents=True, exist_ok=True)
    print(f"extracting panorama segments from {tgz.name} (~{tgz.stat().st_size / 1e9:.1f} GB) ...")
    n = 0
    with tarfile.open(tgz, "r|gz") as tf:
        for member in tf:
            if not member.isfile() or "/" in member.name or not member.name.endswith(".h264"):
                continue
            tf.extract(member, pano_dir, filter="data")
            n += 1
            if n % 100 == 0:
                print(f"  ... {n} segments")
    print(f"  extracted {n} segments -> {pano_dir}")


def pano_segments(raw_root: Path, pano_dir: Path) -> list[Path]:
    """Return the panorama segments in capture order, extracting the tgz if needed."""
    segments = sorted(pano_dir.glob("*.h264")) if pano_dir.is_dir() else []
    if not segments:
        tgz = raw_root / "pano.tgz"
        if not tgz.is_file():
            raise FileNotFoundError(f"neither extracted segments in {pano_dir} nor {tgz}")
        extract_pano(tgz, pano_dir)
        segments = sorted(pano_dir.glob("*.h264"))
    if not segments:
        raise FileNotFoundError(f"no *.h264 segments in {pano_dir}")

    indexed = sorted((parse_segment_name(p.name)[0], p) for p in segments)
    expected = list(range(len(indexed)))
    got = [i for i, _ in indexed]
    if got != expected:
        missing = sorted(set(range(got[-1] + 1)) - set(got))
        print(f"!! panorama segment numbering has gaps; missing {len(missing)} (e.g. {missing[:5]})")
    return [p for _, p in indexed]


def count_frames(path: Path) -> int:
    """Number of coded frames in a raw H.264 segment (packet count -- no decoding)."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip().split(",")[0])


def frame_offsets(segments: list[Path], jobs: int) -> tuple[list[int], list[int]]:
    """Return (per-segment frame counts, exclusive-prefix frame offsets)."""
    print(f"probing frame counts of {len(segments)} segments ...")
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        counts = list(pool.map(count_frames, segments))
    offsets, total = [], 0
    for c in counts:
        offsets.append(total)
        total += c
    return counts, offsets


def build_panorama_mp4(segments: list[Path], dst: Path, fps: float) -> None:
    """Concatenate the raw H.264 segments into ``dst`` (stream copy, no re-encode).

    Each segment is a self-contained elementary stream starting with SPS/PPS and
    an IDR, so the bytes can simply be fed to ffmpeg back to back (this is the
    approach the dataset page itself recommends). Written to a temp file first so
    an interrupted run never leaves a truncated cache behind.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".partial.mp4")
    if tmp.exists():
        tmp.unlink()
    total_bytes = sum(p.stat().st_size for p in segments)
    print(f"concatenating {len(segments)} segments (~{total_bytes / 1e9:.1f} GB) -> {dst} ...")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "h264", "-framerate", str(fps), "-i", "pipe:0",
        "-c", "copy", "-an", str(tmp),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        assert proc.stdin is not None
        for i, seg in enumerate(segments, 1):
            with seg.open("rb") as fh:
                shutil.copyfileobj(fh, proc.stdin, 1 << 22)
            if i % 100 == 0:
                print(f"  ... {i}/{len(segments)} segments")
        proc.stdin.close()
    except BrokenPipeError:
        pass
    if proc.wait() != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed while concatenating panorama segments (rc={proc.returncode})")
    tmp.replace(dst)


# --------------------------------------------------------------------------- #
# Ball annotations
# --------------------------------------------------------------------------- #
def ball_bboxes(
    ball_dir: Path,
    segments: list[Path],
    counts: list[int],
    offsets: list[int],
    box: float,
    width: int | None,
    height: int | None,
) -> tuple[list[Bbox], dict]:
    """Build ball bboxes from the per-segment ``*_track.txt`` pixel positions.

    Each track file has one ``<frame> <x> <y>`` line per frame of its segment
    (1-based frame numbers). The point is expanded to a ``box`` x ``box`` square
    and clamped to the frame, since the annotation is a centre point only.
    """
    half = box / 2.0
    bboxes: list[Bbox] = []
    missing: list[str] = []
    out_of_range = 0
    for seg, count, offset in zip(segments, counts, offsets):
        track = ball_dir / f"{seg.name}_track.txt"
        if not track.is_file():
            missing.append(seg.name)
            continue
        for line in track.read_text().splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            frame, x, y = int(parts[0]), float(parts[1]), float(parts[2])
            if not 1 <= frame <= count:
                out_of_range += 1
                continue
            x1, y1, x2, y2 = x - half, y - half, x + half, y + half
            if width is not None:
                x1, x2 = max(0.0, x1), min(float(width), x2)
            if height is not None:
                y1, y2 = max(0.0, y1), min(float(height), y2)
            bboxes.append(Bbox(fid=offset + frame - 1, oid=0, x1=x1, y1=y1, x2=x2, y2=y2, aid=None))
    stats = {
        "segments_with_ball": len(segments) - len(missing),
        "segments_without_ball": len(missing),
        "lines_out_of_range": out_of_range,
    }
    if missing:
        print(f"!! {len(missing)} segment(s) have no ball track (e.g. {missing[:3]})")
    if out_of_range:
        print(f"!! {out_of_range} ball line(s) referenced a frame outside their segment; dropped")
    return bboxes, stats


# --------------------------------------------------------------------------- #
# ZXY sensor logs
# --------------------------------------------------------------------------- #
def parse_zxy(path: Path) -> tuple[list[Obj], list[Bbox], dict]:
    """Parse a 20 Hz ZXY log into player objects and metre-space bboxes.

    Columns (no header): ``timestamp, tag_id, x_pos, y_pos, heading, direction,
    energy, speed, total_distance``. ``fid`` is the 20 Hz index relative to the
    first sample in the file; for the raw log the timestamps are not on an exact
    grid, so several samples of a tag can quantize onto the same ``fid``.
    """
    per_tag: dict[int, dict] = {}
    rows: list[tuple[float, int, float, float]] = []  # (seconds from t0, tag, x, y)
    t0: datetime | None = None
    x_lo = y_lo = float("inf")
    x_hi = y_hi = float("-inf")

    with path.open(newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 9:
                continue
            ts = datetime.fromisoformat(row[0])
            tag = int(row[1])
            x, y = float(row[2]), float(row[3])
            speed, distance = float(row[7]), float(row[8])
            if t0 is None:
                t0 = ts
            rows.append(((ts - t0).total_seconds(), tag, x, y))

            x_lo, x_hi = min(x_lo, x), max(x_hi, x)
            y_lo, y_hi = min(y_lo, y), max(y_hi, y)
            st = per_tag.get(tag)
            if st is None:
                st = per_tag[tag] = {
                    "samples": 0, "first_time": row[0], "last_time": row[0],
                    "first_distance_m": distance, "last_distance_m": distance,
                    "max_speed_ms": speed, "on_pitch_samples": 0,
                    "first_moving_time": None, "last_moving_time": None,
                }
            st["samples"] += 1
            st["last_time"] = row[0]
            st["last_distance_m"] = distance
            st["max_speed_ms"] = max(st["max_speed_ms"], speed)
            if 0.0 <= x <= PITCH_M[0] and 0.0 <= y <= PITCH_M[1]:
                st["on_pitch_samples"] += 1
            if speed > MOVING_MS:
                if st["first_moving_time"] is None:
                    st["first_moving_time"] = row[0]
                st["last_moving_time"] = row[0]

    if t0 is None:
        raise ValueError(f"no usable rows in {path}")

    for st in per_tag.values():
        st["distance_covered_m"] = round(st["last_distance_m"] - st["first_distance_m"], 2)
        st["max_speed_ms"] = round(st["max_speed_ms"], 3)
        st["on_pitch_fraction"] = round(st["on_pitch_samples"] / st["samples"], 4)
        # A tag only counts as a tracked player if it spends most of its samples
        # inside the pitch AND actually runs: several tags in this game sit on the
        # bench or are never worn (see README.md).
        st["is_player_track"] = bool(
            st["on_pitch_fraction"] >= 0.5 and st["max_speed_ms"] >= PLAYER_MIN_MAX_SPEED_MS
        )

    objects = [Obj(oid=tag, cls=PLAYER_CLASS) for tag in sorted(per_tag)]
    stats = {
        "t0": t0,
        "per_tag": per_tag,
        "field_extent_m": {
            "x": [round(x_lo, 3), round(x_hi, 3)],
            "y": [round(y_lo, 3), round(y_hi, 3)],
        },
        "duration_s": max(r[0] for r in rows),
    }
    return objects, rows, stats


def frame_times(segments: list[Path], counts: list[int], offsets: list[int]) -> list[datetime]:
    """Capture time of every panorama frame, from its own segment's start.

    Each segment's filename timestamp is the capture time of its first frame, and
    segment starts drift from a nominal 3 s cadence by up to ~78 ms, so frame
    times are built per segment rather than extrapolated from the first one.
    """
    times: list[datetime] = []
    for seg, count, _ in zip(segments, counts, offsets):
        start = parse_segment_name(seg.name)[1]
        times.extend(start + timedelta(seconds=i / PANO_FPS) for i in range(count))
    return times


def resample_to_frames(
    rows: list[tuple[float, int, float, float]],
    times: list[datetime],
    zxy_t0: datetime,
    max_gap_s: float,
) -> list[tuple[int, int, float, float]]:
    """Resample 20 Hz ZXY samples onto the panorama's 25 fps frame grid.

    In world mode one db is shared by every camera, and the client turns ``fid``
    into a video timestamp through the camera's frame rate -- so a 20 Hz db
    behind a 25 fps camera drifts 25% and the players would be a quarter of the
    match out of step with the footage. Resampling makes ``fid`` mean exactly one
    thing everywhere: panorama frame index.

    Positions are linearly interpolated between the two ZXY samples bracketing a
    frame, and dropped where the bracketing samples are more than ``max_gap_s``
    apart -- a tag that stops transmitting should leave a hole, not a straight
    line across the pitch to wherever it reappears.
    """
    from bisect import bisect_left

    per_tag: dict[int, list[tuple[float, float, float]]] = {}
    for t, tag, x, y in rows:
        per_tag.setdefault(tag, []).append((t, x, y))
    for series in per_tag.values():
        series.sort(key=lambda s: s[0])

    # Frame times as seconds from the ZXY epoch, matching the series' units.
    frame_ts = [(ft - zxy_t0).total_seconds() for ft in times]

    out: list[tuple[int, int, float, float]] = []
    for tag in sorted(per_tag):
        series = per_tag[tag]
        ts = [s[0] for s in series]
        for fid, ft in enumerate(frame_ts):
            i = bisect_left(ts, ft)
            if i == 0:
                # Before the tag's first sample; accept an exact/near match only.
                if ts[0] - ft > max_gap_s:
                    continue
                out.append((fid, tag, series[0][1], series[0][2]))
                continue
            if i >= len(ts):
                if ft - ts[-1] > max_gap_s:
                    continue
                out.append((fid, tag, series[-1][1], series[-1][2]))
                continue
            t0_, x0_, y0_ = series[i - 1]
            t1_, x1_, y1_ = series[i]
            if t1_ - t0_ > max_gap_s:
                continue
            span = t1_ - t0_
            w = 0.0 if span <= 0 else (ft - t0_) / span
            out.append((fid, tag, x0_ + (x1_ - x0_) * w, y0_ + (y1_ - y0_) * w))
    out.sort(key=lambda r: (r[0], r[1]))
    return out


def pitch_svg(width: int, height: int, pitch_x0: float, pitch_y0: float, upm: float) -> str:
    """A regulation football pitch drawn in the zxy subset's drawing-area space.

    The dashboard has no video to draw a video-less camera's boxes over, so it
    falls back to a blank frame. This is the backdrop it uses instead: the
    ``viewBox`` is the declared drawing area and the pitch sits exactly where the
    shifted, scaled ZXY coordinates put it, so a box at a player's position lands
    on the right patch of grass without the client knowing anything about pitch
    geometry.

    ``pitch_x0``/``pitch_y0`` are the pitch's top-left corner in stored units and
    ``upm`` the stored units per metre; everything below is converted into stored
    units so the drawing shares one coordinate space with the bboxes. Line widths
    stay real-world widths (12 cm markings) scaled the same way.

    Colours are fixed rather than theme-aware: the image is served as a file and
    embedded with ``<img>``, so it cannot see the dashboard's light/dark toggle,
    and green-with-white-lines reads correctly against either.
    """
    pw, ph = PITCH_M[0] * upm, PITCH_M[1] * upm
    x0, y0 = pitch_x0, pitch_y0  # pitch top-left within the drawing area
    x1, y1 = x0 + pw, y0 + ph  # pitch bottom-right
    cx, cy = x0 + pw / 2, y0 + ph / 2

    pa_w, pa_d = PENALTY_AREA_WIDTH_M * upm, PENALTY_AREA_DEPTH_M * upm
    ga_w, ga_d = GOAL_AREA_WIDTH_M * upm, GOAL_AREA_DEPTH_M * upm
    goal_w, goal_d = GOAL_WIDTH_M * upm, GOAL_DEPTH_M * upm
    spot, r = PENALTY_SPOT_M * upm, CENTRE_CIRCLE_R_M * upm
    pa_y = y0 + (ph - pa_w) / 2  # penalty area top edge
    ga_y = y0 + (ph - ga_w) / 2  # goal area top edge
    goal_y = y0 + (ph - goal_w) / 2
    line_w = 0.12 * upm  # 12 cm pitch markings

    # Penalty arc: the part of a penalty-spot-centred circle outside the box.
    dy = math.sqrt(r**2 - (pa_d - spot) ** 2)
    arc_l = f"M {x0 + pa_d:.3f},{cy - dy:.3f} A {r:g} {r:g} 0 0 1 {x0 + pa_d:.3f},{cy + dy:.3f}"
    arc_r = f"M {x1 - pa_d:.3f},{cy - dy:.3f} A {r:g} {r:g} 0 0 0 {x1 - pa_d:.3f},{cy + dy:.3f}"

    c = CORNER_ARC_R_M * upm
    corners = " ".join(
        [
            f"M {x0 + c},{y0} A {c:g} {c:g} 0 0 1 {x0},{y0 + c}",
            f"M {x1 - c},{y0} A {c:g} {c:g} 0 0 0 {x1},{y0 + c}",
            f"M {x0},{y1 - c} A {c:g} {c:g} 0 0 1 {x0 + c},{y1}",
            f"M {x1},{y1 - c} A {c:g} {c:g} 0 0 0 {x1 - c},{y1}",
        ]
    )

    # Mowing stripes, purely decorative, clipped to the pitch.
    band = pw / 7
    stripes = "".join(
        f'<rect x="{x0 + i * band:.3f}" y="{y0:.3f}" width="{band:.3f}" height="{ph:.3f}"/>'
        for i in range(1, 7, 2)
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:g} {height:g}" \
width="{width:g}" height="{height:g}" preserveAspectRatio="none">
  <title>Association football pitch, {PITCH_M[0]:g} x {PITCH_M[1]:g} m</title>
  <defs>
    <clipPath id="pitch">\
<rect x="{x0:.3f}" y="{y0:.3f}" width="{pw:.3f}" height="{ph:.3f}"/></clipPath>
  </defs>
  <!-- Surround: the area outside the touchlines, where out-of-pitch samples land. -->
  <rect x="0" y="0" width="{width:g}" height="{height:g}" fill="#16321f"/>
  <rect x="{x0:.3f}" y="{y0:.3f}" width="{pw:.3f}" height="{ph:.3f}" fill="#2f7d43"/>
  <g clip-path="url(#pitch)" fill="#378c4c">{stripes}</g>
  <g fill="none" stroke="#ffffff" stroke-width="{line_w:g}" stroke-opacity="0.9">
    <!-- Touchlines, halfway line and centre circle -->
    <rect x="{x0:.3f}" y="{y0:.3f}" width="{pw:.3f}" height="{ph:.3f}"/>
    <line x1="{cx:.3f}" y1="{y0:.3f}" x2="{cx:.3f}" y2="{y1:.3f}"/>
    <circle cx="{cx:.3f}" cy="{cy:.3f}" r="{r:g}"/>
    <!-- Penalty areas -->
    <rect x="{x0:.3f}" y="{pa_y:.3f}" width="{pa_d:g}" height="{pa_w:g}"/>
    <rect x="{x1 - pa_d:.3f}" y="{pa_y:.3f}" width="{pa_d:g}" height="{pa_w:g}"/>
    <!-- Goal areas -->
    <rect x="{x0:.3f}" y="{ga_y:.3f}" width="{ga_d:g}" height="{ga_w:g}"/>
    <rect x="{x1 - ga_d:.3f}" y="{ga_y:.3f}" width="{ga_d:g}" height="{ga_w:g}"/>
    <!-- Penalty arcs and corner arcs -->
    <path d="{arc_l}"/>
    <path d="{arc_r}"/>
    <path d="{corners}"/>
  </g>
  <g fill="#ffffff" fill-opacity="0.9">
    <circle cx="{cx:.3f}" cy="{cy:.3f}" r="{0.2 * upm:g}"/>
    <circle cx="{x0 + spot:.3f}" cy="{cy:.3f}" r="{0.2 * upm:g}"/>
    <circle cx="{x1 - spot:.3f}" cy="{cy:.3f}" r="{0.2 * upm:g}"/>
  </g>
  <!-- Goals, drawn outside the goal lines -->
  <g fill="#ffffff" fill-opacity="0.18" stroke="#ffffff" \
stroke-width="{line_w:g}" stroke-opacity="0.8">
    <rect x="{x0 - goal_d:.3f}" y="{goal_y:.3f}" width="{goal_d:g}" height="{goal_w:g}"/>
    <rect x="{x1:.3f}" y="{goal_y:.3f}" width="{goal_d:g}" height="{goal_w:g}"/>
  </g>
</svg>
"""


def parse_zxy_agg(path: Path) -> dict:
    """Per-tag end-of-log values from the 1 Hz statistics file.

    Columns (no header), per the dataset paper section 2.1.2::

        timestamp, tag_id, total_distance, hir_distance, sprint_distance, total_effort

    All four are cumulative, monotonically non-decreasing counters, and
    ``total_distance``/``total_effort`` match the 20 Hz log's ``total_distance``
    and ``energy`` exactly. ``sprint_distance`` is the distance run at
    >= 25.2 km/h (7.0 m/s) and reproduces to within ~1 m per tag by integrating
    the 20 Hz log above that speed. ``hir_distance`` is the paper's
    high-intensity-run category (>= 19.8 km/h) but does *not* reproduce as a
    plain threshold integration -- it lands consistently below the >= 5.5 m/s
    integral, so the category carries extra criteria from the paper's cited
    source (Di Mascio & Bradley). Treat it as reported, not as recomputable.
    """
    first: dict[int, list[float]] = {}
    last: dict[int, list[float]] = {}
    with path.open(newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 6:
                continue
            tag = int(row[1])
            vals = [float(v) for v in row[2:6]]
            first.setdefault(tag, vals)
            last[tag] = vals
    keys = ("total_distance_m", "hir_distance_m", "sprint_distance_m", "total_effort")
    return {
        str(tag): {k: round(last[tag][i] - first[tag][i], 2) for i, k in enumerate(keys)}
        for tag in sorted(last)
    }


# --------------------------------------------------------------------------- #
# Subsets
# --------------------------------------------------------------------------- #
def convert_pano(args, out_dataset: Path) -> bool:
    segments = pano_segments(args.raw, args.pano_dir)
    counts, offsets = frame_offsets(segments, args.jobs)
    total_frames = offsets[-1] + counts[-1]

    if args.pano_video.exists() and not args.force_video:
        print(f"reusing cached panorama video {args.pano_video}")
    else:
        build_panorama_mp4(segments, args.pano_video, PANO_FPS)
    info = probe_video(args.pano_video)
    if info["num_frames"] and info["num_frames"] != total_frames:
        print(
            f"!! panorama video has {info['num_frames']} frames but the segments sum to "
            f"{total_frames}; ball fids may be misaligned (re-run with --force-video)"
        )

    bboxes, ball_stats = ball_bboxes(
        args.ball_dir, segments, counts, offsets, args.ball_box, info["width"], info["height"]
    )
    if not bboxes:
        print(f"!! no ball annotations found under {args.ball_dir}; skipping pano subset")
        return False

    # Each segment's filename timestamp is the capture time of its FIRST frame
    # (paper section 2.2.3), which is what the paper recommends matching ZXY
    # timestamps against. Segment starts drift by up to a frame relative to a
    # nominal 3 s cadence, so record every segment's own start rather than
    # extrapolating one global offset at 25 fps.
    starts = [parse_segment_name(p.name)[1] for p in segments]
    first_idx, first_time = parse_segment_name(segments[0].name)
    end_time = starts[-1] + timedelta(seconds=counts[-1] / PANO_FPS)
    segment_index = [
        [off, ts.isoformat(sep=" ")] for off, ts in zip(offsets, starts)
    ]
    max_drift_s = max(
        abs((ts - first_time).total_seconds() - off / PANO_FPS)
        for off, ts in zip(offsets, starts)
    )
    zxy_t0 = None
    zxy_csv = args.raw / "zxy" / ZXY_SOURCES["interpolated"]
    if zxy_csv.is_file():
        with zxy_csv.open(newline="") as fh:
            row = next(csv.reader(fh), None)
        if row:
            zxy_t0 = datetime.fromisoformat(row[0])

    subset = f"{GAME}_pano"
    out_dir = out_dataset / subset
    metadata = {
        "dataset": DATASET_BALL,
        "source": "Alfheim -- Simula (https://datasets.simula.no/alfheim/), MMSys'14 dataset paper",
        "game": "2013-11-28 Tromsø IL - Tottenham Hotspurs",
        "subset": subset,
        "modality": "stitched panorama video + manual ball annotation",
        "obj_classes": [BALL_CLASS],
        "activity_classes": [],
        "coordinate_space": "panorama image pixels",
        "num_segments": len(segments),
        "segment_frames": sorted(set(counts)),
        "expected_num_frames": total_frames,
        "video_start_time": first_time.isoformat(sep=" "),
        "video_end_time": end_time.isoformat(sep=" "),  # end of the last segment
        "first_segment_index": first_idx,
        "ball_box_px": args.ball_box,
        "ball_stats": ball_stats,
        "zxy_alignment": {
            "zxy_start_time": zxy_t0.isoformat(sep=" ") if zxy_t0 else None,
            "video_start_offset_s": (
                round((first_time - zxy_t0).total_seconds(), 6) if zxy_t0 else None
            ),
            "zxy_hz": ZXY_HZ,
            "max_segment_drift_s": round(max_drift_s, 6),
            "hint": (
                "Approximate: zxy_fid = round(video_start_offset_s * zxy_hz + fid * zxy_hz / fps). "
                "Exact: find the last [fid_offset, start_time] in segment_index with "
                "fid_offset <= fid, then frame_time = start_time + (fid - fid_offset) / fps "
                "and zxy_fid = round((frame_time - zxy_start_time) * zxy_hz). The two differ "
                "by up to max_segment_drift_s because segment starts are not an exact 3 s cadence."
            ),
        },
        # [fid_offset, capture time of that segment's first frame], one per segment.
        "segment_index": segment_index,
        "notes": (
            "The 3-second raw H.264 panorama segments are byte-concatenated and "
            "muxed to mp4 with a stream copy (no re-encode), so fid = "
            "segment_frame_offset + (track line - 1), 0-based over the whole game. "
            "bbox is a fixed ball_box_px square centred on the annotated ball "
            "centre pixel (the raw annotation is a point, not a box) clamped to "
            "the frame. Only the ball is annotated here: the players live in the "
            f"'{DATASET}' WORLD, positioned on a pitch map in metres. No reliable "
            "field->panorama homography is published, so the two cannot be merged "
            "into one coordinate space -- which is why the ball stays a standalone "
            "dataset rather than joining the world db. No activities (aid=NULL)."
        ),
    }
    meta = build_subset(
        out_dir,
        video_src=args.pano_video,
        objects=[Obj(oid=0, cls=BALL_CLASS)],
        bboxes=bboxes,
        activities=(),
        metadata=metadata,
    )
    rc = meta["row_counts"]
    print(
        f"[ok] {subset}: {rc['objects']} object, {rc['bbox']} ball bboxes, "
        f"{meta['width']}x{meta['height']}, fps={meta['fps']:.0f}, "
        f"frames={meta['num_frames']} -> {out_dir}"
    )
    return True


def convert_world(args, out_dataset: Path) -> bool:
    zxy_dir = args.raw / "zxy"
    src = zxy_dir / ZXY_SOURCES[args.zxy_source]
    if not src.is_file():
        print(f"!! {src} not found; skipping world")
        return False

    print(f"parsing {src.name} ...")
    objects, rows, stats = parse_zxy(src)
    radius = args.player_radius

    # The world's coordinate frame: the union of the pitch and every sample,
    # plus a margin, shifted so nothing is negative and scaled so integer
    # storage does not quantize a running player.
    #
    # Stored coordinates are unsigned integers downstream (the engine's
    # BoundingBox is u32), but ZXY goes negative wherever a player steps past the
    # west or north touchline, and 1 unit = 1 m would put a player in a 1 m cell
    # -- at 25 fps a runner covers ~0.35 m per frame, which would read as a stall
    # then a jump. The world frame expresses both as
    # `stored = round(metres * coordinate_scale) + delta`, which is exactly what
    # WorldMeta declares, so the server converts back for queries and map
    # positions on its own.
    (ex_lo, ex_hi), (ey_lo, ey_hi) = (
        stats["field_extent_m"]["x"],
        stats["field_extent_m"]["y"],
    )
    upm = args.units_per_metre
    lo_x = min(0.0, ex_lo - radius) - PITCH_MARGIN_M
    lo_y = min(0.0, ey_lo - radius) - PITCH_MARGIN_M
    hi_x = max(PITCH_M[0], ex_hi + radius) + PITCH_MARGIN_M
    hi_y = max(PITCH_M[1], ey_hi + radius) + PITCH_MARGIN_M
    # Whole stored units, so the deltas the world frame carries are exact.
    x_delta = math.ceil(-lo_x * upm)
    y_delta = math.ceil(hi_y * upm)
    width = math.ceil(hi_x * upm) + x_delta
    height = y_delta - math.floor(lo_y * upm)

    # Y is stored mirrored: `stored_y = y_delta - metres_y * scale`.
    #
    # The server's map frame negates Y (`to_map_pos`), a convention inherited
    # from CARLA. Storing Y the obvious way would therefore put the pitch at
    # map y in [-68, 0], so every spatial query would be written against
    # negative numbers, contradicting the dataset's own documented axis
    # (0 <= y <= 68, growing east -- paper section 2.1.1). Mirroring here makes
    # the two negations cancel, so map and query coordinates ARE pitch metres.
    # The drawing is unaffected: a pitch is symmetric about the halfway line's
    # perpendicular, so the mirrored markings are identical.
    pitch_x0 = x_delta  # metres_x = 0
    pitch_y0 = y_delta - PITCH_M[1] * upm  # metres_y = 68, the small stored Y

    # Put the trajectories on the panorama's frame grid (see resample_to_frames).
    segments = pano_segments(args.raw, args.pano_dir)
    counts_f, offsets_f = frame_offsets(segments, args.jobs)
    times = frame_times(segments, counts_f, offsets_f)
    print(f"resampling {len(rows)} samples onto {len(times)} panorama frames ...")
    framed = resample_to_frames(rows, times, stats["t0"], args.max_gap_s)

    # Mirrored Y means the larger metre value is the smaller stored one, so the
    # box corners swap: y1 (top in stored space) comes from y + radius.
    bboxes = [
        Bbox(
            fid=fid,
            oid=tag,
            x1=(x - radius) * upm + x_delta,
            y1=y_delta - (y + radius) * upm,
            x2=(x + radius) * upm + x_delta,
            y2=y_delta - (y - radius) * upm,
            aid=None,
        )
        for fid, tag, x, y in framed
    ]

    agg_path = zxy_dir / ZXY_AGG
    agg = parse_zxy_agg(agg_path) if agg_path.is_file() else None

    # World layout: one db and one map at the root, cameras as subdirectories.
    out_dir = out_dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = build_duckdb(out_dir / "duckdb.db", objects, bboxes, ())
    (out_dir / SCENE_VIEW_FILE).write_text(pitch_svg(width, height, pitch_x0, pitch_y0, upm))
    (out_dir / "world-metadata.json").write_text(
        json.dumps(
            {
                "coordinate_scale": upm,
                "x_delta": x_delta,
                "y_delta": y_delta,
                "world_view": SCENE_VIEW_FILE,
                "width": width,
                "height": height,
            },
            indent=2,
        )
        + "\n"
    )

    # The panorama as a viewpoint. It carries no camera model on purpose: the
    # dataset publishes no field->panorama homography (the warp is non-linear and
    # the matrices were lost), so world positions cannot be projected into it.
    # The video plays; boxes are not drawn over it.
    cam_dir = out_dir / PANO_CAMERA
    cam_dir.mkdir(parents=True, exist_ok=True)
    video_info = {}
    if not args.pano_video.is_file():
        build_panorama_mp4(segments, args.pano_video, PANO_FPS)
    if args.pano_video.is_file():
        symlink_video(args.pano_video, cam_dir / "video.mp4")
        video_info = probe_video(cam_dir / "video.mp4")
    else:
        print(f"!! {args.pano_video} not built yet; world camera has no video")
    (cam_dir / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "camera": PANO_CAMERA,
                "fps": video_info.get("fps") or PANO_FPS,
                "width": video_info.get("width"),
                "height": video_info.get("height"),
                "num_frames": video_info.get("num_frames"),
                "duration_s": video_info.get("duration_s"),
                "coordinate_scale": upm,
                "projectable": False,
                "notes": (
                    "Viewpoint only. No field->panorama homography is published "
                    "for this dataset, so world positions cannot be projected "
                    "into this video and no bounding boxes are drawn over it; "
                    "the players are shown on the world map instead. fid is this "
                    "video's frame index, which the world db shares."
                ),
            },
            indent=2,
        )
        + "\n"
    )

    # build_duckdb remaps object ids to a continuous 0-based range (sorted by the
    # original id), so record which ZXY tag each oid came from.
    tag_to_oid = {tag: i for i, tag in enumerate(sorted(o.oid for o in objects))}
    t0 = stats["t0"]
    n_players = sum(st["is_player_track"] for st in stats["per_tag"].values())
    metadata = {
        "dataset": DATASET,
        "source": "Alfheim -- Simula (https://datasets.simula.no/alfheim/), MMSys'14 dataset paper",
        "game": "2013-11-28 Tromsø IL - Tottenham Hotspurs",
        "world": DATASET,
        "modality": "ZXY Sport Tracking player positions on a pitch map, plus the panorama as a viewpoint",
        "zxy_source": args.zxy_source,
        "zxy_file": src.name,
        "fps": video_info.get("fps") or PANO_FPS,
        "coordinate_space": (
            "world frame: stored = round(pitch_metres * coordinate_scale) + delta, "
            "declared in world-metadata.json. NOT image pixels."
        ),
        "player_radius_m": radius,
        # The world frame and the map drawn in it. pitch.svg spans stored
        # 0..width / 0..height, which is what view_roadmap maps through the same
        # (stored - delta) / scale transform the object positions use.
        "width": width,
        "height": height,
        "world_view": SCENE_VIEW_FILE,
        "coordinate_scale": upm,
        "deltas": [x_delta, y_delta],
        "y_mirrored": True,
        "cameras": [PANO_CAMERA],
        "obj_classes": [PLAYER_CLASS],
        "activity_classes": [],
        "num_tags": len(objects),
        "num_player_tracks": n_players,
        "t0": t0.isoformat(sep=" "),
        "zxy_hz": ZXY_HZ,
        "resampled_to_video_frames": True,
        "resample_max_gap_s": args.max_gap_s,
        "num_frames": video_info.get("num_frames") or len(times),
        "duration_s": video_info.get("duration_s") or round(len(times) / PANO_FPS, 3),
        "zxy_duration_s": round(stats["duration_s"], 3),
        "field_extent_m": stats["field_extent_m"],
        "pitch_size_m": list(PITCH_M),
        # Paper section 2.1.1.
        "coordinate_system": {
            "origin": "north-western corner of the pitch (lower-left as seen from the cameras)",
            "x_axis": "southwards, along the long side of the pitch; in-field 0 <= x <= 105",
            "y_axis": "eastwards, along the short edge; in-field 0 <= y <= 68",
            "heading_direction": "radians, 0 along the positive y-axis; magnetic north ~300 deg",
            "positional_error_m": 1.0,
        },
        "oid_to_tag": {str(oid): tag for tag, oid in tag_to_oid.items()},
        "per_tag": {
            str(tag_to_oid[tag]): {
                "tag_id": tag,
                **{k: v for k, v in st.items() if k not in ("first_distance_m", "last_distance_m")},
                **({"agg": agg[str(tag)]} if agg and str(tag) in agg else {}),
            }
            for tag, st in sorted(stats["per_tag"].items())
        },
        "row_counts": counts,
        "notes": (
            "A WORLD, not a standalone camera: one db and one map at the root, "
            "with the panorama as a camera subdirectory. bbox coordinates are "
            "pitch positions in the world frame, not image pixels -- a "
            "player_radius_m square footprint around the ZXY (x_pos, y_pos) "
            "sample, mapped by stored = round(metres * coordinate_scale) + delta "
            "so nothing is negative (players do step off the pitch, and stored "
            "coordinates are unsigned) and nothing is quantized by integer "
            "storage. world-metadata.json declares that frame, so the server "
            "converts back on its own; queries are written in PITCH METRES. "
            "fid is the panorama video's frame index: the 20 Hz ZXY samples are "
            f"resampled onto that {PANO_FPS:.0f} fps grid (linear interpolation, "
            "holes left where a tag is silent for more than resample_max_gap_s) "
            "so the world db and the video share one timeline. The panorama "
            "camera carries NO camera model: no field->panorama homography is "
            "published, so world positions cannot be projected into the footage "
            "and no boxes are drawn over it -- the players appear on the map "
            "instead. Only the HOME "
            "side (Tromsø IL) wears ZXY tags -- the visitors are not tracked at "
            "all -- and not every tag is a worn one: the paper (section 2.4) notes "
            "that belts sometimes fail or are removed mid-game, and that unused "
            "belts are sometimes activated remotely by the control system and "
            "transmit anyway; these abnormalities were deliberately left in. See "
            "per_tag.is_player_track, which requires the tag to stay mostly inside "
            f"the pitch and to reach {PLAYER_MIN_MAX_SPEED_MS:.0f} m/s at least "
            "once, filtering those out. ZXY positions carry ~1 m error, comparable "
            "to the synthetic footprint box itself. Player identities are randomized "
            "upstream to protect privacy and carry no team or role information, "
            "so every object is class 'player'. The manually annotated ball is "
            f"NOT here -- it is in panorama pixels, which without a homography "
            f"cannot be expressed in this frame, so it stays a standalone "
            f"dataset under '{DATASET_BALL}'. No activities (aid=NULL)."
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"[ok] world {DATASET}: {counts['objects']} tags ({n_players} real player "
        f"tracks), {counts['bbox']} bboxes on the pitch map, fid = panorama frame "
        f"(0..{metadata['num_frames'] - 1}), camera '{PANO_CAMERA}' -> {out_dir}"
    )
    return True


def convert(args) -> None:
    n = 0
    if "world" in args.subsets:
        n += convert_world(args, args.out / DATASET)
    if "ball" in args.subsets:
        n += convert_pano(args, args.out / DATASET_BALL)
    print(f"\nDone: wrote {n} of {len(args.subsets)} target(s) under {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--raw", type=Path, default=DATASETS_ROOT / "raw" / "alfheim_Totten",
        help="Raw alfheim_Totten root (default: datasets/raw for this dataset).",
    )
    parser.add_argument(
        "--out", type=Path, default=DATASETS_ROOT / "processed",
        help="Output root; an 'alfheim/' subdir is created under it.",
    )
    parser.add_argument(
        "--subsets", nargs="+", choices=["world", "ball"], default=["world", "ball"],
        help="Which outputs to build (default: both). 'world' is the alfheim world "
             "(pitch map + player trajectories + the panorama as a viewpoint); 'ball' "
             "is the standalone alfheim_ball dataset (panorama video + ball in pixels).",
    )
    parser.add_argument(
        "--pano-dir", type=Path, default=None,
        help="Where panorama .h264 segments live / are extracted (default: <raw>/pano).",
    )
    parser.add_argument(
        "--pano-video", type=Path, default=None,
        help="Cached concatenated panorama mp4 (default: <raw>/panorama.mp4). "
             "video.mp4 is a symlink to it, so it is built once and reused.",
    )
    parser.add_argument(
        "--ball-dir", type=Path, default=None,
        help="Directory of *_track.txt ball positions (default: <raw>/ball/track).",
    )
    parser.add_argument("--force-video", action="store_true", help="Rebuild the panorama mp4.")
    parser.add_argument(
        "--ball-box", type=float, default=20.0,
        help="Side length in pixels of the box drawn around the ball point (default: 20).",
    )
    parser.add_argument(
        "--zxy-source", choices=sorted(ZXY_SOURCES), default="interpolated",
        help="Which 20 Hz ZXY log to use (default: interpolated -- an exact 20 Hz grid; "
             "'raw' has irregular timestamps that quantize onto repeated fids).",
    )
    parser.add_argument(
        "--player-radius", type=float, default=0.5,
        help="Half-side in metres of the player footprint box (default: 0.5).",
    )
    parser.add_argument(
        "--max-gap-s", type=float, default=0.5,
        help="Leave a hole in a resampled trajectory when the ZXY samples bracketing a "
             "video frame are further apart than this (default: 0.5 s).",
    )
    parser.add_argument(
        "--units-per-metre", type=float, default=10.0,
        help="Stored world units per metre (default: 10, i.e. decimetres), declared as "
             "the world's coordinate_scale. Stored coordinates are integers downstream, "
             "so 1 unit = 1 m would quantize players into 1 m cells.",
    )
    parser.add_argument(
        "--jobs", type=int, default=min(8, (os.cpu_count() or 4)),
        help="Parallel ffprobe workers when measuring segment lengths.",
    )
    args = parser.parse_args()
    args.raw = args.raw.resolve()
    args.out = args.out.resolve()
    args.pano_dir = (args.pano_dir or args.raw / "pano").resolve()
    args.pano_video = (args.pano_video or args.raw / "panorama.mp4").resolve()
    args.ball_dir = (args.ball_dir or args.raw / "ball" / "track").resolve()
    convert(args)


if __name__ == "__main__":
    main()
