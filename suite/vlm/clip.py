# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Cut the 3:00-3:30 window out of ytstream uav_0006, exactly.

The window is seconds 180-210 of a 5 fps video, so it is frames 900-1050 of the
annotation, and `frame = second * 5` holds on the nose. Everything downstream --
mapping a model's "MM:SS" back to a tracker frame, scoring an interval -- depends
on that, so the cut has to land on the stated boundary rather than near it.

`-c copy` would not: a stream copy can only start at a keyframe, so it silently
slides the window by however far the nearest one is. We re-encode instead and
then verify the result is 150 frames at 5 fps before anyone uses it.

    uv run papers/vldb27/suite/vlm/clip.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path("/home/satyam/dev/frameql")
_SRC = _REPO / "datasets/processed/ytstream/uav_0006/video.mp4"
_OUT = _REPO / "papers/vldb27/suite/vlm/clip"

FPS = 5


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_frames", "-show_entries",
         "stream=width,height,r_frame_rate,nb_read_frames,duration",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(out.stdout)["streams"][0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=180, help="window start, seconds")
    ap.add_argument("--dur", type=int, default=30, help="window length, seconds")
    ap.add_argument("--out-dir", type=Path, default=_OUT)
    args = ap.parse_args()
    START_S, DUR_S = args.start, args.dur
    EXPECT_FRAMES = DUR_S * FPS
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dst = args.out_dir / f"uav0006_{START_S}_{START_S + DUR_S}.mp4"

    # -ss before -i seeks fast; with a re-encode the output still starts exactly
    # at the requested second rather than at the preceding keyframe.
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(START_S), "-i", str(_SRC), "-t", str(DUR_S),
         "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-an", str(dst)],
        check=True, capture_output=True,
    )

    s = probe(dst)
    frames = int(s["nb_read_frames"])
    num, den = (int(v) for v in s["r_frame_rate"].split("/"))
    fps = num / den
    print(f"{dst.relative_to(_REPO)}")
    print(f"  {s['width']}x{s['height']}  {fps:g} fps  {frames} frames  "
          f"{float(s['duration']):.2f} s  {dst.stat().st_size / 2**20:.1f} MB")

    ok = frames == EXPECT_FRAMES and fps == FPS
    print(f"  expected {EXPECT_FRAMES} frames at {FPS} fps -> "
          f"{'OK' if ok else 'MISMATCH'}")
    if not ok:
        sys.exit("clip does not line up with the annotation; refusing to proceed")
    print(f"  annotation frames {START_S * FPS}-{(START_S + DUR_S) * FPS} "
          f"map to clip seconds 0-{DUR_S}")


if __name__ == "__main__":
    main()
