# /// script
# requires-python = ">=3.12"
# dependencies = ["duckdb==1.5.6", "pillow"]
# ///
"""Freeze the set of turns the VLM will be scored against.

Membership comes from a `.trql` region query, run by the engine over the whole
video: a car turns iff its centre is in the first region and later in the
second. That query is the *specification* of the event, not an oracle -- which
is why the regions get rendered for inspection before the label set is used for
anything.

The two regions are read out of the query file rather than repeated here. They
were duplicated as module constants once, with a comment saying the two copies
had to match; with more than one query in the directory that is a bug waiting to
happen.

The interval is computed here rather than taken from the query. `THEN` spans from
the start of the approach spell to the end of the exit spell, and a car that
queues at the junction sits in `approach` for a minute before it moves; that
dwell is not the turn. The manoeuvre is the transition:

    last frame in the first region  ->  first frame in the second

and the boundary rule keeps a turn only when that whole transition falls inside
the clip. Scoring a model against the dwell would credit it for an interval no
one can see.

    uv run suite/vlm/groundtruth.py
    uv run suite/vlm/groundtruth.py --start 180 --end 780 \
        --query queries/UAV-rightturn-top.trql
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import duckdb
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import CLI, PROCESSED, ROOT, VLM

_REPO = ROOT
_HERE = VLM
_DB = PROCESSED / "ytstream/uav_0006/duckdb.db"
_VIDEO = PROCESSED / "ytstream/uav_0006/video.mp4"
_CLI = CLI

FPS = 5
WIN_S, WIN_E = 180, 210          # default window, overridable on the CLI
F0, F1 = WIN_S * FPS, WIN_E * FPS
DEFAULT_QUERY = Path("queries/UAV-rightturn.trql")


def regions(query: str) -> tuple[tuple[int, ...], tuple[int, ...], list[str]]:
    """The two rectangles of the REGIONS clause, in declaration order.

    One source of truth: the engine and this script read the same text, so the
    labels cannot drift from the query they are supposed to specify.
    """
    m = re.search(r"REGIONS\s*\((.*?)\)\s*\n", query, re.S)
    if not m:
        raise SystemExit("query has no REGIONS clause")
    found = re.findall(r"(\w+)\s*=\s*\[([^\]]+)\]", m.group(1))
    if len(found) != 2:
        raise SystemExit(
            f"expected exactly 2 regions, found {len(found)}: "
            f"{', '.join(n for n, _ in found)}")
    boxes = [tuple(int(float(v)) for v in body.split(",")) for _, body in found]
    if any(len(b) != 4 for b in boxes):
        raise SystemExit("this script handles rectangles only, not polygons")
    return boxes[0], boxes[1], [n for n, _ in found]


def engine_oids() -> list[int]:
    """Index i of an engine tuple is `oids[i]` in the database."""
    c = duckdb.connect(str(_DB), read_only=True)
    out = [r[0] for r in c.execute(
        "SELECT DISTINCT oid FROM bbox ORDER BY oid").fetchall()]
    c.close()
    return out


def run_query(q: str) -> list[int]:
    out = _HERE / "clip" / "_rightturn.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(_CLI), "--data-dir", str(PROCESSED), "--dataset", "ytstream", "--sub-dataset", "uav_0006",
         "--max-gap", "10", "--max-error-px", "0", "run", q, str(out)],
        check=True, capture_output=True, text=True)
    idx = engine_oids()
    data = json.loads(out.read_text())
    return sorted({idx[t[0][0]] for t in data["results"]})


def manoeuvre(conn, oid: int, approach, exit_r) -> tuple[int, int] | None:
    """(last frame in approach, first frame in exit) for this track, or None."""
    p = conn.execute(
        "SELECT fid, (x1+x2)/2.0, (y1+y2)/2.0 FROM bbox WHERE oid = ? ORDER BY fid",
        [oid]).fetchall()
    inside = lambda x, y, r: r[0] <= x <= r[2] and r[1] <= y <= r[3]
    in_a = [f for f, x, y in p if inside(x, y, approach)]
    if not in_a:
        return None
    exit_f = next((f for f, x, y in p if f > in_a[0] and inside(x, y, exit_r)), None)
    if exit_f is None:
        return None
    return max(f for f in in_a if f < exit_f), exit_f


def overlay(members: list[dict], path: Path, approach, exit_r, names) -> None:
    """The window's first frame, with both regions and every turn drawn on it."""
    frame = path.with_name("_frame.png")
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(WIN_S), "-i", str(_VIDEO),
         "-frames:v", "1", str(frame)],
        check=True, capture_output=True)
    img = Image.open(frame).convert("RGB")
    d = ImageDraw.Draw(img)
    d.rectangle(approach, outline=(32, 160, 64), width=3)
    d.text((approach[0] + 4, approach[1] + 4), names[0], fill=(32, 160, 64))
    d.rectangle(exit_r, outline=(30, 102, 245), width=3)
    d.text((exit_r[0] + 4, exit_r[1] + 4), names[1], fill=(30, 102, 245))

    conn = duckdb.connect(str(_DB), read_only=True)
    for m in members:
        pts = conn.execute(
            "SELECT (x1+x2)/2.0, (y1+y2)/2.0 FROM bbox "
            "WHERE oid = ? AND fid BETWEEN ? AND ? ORDER BY fid",
            [m["oid"], m["turn_start_frame"], m["turn_end_frame"]]).fetchall()
        if len(pts) > 1:
            d.line([tuple(q) for q in pts], fill=(210, 15, 57), width=3)
            d.text(pts[0], f" {m['oid']}", fill=(210, 15, 57))
    conn.close()
    img.save(path)
    frame.unlink(missing_ok=True)


def main() -> None:
    global WIN_S, WIN_E, F0, F1
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=WIN_S, help="window start, seconds")
    ap.add_argument("--end", type=int, default=WIN_E, help="window end, seconds")
    ap.add_argument("--query", type=Path, default=DEFAULT_QUERY,
                    help="the .trql region query that specifies the event")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    WIN_S, WIN_E = args.start, args.end
    F0, F1 = WIN_S * FPS, WIN_E * FPS

    qpath = args.query if args.query.is_absolute() else _HERE / args.query
    q = qpath.read_text().strip()
    approach, exit_r, names = regions(q)
    # Each query gets its own labels and its own overlay; the default query
    # keeps the names it already had so nothing downstream has to move.
    tag = "" if qpath.stem == DEFAULT_QUERY.stem else f"_{qpath.stem.split('-')[-1]}"
    if args.out is None:
        args.out = _HERE / (f"gt{tag}.json" if (WIN_S, WIN_E) == (180, 210)
                            else f"gt{tag}_{WIN_S}_{WIN_E}.json")
    oids = run_query(q)
    conn = duckdb.connect(str(_DB), read_only=True)

    members, rejected = [], []
    for oid in oids:
        mv = manoeuvre(conn, oid, approach, exit_r)
        if mv is None:
            continue
        a, e = mv
        rec = {"oid": oid, "turn_start_frame": a, "turn_end_frame": e,
               "turn_start_s": a / FPS, "turn_end_s": e / FPS}
        if F0 <= a and e <= F1:
            members.append(rec)
        elif a <= F1 and e >= F0:          # overlaps the window but straddles it
            rejected.append(rec)
    conn.close()

    members.sort(key=lambda m: m["turn_start_frame"])
    gt = {"video": "ytstream/uav_0006", "window_s": [WIN_S, WIN_E],
          "window_frames": [F0, F1], "fps": FPS,
          "query_file": qpath.name, "region_names": names,
          "approach": list(approach), "exit_right": list(exit_r),
          "query": q, "turns_in_whole_video": len(oids), "turns": members,
          "straddling_window_excluded": rejected}
    args.out.write_text(json.dumps(gt, indent=2) + "\n")

    print(f"{qpath.name}: {names[0]} -> {names[1]}")
    print(f"turns in the whole video: {len(oids)}")
    print(f"completing inside {WIN_S}-{WIN_E}s: {len(members)}")
    for m in members:
        print(f"  oid {m['oid']:>4}  turn {m['turn_start_s']:>6.1f}s -> "
              f"{m['turn_end_s']:>6.1f}s  (frames {m['turn_start_frame']}-{m['turn_end_frame']})")
    if rejected:
        print(f"straddling the window, excluded: "
              f"{', '.join(str(r['oid']) for r in rejected)}")
    png = args.out.with_name(f"regions{tag}.png")
    overlay(members, png, approach, exit_r, names)
    print(f"\nwrote {args.out.relative_to(_REPO)} and {png.relative_to(_REPO)}")
    print("inspect regions.png before trusting this label set")


if __name__ == "__main__":
    main()
