#!/usr/bin/env python
"""Convert the SoccerTrack v2 dataset into the TrQL world format.

Ten amateur matches, each shot as a single 4K panorama per half, with three
annotation sources that all key on the same player ids:

    gsr/<m>/<m>_{1st,2nd}.json      per-frame players + goalkeepers, in BOTH
                                    image pixels and pitch metres (~2.7 GB each)
    bas/<m>/<m>_12_class_events.json  ball-action events (PASS, SHOT, GOAL, ...)
    raw/<m>/<m>_tracker_box_data.xml  per-frame positions incl. THE BALL

Each half becomes a world, because a world's cameras share its db and therefore
its timeline:

    <out>/SoccerTrackv2/SoccerTrackv2-<match>-<half>/
        duckdb.db            players + goalkeepers + ball, pitch metres x 100
        world-metadata.json  the frame those coordinates are stored in
        world-view.svg       top view of the pitch, same frame
        panorama/
            video.mp4        symlink to the source 4K panorama
            metadata.json
            duckdb.db        the same players, in the video's own pixels

The panorama of each half is also exposed as its own sub-dataset, which is what
the paper's ST-swap-img query reads (merged by `converters/paper_merges.sh --only
soccertrack-20`):

    <out>/SoccerTrackv2pano/SoccerTrackv2-<match>-<half>/   symlinks to panorama/*

Unlike Alfheim, this dataset ships a usable projection: `gsr` carries
`bbox_image` *and* `bbox_pitch` for every annotation, so the camera's boxes
genuinely line up with the footage while the world db stays in pitch metres for
the map. That is why there are two dbs rather than one.

The ball comes from the tracker XML, whose positions are normalised to [0, 1]
across the pitch. That mapping was verified rather than assumed: converting with
`x = nx*105 - 52.5, y = ny*68 - 34` and matching frames by their ordinal within
the period reproduces `gsr`'s own `bbox_pitch` for the same players to a median
residual of 0.00 m -- the two files are the same measurements. The ball has no
image-space position, so it appears only in the world db.

Raw layout (`--raw`, default `datasets/raw/soccertrack-v2`), as downloaded
from the dataset's Hugging Face repository (`atomscott/soccertrack-v2`)::

    gsr/<m>/<m>_{1st,2nd}.json
    bas/<m>/<m>_12_class_events.json
    raw/<m>/<m>_{tracker_box_data.xml,player_nodes.csv,keypoints.json}
    videos/<m>/<m>_panorama_{1st,2nd}_half.mp4

The videos are needed even with `--no-video`: their frame counts decide which
annotations run past the footage and are dropped, which happens in 19 of the 20
halves.

Run from the repository root with the converter venv::

    .trqs-pyenv/bin/python converters/soccertrack_convert.py      # everything (~30 min, streams 53 GB)
    .trqs-pyenv/bin/python converters/soccertrack_convert.py --matches 117092
    .trqs-pyenv/bin/python converters/soccertrack_convert.py --no-video   # no video symlink
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import duckdb
import ijson

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_ROOT = SCRIPT_DIR.parent / "datasets"
sys.path.insert(0, str(SCRIPT_DIR))

from common import SCHEMA  # noqa: E402

DATASET = "SoccerTrackv2"
#: Family of the per-half panorama sub-datasets (symlinks into each world).
PANO_FAMILY = "SoccerTrackv2pano"

#: Annotation rate of every match in the set.
FPS = 25.0
#: Stored units per metre, as for the other pitch datasets.
SCALE = 100.0
#: Nominal pitch, used to place the world view and to un-normalise the tracker.
PITCH_L, PITCH_W = 105.0, 68.0
#: Footprint side (metres) drawn at a player's ground position, and the ball's.
PLAYER_SIZE_M = 0.7
BALL_SIZE_M = 0.3

HALVES = {"1st": "FIRST_HALF", "2nd": "SECOND_HALF"}
#: `bas` positions are a match clock: the second half is stamped from 45:00 on.
HALF_START_MS = {"1st": 0, "2nd": 2_700_000}

#: The 12 ball-action classes, in a fixed order so `aid` means the same thing in
#: every world. Indexing them per-half instead would be tempting -- two halves
#: never contain quite the same actions, `GOAL` appears in 18 of 20 and `HEADER`
#: in 15 -- but it would give each world its own numbering, and a query written
#: against one would quietly match a different action in the next. Every world
#: therefore gets all 12 rows, including the ones it never uses.
ACTION_CLASSES = (
    "BALL PLAYER BLOCK",
    "CROSS",
    "DRIVE",
    "FREE KICK",
    "GOAL",
    "HEADER",
    "HIGH PASS",
    "OUT",
    "PASS",
    "PLAYER SUCCESSFUL TACKLE",
    "SHOT",
    "THROW IN",
)


# --------------------------------------------------------------------------- #
# Ball actions (`bas`)
# --------------------------------------------------------------------------- #
def read_events(match_dir: Path, half: str, num_frames: int) -> tuple[dict, list, dict]:
    """Events of one half as `{(frame, player_id): aid}`, plus the class list.

    Returns `(by_key, classes, stats)`. `aid` indexes `classes`, so the
    `activities` table is the 12 action classes and a box carries the class it
    was performing -- not one row per event instance.

    Two quirks of the source are handled here rather than left to surprise
    someone later. Positions are a *match* clock, so the second half starts near
    45:00 and has to be rebased. And a few events are simply wrong -- match
    117092 stamps first-half actions at 135:04, well past the end of a 48-minute
    video -- so anything outside the half is dropped and counted.
    """
    files = sorted(match_dir.glob("*_class_events.json"))
    if not files:
        return {}, [], {"events": 0}
    # Prefer the 12-class file when a match ships both 8- and 12-class versions.
    path = next((p for p in files if "12_class" in p.name), files[0])
    actions = json.loads(path.read_text()).get("actions", [])

    period = "1" if half == "1st" else "2"
    base = HALF_START_MS[half]
    classes: list[str] = list(ACTION_CLASSES)
    index: dict[str, int] = {c: i for i, c in enumerate(classes)}
    by_key: dict[tuple[int, int], int] = {}
    n_half = n_kept = n_out = n_nopid = n_unknown = 0

    for a in actions:
        if not str(a.get("gameTime", "")).startswith(period + " "):
            continue
        n_half += 1
        pid = a.get("player_id")
        if not pid:
            n_nopid += 1
            continue
        try:
            fid = int(round((int(a["position"]) - base) / 1000.0 * FPS))
        except (KeyError, TypeError, ValueError):
            continue
        if fid < 0 or fid >= num_frames:
            n_out += 1
            continue
        label = a.get("label") or "UNKNOWN"
        if label not in index:
            # An action outside the documented 12 still gets an id rather than
            # being dropped, appended past them so the fixed block keeps its
            # meaning.
            index[label] = len(classes)
            classes.append(label)
            n_unknown += 1
        # First event wins a given (frame, player): a box carries one aid, and
        # two actions by the same player in the same 40 ms is a labelling
        # artefact rather than two things to represent.
        by_key.setdefault((fid, int(pid)), index[label])
        n_kept += 1

    return by_key, classes, {
        "events_in_half": n_half,
        "events_attached": n_kept,
        "events_outside_video": n_out,
        "events_without_player": n_nopid,
        "classes_beyond_the_documented_12": n_unknown,
    }


# --------------------------------------------------------------------------- #
# Which team a player belongs to
# --------------------------------------------------------------------------- #
def read_team_map(path: Path) -> tuple[dict[int, str], dict[str, str]]:
    """`player_id -> "A"/"B"` from `raw/<m>_player_nodes.csv`, plus club names.

    The side letters come from the club id, **not** from `gsr`'s `team`
    attribute. That attribute records which side of the pitch a player is on,
    and the teams change ends at half time: every player present in both halves
    of 117092 flips left<->right. Keying the class off it would put a team in
    `playerA` for one half and `playerB` for the other, which is exactly the
    kind of error that never announces itself.

    Ids are sorted so the same club is always A, whichever order the file lists
    them in and however often the converter is rerun.
    """
    try:
        rows = list(csv.DictReader(path.open(newline="")))
    except OSError:
        return {}, {}
    if not rows or "team_id" not in rows[0] or "player_id" not in rows[0]:
        return {}, {}

    # A player occasionally appears against two clubs in this file; the one they
    # appear with most often wins.
    seen: dict[int, dict[str, int]] = {}
    names: dict[str, str] = {}
    for r in rows:
        try:
            pid = int(r["player_id"])
        except (TypeError, ValueError):
            continue
        tid = (r.get("team_id") or "").strip()
        if not tid:
            continue
        seen.setdefault(pid, {}).setdefault(tid, 0)
        seen[pid][tid] += 1
        names.setdefault(tid, (r.get("team_name") or "").strip())

    ids = sorted({t for v in seen.values() for t in v})
    letters = {t: chr(ord("A") + i) for i, t in enumerate(ids)}
    team_of = {pid: letters[max(v, key=v.get)] for pid, v in seen.items()}
    return team_of, {letters[t]: names.get(t, t) for t in ids}


def resolve_classes(tracks: dict, team_of: dict[int, str]) -> tuple[dict, dict]:
    """Final class per track: role + team letter, e.g. `playerA`, `goalkeeperB`.

    Tracks whose `player_id` is missing from `player_nodes.csv` are placed by
    the side they were annotated on: the sides map onto the clubs consistently
    *within* one half, so the correspondence is measured from the tracks that
    are known and then applied to the rest. A track with neither keeps its bare
    role rather than being guessed onto a team.
    """
    # side ("left"/"right") -> team letter, learned from the resolved tracks.
    votes: dict[str, dict[str, int]] = {}
    for _, _, pid, side in tracks.values():
        if pid in team_of and side:
            votes.setdefault(side, {}).setdefault(team_of[pid], 0)
            votes[side][team_of[pid]] += 1
    side_letter = {s: max(v, key=v.get) for s, v in votes.items()}

    classes: dict[int, str] = {}
    stats = {"by_player_id": 0, "by_side": 0, "unplaced": 0}
    for tid, (_, role, pid, side) in tracks.items():
        letter = team_of.get(pid) or side_letter.get(side or "")
        if pid in team_of:
            stats["by_player_id"] += 1
        elif letter:
            stats["by_side"] += 1
        else:
            stats["unplaced"] += 1
        classes[tid] = f"{role}{letter}" if letter else role
    return classes, stats


# --------------------------------------------------------------------------- #
# The ball (`raw/<m>_tracker_box_data.xml`)
# --------------------------------------------------------------------------- #
FRAME_RE = re.compile(rb'<frame[^>]*eventPeriod="([A-Z_]+)"')
BALL_RE = re.compile(rb'<ball[^>]*loc="\[([-0-9.]+), ?([-0-9.]+)\]"')


def read_ball(xml_path: Path, half: str) -> list[tuple[int, float, float]]:
    """Ball positions of one half as `(frame, x_m, y_m)` in pitch metres.

    Frames are matched by their **ordinal within the period**, not by the file's
    `frameNumber`: the tracker numbers frames from 252 and the count per period
    equals `gsr`'s image count exactly, so the nth tracker frame is the nth
    annotated frame. Positions are normalised across the pitch and centred, a
    mapping checked against `gsr`'s own `bbox_pitch` (median residual 0.00 m).
    """
    want = HALVES[half]
    out: list[tuple[int, float, float]] = []
    idx = -1
    in_period = False
    with xml_path.open("rb") as fh:
        for line in fh:
            m = FRAME_RE.search(line)
            if m:
                in_period = m.group(1).decode() == want
                if in_period:
                    idx += 1
                continue
            if not in_period:
                continue
            b = BALL_RE.search(line)
            if b:
                nx, ny = float(b.group(1)), float(b.group(2))
                out.append((idx, nx * PITCH_L - PITCH_L / 2.0, ny * PITCH_W - PITCH_W / 2.0))
    return out


# --------------------------------------------------------------------------- #
# Video
# --------------------------------------------------------------------------- #
def read_pitch_model(path: Path) -> dict | None:
    """Derive the pitch's real geometry from the dataset's own keypoints.

    `raw/<m>/<m>_keypoints.json` maps pitch positions **in metres** to image
    pixels, keyed `"(x,y)"` on a corner origin. The 65 keys are the same for
    every match and spell out the markings, so the world view can be drawn from
    what the dataset asserts instead of from a textbook pitch:

        y=0, y=68 spanning x=0..105              touchlines
        y=13.84 / 54.16 at x=16.5 and x=88.5     penalty areas
        y=24.84 / 43.16 at x=5.5  and x=99.5     goal areas
        y=30.34 / 37.66 at x=0    and x=105      goalposts
        y=24.85, 34.0, 43.15 at x=52.5           halfway line + centre circle

    Bands are recognised by structure rather than by their values: a marking
    line shows up at two or more distinct x, while the circle's samples sit on
    the halfway column alone, which is what separates y=24.84 (goal area) from
    y=24.85 (circle). Returns metres on the corner origin; the caller re-centres.
    """
    try:
        keys = json.loads(path.read_text()).keys()
    except (OSError, json.JSONDecodeError):
        return None
    pts = []
    for k in keys:
        nums = re.findall(r"-?[\d.]+", k)
        if len(nums) == 2:
            pts.append((float(nums[0]), float(nums[1])))
    if len(pts) < 8:
        return None

    xs = sorted({x for x, _ in pts})
    ys = sorted({y for _, y in pts})
    length, width = max(xs) - min(xs), max(ys) - min(ys)
    cx, cy = min(xs) + length / 2.0, min(ys) + width / 2.0

    x_at: dict[float, set] = {}
    y_at: dict[float, set] = {}
    for x, y in pts:
        x_at.setdefault(y, set()).add(x)
        y_at.setdefault(x, set()).add(y)

    # The halfway column carries the circle: the x nearest the centre holding a
    # sample at the centre line itself.
    halfway = min((x for x in xs if any(abs(y - cy) < 0.5 for y in y_at[x])),
                  key=lambda x: abs(x - cx), default=cx)
    circle_ys = [y for y in y_at.get(halfway, ()) if len(x_at[y]) == 1]
    radius = max((abs(y - cy) for y in circle_ys), default=0.0)

    # Marking bands: a y appearing at two or more x, off the touchlines. Its
    # depth is how far the line runs in from each goal line.
    bands = []
    for y in ys:
        if abs(y - min(ys)) < 0.01 or abs(y - max(ys)) < 0.01 or len(x_at[y]) < 2:
            continue
        if y >= cy:                      # take each symmetric pair once
            continue
        # How far the marking runs in from the goal line. Only x *strictly*
        # between the two goal lines counts: the goalposts are sampled at x=0
        # and x=105 alone, and treating the far one as an inner edge would draw
        # a box the length of the pitch.
        inner = sorted(x for x in x_at[y]
                       if min(xs) + 0.01 < x < max(xs) - 0.01)
        depth = (inner[0] - min(xs)) if inner else 0.0
        bands.append({"y0": y, "y1": min(ys) + max(ys) - y, "depth": depth})

    return {"x0": min(xs), "y0": min(ys), "length": length, "width": width,
            "cx": cx, "cy": cy, "halfway": halfway, "radius": radius,
            "bands": sorted(bands, key=lambda b: -b["depth"]), "n_keypoints": len(pts)}


def find_video(videos_dir: Path, match: str, half: str) -> Path | None:
    """The match's panorama for a half.

    `117093` also ships a `calibrated_panorama` variant; the plain panorama is
    the one the annotations are in, so the name is matched exactly rather than
    by glob.
    """
    p = videos_dir / match / f"{match}_panorama_{half}_half.mp4"
    return p if p.is_file() else None


def probe(video: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,nb_frames,avg_frame_rate", "-of", "json", str(video)],
        check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = (s.get("avg_frame_rate") or "25/1").split("/")
    fps = float(num) / float(den) if float(den) else FPS
    n = int(s.get("nb_frames") or 0)
    return {"width": int(s["width"]), "height": int(s["height"]),
            "fps": fps, "num_frames": n, "duration_s": n / fps if fps else 0.0}


# --------------------------------------------------------------------------- #
# The main pass over one half's gsr file
# --------------------------------------------------------------------------- #
class Writer:
    """Streams bbox rows to a temp CSV for bulk loading.

    A half is ~1.5 M boxes per db and DuckDB binds `executemany` one statement
    at a time, which is far too slow at this volume.
    """

    def __init__(self, out_dir: Path, tag: str):
        fd, self.path = tempfile.mkstemp(suffix=f".{tag}.csv", dir=str(out_dir))
        self.fh = os.fdopen(fd, "w", newline="")
        self.w = csv.writer(self.fh)
        self.n = 0

    def row(self, fid, oid, x1, y1, x2, y2, aid):
        self.w.writerow((fid, oid, x1, y1, x2, y2, "" if aid is None else aid))
        self.n += 1

    def close(self):
        self.fh.close()

    def unlink(self):
        os.unlink(self.path)


BBOX_COLS = ("{'fid':'BIGINT','oid':'BIGINT','x1':'DOUBLE','y1':'DOUBLE',"
             "'x2':'DOUBLE','y2':'DOUBLE','aid':'BIGINT'}")


def build_db(db_path: Path, csv_path: str, n_rows: int, objects: list,
             classes: list[str]) -> dict:
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    try:
        con.execute(SCHEMA)
        if objects:
            con.executemany("INSERT INTO objects VALUES (?, ?)", objects)
        if classes:
            con.executemany("INSERT INTO activities VALUES (?, ?)",
                            list(enumerate(classes)))
        if n_rows:
            con.execute(
                f"INSERT INTO bbox SELECT * FROM read_csv(?, header=false, "
                f"columns={BBOX_COLS})", [csv_path])
    finally:
        con.close()
    return {"objects": len(objects), "activities": len(classes), "bbox": n_rows}


def convert_half(raw: Path, out_root: Path, match: str, half: str,
                 do_video: bool) -> None:
    gsr_path = raw / "gsr" / match / f"{match}_{half}.json"
    if not gsr_path.is_file():
        print(f"   !! {gsr_path.name} missing, skipping")
        return

    world_dir = out_root / f"{DATASET}-{match}-{half}"
    cam_dir = world_dir / "panorama"
    world_dir.mkdir(parents=True, exist_ok=True)
    cam_dir.mkdir(parents=True, exist_ok=True)

    pitch_model = read_pitch_model(raw / "raw" / match / f"{match}_keypoints.json")
    if pitch_model is None:
        print("   !! no keypoints.json; falling back to a nominal pitch")
        pitch_model = {"x0": 0.0, "y0": 0.0, "length": PITCH_L, "width": PITCH_W,
                 "cx": PITCH_L / 2, "cy": PITCH_W / 2, "halfway": PITCH_L / 2,
                 "radius": 9.15, "n_keypoints": 0,
                 "bands": [{"y0": 13.84, "y1": 54.16, "depth": 16.5},
                           {"y0": 24.84, "y1": 43.16, "depth": 5.5}]}

    team_of, team_names = read_team_map(
        raw / "raw" / match / f"{match}_player_nodes.csv")

    video = find_video(raw / "videos", match, half)
    vinfo = probe(video) if video else None
    # Annotations run a little past the footage on several matches (118575's
    # first half claims 72 101 frames against the video's 71 850). Boxes beyond
    # the last real frame would sit on a video that has ended, so they are
    # dropped and counted rather than written.
    limit = vinfo["num_frames"] if vinfo else None

    # --- events first: the gsr pass needs them to stamp aids as it goes ---
    n_frames_guess = limit or 10 ** 9
    events, classes, ev_stats = read_events(raw / "bas" / match, half, n_frames_guess)

    pitch = Writer(world_dir, "pitch")
    image = Writer(cam_dir, "image")
    # track_id -> [oid, role, player_id, side]; the class needs the whole half
    objects: dict[int, list] = {}
    catname: dict[int, str] = {}
    img_frame: dict[str, int] = {}
    frame_w = frame_h = 0
    px = [math.inf, -math.inf]
    py = [math.inf, -math.inf]
    half_m = PLAYER_SIZE_M / 2.0
    n_clamped = n_nopitch = n_noimage = 0

    with gsr_path.open("rb") as fh:
        for key, obj in ijson.kvitems(fh, ""):
            if key == "categories":
                catname = {int(c["id"]): c["name"] for c in obj}
            elif key == "images":
                for im in obj:
                    img_frame[im["image_id"]] = int(re.sub(r"\D", "", im["file_name"])) - 1
                    frame_w = max(frame_w, int(im["width"]))
                    frame_h = max(frame_h, int(im["height"]))
            elif key == "annotations":
                for a in obj:
                    if a.get("supercategory") != "object":
                        continue
                    fid = img_frame.get(a["image_id"])
                    if fid is None:
                        continue
                    if limit is not None and fid >= limit:
                        n_clamped += 1
                        continue
                    tid = int(a["track_id"])
                    attrs = a.get("attributes") or {}
                    pid = attrs.get("player_id")
                    if tid not in objects:
                        # `attributes.role` rather than the category table: the
                        # `categories` block sits at the END of the file, so
                        # during this pass `catname` is still empty and every
                        # track would fall back to "player" — which is how the
                        # goalkeepers went missing from the first build.
                        role = (attrs.get("role")
                                or catname.get(int(a.get("category_id", 0)))
                                or "player")
                        objects[tid] = [len(objects), role, None, None]
                    if objects[tid][2] is None and pid is not None:
                        objects[tid][2] = int(pid)
                        objects[tid][3] = attrs.get("team")
                    oid = objects[tid][0]
                    aid = events.get((fid, int(pid))) if pid is not None else None

                    b = a.get("bbox_image") or {}
                    if b and b.get("x") is not None:
                        x, y = float(b["x"]), float(b["y"])
                        image.row(fid, oid, x, y, x + float(b["w"]), y + float(b["h"]), aid)
                    else:
                        n_noimage += 1

                    q = a.get("bbox_pitch") or {}
                    qx, qy = q.get("x_bottom_middle"), q.get("y_bottom_middle")
                    if qx is None or qy is None:
                        n_nopitch += 1
                        continue
                    qx, qy = float(qx), float(qy)
                    px[0] = min(px[0], qx - half_m); px[1] = max(px[1], qx + half_m)
                    py[0] = min(py[0], qy - half_m); py[1] = max(py[1], qy + half_m)
                    pitch.row(fid, oid, qx - half_m, qy - half_m,
                              qx + half_m, qy + half_m, aid)

    # --- the ball, world db only (it has no image-space position) ---
    ball_oid = len(objects)
    n_ball = 0
    xml = raw / "raw" / match / f"{match}_tracker_box_data.xml"
    if xml.is_file():
        half_b = BALL_SIZE_M / 2.0
        for fid, bx, by in read_ball(xml, half):
            if limit is not None and fid >= limit:
                continue
            px[0] = min(px[0], bx - half_b); px[1] = max(px[1], bx + half_b)
            py[0] = min(py[0], by - half_b); py[1] = max(py[1], by + half_b)
            pitch.row(fid, ball_oid, bx - half_b, by - half_b,
                      bx + half_b, by + half_b, None)
            n_ball += 1

    pitch.close()
    image.close()

    # --- shift the pitch onto the origin, as the other pitch datasets do ---
    shift_x, shift_y = px[0], py[0]
    shifted = tempfile.mkstemp(suffix=".shift.csv", dir=str(world_dir))
    os.close(shifted[0])
    with open(pitch.path, newline="") as src, open(shifted[1], "w", newline="") as dst:
        r, w = csv.reader(src), csv.writer(dst)
        for row in r:
            w.writerow((row[0], row[1],
                        int(math.floor((float(row[2]) - shift_x) * SCALE)),
                        int(math.floor((float(row[3]) - shift_y) * SCALE)),
                        int(math.ceil((float(row[4]) - shift_x) * SCALE)),
                        int(math.ceil((float(row[5]) - shift_y) * SCALE)),
                        row[6]))

    cls_of, team_stats = resolve_classes(objects, team_of)
    world_objs = [(rec[0], cls_of[tid])
                  for tid, rec in sorted(objects.items(), key=lambda kv: kv[1][0])]
    cam_objs = list(world_objs)
    if n_ball:
        world_objs.append((ball_oid, "ball"))

    wcounts = build_db(world_dir / "duckdb.db", shifted[1], pitch.n, world_objs, classes)
    ccounts = build_db(cam_dir / "duckdb.db", image.path, image.n, cam_objs, classes)
    os.unlink(shifted[1])
    pitch.unlink()
    image.unlink()

    # --- video + camera metadata ---
    if do_video and video:
        link = cam_dir / "video.mp4"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(video.resolve())

    num_frames = limit or (max(img_frame.values()) + 1 if img_frame else 0)
    cam_meta = {
        "dataset": DATASET,
        "source": "SoccerTrack v2 (SoccerNet game-state annotations over the 4K panorama)",
        "match": match,
        "half": half,
        "fps": vinfo["fps"] if vinfo else FPS,
        "num_frames": num_frames,
        "duration_s": vinfo["duration_s"] if vinfo else num_frames / FPS,
        "width": vinfo["width"] if vinfo else frame_w,
        "height": vinfo["height"] if vinfo else frame_h,
        "coordinate_space": "image pixels",
        "obj_classes": sorted({c for _, c in cam_objs}),
        "activity_classes": classes,
        "row_counts": ccounts,
        "teams": team_names,
        "team_assignment": team_stats,
        "gsr_declared_frame_size": [frame_w, frame_h],
        "notes": (
            "Boxes are gsr's bbox_image, in this video's own pixels, so they "
            "overlay the footage directly with no scaling. Ignore "
            "gsr_declared_frame_size: every gsr file states 3840x1504 whatever "
            "the match was actually shot at (4096x1080 for most, 3840x1906 for "
            "117092), so it is a template value, not a measurement — verified "
            "by rendering boxes over frames of both a 3840x1906 and a 4096x1080 "
            "match, where they land on the players unscaled and applying the "
            "size ratio puts them in empty grass. The ball is NOT here: the "
            "tracker gives it a pitch position only, so it lives in the world's "
            "duckdb.db. Activities are the bas action classes, stamped on the "
            "acting player's box at the event frame."
        ),
    }
    (cam_dir / "metadata.json").write_text(json.dumps(cam_meta, indent=2) + "\n")

    write_world(world_dir, f"{match}-{half}", shift_x, shift_y, px, py,
                num_frames, wcounts, classes, n_ball, ev_stats,
                {"clamped_past_video": n_clamped, "no_pitch_coords": n_nopitch,
                 "no_image_coords": n_noimage}, pitch_model,
                sorted({c for _, c in world_objs}), team_names, team_stats)

    print(f"   world {wcounts['objects']} objs / {wcounts['bbox']} boxes"
          f" | panorama {ccounts['objects']} objs / {ccounts['bbox']} boxes"
          f" | ball {n_ball} | acts {len(classes)}"
          f"{f' | clamped {n_clamped}' if n_clamped else ''}", flush=True)


# --------------------------------------------------------------------------- #
# World view + metadata
# --------------------------------------------------------------------------- #
def write_world(world_dir: Path, label: str, shift_x: float, shift_y: float,
                px, py, num_frames: int, counts: dict, classes: list,
                n_ball: int, ev_stats: dict, drops: dict, pitch: dict,
                obj_classes: list, team_names: dict, team_stats: dict) -> None:
    """`world-view.svg` + `world-metadata.json` for one half.

    The pitch is centred on the origin in source coordinates, so after the shift
    its corner sits at `-shift`; the drawing is placed through the same
    transform as the boxes so the lines land where the players do.
    """
    s = SCALE

    def X(m):
        return (m - shift_x) * s

    def Y(m):
        return (m - shift_y) * s

    # The keypoints are on a corner origin; the boxes are centred on the halfway
    # spot, so the drawing is re-centred through the model's own centre rather
    # than through an assumed half-pitch.
    p = pitch

    def kx_(kx):
        return X(kx - p["cx"])

    def ky_(ky):
        return Y(ky - p["cy"])

    parts = [
        f'<rect x="{kx_(p["x0"]):.0f}" y="{ky_(p["y0"]):.0f}" '
        f'width="{p["length"]*s:.0f}" height="{p["width"]*s:.0f}" fill="none"/>',
        f'<line x1="{kx_(p["halfway"]):.0f}" y1="{ky_(p["y0"]):.0f}" '
        f'x2="{kx_(p["halfway"]):.0f}" y2="{ky_(p["y0"]+p["width"]):.0f}"/>',
    ]
    if p["radius"] > 0:
        parts.append(
            f'<circle cx="{kx_(p["cx"]):.0f}" cy="{ky_(p["cy"]):.0f}" '
            f'r="{p["radius"]*s:.0f}" fill="none"/>')
    for band in p["bands"]:
        h = (band["y1"] - band["y0"]) * s
        if band["depth"] <= 0:
            # A band sampled only on the goal lines is the goal mouth itself;
            # drawn as a stroke across each one rather than as a box.
            for x0 in (p["x0"], p["x0"] + p["length"]):
                parts.append(
                    f'<line x1="{kx_(x0):.0f}" y1="{ky_(band["y0"]):.0f}" '
                    f'x2="{kx_(x0):.0f}" y2="{ky_(band["y1"]):.0f}" '
                    f'stroke-width="{0.4*s:.0f}"/>')
            continue
        for x0 in (p["x0"], p["x0"] + p["length"] - band["depth"]):
            parts.append(
                f'<rect x="{kx_(x0):.0f}" y="{ky_(band["y0"]):.0f}" '
                f'width="{band["depth"]*s:.0f}" height="{h:.0f}" fill="none"/>')

    width = math.ceil((px[1] - shift_x) * s)
    height = math.ceil((py[1] - shift_y) * s)
    (world_dir / "world-view.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">'
        f'<g stroke="#2e7d32" stroke-width="{0.12*s:.0f}" fill="none">'
        + "".join(parts) + "</g></svg>\n")

    meta = {
        "dataset": DATASET,
        "world": label,
        "source": "SoccerTrack v2 (gsr pitch coordinates + tracker ball)",
        "coordinate_scale": SCALE,
        "x_delta": -shift_x * SCALE,
        "y_delta": -shift_y * SCALE,
        "coordinate_space": f"pitch metres x {SCALE:.0f} (NOT image pixels)",
        "pitch_length_m": pitch["length"],
        "pitch_width_m": pitch["width"],
        "pitch_model": {
            "source": ("derived from the match's keypoints.json"
                       if pitch["n_keypoints"] else "nominal fallback"),
            "keypoints": pitch["n_keypoints"],
            "centre_circle_radius_m": pitch["radius"],
            "marking_bands": pitch["bands"],
        },
        "world_view": "world-view.svg",
        "fps": FPS,
        "num_frames": num_frames,
        "duration_s": num_frames / FPS,
        "width": width,
        "height": height,
        "obj_classes": sorted(set(obj_classes) | ({"ball"} if n_ball else set())),
        "activity_classes": classes,
        "teams": team_names,
        "team_assignment": team_stats,
        "row_counts": counts,
        "event_stats": ev_stats,
        "dropped": drops,
        "notes": (
            "Classes carry the team: playerA/playerB and goalkeeperA/"
            "goalkeeperB, where the letter is the club id from "
            "player_nodes.csv (see `teams`) and NOT gsr's `team` attribute -- "
            "that one records which side of the pitch a player is on and flips "
            "at half time, so it would swap the teams between the two halves "
            "of a match. "
            f"Player and goalkeeper footprints ({PLAYER_SIZE_M} m square) from "
            f"gsr's bbox_pitch, plus the ball ({BALL_SIZE_M} m) from the "
            "tracker XML, whose normalised positions reproduce gsr's own pitch "
            "coordinates exactly. Stored coordinates are round(metres * "
            f"{SCALE:.0f}) shifted so the extent of play starts at the origin; "
            "pitch metres are (stored - delta) / coordinate_scale. These are "
            "NOT image pixels — the same objects in the video's pixel space are "
            "in panorama/duckdb.db. fid is a 25 Hz index shared with the video."
        ),
    }
    (world_dir / "world-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


# --------------------------------------------------------------------------- #
def link_panorama(out: Path, name: str) -> None:
    """Expose `<out>/SoccerTrackv2/<name>/panorama` as `<out>/SoccerTrackv2pano/<name>`."""
    cam_dir = out / DATASET / name / "panorama"
    if not (cam_dir / "duckdb.db").is_file():
        return
    pano = out / PANO_FAMILY / name
    pano.mkdir(parents=True, exist_ok=True)
    for f in ("duckdb.db", "metadata.json", "video.mp4"):
        link, target = pano / f, cam_dir / f
        if link.is_symlink() or link.exists():
            link.unlink()
        if target.exists():
            link.symlink_to(target.resolve())


def convert(raw: Path, out: Path, matches: list[str], halves: list[str],
            do_video: bool) -> None:
    out_root = out / DATASET
    out_root.mkdir(parents=True, exist_ok=True)
    found = sorted(p.name for p in (raw / "gsr").iterdir() if p.is_dir())
    for match in found:
        if matches and match not in matches:
            continue
        for half in halves:
            print(f"== {DATASET}-{match}-{half}", flush=True)
            convert_half(raw, out_root, match, half, do_video)
            link_panorama(out, f"{DATASET}-{match}-{half}")
    print(f"\nDone -> {out_root}/{DATASET}-* and {out / PANO_FAMILY}/{DATASET}-*")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=DATASETS_ROOT / "raw" / "soccertrack-v2")
    ap.add_argument("--out", type=Path, default=DATASETS_ROOT / "processed")
    ap.add_argument("--matches", nargs="*", default=[])
    ap.add_argument("--halves", nargs="*", default=list(HALVES))
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    convert(args.raw.resolve(), args.out.resolve(), args.matches, args.halves,
            not args.no_video)


if __name__ == "__main__":
    main()
