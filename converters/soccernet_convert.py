#!/usr/bin/env python
"""Convert the raw SoccerNet-Tracking (SNMOT) dataset into the TrQL common format.

SoccerNet-Tracking is MOTChallenge-style soccer player/ball tracking. Each
sequence ships frames (not a video), a MOT ``gt/gt.txt``, and a ``gameinfo.ini``
that maps each track id to a role (player/goalkeeper per team, referee, ball).

Raw layout expected (see README.md)::

    raw/soccernet/<split>/<SEQ>/img1/000001.jpg ...
    raw/soccernet/<split>/<SEQ>/gt/gt.txt
    raw/soccernet/<split>/<SEQ>/seqinfo.ini
    raw/soccernet/<split>/<SEQ>/gameinfo.ini

where ``<split>`` is ``train`` or ``test``. The ``challenge`` split has no
ground truth and is skipped.

Each sequence becomes one subset::

    <out>/soccernet/<split>_<SEQ>/video.mp4   (encoded from img1 at seqinfo fps)
                                 /duckdb.db
                                 /metadata.json

gt.txt columns (MOT): frame, id, bb_left, bb_top, bb_width, bb_height, conf, ...

  * objects.oid   = track id; objects.class = role from gameinfo.ini.
  * bbox          = one row per gt detection; aid is always NULL.
  * activities    = empty (pure tracking dataset).

Run with the project venv::

    .trqs-pyenv/bin/python converters/soccernet_convert.py
"""

from __future__ import annotations

import argparse
import configparser
import re
import sys
from pathlib import Path

# Make the shared helpers importable regardless of CWD.
SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_ROOT = SCRIPT_DIR.parent / "datasets"
sys.path.insert(0, str(SCRIPT_DIR))

from common import Obj, Bbox, build_subset  # noqa: E402

DATASET = "soccernet"
SPLITS = ("train", "test")  # "challenge" has no gt -> skipped
TRACKLET_RE = re.compile(r"^trackletID_(\d+)\s*=\s*([^;]+);?(.*)$")


def role_to_class(role: str) -> str:
    """Normalize a gameinfo role string to a class name."""
    r = role.strip().lower()
    if r.startswith("player"):
        if "left" in r:
            return "player_left"
        if "right" in r:
            return "player_right"
        return "player"
    if r.startswith("goalkeeper"):  # note: source has both "goalkeeper"/"goalkeepers"
        if "left" in r:
            return "goalkeeper_left"
        if "right" in r:
            return "goalkeeper_right"
        return "goalkeeper"
    if "referee" in r:
        return "referee"
    if "ball" in r:
        return "ball"
    return "other"


def parse_gameinfo(path: Path) -> dict[int, dict]:
    """Map track id -> {'class', 'role', 'jersey'} from gameinfo.ini."""
    tracklets: dict[int, dict] = {}
    if not path.is_file():
        return tracklets
    for line in path.read_text().splitlines():
        m = TRACKLET_RE.match(line.strip())
        if not m:
            continue
        tid = int(m.group(1))
        role = m.group(2).strip()
        jersey = m.group(3).strip() or None
        tracklets[tid] = {"class": role_to_class(role), "role": role, "jersey": jersey}
    return tracklets


def parse_gt(path: Path, tracklets: dict[int, dict]) -> tuple[list[Obj], list[Bbox]]:
    """Parse gt.txt into objects and bboxes; class comes from the tracklet map."""
    objects: dict[int, Obj] = {}
    bboxes: list[Bbox] = []
    for line in path.read_text().splitlines():
        p = line.strip().split(",")
        if not p or not p[0]:
            continue
        frame = int(p[0])
        tid = int(p[1])
        left, top, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
        conf = float(p[6]) if len(p) > 6 and p[6] != "" else 1.0
        if conf == 0 or w <= 0 or h <= 0:
            continue
        if tid not in objects:
            cls = tracklets.get(tid, {}).get("class", "unknown")
            objects[tid] = Obj(oid=tid, cls=cls)
        bboxes.append(
            Bbox(fid=frame - 1, oid=tid, x1=left, y1=top, x2=left + w, y2=top + h, aid=None)
        )
    return list(objects.values()), bboxes


def read_ini(path: Path) -> dict:
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return {k: v for section in cfg.sections() for k, v in cfg.items(section)}


def iter_sequences(raw_root: Path, splits):
    for split in splits:
        split_dir = raw_root / split
        if not split_dir.is_dir():
            print(f"!! split '{split}' not found under {raw_root}; skipping")
            continue
        for seq_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            yield split, seq_dir


def convert(raw_root: Path, out_root: Path, splits) -> None:
    out_dataset = out_root / DATASET
    n = 0
    for split, seq_dir in iter_sequences(raw_root, splits):
        seq = seq_dir.name
        gt = seq_dir / "gt" / "gt.txt"
        img_dir = seq_dir / "img1"
        if not gt.is_file():
            print(f"!! {split}/{seq}: no gt.txt; skipping")
            continue
        if not img_dir.is_dir():
            print(f"!! {split}/{seq}: no img1/; skipping")
            continue

        seqinfo = read_ini(seq_dir / "seqinfo.ini")
        fps = float(seqinfo.get("framerate", 25))
        tracklets = parse_gameinfo(seq_dir / "gameinfo.ini")
        objects, bboxes = parse_gt(gt, tracklets)
        obj_classes = sorted({o.cls for o in objects})

        subset = f"{split}_{seq}"
        out_dir = out_dataset / subset
        gameinfo = read_ini(seq_dir / "gameinfo.ini")
        metadata = {
            "dataset": DATASET,
            "source": "SoccerNet-Tracking (SNMOT)",
            "split": split,
            "sequence": seq,
            "obj_classes": obj_classes,
            "activity_classes": [],
            "action_class": gameinfo.get("actionclass"),
            "tracklets": {
                str(o.oid): {
                    "class": o.cls,
                    "role": tracklets.get(o.oid, {}).get("role"),
                    "jersey": tracklets.get(o.oid, {}).get("jersey"),
                }
                for o in objects
            },
            "notes": (
                "Frames encoded to H.264 at the seqinfo.ini frameRate. fid is "
                "0-based (gt frame - 1). Object class is the gameinfo.ini role "
                "(team encoded, e.g. player_left); jersey numbers are kept in "
                "'tracklets'. No activities (aid=NULL)."
            ),
        }
        meta = build_subset(
            out_dir,
            frames_dir=img_dir,
            fps=fps,
            frames_pattern="*.jpg",
            objects=objects,
            bboxes=bboxes,
            activities=(),
            metadata=metadata,
        )
        rc = meta["row_counts"]
        print(
            f"[ok] {subset}: {rc['objects']} objects, {rc['bbox']} bboxes, "
            f"classes={obj_classes}, fps={meta['fps']:.0f}, frames={meta['num_frames']} -> {out_dir}"
        )
        n += 1
    print(f"\nDone: wrote {n} subset(s) under {out_dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=DATASETS_ROOT / "raw" / "soccernet",
        help="Raw soccernet root (default: datasets/raw for this dataset).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATASETS_ROOT / "processed",
        help="Output root; a 'soccernet/' subdir is created under it.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLITS),
        help="Splits to convert (default: train test). 'challenge' has no gt.",
    )
    args = parser.parse_args()
    convert(args.raw.resolve(), args.out.resolve(), args.splits)


if __name__ == "__main__":
    main()
