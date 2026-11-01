#!/usr/bin/env python3
"""
detect_track.py — Run a detector+tracker over one video and write a FrameQL
DuckDB (the `objects` + `bbox` tables the rest of the pipeline expects).

This is the worker invoked by the frameql-tui "detect" action, one process per
video. It is intentionally machine-friendly: progress is reported on stdout as
`PROGRESS <done> <total>` lines and a final `DONE <n_objs> <n_bbox>` line, so
the TUI can drive a progress bar. Everything else (model load chatter, etc.)
goes to stderr.

Output schema (matches frameql-tui/src/convert.rs expectations):
  objects(oid INTEGER, class VARCHAR)         -- one row per track, majority class
  bbox(oid INTEGER, fid INTEGER, x1,y1,x2,y2 INTEGER)  -- fid is 0-based

Requires: ultralytics, opencv-python, duckdb (pip install in the inference env).
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2


def log(*args) -> None:
    """Human-facing chatter goes to stderr; stdout is reserved for PROGRESS/DONE."""
    print(*args, file=sys.stderr, flush=True)


def load_model(model: str):
    """Load a YOLO or RT-DETR model by name/path (Ultralytics auto-downloads)."""
    if "rtdetr" in Path(model).name.lower():
        from ultralytics import RTDETR

        return RTDETR(model)
    from ultralytics import YOLO

    return YOLO(model)


def run(args: argparse.Namespace) -> None:
    video = Path(args.video)
    if not video.exists():
        sys.exit(f"video not found: {video}")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"cannot open video: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()

    classes = args.classes if args.classes else None  # None => all classes
    names = None  # filled from the model on first result

    model = load_model(args.model)
    log(f"model={args.model} tracker={args.tracker} conf={args.conf} "
        f"imgsz={args.imgsz} half={args.half} classes={classes or 'all'}")

    results = model.track(
        source=str(video),
        tracker=args.tracker,
        classes=classes,
        conf=args.conf,
        imgsz=args.imgsz,
        half=args.half,
        persist=True,
        stream=True,
        verbose=False,
    )

    # Stream detections to a temp CSV, then bulk-load into DuckDB. Building the
    # tables from a CSV keeps memory flat and the load fast for long videos.
    tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
    tmp_path = Path(tmp.name)
    n_bbox = 0
    try:
        writer = csv.writer(tmp)
        writer.writerow(["fid", "oid", "class", "x1", "y1", "x2", "y2"])

        # fid is 0-based to match the ground-truth datasets' convention.
        for fid, result in enumerate(results):
            if names is None:
                names = result.names
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                for xyxy, tid, cls_id in zip(
                    boxes.xyxy.cpu().tolist(),
                    boxes.id.cpu().tolist(),
                    boxes.cls.cpu().tolist(),
                ):
                    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
                    label = names.get(int(cls_id), str(int(cls_id))) if names else str(int(cls_id))
                    writer.writerow([fid, int(tid), label, x1, y1, x2, y2])
                    n_bbox += 1

            done = fid + 1
            if done % args.progress_every == 0 or done == total:
                print(f"PROGRESS {done} {total}", flush=True)
        tmp.close()

        n_objs = write_duckdb(tmp_path, Path(args.out))
    finally:
        tmp_path.unlink(missing_ok=True)

    print(f"DONE {n_objs} {n_bbox}", flush=True)


def write_duckdb(csv_path: Path, out: Path) -> int:
    """Load the detection CSV into `objects` + `bbox` tables, returning n_objs."""
    import duckdb as ddb

    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    con = ddb.connect(str(out))
    try:
        con.execute(
            f"""
            CREATE TABLE raw AS SELECT * FROM read_csv(
                '{csv_path}', header = true,
                columns = {{
                    'fid':'INTEGER','oid':'INTEGER','class':'VARCHAR',
                    'x1':'INTEGER','y1':'INTEGER','x2':'INTEGER','y2':'INTEGER'
                }})
            """
        )
        # One row per track: the most-common class across all of its frames.
        con.execute(
            """
            CREATE TABLE objects AS
            SELECT oid, arg_max(class, cnt) AS class FROM (
                SELECT oid, class, COUNT(*) AS cnt FROM raw GROUP BY oid, class
            ) GROUP BY oid ORDER BY oid
            """
        )
        con.execute(
            "CREATE TABLE bbox AS SELECT oid, fid, x1, y1, x2, y2 FROM raw ORDER BY oid, fid"
        )
        con.execute("DROP TABLE raw")
        return con.execute("SELECT COUNT(*) FROM objects").fetchone()[0]
    finally:
        con.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, help="Input video file")
    p.add_argument("--out", required=True, help="Output duckdb.db path")
    p.add_argument("--model", required=True, help="Model name or weights path (e.g. yolo11l.pt)")
    p.add_argument("--tracker", default="bytetrack.yaml", help="Tracker yaml (bytetrack/botsort)")
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--half", action="store_true", help="FP16 inference")
    p.add_argument("--classes", type=int, nargs="*", default=None,
                   help="Class ids to keep (default: all)")
    p.add_argument("--progress-every", type=int, default=30,
                   help="Emit a PROGRESS line every N frames")
    run(p.parse_args())


if __name__ == "__main__":
    main()
