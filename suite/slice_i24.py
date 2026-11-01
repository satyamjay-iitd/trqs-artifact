#!/usr/bin/env python3
"""Cut i24_motion's 4-hour sub-datasets into fixed-length slices.

`post1` is 359 777 frames of continuous freeway -- 830 366 trajectories over 280M
boxes in one unbroken 4-hour window. A single-object query runs fine on that
(I24-Q1: 560 ms). A two-object query does not, in either evaluator: the query is
over *pairs of co-appearing objects*, and on a continuous 4-hour recording with
hundreds of vehicles on the road at any instant, essentially every vehicle
co-appears with every vehicle that shares any part of its journey.

Slicing fixes the shape of the problem rather than the query. It also makes i24
report like soccernet and nuscenes -- a distribution over sub-datasets rather
than one number -- which is what the evaluation section wants anyway.

    ./slice_i24.py --minutes 10                 # every slice of post1
    ./slice_i24.py --minutes 10 --only 0 1 2    # just the first three

Each slice becomes `<dataset>/<sub>_m<NN>/` with its own `duckdb.db` and
`metadata.json`. `fid` is rebased to 0 within the slice so query time windows
mean the same thing everywhere; `frame0_epoch` is advanced to match, so a slice
still says truthfully when it was recorded. Objects are carried over only if
they have a box inside the slice, and their `oid`s are left alone -- the same
vehicle keeps its id across slices, which is what makes a per-slice result
aggregable.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frames import PROCESSED  # noqa: E402

HZ = 25.0


def slice_one(src_dir: Path, out_dir: Path, fid_lo: int, fid_hi: int) -> dict:
    """Write the `[fid_lo, fid_hi)` window of `src_dir` as its own sub-dataset."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    con = duckdb.connect(str(out_dir / "duckdb.db"))
    con.execute("SET memory_limit='24GB'")
    con.execute(f"ATTACH '{src_dir / 'duckdb.db'}' AS src (READ_ONLY)")
    con.execute(f"""
        CREATE TABLE bbox AS
        SELECT fid - {fid_lo} AS fid, oid, x1, y1, x2, y2, aid
        FROM src.bbox WHERE fid >= {fid_lo} AND fid < {fid_hi}
    """)
    con.execute("""
        CREATE TABLE objects AS
        SELECT o.* FROM src.objects o
        WHERE o.oid IN (SELECT DISTINCT oid FROM bbox)
    """)
    con.execute("CREATE TABLE activities AS SELECT * FROM src.activities LIMIT 0")
    counts = con.execute(
        "SELECT (SELECT count(*) FROM objects), (SELECT count(*) FROM bbox), "
        "(SELECT max(fid) FROM bbox)").fetchone()
    con.close()

    meta = json.loads((src_dir / "metadata.json").read_text())
    meta["subset"] = out_dir.name
    meta["sliced_from"] = src_dir.name
    meta["source_fid_range"] = [fid_lo, fid_hi]
    if meta.get("frame0_epoch") is not None:
        meta["frame0_epoch"] = meta["frame0_epoch"] + fid_lo / HZ
    meta["num_frames"] = (counts[2] + 1) if counts[2] is not None else 0
    meta["row_counts"] = {"objects": counts[0], "activities": 0, "bbox": counts[1]}
    meta["notes"] = (
        f"Slice [{fid_lo}, {fid_hi}) of {src_dir.name}, fid rebased to 0. "
        + meta.get("notes", "")
    )
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return {"objects": counts[0], "bbox": counts[1], "frames": meta["num_frames"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="i24_motion")
    ap.add_argument("--sub", default="post1")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--only", type=int, nargs="+", help="slice indices to write")
    args = ap.parse_args()

    src = PROCESSED / args.dataset / args.sub
    con = duckdb.connect(str(src / "duckdb.db"), read_only=True)
    fid_max = con.execute("SELECT max(fid) FROM bbox").fetchone()[0]
    con.close()

    width = int(round(args.minutes * 60 * HZ))
    n = (fid_max + 1 + width - 1) // width
    print(f"{args.sub}: {fid_max + 1} frames -> {n} slices of {width} frames "
          f"({args.minutes:g} min)")

    for i in range(n):
        if args.only is not None and i not in args.only:
            continue
        out = src.parent / f"{args.sub}_m{i:02d}"
        t0 = time.time()
        r = slice_one(src, out, i * width, (i + 1) * width)
        print(f"  {out.name}: {r['objects']:7d} objects  {r['bbox']:10d} boxes  "
              f"{r['frames']:6d} frames  ({time.time() - t0:.1f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
