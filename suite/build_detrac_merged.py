# /// script
# requires-python = ">=3.13"
# dependencies = ["duckdb>=1.5.6"]
# ///
"""Merge every UA-DETRAC sequence into one dataset, quantised to integer pixels.

A single DETRAC sequence is ~38 k boxes, three orders of magnitude smaller than
the other datasets in the suite, so a query over one of them measures the
engine's fixed per-query cost as much as its algebra. Concatenating all 100
gives 1.27 M boxes, which is the figure the dataset table reports.

Sequences are laid end to end: replica i's frames shift past every earlier
sequence's last frame and its oids past their last oid, so two objects from
different sequences never share a frame and a pair query cannot match across
them -- the same construction `clevrer_merged` uses for its 10,000 videos.

Coordinates are truncated on the way in, for the reason `build_detrac_px.py`
gives: UA-DETRAC annotates in floats, `trql-storage` stores integers, and a
reference query reading the floats asks a question of data the engine never
sees.

    uv run suite/build_detrac_merged.py
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from paths import PROCESSED

_ROOT = PROCESSED
_SRC = _ROOT / "detrac"
_DST = _ROOT / "detrac_px_merged" / "all"


def main() -> None:
    seqs = sorted(d for d in _SRC.iterdir() if (d / "duckdb.db").is_file())
    _DST.mkdir(parents=True, exist_ok=True)
    out = _DST / "duckdb.db"
    for stale in (out, out.with_suffix(".db.wal")):
        stale.unlink(missing_ok=True)

    conn = duckdb.connect(str(out))
    conn.execute(
        """CREATE TABLE objects(oid BIGINT, class VARCHAR);
           CREATE TABLE bbox(fid BIGINT, oid BIGINT, x1 BIGINT, y1 BIGINT,
                             x2 BIGINT, y2 BIGINT, aid BIGINT)"""
    )
    fid_base = oid_base = 0
    for seq in seqs:
        conn.execute(f"ATTACH '{seq / 'duckdb.db'}' AS src (READ_ONLY)")
        conn.execute(
            f"""INSERT INTO objects
                SELECT oid + {oid_base}, class FROM src.objects"""
        )
        conn.execute(
            f"""INSERT INTO bbox
                SELECT fid + {fid_base}, oid + {oid_base},
                       CAST(greatest(trunc(x1), 0) AS BIGINT),
                       CAST(greatest(trunc(y1), 0) AS BIGINT),
                       CAST(greatest(trunc(x2), 0) AS BIGINT),
                       CAST(greatest(trunc(y2), 0) AS BIGINT),
                       aid
                FROM src.bbox"""
        )
        nf, no = conn.execute(
            "SELECT COALESCE(MAX(fid), -1) + 1, COALESCE(MAX(oid), -1) + 1 FROM src.bbox"
        ).fetchone()
        conn.execute("DETACH src")
        fid_base += nf
        oid_base += max(no, 1)

    rows, objs, frames = conn.execute(
        "SELECT count(*), count(DISTINCT oid), count(DISTINCT fid) FROM bbox"
    ).fetchone()
    cls = dict(conn.execute("SELECT class, count(*) FROM objects GROUP BY 1").fetchall())
    conn.close()
    print(f"{_DST.relative_to(_ROOT)}: {len(seqs)} sequences, {rows:,} boxes, "
          f"{objs:,} objects, {frames:,} frames\n    classes {cls}")

    (_DST / "metadata.json").write_text(json.dumps({
        "fps": 25.0, "width": 960, "height": 540,
        "num_frames": frames,
        "row_counts": {"objects": objs, "activities": 0, "bbox": rows},
        "dataset": "detrac_px_merged",
        "source": "UA-DETRAC, all train sequences, concatenated",
        "notes": "Coordinates truncated to integers to match trql-storage's ingest; "
                 "sequences laid end to end so none co-appears with another.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
