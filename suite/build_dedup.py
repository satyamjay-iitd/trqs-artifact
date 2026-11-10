# /// script
# requires-python = ">=3.13"
# dependencies = ["duckdb==1.5.6"]
# ///
"""Build the frame-deduplicated copy of a clip: the same detections, minus the
rows that repeat the one before them.

A frame-expanded store writes one row per object per frame however the object
behaves, and a parked car writes the same four numbers for every frame it is
parked. The obvious relational answer to that is run-length encoding, and this
builds it: a row survives only where it says something the previous frame did
not, and carries `e_fid`, the last frame through which it holds.

The surviving rows are a strict subset of the source rows -- every one is a real
detection, at its own frame -- so this is a *dropped-frames* copy of the dataset
rather than a re-encoding of it, and `e_fid` is recoverable from the drop (it is
the frame before the next surviving row of that object, or the run's own end).
It is carried as a column only so a query need not re-derive it, and because
without it the drop is ambiguous: a run's last row and a gap's last row look
alike, and reading one as the other asserts an object was present while it was
absent.

    uv run suite/build_dedup.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import duckdb

from paths import PROCESSED

_ROOT = PROCESSED

# source sub-dataset -> the dedup copy written beside it. The pair the figure
# draws: a parking lot where almost nothing moves, and a motorway junction where
# almost nothing holds still. DETRAC is read from its integer-pixel copy
# (`build_detrac_px.py`), so all three encodings see the same numbers.
_CLIPS = {
    "virat/VIRAT_S_010109_04_000487_000538": "virat_dedup/VIRAT_S_010109_04_000487_000538",
    "detrac_px/MVI_40191": "detrac_px_dedup/MVI_40191",
}

# `newrun` marks a row that cannot be folded into its predecessor: the object was
# absent in the previous frame (a gap, or the start of its track) or the box
# moved. Running that flag through a `sum() OVER` numbers the runs, and the
# grouping collapses each to one row.
_BUILD = """
CREATE TABLE objects AS SELECT * FROM src.objects;

CREATE TABLE bbox AS
WITH s AS (
    SELECT oid, fid, x1, y1, x2, y2,
           lag(fid) OVER w AS pf,
           lag(x1)  OVER w AS px1, lag(y1) OVER w AS py1,
           lag(x2)  OVER w AS px2, lag(y2) OVER w AS py2
    FROM src.bbox
    WINDOW w AS (PARTITION BY oid ORDER BY fid)
),
marked AS (
    SELECT *, CASE WHEN pf IS NULL OR pf <> fid - 1
                     OR x1 <> px1 OR y1 <> py1 OR x2 <> px2 OR y2 <> py2
                   THEN 1 ELSE 0 END AS newrun
    FROM s
),
grouped AS (
    SELECT *, sum(newrun) OVER (PARTITION BY oid ORDER BY fid ROWS UNBOUNDED PRECEDING) AS run
    FROM marked
)
SELECT oid,
       min(fid) AS fid,    -- the surviving detection: a real row of the source
       max(fid) AS e_fid,  -- through which frame it holds
       any_value(x1) AS x1, any_value(y1) AS y1,
       any_value(x2) AS x2, any_value(y2) AS y2
FROM grouped
GROUP BY oid, run
ORDER BY oid, fid;
"""


def build(src_rel: str, dst_rel: str) -> None:
    src = _ROOT / src_rel / "duckdb.db"
    dst_dir = _ROOT / dst_rel
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "duckdb.db"
    for stale in (dst, dst.with_suffix(".db.wal")):
        stale.unlink(missing_ok=True)

    conn = duckdb.connect(str(dst))
    conn.execute(f"ATTACH '{src}' AS src (READ_ONLY)")
    conn.execute(_BUILD)
    conn.execute("DETACH src")

    kept = conn.execute("SELECT count(*) FROM bbox").fetchone()[0]
    covered = conn.execute("SELECT sum(e_fid - fid + 1) FROM bbox").fetchone()[0]
    conn.close()

    rows = duckdb.connect(str(src), read_only=True).execute(
        "SELECT count(*) FROM bbox"
    ).fetchone()[0]
    # Every source row is covered exactly once, or the drop lost a detection.
    assert covered == rows, f"{dst_rel}: covers {covered} frames, source has {rows}"
    print(f"{dst_rel}\n    {rows:>8,} rows -> {kept:>8,} ({rows / kept:.1f}x fewer, "
          f"{100 * (1 - kept / rows):.1f}% dropped)")

    meta = _ROOT / src_rel / "metadata.json"
    if meta.is_file():
        shutil.copy(meta, dst_dir / "metadata.json")


if __name__ == "__main__":
    for s, d in _CLIPS.items():
        build(s, d)
