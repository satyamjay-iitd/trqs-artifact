# /// script
# requires-python = ">=3.13"
# dependencies = ["duckdb>=1.5.6"]
# ///
"""Quantise a DETRAC sequence to integer pixels, which is what the engine stores.

UA-DETRAC annotates boxes as floats; `trql-storage`'s ingest casts each
coordinate with `as u32` (`convert.rs:197`), a truncation, and evaluates
`cx = (x1 + x2) / 2.0` over the result (`eval.rs:163`). A reference query reading
the original doubles is therefore asking a question of data the engine never
sees: a centroid half a pixel out flips a threshold test at the boundary, and the
two answers differ for reasons that have nothing to do with the algebra. On
MVI_40191 that is one pair in fifty at eps=0, and no fitting budget can fix it
because the discrepancy is in the storage, not the fit.

Quantising the source once puts all three encodings on the same numbers. Every
other dataset in the suite already stores integer coordinates, so this only
restates for DETRAC what is true elsewhere by construction.

    uv run papers/vldb27/suite/build_detrac_px.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import duckdb

_ROOT = Path("/home/satyam/dev/frameql/datasets/processed")
_CLIPS = {"detrac/MVI_40191": "detrac_px/MVI_40191"}

# `as u32` truncates toward zero and saturates at 0 for a negative, so
# `greatest(trunc(v), 0)` is the same function over doubles.
_BUILD = """
CREATE TABLE objects AS SELECT * FROM src.objects;
CREATE TABLE bbox AS
SELECT fid, oid,
       CAST(greatest(trunc(x1), 0) AS BIGINT) AS x1,
       CAST(greatest(trunc(y1), 0) AS BIGINT) AS y1,
       CAST(greatest(trunc(x2), 0) AS BIGINT) AS x2,
       CAST(greatest(trunc(y2), 0) AS BIGINT) AS y2,
       aid
FROM src.bbox ORDER BY oid, fid;
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
    neg = conn.execute(
        "SELECT count(*) FROM src.bbox WHERE x1 < 0 OR y1 < 0 OR x2 < 0 OR y2 < 0"
    ).fetchone()[0]
    conn.execute(_BUILD)
    conn.execute("DETACH src")
    rows = conn.execute("SELECT count(*) FROM bbox").fetchone()[0]
    moved = conn.execute("SELECT count(*) FROM bbox WHERE x2 <= x1 OR y2 <= y1").fetchone()[0]
    conn.close()
    print(f"{dst_rel}: {rows:,} rows, {neg} negative coordinates clamped, "
          f"{moved} degenerate boxes after truncation")

    meta = _ROOT / src_rel / "metadata.json"
    if meta.is_file():
        shutil.copy(meta, dst_dir / "metadata.json")


if __name__ == "__main__":
    for s, d in _CLIPS.items():
        build(s, d)
