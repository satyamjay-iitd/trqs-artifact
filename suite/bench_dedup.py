# /// script
# requires-python = ">=3.13"
# dependencies = ["duckdb>=1.5.6"]
# ///
"""The frame-deduplicated DuckDB baseline, against the frame-expanded one.

Both sides run the *same query* at `threads=1`: `sql-ref/VIRAT-approach.sql` over
one row per object per frame, and `sql-dedup/VIRAT-approach.sql` over the copy
built by `build_dedup.py`, where a row survives only if it says something the
previous frame did not. The answers are required to be identical -- the drop is
lossless, so a difference is a bug in one of the two queries, not a trade.

Tiling matches `trql-cli`'s `--db-repeat` (`sqlrun::tiled_duckdb`): replica i
shifts frames by i * (max_fid + 1) and oids by i * (max_oid + 1), so the
baseline sees the same grown database the engine was measured on.

    uv run papers/vldb27/suite/bench_dedup.py --repeat 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb

_ROOT = Path("/home/satyam/dev/frameql/datasets/processed")
_SUITE = Path("/home/satyam/dev/frameql/papers/vldb27/suite")

# query id -> (frame-expanded sub-dataset, deduplicated sub-dataset). Each id
# has its own `sql-ref/<id>.sql` and `sql-dedup/<id>.sql`; both clips here run
# the same query, which is what makes their costs comparable.
_CLIPS = {
    "VIRAT-meet": (
        "virat/VIRAT_S_010109_04_000487_000538",
        "virat_dedup/VIRAT_S_010109_04_000487_000538",
    ),
    "DETRAC-meet": (
        "detrac_px/MVI_40191",
        "detrac_px_dedup/MVI_40191",
    ),
}


def _tiled(src: Path, times: int, out: Path) -> None:
    """`times` copies of `src`, laid out exactly as `sqlrun::tiled_duckdb` does."""
    for stale in (out, out.with_suffix(".db.wal")):
        stale.unlink(missing_ok=True)
    conn = duckdb.connect(str(out))
    conn.execute(f"ATTACH '{src}' AS src (READ_ONLY)")
    stride = conn.execute("SELECT COALESCE(MAX(fid), -1) + 1 FROM src.bbox").fetchone()[0]
    oid_stride = conn.execute("SELECT COALESCE(MAX(oid), -1) + 1 FROM src.objects").fetchone()[0]
    has_e = "e_fid" in {c[0] for c in conn.execute("DESCRIBE src.bbox").fetchall()}
    # The dedup copy carries `e_fid`, which is a frame index and shifts with the
    # rest of the replica; without this the last replica's intervals would still
    # end in the first replica's frame range.
    shift_e = f", b.e_fid + t.i * {stride} AS e_fid" if has_e else ""
    exclude = "(oid, fid, e_fid)" if has_e else "(oid, fid)"
    conn.execute(
        f"""CREATE TABLE objects AS
              SELECT o.* EXCLUDE (oid), o.oid + t.i * {oid_stride} AS oid
              FROM src.objects o CROSS JOIN generate_series(0, {times - 1}) t(i)"""
    )
    conn.execute(
        f"""CREATE TABLE bbox AS
              SELECT b.* EXCLUDE {exclude},
                     b.oid + t.i * {oid_stride} AS oid,
                     b.fid + t.i * {stride}     AS fid{shift_e}
              FROM src.bbox b CROSS JOIN generate_series(0, {times - 1}) t(i)"""
    )
    conn.execute("DETACH src")
    conn.close()


def _run(db: Path, sql: str, repeats: int) -> tuple[float, list]:
    """Best of `repeats` wall times, and the answer, at one thread."""
    conn = duckdb.connect(str(db), read_only=True)
    conn.execute("SET threads=1")
    best, rows = None, None
    for _ in range(repeats):
        t0 = time.perf_counter()
        rows = conn.execute(sql).fetchall()
        ms = (time.perf_counter() - t0) * 1000.0
        best = ms if best is None else min(best, ms)
    conn.close()
    return best, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=10, help="tiling factor (--db-repeat)")
    ap.add_argument("--runs", type=int, default=3, help="timed repeats per query")
    ap.add_argument("--out", type=Path, default=_SUITE / "results-dedup.json")
    args = ap.parse_args()

    tmp = Path("/home/satyam/.claude/jobs/c5f85dd3/tmp")

    out = {}
    for name, (full_rel, ded_rel) in _CLIPS.items():
        ref_sql = (_SUITE / "sql-ref" / f"{name}.sql").read_text()
        ded_sql = (_SUITE / "sql-dedup" / f"{name}.sql").read_text()
        full_src, ded_src = _ROOT / full_rel / "duckdb.db", _ROOT / ded_rel / "duckdb.db"
        full_db, ded_db = tmp / f"{name}-full.db", tmp / f"{name}-dedup.db"
        _tiled(full_src, args.repeat, full_db)
        _tiled(ded_src, args.repeat, ded_db)

        rows_full = duckdb.connect(str(full_db), read_only=True).execute(
            "SELECT count(*) FROM bbox").fetchone()[0]
        rows_ded = duckdb.connect(str(ded_db), read_only=True).execute(
            "SELECT count(*) FROM bbox").fetchone()[0]

        ms_full, ans_full = _run(full_db, ref_sql, args.runs)
        ms_ded, ans_ded = _run(ded_db, ded_sql, args.runs)
        same = sorted(ans_full) == sorted(ans_ded)
        # Scored against the frame-expanded answer, which is the reference the
        # engine's own precision/recall in bench.db is measured against -- so the
        # three encodings carry comparable numbers.
        ref_set, ded_set = set(ans_full), set(ans_ded)
        hit = len(ref_set & ded_set)
        recall = hit / len(ref_set) if ref_set else 1.0
        precision = hit / len(ded_set) if ded_set else 1.0

        out[name] = {
            "scale": args.repeat,
            "rows_frame_expanded": rows_full,
            "rows_deduped": rows_ded,
            "ms_frame_expanded": ms_full,
            "ms_deduped": ms_ded,
            "answer_rows": len(ans_full),
            "answers_identical": same,
            "recall_deduped": recall,
            "precision_deduped": precision,
        }
        print(
            f"{name:14s} rows {rows_full:>10,} -> {rows_ded:>9,} ({rows_full / rows_ded:5.1f}x)   "
            f"time {ms_full:8.1f} ms -> {ms_ded:7.1f} ms ({ms_full / ms_ded:5.1f}x)   "
            f"{len(ans_full):3d} answer rows, recall {recall:.4f}, identical: {same}"
        )
        for stale in (full_db, ded_db):
            stale.unlink(missing_ok=True)

    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
