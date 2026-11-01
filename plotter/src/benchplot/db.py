"""Reading a `trql-cli bench.db`.

The database is SQLite (written by `trql-cli/src/benchdb.rs`), so this needs no
driver beyond the standard library. `Dataset` and `Query` are upserted by natural
key, which means a query's whole history — every error budget, every repeat — is
a set of `Run` rows sharing one `query_id`.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Literal

import pandas as pd

Aggregate = Literal["latest", "median", "min", "best"]

_RUNS_SQL = """
SELECT r.id            AS run_id,
       r.timestamp     AS timestamp,
       q.name          AS query,
       d.name          AS dataset,
       d.num_obj       AS num_obj,
       r.trql_error_px AS err,
       r.trql_gap      AS gap,
       -- Tiling factor (`--db-repeat`). `{scale}` is `r.scale` where the column
       -- exists and the literal 1 where it does not — see `_column_or`.
       {scale} AS scale,
       r.trql_time     AS trql_ms,
       r.sql_time      AS sql_ms,
       r.trql_oid_precision AS precision,
       r.trql_oid_recall    AS recall,
       r.trql_iou           AS iou,
       -- Whole-process RSS, so it includes the engine's resident `.trql`.
       r.trql_peak_mem_mb   AS trql_mem_mb,
       -- DuckDB's own `duckdb_memory()` accounting; a different quantity from
       -- the line above, not a like-for-like subtraction.
       r.sql_peak_mem_mb    AS sql_mem_mb,
       -- What the relational encoding stores (`d.num_box`: one bbox row per object
       -- per frame) against what the engine stores instead (`r.trql_segments`:
       -- polynomial pieces). Boxes are invariant so they sit on `Dataset`; pieces
       -- are on `Run` because the error budget moves them.
       -- What produced the rows the query ran over: a detector config id, or GT.
       {detector} AS detector,
       -- The query's answer against its answer on ground truth, and the same for
       -- the reference SQL. Distinct from `precision`/`recall` above, which
       -- compare the engine with SQL *on the same data*.
       {gt_oid_precision} AS gt_precision,
       {gt_oid_recall} AS gt_recall,
       {gt_iou} AS gt_iou,
       {sql_gt_precision} AS sql_gt_precision,
       {sql_gt_recall} AS sql_gt_recall,
       {sql_gt_iou} AS sql_gt_iou,
       {segments} AS segments,
       {poly_segments} AS poly_segments,
       {direct_segments} AS direct_segments,
       {num_box} AS num_box
FROM Run r
JOIN Query   q ON q.id = r.query_id
JOIN Dataset d ON d.id = q.dataset_id
"""


def _column_or(conn: sqlite3.Connection, table: str, column: str, missing: str) -> str:
    """`table.column` when the database has it, else the literal `missing`.

    `trql-cli` adds columns to `bench.db` as the benchmark grows what it records,
    and old rows read them as NULL. A database written before a column existed
    lacks it entirely, though, and selecting it is an error rather than a NULL —
    so a plotter that named it unconditionally would refuse to open any database
    older than itself.
    """
    names = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    return f"{table[0].lower()}.{column}" if column in names else missing


def load_runs(db_path: Path) -> pd.DataFrame:
    """Every run in `db_path`, one row each, with `speedup` derived.

    Rows whose SQL side is missing are dropped: `--sql` is optional for
    `trql-cli bench`, and a run without it has no baseline to be a speedup over.
    """
    if not db_path.is_file():
        raise FileNotFoundError(f"no bench database at {db_path}")

    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        sql = _RUNS_SQL.format(
            scale=f"COALESCE({_column_or(conn, 'Run', 'scale', 'NULL')}, 1)",
            segments=_column_or(conn, "Run", "trql_segments", "NULL"),
            poly_segments=_column_or(conn, "Run", "trql_poly_segments", "NULL"),
            direct_segments=_column_or(conn, "Run", "trql_direct_segments", "NULL"),
            num_box=_column_or(conn, "Dataset", "num_box", "NULL"),
            # NULL where the run predates the column, which is what a run over
            # ground truth records — same treatment as `scale` above, and
            # without it such a run does not collapse with a later repeat.
            detector=f"COALESCE({_column_or(conn, 'Run', 'detector', chr(39) + 'GT' + chr(39))}, 'GT')",
            gt_oid_precision=_column_or(conn, "Run", "gt_oid_precision", "NULL"),
            gt_oid_recall=_column_or(conn, "Run", "gt_oid_recall", "NULL"),
            gt_iou=_column_or(conn, "Run", "gt_iou", "NULL"),
            sql_gt_precision=_column_or(conn, "Run", "sql_gt_precision", "NULL"),
            sql_gt_recall=_column_or(conn, "Run", "sql_gt_recall", "NULL"),
            sql_gt_iou=_column_or(conn, "Run", "sql_gt_iou", "NULL"),
        )
        df = pd.read_sql_query(sql, conn)

    df = df[df["sql_ms"].notna() & df["trql_ms"].notna() & (df["trql_ms"] > 0)]
    if df.empty:
        raise ValueError(
            f"{db_path} holds no run with both timings — was `bench` given `--sql`?"
        )

    df = df.copy()
    df["speedup"] = df["sql_ms"] / df["trql_ms"]
    # An answer counts as exact only when it matches the hand-written SQL on both
    # object tuples and their intervals; `iou < 1` means edges moved.
    df["exact"] = (
        (df["precision"] == 1.0) & (df["recall"] == 1.0) & (df["iou"] == 1.0)
    )
    return df


def collapse_repeats(df: pd.DataFrame, how: Aggregate = "latest") -> pd.DataFrame:
    """One row per `(query, err)`, since a point may have been measured more than once.

    `latest` is the default because repeated measurements usually straddle a code
    change, and the newest is then the only one describing the current engine.
    `median` and `min` are for the other case — several runs of one build, where
    the spread is measurement noise.

    `best` keeps the most accurate repeat: highest ground-truth recall, then
    precision, then the newest. Unlike the others it is a *biased* choice — where
    repeats disagree it reports the best of them rather than the typical one — so
    it answers "what is achievable here", not "what does a run of this look
    like". Prefer `latest` or `median` for a headline number, and say which was
    used either way.
    """
    # What makes two runs *the same point* rather than alternatives: same query,
    # same sub-dataset, same fitting budget, same tiling, same gap smoothing,
    # same input. A run at `--db-repeat 100` read a different database from one
    # at 1; a run at `max_gap=25` asked a different question of the same one; and
    # a run over detector output is not a repeat of one over ground truth.
    #
    # `dataset` matters wherever one query name spans many sub-datasets — DETRAC
    # runs it over 100 sequences — and without it they collapse to whichever
    # happened to be measured last.
    keys = ["query", "dataset", "err", "scale", "gap", "detector"]
    if how == "latest":
        # `drop_duplicates` keeps whole rows; a `groupby().last()` would take the
        # last non-null value column by column, mixing runs where one has a gap.
        ordered = df.sort_values(["timestamp", "run_id"])
        return ordered.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    if how == "best":
        # Sorted worst-to-best so `keep="last"` takes the best whole row; the
        # trailing run_id makes the choice deterministic when repeats tie, which
        # they do whenever the work was deterministic and nothing changed between
        # them.
        by = [c for c in ("gt_recall", "gt_precision") if c in df.columns]
        if not by:
            raise ValueError(
                "aggregate 'best' needs ground-truth scores; this database has "
                "none, so there is no accuracy to rank repeats by"
            )
        ordered = df.sort_values([*by, "timestamp", "run_id"], na_position="first")
        return ordered.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    if how in ("median", "min"):
        agg = {
            "trql_ms": how,
            "sql_ms": how,
            # Memory follows the same statistic as time: a run that was slower
            # because it did more work generally also held more.
            "trql_mem_mb": how,
            "sql_mem_mb": how,
            # Properties of the artifact, not of the run: every repeat of a
            # (query, err) point read the same one, so any of them will do.
            "gt_precision": how,
            "gt_recall": how,
            "gt_iou": how,
            "sql_gt_precision": how,
            "sql_gt_recall": how,
            "sql_gt_iou": how,
            "num_box": "last",
            "segments": "last",
            "poly_segments": "last",
            "direct_segments": "last",
            "precision": "min",
            "recall": "min",
            "iou": "min",
            "exact": "all",
            "num_obj": "last",
            "gap": "last",
        }
        out = df.groupby(keys, as_index=False).agg(agg)
        out["speedup"] = out["sql_ms"] / out["trql_ms"]
        return out
    raise ValueError(f"unknown aggregate {how!r}")
