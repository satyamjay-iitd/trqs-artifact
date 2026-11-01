//! The benchmark results database: `Dataset`, `Query`, `Run`.
//!
//! Separate from [`crate::results`], which mirrors the evaluator's schema for the
//! plot scripts. This one records a *paired* run — the engine and its own
//! compiled SQL over the same sub-dataset — so every row carries both timings
//! and the agreement between them.
//!
//! `Dataset` and `Query` are upserted by natural key (dataset name; query name
//! within a dataset), so repeated runs accumulate as `Run` rows against stable
//! ids and a query's history is a single `WHERE query_id = …`.

use std::path::Path;

use anyhow::Context as _;
use rusqlite::{Connection, params};

/// Trajectory-length distribution and size of one sub-dataset.
pub struct DatasetRow<'a> {
    pub name: &'a str,
    /// Rows in `objects`.
    pub num_obj: u64,
    /// Distinct objects that actually carry boxes.
    pub num_traj: u64,
    /// Rows in `bbox`: one per object per frame. The relational encoding stores
    /// exactly this many, whatever the objects do — which is what makes the ratio
    /// against `Run.trql_segments` a measure of the redundancy it carries.
    pub num_box: u64,
    /// Frames covered per trajectory.
    pub avg_traj_len: f64,
    pub median_traj_len: f64,
    pub p99_traj_len: f64,
    pub max_traj_len: u64,
    pub min_traj_len: u64,
}

/// A query, and the SQL it compiles to, against one dataset.
pub struct QueryRow<'a> {
    pub name: &'a str,
    /// Dataset name, denormalised so a `Query` reads without a join.
    pub dataset: &'a str,
    pub trql_query_text: &'a str,
    pub sql_query_text: &'a str,
    pub dataset_id: i64,
}

/// One paired execution.
pub struct RunRow {
    pub query_id: i64,
    /// Tiling factor this run used (`--db-repeat`), 1 for an untiled database.
    ///
    /// Two runs at different scales are not repeats of one another — they read
    /// different databases — so anything collapsing repeats has to key on this as
    /// well as the query and the error budget.
    pub scale: u32,
    /// DuckDB query time, milliseconds, `threads=1`, excluding connection open.
    pub sql_time: f64,
    /// Engine query time, milliseconds, excluding the `.trql` load.
    pub trql_time: f64,
    pub trql_error_px: f64,
    pub trql_gap: u32,
    /// Of the tuples SQL found, the fraction the engine also found.
    pub trql_oid_recall: f64,
    /// Of the tuples the engine found, the fraction SQL also found.
    pub trql_oid_precision: f64,
    /// Mean temporal IoU over the tuples both found.
    pub trql_iou: f64,
    /// Per-stage self-time in milliseconds, averaged per run. `None` unless the
    /// binary was built with the `instrument` feature.
    pub trql_selection_ms: Option<f64>,
    pub trql_poly_ms: Option<f64>,
    /// Turning solved frames into interval sets, and clipping memoised results
    /// to a tuple's co-appearance — per leaf per tuple.
    pub trql_lift_ms: Option<f64>,
    pub trql_temporal_ms: Option<f64>,
    /// Residual: planning, optimisation, tuple materialisation, timing overhead.
    pub trql_other_ms: Option<f64>,
    /// Peak process RSS during the engine's query, in megabytes. Process-wide,
    /// so it includes the loaded `.trql` — which is the engine's data structure,
    /// not overhead, and the figure the evaluator also reports.
    pub trql_peak_mem_mb: Option<f64>,
    /// Peak of DuckDB's own memory accounting (`duckdb_memory()`) during its
    /// query, in megabytes — its buffer manager alone.
    ///
    /// Not the same kind of measurement as `trql_peak_mem_mb`: that is process
    /// RSS including the engine's resident `.trql`, this is one engine's internal
    /// accounting. Comparable in spirit, not by subtraction.
    pub sql_peak_mem_mb: Option<f64>,
    /// Peak RSS *growth* during DuckDB's query, in megabytes — the cruder
    /// cross-check on `sql_peak_mem_mb`.
    ///
    /// Counts pages DuckDB touched that its own accounting may not tag, but is a
    /// lower bound: memory the allocator holds is reused without RSS moving.
    pub sql_peak_rss_mb: Option<f64>,
    /// Polynomial pieces in the `.trql` this run read. On `Run` rather than
    /// `Dataset` because it is the one quantity here that the error budget moves:
    /// the same trajectories fit into fewer pieces as the tolerance grows.
    pub trql_segments: Option<u64>,
    /// Of those, the pieces whose four channels are all fitted polynomials.
    pub trql_poly_segments: Option<u64>,
    /// Of those, the pieces storing raw samples — no polynomial was cheap enough.
    /// At `max_error_px = 0` these dominate, which is why the representation is
    /// inert there.
    pub trql_direct_segments: Option<u64>,
    /// What produced the rows this query ran over: a detector configuration id,
    /// or `GT` for human annotation.
    ///
    /// Without it two rows of the same query at the same budget are
    /// indistinguishable, and the whole point of the accuracy experiment is that
    /// they differ only in where their input came from.
    pub detector: String,
    /// The query's answer scored against its answer on ground truth, when `--gt`
    /// supplied one.
    ///
    /// Distinct from `trql_oid_*`, which is the engine against its reference SQL
    /// **on the same data** — a correctness check that should read 1.0 whatever
    /// the input. These say how much detection error changed the answer.
    pub gt_oid_precision: Option<f64>,
    pub gt_oid_recall: Option<f64>,
    pub gt_iou: Option<f64>,
    /// The same scores for the reference SQL's answer. SQL has no notion of gap
    /// smoothing, so this is the fixed baseline the engine's `gt_*` are read
    /// against as `max_gap` varies.
    pub sql_gt_precision: Option<f64>,
    pub sql_gt_recall: Option<f64>,
    pub sql_gt_iou: Option<f64>,
}

fn schema(conn: &Connection) -> anyhow::Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS Dataset (
             id               INTEGER PRIMARY KEY,
             name             TEXT UNIQUE NOT NULL,
             num_obj          INTEGER NOT NULL,
             num_traj         INTEGER NOT NULL,
             avg_traj_len     REAL    NOT NULL,
             median_traj_len  REAL    NOT NULL,
             p99_traj_len     REAL    NOT NULL,
             max_traj_len     INTEGER NOT NULL,
             min_traj_len     INTEGER NOT NULL,
             num_box          INTEGER
         );
         CREATE TABLE IF NOT EXISTS Query (
             id               INTEGER PRIMARY KEY,
             name             TEXT NOT NULL,
             dataset          TEXT NOT NULL,
             trql_query_text  TEXT NOT NULL,
             sql_query_text   TEXT NOT NULL,
             dataset_id       INTEGER NOT NULL REFERENCES Dataset(id),
             UNIQUE (name, dataset_id)
         );
         CREATE TABLE IF NOT EXISTS Run (
             id                  INTEGER PRIMARY KEY,
             timestamp           TEXT NOT NULL DEFAULT (datetime('now')),
             query_id            INTEGER NOT NULL REFERENCES Query(id),
             sql_time            REAL NOT NULL,
             trql_time           REAL NOT NULL,
             trql_error_px       REAL NOT NULL,
             trql_gap            INTEGER NOT NULL,
             trql_oid_recall     REAL NOT NULL,
             trql_oid_precision  REAL NOT NULL,
             trql_iou            REAL NOT NULL,
             trql_selection_ms   REAL,
             trql_poly_ms        REAL,
             trql_lift_ms        REAL,
             trql_temporal_ms    REAL,
             trql_other_ms       REAL,
             trql_peak_mem_mb    REAL,
             sql_peak_mem_mb     REAL,
             sql_peak_rss_mb     REAL,
             trql_segments        INTEGER,
             trql_poly_segments   INTEGER,
             trql_direct_segments INTEGER,
             scale                INTEGER,
             detector             TEXT,
             gt_oid_precision     REAL,
             gt_oid_recall        REAL,
             gt_iou               REAL,
             sql_gt_precision     REAL,
             sql_gt_recall        REAL,
             sql_gt_iou           REAL
         );",
    )
    .context("failed to create bench schema")?;
    // `CREATE TABLE IF NOT EXISTS` leaves an existing table untouched, so a
    // database written before these columns existed needs them added. Every one
    // is nullable, so rows measured before it was recorded stay readable and read
    // as "not measured" rather than as zero.
    for (table, col, ty) in [
        ("Run", "trql_selection_ms", "REAL"),
        ("Run", "trql_poly_ms", "REAL"),
        ("Run", "trql_lift_ms", "REAL"),
        ("Run", "trql_temporal_ms", "REAL"),
        ("Run", "trql_other_ms", "REAL"),
        ("Run", "trql_peak_mem_mb", "REAL"),
        ("Run", "sql_peak_mem_mb", "REAL"),
        ("Run", "sql_peak_rss_mb", "REAL"),
        ("Run", "trql_segments", "INTEGER"),
        ("Run", "trql_poly_segments", "INTEGER"),
        ("Run", "trql_direct_segments", "INTEGER"),
        ("Run", "scale", "INTEGER"),
        ("Run", "detector", "TEXT"),
        ("Run", "gt_oid_precision", "REAL"),
        ("Run", "gt_oid_recall", "REAL"),
        ("Run", "gt_iou", "REAL"),
        ("Run", "sql_gt_precision", "REAL"),
        ("Run", "sql_gt_recall", "REAL"),
        ("Run", "sql_gt_iou", "REAL"),
        ("Dataset", "num_box", "INTEGER"),
    ] {
        let present: bool = conn
            .prepare("SELECT 1 FROM pragma_table_info(?1) WHERE name = ?2")?
            .exists([table, col])?;
        if !present {
            conn.execute_batch(&format!("ALTER TABLE {table} ADD COLUMN {col} {ty}"))
                .with_context(|| format!("failed to add `{col}` to {table}"))?;
        }
    }
    Ok(())
}

/// Insert or update the dataset row, returning its id.
///
/// Stats are refreshed on every run rather than kept from the first: a
/// sub-dataset can be reconverted between runs, and a stale row would silently
/// describe data that no longer exists.
pub fn upsert_dataset(conn: &Connection, d: &DatasetRow<'_>) -> anyhow::Result<i64> {
    conn.execute(
        "INSERT INTO Dataset(name, num_obj, num_traj, avg_traj_len, median_traj_len,
                             p99_traj_len, max_traj_len, min_traj_len, num_box)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)
         ON CONFLICT(name) DO UPDATE SET
             num_obj = excluded.num_obj, num_traj = excluded.num_traj,
             avg_traj_len = excluded.avg_traj_len, median_traj_len = excluded.median_traj_len,
             p99_traj_len = excluded.p99_traj_len, max_traj_len = excluded.max_traj_len,
             min_traj_len = excluded.min_traj_len, num_box = excluded.num_box",
        params![
            d.name, d.num_obj, d.num_traj, d.avg_traj_len, d.median_traj_len,
            d.p99_traj_len, d.max_traj_len, d.min_traj_len, d.num_box
        ],
    )
    .context("failed to upsert Dataset")?;
    let id: i64 = conn
        .query_row("SELECT id FROM Dataset WHERE name = ?1", params![d.name], |r| r.get(0))
        .context("failed to read Dataset id")?;
    Ok(id)
}

/// Insert or update the query row, returning its id.
///
/// The SQL text is refreshed too, so a change in the compiler shows up against
/// the same query rather than creating a second one.
pub fn upsert_query(conn: &Connection, q: &QueryRow<'_>) -> anyhow::Result<i64> {
    conn.execute(
        "INSERT INTO Query(name, dataset, trql_query_text, sql_query_text, dataset_id)
         VALUES (?1,?2,?3,?4,?5)
         ON CONFLICT(name, dataset_id) DO UPDATE SET
             trql_query_text = excluded.trql_query_text,
             sql_query_text  = excluded.sql_query_text,
             dataset         = excluded.dataset",
        params![q.name, q.dataset, q.trql_query_text, q.sql_query_text, q.dataset_id],
    )
    .context("failed to upsert Query")?;
    let id: i64 = conn
        .query_row(
            "SELECT id FROM Query WHERE name = ?1 AND dataset_id = ?2",
            params![q.name, q.dataset_id],
            |r| r.get(0),
        )
        .context("failed to read Query id")?;
    Ok(id)
}

pub fn insert_run(conn: &Connection, r: &RunRow) -> anyhow::Result<i64> {
    conn.execute(
        "INSERT INTO Run(query_id, sql_time, trql_time, trql_error_px, trql_gap,
                         trql_oid_recall, trql_oid_precision, trql_iou,
                         trql_selection_ms, trql_poly_ms, trql_lift_ms, trql_temporal_ms, trql_other_ms,
                         trql_peak_mem_mb, sql_peak_mem_mb, sql_peak_rss_mb,
                         trql_segments, trql_poly_segments, trql_direct_segments, scale,
                         detector, gt_oid_precision, gt_oid_recall, gt_iou,
                         sql_gt_precision, sql_gt_recall, sql_gt_iou)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,
                 ?20,?21,?22,?23,?24,?25,?26,?27)",
        params![
            r.query_id, r.sql_time, r.trql_time, r.trql_error_px, r.trql_gap,
            r.trql_oid_recall, r.trql_oid_precision, r.trql_iou,
            r.trql_selection_ms, r.trql_poly_ms, r.trql_lift_ms, r.trql_temporal_ms, r.trql_other_ms,
            r.trql_peak_mem_mb, r.sql_peak_mem_mb, r.sql_peak_rss_mb,
            r.trql_segments, r.trql_poly_segments, r.trql_direct_segments, r.scale,
            &r.detector, r.gt_oid_precision, r.gt_oid_recall, r.gt_iou,
            r.sql_gt_precision, r.sql_gt_recall, r.sql_gt_iou
        ],
    )
    .context("failed to insert Run")?;
    Ok(conn.last_insert_rowid())
}

/// Open (creating if needed) the bench database at `path`.
pub fn open(path: &Path) -> anyhow::Result<Connection> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let conn = Connection::open(path)
        .with_context(|| format!("failed to open bench db {}", path.display()))?;
    schema(&conn)?;
    Ok(conn)
}
