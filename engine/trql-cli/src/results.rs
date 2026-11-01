//! Append benchmark runs to the SQLite results database.
//!
//! Mirrors `evaluator/src/db.rs` so the plot scripts under `papers/plotter/`
//! (see its `plotter/models.py`) read rows written here exactly as they read
//! the evaluator's: the same `systems` / `queries` / `runs` tables, with the
//! same column names and types.
//!
//! One addition: a nullable `mean_iou` column, populated when a benchmark is
//! given ground truth to score against. Databases created by the evaluator
//! predate that column, so [`record_run`] adds it on the fly.

use std::path::Path;

use anyhow::Context as _;
use rusqlite::{Connection, params};

/// One row of the `runs` table.
pub struct RunRecord<'a> {
    /// `systems.name`, e.g. `TrQS`.
    pub system: &'a str,
    /// `queries.name` — the short query id, e.g. `Q1`.
    pub query: &'a str,
    /// Database tiling factor the run was measured at.
    pub scale: u32,
    /// Mean wall-clock query time, in milliseconds.
    pub exec_ms: f64,
    /// Peak process RSS during the run, in megabytes.
    pub memory_mb: f64,
    pub precision: f64,
    pub recall: f64,
    pub max_gap: u32,
    pub max_error_px: f64,
    /// Mean polynomial segments per trajectory.
    pub num_segments: Option<f64>,
    /// On-disk size of the `.trql` under test, in megabytes.
    pub size_mb: u32,
    /// Mean temporal IoU; `None` when the run was not scored.
    pub mean_iou: Option<f64>,
}

/// Create the schema if needed, then append `rec` as a new run.
///
/// `systems` and `queries` are upserted by name, so repeated runs of the same
/// query accumulate as rows sharing those ids.
pub fn record_run(db_path: &Path, rec: &RunRecord<'_>) -> anyhow::Result<()> {
    let conn = Connection::open(db_path)
        .with_context(|| format!("failed to open results db {}", db_path.display()))?;

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS systems (
             id   INTEGER PRIMARY KEY,
             name TEXT UNIQUE NOT NULL
         );
         CREATE TABLE IF NOT EXISTS queries (
             id          INTEGER PRIMARY KEY,
             name        TEXT UNIQUE NOT NULL,
             description TEXT
         );
         CREATE TABLE IF NOT EXISTS runs (
             id           INTEGER PRIMARY KEY,
             system_id    INTEGER REFERENCES systems(id),
             query_id     INTEGER REFERENCES queries(id),
             exec_time_ms REAL    NOT NULL,
             memory_mb    REAL    NOT NULL,
             scale        INTEGER NOT NULL DEFAULT 1,
             precision    REAL    NOT NULL,
             recall       REAL    NOT NULL,
             max_gap      INTEGER NOT NULL,
             max_error_px REAL    NOT NULL,
             num_segments REAL,
             size_mb      INTEGER NOT NULL,
             mean_iou     REAL,
             created_at   TEXT    DEFAULT (datetime('now'))
         );",
    )
    .with_context(|| format!("failed to create tables in {}", db_path.display()))?;

    // `CREATE TABLE IF NOT EXISTS` leaves an existing `runs` untouched, so a
    // database written by the evaluator needs `mean_iou` added.
    ensure_mean_iou_column(&conn, db_path)?;

    conn.execute(
        "INSERT OR IGNORE INTO systems(name) VALUES (?1)",
        params![rec.system],
    )
    .context("failed to upsert system")?;
    conn.execute(
        "INSERT OR IGNORE INTO queries(name) VALUES (?1)",
        params![rec.query],
    )
    .context("failed to upsert query")?;

    conn.execute(
        "INSERT INTO runs(
             system_id, query_id, exec_time_ms, memory_mb, scale,
             precision, recall, max_gap, max_error_px, num_segments, size_mb, mean_iou
         ) VALUES (
             (SELECT id FROM systems WHERE name = ?1),
             (SELECT id FROM queries WHERE name = ?2),
             ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12
         )",
        params![
            rec.system,
            rec.query,
            rec.exec_ms,
            rec.memory_mb,
            rec.scale,
            rec.precision,
            rec.recall,
            rec.max_gap,
            rec.max_error_px,
            rec.num_segments,
            rec.size_mb,
            rec.mean_iou,
        ],
    )
    .with_context(|| format!("failed to insert run into {}", db_path.display()))?;

    Ok(())
}

/// Add the `mean_iou` column to a pre-existing `runs` table that lacks it.
fn ensure_mean_iou_column(conn: &Connection, db_path: &Path) -> anyhow::Result<()> {
    let mut stmt = conn
        .prepare("SELECT 1 FROM pragma_table_info('runs') WHERE name = 'mean_iou'")
        .context("failed to inspect `runs` columns")?;
    let present = stmt
        .exists([])
        .context("failed to inspect `runs` columns")?;
    if !present {
        conn.execute_batch("ALTER TABLE runs ADD COLUMN mean_iou REAL")
            .with_context(|| format!("failed to add `mean_iou` column to {}", db_path.display()))?;
    }
    Ok(())
}
