//! Run a hand-supplied SQL query against the sub-dataset's own `duckdb.db`.
//!
//! The query is written by hand against the DSL's *denotation*, over the same
//! `bbox`/`objects` tables the `.trql` was built from, and is expected to share
//! nothing with the engine's implementation.
//!
//! That independence is the whole point. `trql-sql` can generate SQL from the
//! same algebra the engine evaluates, but comparing against *that* only checks
//! that the interval-set encoding was transcribed faithfully into SQL — both
//! sides carry the same formulation, so both are wrong together if it is wrong.
//! A query someone wrote from the semantics instead is the only comparison that
//! can find a fault in the formulation.
//!
//! Three details make the comparison meaningful rather than approximate.
//!
//! **One thread.** The engine evaluates on a single core. DuckDB defaults to one
//! thread per core, so a default-configured comparison measures parallelism
//! rather than the two evaluation strategies. [`SQL_PRELUDE`] pins it to one.
//!
//! **Object identity.** The engine numbers objects densely from 0, in the order
//! `trql-storage` reads them (`… ORDER BY b.oid`), while the SQL reports the
//! `oid` column itself. Those coincide only when a source happens to number its
//! objects `0..N-1`. [`engine_oids`] maps between them; without it the two sides
//! disagree on every row while describing identical spans.
//!
//! **Span merging.** The engine reports an interval *set*, several of whose
//! 5-tuples can describe overlapping spans; `IntervalSet::maximal()` merges
//! them. The supplied SQL therefore need not merge its own output — both sides
//! go through the same merge in [`crate::compare::QueryOutput`] before being
//! compared. Comparing raw, unmerged spans reports differences that are not
//! there.

use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::Context as _;
use duckdb::Connection;

/// Pinned to one thread — see the module docs.
const SQL_PRELUDE: &str = "SET threads=1;";

/// One SQL run: the grouped result and how long the query itself took.
pub struct SqlRun {
    /// `(oid tuple, maximal intervals)` per matched tuple, ready for
    /// [`crate::compare::QueryOutput::from_rows`].
    pub rows: Vec<(Vec<u32>, Vec<(u32, u32)>)>,
    /// Query execution only. Opening the database is excluded: DuckDB mmaps the
    /// file and reads what it touches, so an open is not comparable to the
    /// engine's `.trql` load and would flatter neither side honestly.
    pub elapsed_ms: f64,
    /// Rows before grouping — one per `(tuple, interval)`.
    pub raw_rows: usize,
    /// Peak of DuckDB's own memory accounting over the query, in megabytes.
    ///
    /// Sampled from `duckdb_memory()` on a second connection to the same open
    /// database, so it is DuckDB's buffer manager alone — immune to anything
    /// this process allocates around it, unlike [`Self::peak_rss_mb`].
    pub peak_mem_mb: f64,
    /// Peak RSS *growth* over the query, in megabytes.
    ///
    /// Growth rather than total because DuckDB runs in this process, after the
    /// engine's `.trql` is resident — see [`crate::metrics::measure_rss_kb`].
    /// Kept as a cross-check on [`Self::peak_mem_mb`]: it counts pages DuckDB
    /// touched that its own accounting may not tag, but it is a lower bound
    /// (allocator-held memory is reused without RSS moving) and it still
    /// includes the flat row buffer this function fills while draining.
    pub peak_rss_mb: f64,
}

/// How often the `duckdb_memory()` sampler polls.
///
/// Each poll is a query on a second connection competing for the same
/// single-threaded scheduler as the query being measured, so this trades
/// resolution against perturbing the very timing it runs alongside.
const MEM_POLL: Duration = Duration::from_millis(10);

/// Peak `duckdb_memory()` over a run, sampled on a background thread.
///
/// `temporary_storage_bytes` is deliberately excluded: that is what DuckDB
/// spilled to disk, which is the opposite of memory held.
struct MemSampler {
    peak_bytes: Arc<AtomicU64>,
    done: Arc<AtomicBool>,
    handle: thread::JoinHandle<()>,
}

impl MemSampler {
    fn start(conn: &Connection) -> anyhow::Result<Self> {
        // A second connection to the *already-open* database, so it reports the
        // same buffer manager. Opening the file again would make a separate
        // instance, whose memory is unrelated to the query under measurement.
        let probe = conn
            .try_clone()
            .context("failed to clone the DuckDB connection for memory sampling")?;
        let peak_bytes = Arc::new(AtomicU64::new(0));
        let done = Arc::new(AtomicBool::new(false));
        let (peak2, done2) = (Arc::clone(&peak_bytes), Arc::clone(&done));

        let handle = thread::spawn(move || {
            while !done2.load(Ordering::Relaxed) {
                // A failed poll is not fatal: it costs one sample, and losing the
                // memory figure must not fail a run that is otherwise sound.
                if let Ok(bytes) = probe.query_row(
                    "SELECT coalesce(sum(memory_usage_bytes), 0) FROM duckdb_memory()",
                    [],
                    |r| r.get::<_, i64>(0),
                ) {
                    peak2.fetch_max(bytes.max(0) as u64, Ordering::Relaxed);
                }
                thread::sleep(MEM_POLL);
            }
        });
        Ok(Self {
            peak_bytes,
            done,
            handle,
        })
    }

    fn finish(self) -> f64 {
        self.done.store(true, Ordering::Relaxed);
        let _ = self.handle.join();
        self.peak_bytes.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0)
    }
}

/// The db `oid` behind each of the engine's object indices, ascending.
///
/// Index `i` of the engine's tuples is `oids[i]` in the database.
pub fn engine_oids(duckdb_path: &Path) -> anyhow::Result<Vec<u32>> {
    let conn = open(duckdb_path)?;
    let mut stmt = conn.prepare("SELECT DISTINCT oid FROM bbox ORDER BY oid")?;
    let mut rows = stmt.query([])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        let oid: i64 = row.get(0)?;
        out.push(u32::try_from(oid).context("oid does not fit in u32")?);
    }
    Ok(out)
}

fn open(duckdb_path: &Path) -> anyhow::Result<Connection> {
    let cfg = duckdb::Config::default().access_mode(duckdb::AccessMode::ReadOnly)?;
    Connection::open_with_flags(duckdb_path, cfg)
        .with_context(|| format!("failed to open {}", duckdb_path.display()))
}

/// Runs `sql` and groups its `(oid…, s, e)` rows by tuple.
///
/// The generated query projects one row per `(tuple, interval)`; the engine
/// reports one entry per tuple carrying all of its intervals. Grouping here is
/// what puts the two in the same shape.
pub fn run(duckdb_path: &Path, sql: &str, num_slots: usize) -> anyhow::Result<SqlRun> {
    let conn = open(duckdb_path)?;
    conn.execute_batch(SQL_PRELUDE)
        .context("failed to set threads=1")?;

    // A supplied file could set `threads` itself and quietly undo the pin, which
    // would turn the comparison into a measurement of DuckDB's parallelism
    // against a single-threaded engine. Check rather than trust.
    let threads: i64 = conn
        .query_row("SELECT current_setting('threads')", [], |r| r.get(0))
        .context("failed to read DuckDB's thread setting")?;
    if threads != 1 {
        anyhow::bail!("expected DuckDB at threads=1, found {threads}");
    }

    // Time the query alone: preparing and draining it, with the connection
    // already open, so the number is comparable to the engine's query time.
    // Both memory figures are sampled over the same window, for the same reason.
    //
    // Only the *reading* of rows happens here. Grouping them into a map is this
    // harness's bookkeeping, not DuckDB's work, and it allocates in proportion to
    // the answer — so it runs below, after both samplers have stopped, and stays
    // out of the memory figures. What remains inside is one flat `Vec<u32>`, the
    // smallest buffer that can hold rows that must be read to execute the query
    // at all.
    let sampler = MemSampler::start(&conn)?;
    let row_width = num_slots + 2;
    let t0 = Instant::now();
    let (drained, peak_kb, baseline_kb) = crate::metrics::measure_rss_kb(|| -> anyhow::Result<_> {
        let mut stmt = conn.prepare(sql).with_context(|| {
            "failed to prepare the supplied SQL (it must be a single SELECT statement)".to_string()
        })?;
        let mut rows = stmt.query([]).context("failed to run the supplied SQL")?;

        let mut flat: Vec<u32> = Vec::new();
        while let Some(row) = rows.next()? {
            for i in 0..num_slots {
                let oid: i64 = row.get(i)?;
                flat.push(u32::try_from(oid).context("oid does not fit in u32")?);
            }
            let s: i64 = row.get(num_slots).with_context(|| {
                format!(
                    "the supplied SQL must project {} object column(s) then `s`, `e` — \
                     could not read `s` at column {}",
                    num_slots, num_slots
                )
            })?;
            let e: i64 = row.get(num_slots + 1).context("could not read `e`")?;
            flat.push(s.max(0) as u32);
            flat.push(e.max(0) as u32);
        }
        Ok(flat)
    });
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let peak_mem_mb = sampler.finish();
    let flat = drained?;

    // Outside every measurement window: reshape the flat rows into the per-tuple
    // form `compare` expects.
    let mut grouped: std::collections::BTreeMap<Vec<u32>, Vec<(u32, u32)>> = Default::default();
    for chunk in flat.chunks_exact(row_width) {
        let (tuple, span) = chunk.split_at(num_slots);
        grouped
            .entry(tuple.to_vec())
            .or_default()
            .push((span[0], span[1]));
    }

    Ok(SqlRun {
        rows: grouped.into_iter().collect(),
        elapsed_ms,
        raw_rows: flat.len() / row_width,
        peak_mem_mb,
        peak_rss_mb: (peak_kb.saturating_sub(baseline_kb) / 1024) as f64,
    })
}

/// Trajectory-length distribution, read from the source `duckdb.db`.
///
/// Length is *frames covered*, `count(*)` per `oid`, not the first-to-last span
/// — a track with a hole is shorter than its span, and it is the covered count
/// the engine's `avg_traj_len` also reports.
pub struct TrajStats {
    pub num_obj: u64,
    pub num_traj: u64,
    /// Rows in `bbox` — one per object per frame, which is what the relational
    /// encoding stores and what the engine's segment count is measured against.
    pub num_box: u64,
    pub avg: f64,
    pub median: f64,
    pub p99: f64,
    pub max: u64,
    pub min: u64,
}

pub fn traj_stats(duckdb_path: &Path) -> anyhow::Result<TrajStats> {
    let conn = open(duckdb_path)?;
    let num_obj: i64 = conn
        .query_row("SELECT count(*) FROM objects", [], |r| r.get(0))
        .context("failed to count objects")?;
    let (num_traj, num_box, avg, median, p99, max, min): (i64, i64, f64, f64, f64, i64, i64) = conn
        .query_row(
            "WITH len AS (SELECT oid, count(*) AS n FROM bbox GROUP BY oid)
             SELECT count(*), sum(n), avg(n), median(n), quantile_cont(n, 0.99), max(n), min(n)
             FROM len",
            [],
            |r| {
                Ok((
                    r.get(0)?,
                    r.get(1)?,
                    r.get(2)?,
                    r.get(3)?,
                    r.get(4)?,
                    r.get(5)?,
                    r.get(6)?,
                ))
            },
        )
        .context("failed to compute trajectory length stats")?;
    Ok(TrajStats {
        num_obj: num_obj as u64,
        num_traj: num_traj as u64,
        num_box: num_box as u64,
        avg,
        median,
        p99,
        max: max as u64,
        min: min as u64,
    })
}

/// A temporary `duckdb.db` holding `times` tiled copies of `source`, so the SQL
/// side sees the same grown database `Trqs::repeat` gives the engine.
///
/// The tiling must match [`trql::storage::db::Trqs::repeat`] exactly or the two
/// answers are not comparable. That function shifts replica `i`'s frames by
/// `i * stride` where `stride = max_frame + 1`, and appends each replica's
/// objects after the previous one's. Here the same stride comes from
/// `MAX(fid) + 1` — the engine derives it from the fitted polynomials, which
/// cover exactly the frames the boxes do — and `oid` is offset by
/// `i * (MAX(oid) + 1)`, which keeps replicas disjoint and, crucially, keeps them
/// in the same *sorted order* the engine assigns its dense indices in. Sparse
/// oids (an i24 slice) still work: replica `i`'s ids all sit below replica
/// `i+1`'s, so sorting groups by replica and preserves order within one.
///
/// Returned as a [`TempDb`] that deletes the file on drop — a tiled copy is
/// `times` times the source and there is no reason to keep it.
pub struct TempDb {
    pub path: PathBuf,
}

impl Drop for TempDb {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
        // DuckDB may leave a write-ahead log beside it.
        let _ = std::fs::remove_file(self.path.with_extension("db.wal"));
    }
}

pub fn tiled_duckdb(source: &Path, times: u32) -> anyhow::Result<TempDb> {
    let path = std::env::temp_dir().join(format!(
        "trql-tiled-{}-x{times}.db",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);

    let conn = Connection::open(&path)
        .with_context(|| format!("failed to create tiled db at {}", path.display()))?;
    conn.execute_batch(&format!(
        "ATTACH '{}' AS src (READ_ONLY);",
        source.to_string_lossy().replace('\'', "''")
    ))
    .with_context(|| format!("failed to attach {}", source.display()))?;

    let stride: i64 = conn
        .query_row("SELECT COALESCE(MAX(fid), -1) + 1 FROM src.bbox", [], |r| r.get(0))
        .context("failed to read the frame stride")?;
    let oid_stride: i64 = conn
        .query_row("SELECT COALESCE(MAX(oid), -1) + 1 FROM src.objects", [], |r| r.get(0))
        .context("failed to read the object stride")?;

    // One row per (source row, replica). `generate_series` keeps this a single
    // statement rather than `times` UNION ALL branches.
    conn.execute_batch(&format!(
        "CREATE TABLE objects AS
           SELECT o.* EXCLUDE (oid), o.oid + t.i * {oid_stride} AS oid
           FROM src.objects o CROSS JOIN generate_series(0, {last}) t(i);
         CREATE TABLE bbox AS
           SELECT b.* EXCLUDE (oid, fid),
                  b.oid + t.i * {oid_stride} AS oid,
                  b.fid + t.i * {stride}     AS fid
           FROM src.bbox b CROSS JOIN generate_series(0, {last}) t(i);
         DETACH src;",
        last = times as i64 - 1
    ))
    .context("failed to build the tiled tables")?;
    drop(conn);
    Ok(TempDb { path })
}
