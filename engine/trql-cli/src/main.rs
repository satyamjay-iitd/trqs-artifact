//! Command-line front end for TrQL: build `.trql` artifacts from processed
//! datasets, then run, benchmark, and score queries against them.
//!
//! Every subcommand operates on one sub-dataset, selected by `--dataset` and
//! `--sub-dataset` within the tree `--data-dir` points at (the same tree
//! [`trql::storage::datasets::Datasets`] walks). `--max-gap` and
//! `--max-error-px` pick which `.trql` of that sub-dataset to use, matching the
//! `<max_gap>_<max_error_px>.trql` naming convention.

mod benchdb;
mod compare;
mod detectcmd;
mod metrics;
mod results;
mod sqlrun;
mod ui;

use std::{
    fs::File,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::Context as _;
use clap::{Parser, Subcommand};
use serde::Serialize;
use trql::{
    db::Trqs,
    dsl::parse_query,
    eval::AlgebraCtx,
    storage::{
        datasets::{ConvertProgress, SubDataset, TrqlParams},
        merge,
        stats::TrqsStats,
    },
};
use trql::storage::eval::Metrics;
use trql_types::QueryResult;

use crate::{
    compare::{CompareMetrics, QueryOutput},
    ui::{Format, Ui, group_digits},
};

#[derive(Parser)]
#[command(version, about = "Build, query, and benchmark TrQL datasets")]
struct Cli {
    /// Root of the processed dataset tree.
    #[arg(long, default_value = "datasets/processed")]
    data_dir: PathBuf,

    /// Dataset family, e.g. `cityflow`.
    #[arg(long)]
    dataset: String,

    /// Sub-dataset within the family, e.g. `train_S01_c001`.
    ///
    /// Required by every command that operates on one sub-dataset. `merge` and
    /// `detect` work across several and name their own sources, so they do not
    /// take it — it used to be mandatory regardless, which meant passing a value
    /// that was then ignored.
    #[arg(long)]
    sub_dataset: Option<String>,

    /// Max frame gap before a trajectory is split into separate polynomial
    /// segments. First half of the `<max_gap>_<max_error_px>.trql` name.
    #[arg(long, default_value_t = 10)]
    max_gap: u32,

    /// Max allowed polynomial fitting error, in pixels. Second half of the
    /// `<max_gap>_<max_error_px>.trql` name.
    #[arg(long, default_value_t = 0.0)]
    max_error_px: f64,

    /// Output format. `json` prints one document per command and suppresses
    /// progress bars.
    #[arg(long, value_enum, default_value_t = Format::Text)]
    format: Format,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Manage the `.trql` artifacts of the selected sub-dataset.
    #[command(subcommand)]
    Dataset(DatasetCRUD),

    /// Merge sub-datasets of the selected family. `--sub-dataset` is ignored;
    /// the sources are named by `--sub` / `--from-file`.
    #[command(subcommand)]
    Merge(MergeOp),

    /// Run a query against the selected `.trql` and write its result rows to
    /// `output`.
    Run {
        /// Query in the TrQL DSL.
        query: String,
        /// File to write the matching rows to.
        output: PathBuf,
        /// Also write the answer as an `oid0,…,start,end` CSV — the format
        /// `bench --gt` and `compare` read, so one run's answer can be the
        /// reference another is scored against.
        #[arg(long)]
        csv: Option<PathBuf>,
    },

    /// Time a query over repeated runs, writing a JSON report to the
    /// sub-dataset's `benchmark/` folder.
    Bench {
        /// Short identifier for the query, e.g. `Q1`. Used in the report file
        /// name.
        query_id: String,
        /// Query in the TrQL DSL.
        query: String,
        /// Dataset replication factor: 1 uses the original data; N tiles it N
        /// times with offset object and frame ids to benchmark larger inputs.
        db_repeat: u32,
        /// How many times to run the query per loaded database.
        run_repeat: u32,
        /// Ground-truth CSV (`oid0,…,start,end`). Scores the run's output,
        /// adding precision/recall/IoU to the report.
        #[arg(long)]
        gt: Option<PathBuf>,
        /// Append this run to the suite's SQLite results schema.
        /// Use --bench-db with --sql for the current plotter's paired schema.
        #[arg(long)]
        results_db: Option<PathBuf>,
        /// Equivalent SQL for this query, as a file containing one `SELECT`
        /// projecting `oid0, …, oidN, s, e`. Run against the sub-dataset's own
        /// `duckdb.db` at `threads=1` and compared to the engine's answer.
        ///
        /// Deliberately supplied rather than generated: `trql-sql` emits our own
        /// interval-set encoding, so comparing against it tests the
        /// transcription of our algebra rather than the semantics. An
        /// independently written query is the only comparison that can find a
        /// fault in the formulation itself.
        #[arg(long)]
        sql: Option<PathBuf>,
        /// Paired benchmark database, in the `Dataset` / `Query` / `Run` schema.
        /// Only written when `--sql` supplies something to compare against.
        #[arg(long, default_value = "bench.db")]
        bench_db: PathBuf,
    },

    /// Run a detector+tracker over sub-dataset videos and score the predictions
    /// against their ground truth. `--sub-dataset` is ignored; sources are named
    /// by `--sub` / `--from-file` / `--all`.
    #[command(subcommand)]
    Detect(DetectOp),

    /// Score a query against ground truth by matching `gt`'s rows to the rows
    /// the query returns.
    Compare {
        /// Query in the TrQL DSL.
        query: String,
        /// Ground-truth rows to match against.
        gt: PathBuf,
    },
}

#[derive(Subcommand)]
enum DetectOp {
    /// Detect, align predicted track ids to ground truth, and score.
    ///
    /// Detection needs a GPU and a Python environment outside this workspace, so
    /// the inference itself is a subprocess — see `--python` / `--script`.
    Run {
        /// Preset to run. `trql-cli … detect configs` lists them.
        #[arg(long, default_value = "yolo11l-bytetrack-traffic")]
        config: String,
        /// Sub-datasets to run on. Repeat the flag, or use `--from-file` /
        /// `--all`.
        #[arg(long = "sub", value_name = "NAME")]
        subs: Vec<String>,
        /// Read sub-dataset names from a file, one per line (blank lines and
        /// lines starting with `#` are ignored).
        #[arg(long)]
        from_file: Option<PathBuf>,
        /// Every sub-dataset of `--dataset` that has a video.
        #[arg(long)]
        all: bool,
        /// Stop after this many sub-datasets — a cheap first pass over a large
        /// family before committing to all of it.
        #[arg(long)]
        limit: Option<usize>,
        /// Python interpreter with ultralytics, opencv and duckdb installed.
        #[arg(long, default_value = ".venv-detect/bin/python")]
        python: String,
        /// The detect+track worker script.
        #[arg(long, default_value = "scripts/detect_track.py")]
        script: PathBuf,
        /// IoU threshold for both id alignment and scoring.
        #[arg(long, default_value_t = 0.5)]
        iou: f64,
        /// Override the preset's model weights, e.g. `rtdetr-l.pt`.
        #[arg(long)]
        model: Option<String>,
        /// Override the preset's tracker, e.g. `botsort.yaml`.
        #[arg(long)]
        tracker: Option<String>,
        /// Override the preset's detection confidence threshold.
        #[arg(long)]
        conf: Option<f64>,
        /// Override the preset's inference image size.
        #[arg(long)]
        imgsz: Option<u32>,
        /// Override the COCO class ids kept, e.g. `--classes 2 5 7` for
        /// car/bus/truck. Empty keeps every class.
        #[arg(long, num_args = 0.., value_name = "ID")]
        classes: Option<Vec<u32>>,
        /// Rename a predicted class, as `FROM=TO`. Repeatable. The detector
        /// speaks COCO; use this where the dataset calls the same thing something
        /// else, e.g. `--class-alias person=pedestrian` for MOT17, so one query
        /// text runs against both ground truth and predictions.
        #[arg(long = "class-alias", value_name = "FROM=TO")]
        class_alias: Vec<String>,
        /// Re-run sub-datasets whose predictions already exist.
        #[arg(long)]
        force: bool,
    },

    /// List the detector+tracker presets.
    Configs,
}

#[derive(Subcommand)]
enum DatasetCRUD {
    /// Build the `<max_gap>_<max_error_px>.trql` for this sub-dataset from its
    /// `duckdb.db`. Does nothing if that artifact already exists.
    #[command(alias = "create")]
    Build,

    /// Print the trajectory statistics recorded in the selected `.trql`.
    Stats,

    /// Delete only the `<max_gap>_<max_error_px>.trql` artifact. The source
    /// `duckdb.db`, video, and metadata are left untouched.
    Delete,

    /// Delete every `.trql` artifact of this sub-dataset, whatever parameters
    /// they were built with. Ignores `--max-gap` / `--max-error-px`; sources
    /// are left untouched.
    DeleteAll,
}

#[derive(Subcommand)]
enum MergeOp {
    /// Fold several sub-datasets of one family into a new merged sub-dataset,
    /// offsetting `oid` and `fid` so ids stay continuous and non-overlapping.
    ///
    /// Non-interactive on purpose: `trql-tui` can only mark sub-datasets one at a
    /// time or a whole family at once, which cannot express a selection like "the
    /// 467 nuScenes scenes on the boston-seaport map". Both front ends call the
    /// same `trql-storage::merge`, so a merge made here is identical to one made
    /// in the TUI.
    Run {
        /// Sub-datasets to merge, in merge order. Repeat the flag, or pass a file
        /// of names with `--from-file`.
        #[arg(long = "sub", value_name = "NAME")]
        subs: Vec<String>,
        /// Read sub-dataset names from a file, one per line (blank lines and
        /// lines starting with `#` are ignored).
        #[arg(long)]
        from_file: Option<PathBuf>,
        /// Output directory. Defaults to `<data-dir>/merged/<first>+<n>more`,
        /// matching what the TUI would choose.
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

/// What `dataset build` did.
#[derive(Serialize)]
struct BuildReport {
    path: PathBuf,
    /// False when the artifact already existed and nothing was rebuilt.
    built: bool,
    num_objs: Option<u32>,
}

/// A `.trql` already built for a sub-dataset.
#[derive(Serialize)]
struct ArtifactReport {
    file: String,
    max_gap: u32,
    max_error_px: f64,
    stale: bool,
}

impl From<&TrqlParams> for ArtifactReport {
    fn from(p: &TrqlParams) -> Self {
        Self {
            file: p.file_name(),
            max_gap: p.max_gap,
            max_error_px: p.max_error_px,
            stale: p.stale,
        }
    }
}

/// The JSON document written to a `run`'s `--output`.
#[derive(Serialize)]
struct ResultFile {
    query: String,
    count: usize,
    results: QueryResult,
}

/// What `run` reported to the user (the rows themselves go to `output`).
#[derive(Serialize)]
struct RunReport {
    query: String,
    output: PathBuf,
    matches: usize,
    elapsed_ms: f64,
}

/// The engine's name in bench output and the results database. The
/// `naive-intervals` ablation build reports under its own name so its runs are
/// never averaged in with the real engine's.
#[cfg(not(feature = "naive-intervals"))]
const ENGINE: &str = "TrQS";
#[cfg(feature = "naive-intervals")]
const ENGINE: &str = "TrQS-naive";
#[cfg(not(feature = "naive-intervals"))]
const INTERVAL_REPR: &str = "5tuple";
#[cfg(feature = "naive-intervals")]
const INTERVAL_REPR: &str = "naive";

/// Runtime metrics for a benchmarked query — the same set the evaluator
/// records for the TrQS engine (`evaluator/src/runner.rs`). Correctness scores
/// are included only when `--gt` supplies ground truth to compare against.
#[derive(Serialize)]
struct BenchReport {
    query_id: String,
    query: String,
    /// Interval-set representation the engine was built with — `5tuple`, or
    /// `naive` under the `naive-intervals` ablation feature.
    interval_repr: &'static str,
    /// Tiling factor the database was loaded at (`--db-repeat`).
    scale: u32,
    max_gap: u32,
    max_error_px: f64,
    /// Objects in the (tiled) database under test.
    objects: u32,
    /// Result rows the query matched.
    rows: usize,
    /// Mean wall-clock time across `run_repeat` runs.
    avg_ms: f64,
    /// Per-run wall-clock times, in order.
    times_ms: Vec<f64>,
    /// Peak process RSS observed during a run.
    peak_mem_mb: f64,
    /// On-disk size of the `.trql` backing the benchmark.
    storage_mb: u32,
    /// Trajectory statistics of the (tiled) database under test.
    db_stats: TrqsStats,
    /// Correctness scores, when `--gt` was given.
    #[serde(skip_serializing_if = "Option::is_none")]
    metrics: Option<CompareMetrics>,
    /// The paired DuckDB run, when one was made.
    #[serde(skip_serializing_if = "Option::is_none")]
    sql: Option<SqlSummary>,
    /// Per-stage self-time, when built with the `instrument` feature.
    #[serde(skip_serializing_if = "Option::is_none")]
    stages: Option<Stages>,
    /// Results database this run was appended to, when `--results-db` was given.
    #[serde(skip_serializing_if = "Option::is_none")]
    results_db: Option<PathBuf>,
}

/// `FROM=TO` strings as pairs, rejecting anything that is not one.
///
/// A silently-ignored malformed alias would leave the predictions unrelabelled
/// and the query returning nothing, which reads as a bad detector rather than a
/// typo.
fn parse_class_aliases(raw: &[String]) -> anyhow::Result<Vec<(String, String)>> {
    raw.iter()
        .map(|s| {
            s.split_once('=')
                .filter(|(f, t)| !f.is_empty() && !t.is_empty())
                .map(|(f, t)| (f.to_owned(), t.to_owned()))
                .ok_or_else(|| anyhow::anyhow!("--class-alias expects FROM=TO, got {s:?}"))
        })
        .collect()
}

/// The bench database's name for a sub-dataset at a given tiling factor.
///
/// Untiled runs keep the bare path, so the common case reads as it always has and
/// no existing row is orphaned. A `--db-repeat N` run gets its own name because it
/// measured its own database — `Dataset` is upserted by name, and sharing one
/// would let a tiled run's object and box counts overwrite the untiled row that
/// earlier runs' stored ratios are computed against.
fn scaled_dataset_name(rel_path: &Path, scale: u32) -> String {
    let base = rel_path.display().to_string();
    if scale <= 1 {
        base
    } else {
        format!("{base} x{scale}")
    }
}

/// What `compare` scored — the query against the ground-truth CSV.
#[derive(Serialize)]
struct CompareReport {
    query: String,
    gt: PathBuf,
    #[serde(flatten)]
    metrics: CompareMetrics,
}

/// What `merge run` produced.
#[derive(Serialize)]
struct MergeReport {
    out_dir: PathBuf,
    num_sources: usize,
    num_objs: u32,
    total_frames: u32,
}

/// What `dataset stats` found.
#[derive(Serialize)]
struct StatsReport {
    name: String,
    rel_path: PathBuf,
    is_gt: bool,
    video: String,
    /// The `.trql` selected by `--max-gap` / `--max-error-px`.
    selected: String,
    /// `None` when the selected `.trql` has not been built.
    stats: Option<TrqsStats>,
    artifacts: Vec<ArtifactReport>,
}

/// The paired SQL run as it appears in the JSON report. The generated SQL is not
/// repeated here — it lives in the bench db's `Query.sql_query_text`, stored once
/// per query rather than once per run.
#[derive(Serialize)]
struct SqlSummary {
    /// DuckDB query time (ms, `threads=1`, connection already open).
    elapsed_ms: f64,
    /// Peak of DuckDB's own `duckdb_memory()` accounting over the query, in MB.
    peak_mem_mb: f64,
    /// Peak RSS growth over the query, in MB — the cross-check on `peak_mem_mb`.
    peak_rss_mb: f64,
    /// Rows before grouping by tuple.
    raw_rows: usize,
    engine_tuples: usize,
    sql_tuples: usize,
    matched_tuples: usize,
    precision: f64,
    recall: f64,
    mean_iou: f64,
    interval_mismatches: usize,
    /// True when the two answers are identical.
    exact: bool,
}

/// Where a query's time went, per run, in milliseconds.
///
/// Only populated when the binary is built with the `instrument` feature. The
/// scopes cost a little on every predicate, so an instrumented run's absolute
/// times are inflated relative to a clean one — read the proportions, not the
/// totals, and take `trql_time` from an uninstrumented run.
#[derive(Serialize, Clone, Copy)]
struct Stages {
    /// Resolving each slot's candidates and enumerating co-appearing tuples.
    selection_ms: f64,
    /// Analytic root-finding plus the segment sweep that invokes it.
    poly_ms: f64,
    /// Building interval-sets from solved frames, and clipping memoised results
    /// to the tuple's co-appearance. Per leaf per tuple.
    lift_ms: f64,
    /// Interval-set algebra: THEN / CONTAINS / OVERLAP / DURATION.
    temporal_ms: f64,
    /// Residual: planning, optimisation, tuple materialisation, timing overhead.
    other_ms: f64,
}

/// One paired SQL run alongside the engine's answer.
struct SqlPairing {
    sql: String,
    /// SQL's answer scored against ground truth — the baseline the engine's
    /// `gt_*` numbers are read against, since SQL has no gap-smoothing notion.
    gt_metrics: Option<CompareMetrics>,
    elapsed_ms: f64,
    /// DuckDB's own peak memory over its query, in MB — see [`sqlrun::SqlRun`].
    peak_mem_mb: f64,
    /// Peak RSS growth over DuckDB's query, in MB — the cross-check.
    peak_rss_mb: f64,
    raw_rows: usize,
    traj: sqlrun::TrajStats,
    metrics: CompareMetrics,
}

/// The selected sub-dataset and how to report on it: the shared context every
/// subcommand needs, so the commands don't thread five arguments each.
struct Ctx {
    ui: Ui,
    /// Root of the processed dataset tree.
    root: PathBuf,
    /// The selected sub-dataset directory, `<root>/<dataset>/<sub-dataset>`.
    /// The selected sub-dataset's directory, absent for the commands that do not
    /// take one.
    dir: Option<PathBuf>,
    max_gap: u32,
    max_error_px: f64,
}

impl Ctx {
    fn from_cli(cli: &Cli) -> Self {
        let root = cli.data_dir.clone();
        let dir = cli
            .sub_dataset
            .as_ref()
            .map(|sub| root.join(&cli.dataset).join(sub));
        Self {
            ui: Ui::new(cli.format),
            root,
            dir,
            max_gap: cli.max_gap,
            max_error_px: cli.max_error_px,
        }
    }

    /// The selected sub-dataset's directory, or an error naming the flag that
    /// would have chosen it.
    fn dir(&self) -> anyhow::Result<&Path> {
        self.dir
            .as_deref()
            .context("this command needs --sub-dataset to say which one to act on")
    }

    fn load_sub_dataset(&self) -> anyhow::Result<SubDataset> {
        SubDataset::load(&self.root, self.dir()?)
    }

    fn load_database(&self, repeat: Option<u32>) -> anyhow::Result<Trqs> {
        let sub_dataset = self.load_sub_dataset()?;
        sub_dataset.load_database(self.max_gap, self.max_error_px, repeat)
    }

    /// Load the selected database, tiled `repeat` times, showing progress. The
    /// bar starts as a spinner and, if the `.trql` has to be built from source
    /// first, switches to a real bar during trajectory fitting.
    fn load_database_with_bar(
        &self,
        sub_dataset: &SubDataset,
        repeat: Option<u32>,
    ) -> anyhow::Result<Trqs> {
        let bar = self.ui.spinner("loading database");
        let mut fitting = false;
        let db = sub_dataset.load_database_with_progress(
            self.max_gap,
            self.max_error_px,
            repeat,
            &mut |progress| match progress {
                ConvertProgress::Inspecting => bar.set_message("inspecting schema"),
                ConvertProgress::Reading => bar.set_message("reading objects"),
                ConvertProgress::Fitting { done, total } => {
                    if !fitting {
                        fitting = true;
                        ui::into_bar(&bar, total as u64, "trajectories fitted");
                    }
                    bar.set_position(done as u64);
                }
                ConvertProgress::Saving => bar.set_message("saving"),
            },
        )?;
        bar.finish_and_clear();
        Ok(db)
    }
}

fn main() -> anyhow::Result<()> {
    let args = Cli::parse();
    let ctx = Ctx::from_cli(&args);

    match args.command {
        Commands::Dataset(op) => match op {
            DatasetCRUD::Build => ctx.build()?,
            DatasetCRUD::Stats => ctx.stats()?,
            DatasetCRUD::Delete => todo!(),
            DatasetCRUD::DeleteAll => todo!(),
        },
        Commands::Merge(op) => match op {
            MergeOp::Run {
                subs,
                from_file,
                out,
            } => ctx.merge(&args.dataset, &subs, from_file.as_deref(), out.as_deref())?,
        },
        Commands::Detect(op) => match op {
            DetectOp::Configs => ctx.detect_configs()?,
            DetectOp::Run {
                config,
                subs,
                from_file,
                all,
                limit,
                python,
                script,
                iou,
                model,
                tracker,
                conf,
                imgsz,
                classes,
                class_alias,
                force,
            } => ctx.detect(
                &args.dataset,
                &config,
                &subs,
                from_file.as_deref(),
                all,
                limit,
                &python,
                &script,
                iou,
                detectcmd::Overrides {
                    model,
                    tracker,
                    conf,
                    imgsz,
                    classes,
                    aliases: parse_class_aliases(&class_alias)?,
                },
                force,
            )?,
        },
        Commands::Run { query, output, csv } => ctx.run(&query, &output, csv.as_deref())?,
        Commands::Bench {
            query_id,
            query,
            db_repeat,
            run_repeat,
            gt,
            results_db,
            sql,
            bench_db,
        } => ctx.bench(
            &query_id,
            &query,
            db_repeat,
            run_repeat,
            gt.as_deref(),
            results_db.as_deref(),
            sql.as_deref(),
            &bench_db,
        )?,
        Commands::Compare { query, gt } => ctx.compare(&query, &gt)?,
    }
    Ok(())
}

impl Ctx {
    fn build(&self) -> anyhow::Result<()> {
        // Every command operates on the same selected sub-dataset, so load it once.
        let sub_dataset = self.load_sub_dataset()?;

        let ui = &self.ui;
        let dir = self.dir()?;
        let (max_gap, max_error_px) = (self.max_gap, self.max_error_px);

        if sub_dataset.has_trql(max_gap, max_error_px) {
            let report = BuildReport {
                path: dir.to_path_buf(),
                built: false,
                num_objs: None,
            };
            return ui.emit(&report, |ui, r| {
                ui.skip(format!(
                    "Trql File with max_gap={} max_error_px={} already exists in {}",
                    max_gap,
                    max_error_px,
                    r.path.display()
                ));
            });
        }

        // One bar, reused across phases: a spinner while DuckDB works (it reports
        // nothing of its own), switching to a real bar once fitting starts.
        let bar = ui.spinner("inspecting schema");
        let db = self
            .load_database_with_bar(&sub_dataset, None)
            .with_context(|| {
                format!(
                    "failed to build trql file with max_gap={} max_error_px={} from dataset in {}",
                    max_gap,
                    max_error_px,
                    dir.to_path_buf().display()
                )
            })?;
        bar.finish_and_clear();

        let report = BuildReport {
            path: dir.to_path_buf(),
            built: true,
            num_objs: Some(db.num_objs()),
        };
        ui.emit(&report, |ui, r| {
            ui.success(format!(
                "built {} ({} objects)",
                r.path.display(),
                group_digits(r.num_objs.unwrap_or(0) as u64)
            ));
        })
    }

    /// Merge the named sub-datasets of `dataset` into one.
    ///
    /// The names are resolved against the tree rather than taken as paths, so a
    /// typo fails here with the available names listed instead of producing a
    /// merge that silently skipped a source.
    /// `detect configs`: the presets, so a `--config` id can be picked without
    /// reading the source.
    fn detect_configs(&self) -> anyhow::Result<()> {
        let configs = trql::storage::detect::predefined_configs();
        if self.ui.is_json() {
            let rows: Vec<_> = configs
                .iter()
                .map(|c| {
                    serde_json::json!({
                        "id": c.id, "name": c.name, "description": c.description,
                        "model": c.model, "tracker": c.tracker,
                        "conf": c.conf, "imgsz": c.imgsz, "half": c.half,
                        "classes": c.classes,
                    })
                })
                .collect();
            return self.ui.emit(&rows, |_, _| {});
        }
        self.ui.table(
            &["id", "model", "tracker", "imgsz", "description"],
            configs
                .iter()
                .map(|c| {
                    vec![
                        c.id.to_owned(),
                        c.model.to_owned(),
                        c.tracker.to_owned(),
                        c.imgsz.to_string(),
                        c.description.to_owned(),
                    ]
                })
                .collect(),
        );
        Ok(())
    }

    /// `detect run`: detect, align, score, across the selected sub-datasets.
    ///
    /// One sub-dataset failing does not stop the batch — a hundred-sequence run
    /// that aborts on the one video with no readable stream has wasted the other
    /// ninety-nine. Failures are recorded per job and summarised at the end.
    #[allow(clippy::too_many_arguments)]
    fn detect(
        &self,
        dataset: &str,
        config_id: &str,
        subs: &[String],
        from_file: Option<&Path>,
        all: bool,
        limit: Option<usize>,
        python: &str,
        script: &Path,
        iou: f64,
        overrides: detectcmd::Overrides,
        force: bool,
    ) -> anyhow::Result<()> {
        let ui = &self.ui;
        let config = overrides.apply(&detectcmd::config_by_id(config_id)?);

        let mut names: Vec<String> = subs.to_vec();
        if let Some(path) = from_file {
            let raw = std::fs::read_to_string(path)
                .with_context(|| format!("failed to read {}", path.display()))?;
            names.extend(
                raw.lines()
                    .map(str::trim)
                    .filter(|l| !l.is_empty() && !l.starts_with('#'))
                    .map(str::to_owned),
            );
        }
        let family_dir = self.root.join(dataset);
        if all {
            if !names.is_empty() {
                anyhow::bail!("--all takes the whole family; drop --sub / --from-file");
            }
            let mut found: Vec<String> = std::fs::read_dir(&family_dir)
                .with_context(|| format!("failed to read {}", family_dir.display()))?
                .filter_map(Result::ok)
                .filter(|e| e.path().is_dir())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect();
            found.sort();
            names = found;
        }
        if names.is_empty() {
            anyhow::bail!(
                "no sub-datasets selected; pass --sub NAME, --from-file, or --all"
            );
        }
        if let Some(n) = limit {
            names.truncate(n);
        }

        // Resolve every name before running anything: a typo should surface now,
        // not after twenty minutes of inference.
        let mut jobs = Vec::new();
        let mut skipped: Vec<(String, &str)> = Vec::new();
        for name in &names {
            let dir = family_dir.join(name);
            let sub = SubDataset::load(&self.root, &dir).with_context(|| {
                format!(
                    "{name}: not a sub-dataset of {dataset} (looked in {})",
                    dir.display()
                )
            })?;
            match detectcmd::job_for(&self.root, &sub, &config) {
                None => skipped.push((name.clone(), "no video.mp4")),
                Some(job) if !force && job.out_dir.join("duckdb.db").is_file() => {
                    skipped.push((name.clone(), "already detected (--force to redo)"))
                }
                Some(job) => jobs.push(job),
            }
        }
        // Progress chatter goes to stderr or nowhere under `--format json`: stdout
        // is the document, and a stray line makes it unparseable.
        if !ui.is_json() {
            for (name, why) in &skipped {
                ui.skip(format!("{name}: {why}"));
            }
        }
        if jobs.is_empty() {
            anyhow::bail!("nothing to do: every selected sub-dataset was skipped");
        }

        if !ui.is_json() {
            ui.section(&format!("Detecting with {} ({})", config.name, config.id));
        }
        let total_jobs = jobs.len();
        let bar = ui.spinner("detecting");
        let mut reports = Vec::with_capacity(total_jobs);
        for (i, job) in jobs.iter().enumerate() {
            let label = job.label.clone();
            let mut on_event = |ev: trql::storage::detect::DetectEvent| match ev {
                trql::storage::detect::DetectEvent::Progress { done, total } => {
                    bar.set_message(format!(
                        "[{}/{total_jobs}] {label}  {done}/{total} frames",
                        i + 1
                    ));
                }
            };
            match detectcmd::run_job(
                python,
                script,
                job,
                &config,
                iou,
                &overrides.aliases,
                &mut on_event,
            ) {
                Ok((tracks, stats, metrics)) => reports.push(detectcmd::JobReport {
                    source: job.label.clone(),
                    out_dir: job.out_dir.clone(),
                    tracks,
                    matched: Some(stats.matched),
                    metrics: Some(metrics),
                    error: None,
                }),
                Err(e) => reports.push(detectcmd::JobReport {
                    source: job.label.clone(),
                    out_dir: job.out_dir.clone(),
                    tracks: 0,
                    matched: None,
                    metrics: None,
                    error: Some(format!("{e:#}")),
                }),
            }
        }
        bar.finish_and_clear();

        let report = detectcmd::DetectReport {
            config: config.id.to_owned(),
            model: config.model.to_owned(),
            tracker: config.tracker.to_owned(),
            conf: config.conf,
            imgsz: config.imgsz,
            half: config.half,
            classes: config.classes.clone(),
            iou_threshold: iou,
            jobs: reports,
        };
        ui.emit(&report, |ui, r| {
            ui.table(
                &["sub-dataset", "tracks", "matched", "precision", "recall", "f1"],
                r.jobs
                    .iter()
                    .map(|j| match (&j.metrics, j.error.as_deref()) {
                        (Some(m), _) => vec![
                            j.source.clone(),
                            group_digits(j.tracks as u64),
                            j.matched.map(group_digits).unwrap_or_default(),
                            format!("{:.4}", m.precision),
                            format!("{:.4}", m.recall),
                            format!("{:.4}", m.f1),
                        ],
                        (None, err) => vec![
                            j.source.clone(),
                            "-".to_owned(),
                            "-".to_owned(),
                            err.unwrap_or("failed").to_string(),
                            String::new(),
                            String::new(),
                        ],
                    })
                    .collect(),
            );
            let ok: Vec<&Metrics> = r.jobs.iter().filter_map(|j| j.metrics.as_ref()).collect();
            if !ok.is_empty() {
                let mean = |f: fn(&Metrics) -> f64| {
                    ok.iter().map(|m| f(m)).sum::<f64>() / ok.len() as f64
                };
                ui.kv("sequences scored", format!("{} of {}", ok.len(), r.jobs.len()));
                ui.kv("mean precision", format!("{:.4}", mean(|m| m.precision)));
                ui.kv("mean recall", format!("{:.4}", mean(|m| m.recall)));
                ui.kv("mean F1", format!("{:.4}", mean(|m| m.f1)));
            }
            let failed = r.jobs.iter().filter(|j| j.error.is_some()).count();
            if failed > 0 {
                ui.warn(format!("{failed} of {} sub-datasets failed", r.jobs.len()));
            }
        })
    }

    fn merge(
        &self,
        dataset: &str,
        subs: &[String],
        from_file: Option<&Path>,
        out: Option<&Path>,
    ) -> anyhow::Result<()> {
        let ui = &self.ui;
        let mut names: Vec<String> = subs.to_vec();
        if let Some(path) = from_file {
            let raw = std::fs::read_to_string(path)
                .with_context(|| format!("failed to read {}", path.display()))?;
            names.extend(
                raw.lines()
                    .map(str::trim)
                    .filter(|l| !l.is_empty() && !l.starts_with('#'))
                    .map(str::to_owned),
            );
        }
        if names.len() < 2 {
            anyhow::bail!(
                "merge needs at least 2 sub-datasets (got {}); pass --sub NAME repeatedly \
                 or --from-file with one name per line",
                names.len()
            );
        }
        // A repeated source would be folded in twice under different ids, which is
        // silent corruption rather than an error, so reject it.
        let mut seen = std::collections::HashSet::new();
        if let Some(dup) = names.iter().find(|n| !seen.insert((*n).clone())) {
            anyhow::bail!("sub-dataset {dup:?} listed more than once");
        }

        let family_dir = self.root.join(dataset);
        let sources: Vec<merge::MergeSource> = names
            .iter()
            .map(|name| {
                let dir = family_dir.join(name);
                let sub = SubDataset::load(&self.root, &dir).with_context(|| {
                    format!("{name}: not a sub-dataset of {dataset} (looked in {})", dir.display())
                })?;
                let video = sub.dir().join("video.mp4");
                Ok(merge::MergeSource {
                    key: sub.duckdb_path.clone(),
                    label: sub.rel_path.display().to_string(),
                    duckdb_path: sub.duckdb_path.clone(),
                    video_path: video.is_file().then_some(video),
                    num_frames: sub.metadata.as_ref().and_then(|m| m.num_frames),
                    metadata_path: sub.metadata_path.clone(),
                })
            })
            .collect::<anyhow::Result<_>>()?;

        // Same naming the TUI picks, so a merge made here is indistinguishable.
        let out_dir = match out {
            Some(p) => p.to_path_buf(),
            None => self.root.join("merged").join(format!(
                "{}+{}more",
                names[0],
                names.len() - 1
            )),
        };
        if out_dir.join("duckdb.db").is_file() {
            anyhow::bail!(
                "{} already holds a duckdb.db — remove it or pass --out",
                out_dir.display()
            );
        }

        let bar = ui.spinner("merging");
        let mut done = 0usize;
        let total = sources.len();
        let summary = merge::merge_datasets(&sources, &out_dir, &mut |ev| {
            if let merge::MergeEvent::Finished { label, result, .. } = ev {
                done += 1;
                match result {
                    Ok(_) => bar.set_message(format!("merged {done}/{total}: {label}")),
                    Err(e) => bar.set_message(format!("FAILED {label}: {e}")),
                }
            }
        })?;
        bar.finish_and_clear();

        let report = MergeReport {
            out_dir: summary.out_dir,
            num_sources: summary.num_sources,
            num_objs: summary.num_objs,
            total_frames: summary.total_frames,
        };
        ui.emit(&report, |ui, r| {
            ui.success(format!("merged {} sub-datasets", r.num_sources));
            ui.kv("output", r.out_dir.display());
            ui.kv("objects", group_digits(r.num_objs as u64));
            ui.kv("frames", group_digits(r.total_frames as u64));
        })
    }

    fn stats(&self) -> anyhow::Result<()> {
        let sub_dataset = self.load_sub_dataset()?;

        let ui = &self.ui;
        let (max_gap, max_error_px) = (self.max_gap, self.max_error_px);
        let selected = sub_dataset.trql_name(max_gap, max_error_px);

        // `read_stats` only touches the file header, so this is cheap even for a
        // multi-gigabyte `.trql`.
        let stats = if sub_dataset.has_trql(max_gap, max_error_px) {
            let path = sub_dataset.trql_path(max_gap, max_error_px);
            Some(
                Trqs::read_stats(&path)
                    .with_context(|| format!("failed to read {}", path.display()))?,
            )
        } else {
            None
        };

        let report = StatsReport {
            name: sub_dataset.name.clone(),
            rel_path: sub_dataset.rel_path.clone(),
            is_gt: sub_dataset.is_gt,
            video: sub_dataset.video_info.to_string(),
            selected,
            stats,
            artifacts: sub_dataset.trql_files.iter().map(Into::into).collect(),
        };

        ui.emit(&report, |ui, r| {
            ui.section(&r.name);
            ui.kv("path", r.rel_path.display());
            ui.kv("source", if r.is_gt { "ground truth" } else { "predicted" });
            ui.kv("video", &r.video);

            ui.section("Trajectory stats");
            match &r.stats {
                Some(s) => {
                    ui.kv("file", &r.selected);
                    ui.kv("objects", group_digits(s.num_objs as u64));
                    ui.kv("avg len", format!("{:.1} frames covered", s.avg_traj_len));
                    ui.kv("avg span", format!("{:.1} frames (incl. gaps)", s.avg_span));
                    ui.kv(
                        "pieces",
                        format!(
                            "{} total ({:.2}/obj)",
                            group_digits(s.total_segments),
                            s.avg_segments
                        ),
                    );
                    let mut seg =
                        format!("{} poly · {} direct", s.poly_segments, s.direct_segments);
                    if s.mixed_segments > 0 {
                        seg.push_str(&format!(" · {} mixed", s.mixed_segments));
                    }
                    ui.kv("segments", seg);
                    ui.kv("canonical", group_digits(s.total_canonical));
                }
                None => ui.warn(format!("no {} built (run `dataset build`)", r.selected)),
            }

            ui.section("Built artifacts");
            if r.artifacts.is_empty() {
                ui.skip("(none yet)");
            } else {
                ui.table(
                    &["file", "max gap", "max error px", "state"],
                    r.artifacts
                        .iter()
                        .map(|a| {
                            vec![
                                a.file.clone(),
                                a.max_gap.to_string(),
                                a.max_error_px.to_string(),
                                if a.stale { "stale" } else { "ok" }.to_owned(),
                            ]
                        })
                        .collect(),
                );
            }
        })
    }

    fn run(&self, query: &str, output: &Path, csv: Option<&Path>) -> anyhow::Result<()> {
        let ui = &self.ui;
        // The `.trql` can be multi-gigabyte, so give some feedback while it loads.
        let bar = ui.spinner("loading database");
        let db = self.load_database(None)?;
        bar.finish_and_clear();

        let (constraints, expr) = parse_query(query, db.schema())?;
        let ctx = AlgebraCtx {
            db: &db,
            poly_trajs: None,
        };

        let t0 = Instant::now();
        let results: QueryResult = ctx.run(&constraints, &expr).into_iter().collect();
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // The result file is always JSON; `--format` governs only the summary.
        if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        if let Some(csv_path) = csv {
            QueryOutput::from_rows(
                results
                    .iter()
                    .map(|(tuple, ivs)| (tuple.to_vec(), ivs.maximal())),
            )
            .write_csv(csv_path, constraints.num_slots())?;
        }
        let result_file = ResultFile {
            query: query.to_owned(),
            count: results.len(),
            results,
        };
        let file = File::create(output)
            .with_context(|| format!("failed to write {}", output.display()))?;
        serde_json::to_writer_pretty(file, &result_file)
            .with_context(|| format!("failed to write {}", output.display()))?;

        let report = RunReport {
            query: query.to_owned(),
            output: output.to_path_buf(),
            matches: result_file.count,
            elapsed_ms,
        };
        ui.emit(&report, |ui, r| {
            ui.success(format!(
                "{} matches written to {}",
                group_digits(r.matches as u64),
                r.output.display()
            ));
            ui.kv("elapsed", format!("{:.1} ms", r.elapsed_ms));
            ui.kv("query", &r.query);
        })
    }

    fn bench(
        &self,
        query_id: &str,
        query: &str,
        db_repeat: u32,
        run_repeat: u32,
        gt: Option<&Path>,
        results_db: Option<&Path>,
        sql_path: Option<&Path>,
        bench_db: &Path,
    ) -> anyhow::Result<()> {
        if db_repeat == 0 || run_repeat == 0 {
            return Err(anyhow::anyhow!("--db-repeat and --run-repeat must be >= 1"));
        }
        let ui = &self.ui;
        let (max_gap, max_error_px) = (self.max_gap, self.max_error_px);
        let sub_dataset = self.load_sub_dataset()?;

        // The `.trql` can be large, so give some feedback while it loads (and
        // while it is built, if it hasn't been yet).
        let db = self.load_database_with_bar(&sub_dataset, Some(db_repeat))?;

        // Storage metric: the file backing the benchmark — a persisted tiling if
        // one exists, otherwise the base `.trql` (tiling was done in memory).
        let tiled = sub_dataset.dir().join(
            TrqlParams {
                max_gap,
                max_error_px,
                repeat: db_repeat,
                stale: false,
            }
            .file_name(),
        );
        let storage_path = if tiled.is_file() {
            tiled
        } else {
            sub_dataset.trql_path(max_gap, max_error_px)
        };
        let storage_mb = metrics::get_file_size_mb(&storage_path)?;

        // Read ground truth up front, so a bad CSV fails before the timed runs.
        let gt_output = gt.map(QueryOutput::from_csv).transpose()?;

        let (constraints, expr) = parse_query(query, db.schema())?;

        // Time the query `run_repeat` times, sampling peak RSS around each run —
        // the same loop as `evaluator/src/runner.rs`.
        let mut times_ms = Vec::with_capacity(run_repeat as usize);
        let mut rows = 0usize;
        let mut peak_mem_mb = 0f64;
        let mut last_results = None;
        trql::instrument::reset();
        for run in 1..=run_repeat {
            let ctx = AlgebraCtx {
                db: &db,
                poly_trajs: Some(db.poly_trajs()),
            };
            let start = Instant::now();
            let (results, peak_rss) = metrics::measure_peak_rss_kb(|| ctx.run(&constraints, &expr));
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            peak_mem_mb = (peak_rss / 1024) as f64;
            rows = results.len();
            times_ms.push(elapsed_ms);
            if !ui.is_json() {
                eprintln!("  run {run}/{run_repeat}: {elapsed_ms:.1} ms  rows: {rows}");
            }
            // Keep the final run's rows so scoring needs no extra execution.
            if gt_output.is_some() && run == run_repeat {
                last_results = Some(results);
            }
        }
        let avg_ms = times_ms.iter().sum::<f64>() / times_ms.len() as f64;

        // Attribute the query's time to its three stages. Buckets are exclusive
        // self-time, so they sum to the instrumented total; dividing by
        // `run_repeat` puts them on the same per-run footing as `avg_ms`.
        let stages = if trql::instrument::ENABLED {
            use trql::instrument::Bucket;
            let b = trql::instrument::snapshot();
            let ms = |x: std::time::Duration| x.as_secs_f64() * 1000.0 / run_repeat as f64;
            Some(Stages {
                selection_ms: ms(b[Bucket::Selection as usize]),
                poly_ms: ms(b[Bucket::RootFinding as usize] + b[Bucket::SegmentIter as usize]),
                lift_ms: ms(b[Bucket::IntervalLift as usize]),
                temporal_ms: ms(b[Bucket::Temporal as usize]),
                other_ms: ms(b[Bucket::Other as usize]),
            })
        } else {
            None
        };

        // Paired SQL run: compile this same query to DuckDB SQL, run it against
        // the sub-dataset's own `duckdb.db`, and compare. `sql_report` is None
        // when `--no-sql` was passed or the database was tiled — a tiled `.trql`
        // holds `repeat` copies of every object, which the source db does not,
        // so there is nothing to compare it against.
        // With `--db-repeat N` the engine runs against an N-times tiled database,
        // so the SQL must too or the two are answering different questions. The
        // tiled copy is built to the same rule `Trqs::repeat` uses and thrown away
        // afterwards.
        let tiled = match sql_path {
            Some(_) if db_repeat > 1 => {
                let bar = ui.spinner("tiling duckdb for the SQL side");
                let t = sqlrun::tiled_duckdb(&sub_dataset.duckdb_path, db_repeat)?;
                bar.finish_and_clear();
                Some(t)
            }
            _ => None,
        };
        let sql_report = match sql_path {
            Some(p) => Some(self.run_paired_sql(
                &sub_dataset,
                p,
                tiled.as_ref().map(|t| t.path.as_path()),
                &db,
                &constraints,
                &expr,
                gt_output.as_ref(),
            )?),
            None => None,
        };

        // Score against ground truth, if given.
        let compare_metrics = match (&gt_output, last_results) {
            (Some(gt_output), Some(results)) => {
                let produced = QueryOutput::from_rows(
                    results
                        .into_iter()
                        .map(|(tuple, intervals)| (tuple.to_vec(), intervals.maximal())),
                );
                Some(produced.compare(gt_output))
            }
            _ => None,
        };

        // Pulled out before `BenchReport` takes ownership: the bench db wants the
        // three scalars, not the samples and counts the report carries.
        let gt_scores = compare_metrics
            .as_ref()
            .map(|m| (m.precision, m.recall, m.mean_iou));

        let report = BenchReport {
            query_id: query_id.to_owned(),
            query: query.to_owned(),
            interval_repr: INTERVAL_REPR,
            scale: db_repeat,
            max_gap,
            max_error_px,
            objects: db.num_objs(),
            rows,
            avg_ms,
            times_ms,
            peak_mem_mb,
            storage_mb,
            db_stats: db.stats().clone(),
            metrics: compare_metrics,
            stages,
            sql: sql_report.as_ref().map(|s| {
                let m = &s.metrics;
                SqlSummary {
                    elapsed_ms: s.elapsed_ms,
                    peak_mem_mb: s.peak_mem_mb,
                    peak_rss_mb: s.peak_rss_mb,
                    raw_rows: s.raw_rows,
                    engine_tuples: m.query_tuples,
                    sql_tuples: m.gt_tuples,
                    matched_tuples: m.tp,
                    precision: m.precision,
                    recall: m.recall,
                    mean_iou: m.mean_iou,
                    interval_mismatches: m.interval_mismatches,
                    exact: m.fp == 0 && m.fn_ == 0 && m.interval_mismatches == 0,
                }
            }),
            results_db: results_db.map(Path::to_path_buf),
        };

        // Append to the results database the plot scripts read. Precision and
        // recall are only real when scored; without `--gt` they default to 1.0,
        // matching what the evaluator records for an unscored engine.
        if let Some(db_path) = results_db {
            let m = report.metrics.as_ref();
            results::record_run(
                db_path,
                &results::RunRecord {
                    system: ENGINE,
                    query: query_id,
                    scale: db_repeat,
                    exec_ms: avg_ms,
                    memory_mb: peak_mem_mb,
                    precision: m.map_or(1.0, |m| m.precision),
                    recall: m.map_or(1.0, |m| m.recall),
                    max_gap,
                    max_error_px,
                    num_segments: Some(report.db_stats.avg_segments),
                    size_mb: storage_mb,
                    mean_iou: m.map(|m| m.mean_iou),
                },
            )?;
        }

        // Record the paired run: dataset shape, both query texts, both timings,
        // and the agreement between them.
        if let Some(sql) = &sql_report {
            let conn = benchdb::open(bench_db)?;
            let t = &sql.traj;
            let dataset_id = benchdb::upsert_dataset(
                &conn,
                &benchdb::DatasetRow {
                    // The tiled database is a different database: more objects,
                    // more boxes, a longer timeline. Naming it apart keeps a
                    // `--db-repeat` run from overwriting the untiled row's counts,
                    // which every stored ratio against them depends on.
                    name: &scaled_dataset_name(&sub_dataset.rel_path, db_repeat),
                    num_obj: t.num_obj,
                    num_traj: t.num_traj,
                    avg_traj_len: t.avg,
                    median_traj_len: t.median,
                    p99_traj_len: t.p99,
                    max_traj_len: t.max,
                    min_traj_len: t.min,
                    num_box: t.num_box,
                },
            )?;
            let q_id = benchdb::upsert_query(
                &conn,
                &benchdb::QueryRow {
                    name: query_id,
                    dataset: &scaled_dataset_name(&sub_dataset.rel_path, db_repeat),
                    trql_query_text: query,
                    sql_query_text: &sql.sql,
                    dataset_id,
                },
            )?;
            benchdb::insert_run(
                &conn,
                &benchdb::RunRow {
                    query_id: q_id,
                    sql_time: sql.elapsed_ms,
                    trql_time: avg_ms,
                    trql_error_px: max_error_px,
                    trql_gap: max_gap,
                    trql_oid_recall: sql.metrics.recall,
                    trql_oid_precision: sql.metrics.precision,
                    trql_iou: sql.metrics.mean_iou,
                    trql_selection_ms: stages.map(|s| s.selection_ms),
                    trql_poly_ms: stages.map(|s| s.poly_ms),
                    trql_lift_ms: stages.map(|s| s.lift_ms),
                    trql_temporal_ms: stages.map(|s| s.temporal_ms),
                    trql_other_ms: stages.map(|s| s.other_ms),
                    trql_peak_mem_mb: Some(peak_mem_mb),
                    sql_peak_mem_mb: Some(sql.peak_mem_mb),
                    sql_peak_rss_mb: Some(sql.peak_rss_mb),
                    trql_segments: Some(report.db_stats.total_segments),
                    trql_poly_segments: Some(report.db_stats.poly_segments),
                    trql_direct_segments: Some(report.db_stats.direct_segments),
                    scale: db_repeat,
                    // Predictions record the configuration that made them in
                    // their metadata; human annotation has no such field.
                    detector: sub_dataset
                        .metadata
                        .as_ref()
                        .and_then(|m| m.detector.clone())
                        .unwrap_or_else(|| "GT".to_owned()),
                    gt_oid_precision: gt_scores.map(|(p, _, _)| p),
                    gt_oid_recall: gt_scores.map(|(_, r, _)| r),
                    gt_iou: gt_scores.map(|(_, _, i)| i),
                    sql_gt_precision: sql.gt_metrics.as_ref().map(|m| m.precision),
                    sql_gt_recall: sql.gt_metrics.as_ref().map(|m| m.recall),
                    sql_gt_iou: sql.gt_metrics.as_ref().map(|m| m.mean_iou),
                },
            )?;

            // The exactness contract. With no fitting error and no gap
            // smoothing the engine represents the trajectory exactly and breaks
            // segments at every missing frame -- which is precisely the SQL's
            // island semantics. The two must then agree completely, so a
            // disagreement here is a bug in one of them, not a modelling
            // difference, and is worth stopping for rather than recording.
            if max_error_px == 0.0 && max_gap == 1 {
                let m = &sql.metrics;
                let exact = m.precision == 1.0
                    && m.recall == 1.0
                    && m.interval_mismatches == 0
                    && m.fp == 0
                    && m.fn_ == 0;
                if !exact {
                    anyhow::bail!(
                        "exactness check FAILED at max_error_px=0, max_gap=1 -- engine and \
                         compiled SQL must agree exactly here.\n  \
                         engine tuples {} | sql tuples {} | both {}\n  \
                         precision {:.6} recall {:.6} mean IoU {:.6}\n  \
                         tuples only in engine {} | only in sql {} | \
                         matched tuples with differing intervals {}\n  \
                         sample engine-only: {:?}\n  sample sql-only: {:?}\n\
                         One of the two is wrong: the engine, or the supplied SQL. \
                         Resolve it before trusting any numbers from this run.",
                        m.query_tuples, m.gt_tuples, m.tp,
                        m.precision, m.recall, m.mean_iou,
                        m.fp, m.fn_, m.interval_mismatches,
                        m.sample_fp, m.sample_fn,
                    );
                }
            }
        }

        // Persist the per-run report as JSON in the sub-dataset's `benchmark/`
        // folder (alongside its `.trql` files), named by query id and params.
        let out_dir = sub_dataset.dir().join("benchmark");
        std::fs::create_dir_all(&out_dir)
            .with_context(|| format!("failed to create {}", out_dir.display()))?;
        let out = out_dir.join(format!(
            "{query_id}_{max_gap}_{max_error_px}_x{db_repeat}.json"
        ));
        let file =
            File::create(&out).with_context(|| format!("failed to write {}", out.display()))?;
        serde_json::to_writer_pretty(file, &report)
            .with_context(|| format!("failed to write {}", out.display()))?;

        ui.emit(&report, |ui, r| {
            ui.table(
                &["engine", "rows", "avg time (ms)", "peak mem (mb)"],
                vec![vec![
                    ENGINE.to_owned(),
                    group_digits(r.rows as u64),
                    format!("{:.1}", r.avg_ms),
                    format!("{:.0}", r.peak_mem_mb),
                ]],
            );
            ui.section("Details");
            ui.kv("query", &r.query_id);
            ui.kv("intervals", r.interval_repr);
            ui.kv("scale", format!("x{}", r.scale));
            ui.kv("objects", group_digits(r.objects as u64));
            ui.kv("avg segments", format!("{:.2}", r.db_stats.avg_segments));
            ui.kv("storage", format!("{} MB", r.storage_mb));
            ui.kv("runs", r.times_ms.len());
            ui.kv("report", out.display());
            if let Some(db) = &r.results_db {
                ui.kv("results db", db.display());
            }

            if let Some(st) = &r.stages {
                ui.section("Query-time breakdown (per run)");
                let total =
                    st.selection_ms + st.poly_ms + st.lift_ms + st.temporal_ms + st.other_ms;
                let pct = |v: f64| if total > 0.0 { v / total * 100.0 } else { 0.0 };
                ui.table(
                    &["stage", "ms", "share"],
                    vec![
                        vec![
                            "object selection".to_owned(),
                            format!("{:.2}", st.selection_ms),
                            format!("{:.1}%", pct(st.selection_ms)),
                        ],
                        vec![
                            "polynomial solving".to_owned(),
                            format!("{:.2}", st.poly_ms),
                            format!("{:.1}%", pct(st.poly_ms)),
                        ],
                        vec![
                            "interval lifting".to_owned(),
                            format!("{:.2}", st.lift_ms),
                            format!("{:.1}%", pct(st.lift_ms)),
                        ],
                        vec![
                            "temporal relations".to_owned(),
                            format!("{:.2}", st.temporal_ms),
                            format!("{:.1}%", pct(st.temporal_ms)),
                        ],
                        vec![
                            "other (residual)".to_owned(),
                            format!("{:.2}", st.other_ms),
                            format!("{:.1}%", pct(st.other_ms)),
                        ],
                    ],
                );
                ui.kv("instrumented total", format!("{total:.2} ms/run"));
            }

            if let Some(s) = &r.sql {
                ui.section("Engine vs compiled SQL");
                // The two memory figures are different measurements — the engine's
                // is process-wide and includes its resident `.trql`, DuckDB's is
                // the growth its query caused. Labelled, not silently columned
                // together, because subtracting one from the other is meaningless.
                ui.table(
                    &["system", "query time (ms)", "tuples", "peak mem (mb)"],
                    vec![
                        vec![
                            ENGINE.to_owned(),
                            format!("{:.1}", r.avg_ms),
                            group_digits(s.engine_tuples as u64),
                            format!("{:.0} (total RSS)", r.peak_mem_mb),
                        ],
                        vec![
                            "DuckDB (1 thread)".to_owned(),
                            format!("{:.1}", s.elapsed_ms),
                            group_digits(s.sql_tuples as u64),
                            format!("{:.0} (duckdb_memory)", s.peak_mem_mb),
                        ],
                    ],
                );
                ui.kv("oid precision", format!("{:.6}", s.precision));
                ui.kv("oid recall", format!("{:.6}", s.recall));
                ui.kv("mean IoU", format!("{:.6} (over matched)", s.mean_iou));
                ui.kv(
                    "interval mismatches",
                    format!("{} of {} matched", s.interval_mismatches, s.matched_tuples),
                );
                if s.exact {
                    ui.success("engine and SQL agree exactly".to_owned());
                } else {
                    ui.warn("engine and SQL DISAGREE".to_owned());
                }
            }

            if let Some(m) = &r.metrics {
                ui.section("Scores");
                ui.kv("precision", format!("{:.4}", m.precision));
                ui.kv("recall", format!("{:.4}", m.recall));
                ui.kv("f1", format!("{:.4}", m.f1));
                ui.kv("mean IoU", format!("{:.4} (over matched)", m.mean_iou));
                ui.kv("tp/fp/fn", format!("{} / {} / {}", m.tp, m.fp, m.fn_));
            }
        })
    }

    /// Run the supplied SQL against the sub-dataset's own `duckdb.db` and
    /// compare it to the engine's answer.
    ///
    /// The SQL is *given*, not generated. Generating it from the same algebra the
    /// engine evaluates only ever checks that the transcription is faithful; a
    /// query written independently against the language's denotation is what can
    /// catch an error in the formulation itself.
    ///
    /// The engine is re-run once here rather than reusing a timed run's rows, so
    /// the timed loop stays free of result-retention cost. Its object indices are
    /// mapped back to database `oid`s before comparing — see [`sqlrun`].
    ///
    /// SQL is treated as the reference: `recall` is the fraction of SQL's tuples
    /// the engine also found, `precision` the fraction of the engine's that SQL
    /// agrees with. Both sides are merged the same way before comparing, so the
    /// SQL need not merge overlapping spans itself.
    #[allow(clippy::too_many_arguments)]
    fn run_paired_sql(
        &self,
        sub_dataset: &SubDataset,
        sql_path: &Path,
        tiled: Option<&Path>,
        db: &Trqs,
        constraints: &trql::plan::VariablesWithConstraints,
        expr: &trql::expr::TrajExpr,
        gt: Option<&QueryOutput>,
    ) -> anyhow::Result<SqlPairing> {
        // Against a tiled run the SQL reads the tiled copy, so both sides see the
        // same database — including `engine_oids`, whose dense-index mapping must
        // come from whichever database the engine actually loaded.
        let duckdb_path = tiled.unwrap_or(&sub_dataset.duckdb_path);
        let sql = std::fs::read_to_string(sql_path)
            .with_context(|| format!("failed to read SQL from {}", sql_path.display()))?;
        let num_slots = constraints.num_slots();

        let sql_run = sqlrun::run(duckdb_path, &sql, num_slots)?;
        let traj = sqlrun::traj_stats(duckdb_path)?;
        let oids = sqlrun::engine_oids(duckdb_path)?;

        let ctx = AlgebraCtx {
            db,
            poly_trajs: Some(db.poly_trajs()),
        };
        let engine: QueryResult = ctx.run(constraints, expr).into_iter().collect();
        let engine_rows = engine.into_iter().map(|(tuple, ivs)| {
            let mapped = tuple
                .iter()
                .map(|&i| oids.get(i as usize).copied().unwrap_or(i))
                .collect::<Vec<u32>>();
            (mapped, ivs.maximal())
        });

        let produced = QueryOutput::from_rows(engine_rows);
        let reference = QueryOutput::from_rows(sql_run.rows.clone().into_iter());
        let metrics = produced.compare(&reference);

        // SQL's own answer scored against ground truth, when one was supplied.
        //
        // Ground truth is written in the engine's *dense* object indices, so the
        // SQL rows — which carry database `oid`s — are mapped back through the
        // inverse of `oids` before comparing. Comparing the two id spaces
        // directly would report a disagreement on every row while both describe
        // the same events.
        let gt_metrics = gt.map(|gt| {
            let to_index: std::collections::HashMap<u32, u32> = oids
                .iter()
                .enumerate()
                .map(|(i, &oid)| (oid, i as u32))
                .collect();
            let mapped = sql_run.rows.iter().map(|(tuple, ivs)| {
                let t = tuple
                    .iter()
                    .map(|o| to_index.get(o).copied().unwrap_or(*o))
                    .collect::<Vec<u32>>();
                (t, ivs.clone())
            });
            QueryOutput::from_rows(mapped).compare(gt)
        });

        Ok(SqlPairing {
            sql,
            gt_metrics,
            elapsed_ms: sql_run.elapsed_ms,
            peak_mem_mb: sql_run.peak_mem_mb,
            peak_rss_mb: sql_run.peak_rss_mb,
            raw_rows: sql_run.raw_rows,
            traj,
            metrics,
        })
    }

    fn compare(&self, query: &str, gt: &Path) -> anyhow::Result<()> {
        let ui = &self.ui;

        // Ground truth first, so a bad CSV fails before the query runs.
        let gt_output = QueryOutput::from_csv(gt)?;

        let sub_dataset = self.load_sub_dataset()?;
        let db = self.load_database_with_bar(&sub_dataset, None)?;
        let (constraints, expr) = parse_query(query, db.schema())?;
        let ctx = AlgebraCtx {
            db: &db,
            poly_trajs: None,
        };
        let query_output = QueryOutput::from_rows(
            ctx.run(&constraints, &expr)
                .into_iter()
                .map(|(tuple, intervals)| (tuple.to_vec(), intervals.maximal())),
        );

        let report = CompareReport {
            query: query.to_owned(),
            gt: gt.to_path_buf(),
            metrics: query_output.compare(&gt_output),
        };

        ui.emit(&report, |ui, r| {
            let m = &r.metrics;
            ui.section("Tuples");
            ui.kv("query", group_digits(m.query_tuples as u64));
            ui.kv("ground truth", group_digits(m.gt_tuples as u64));
            ui.kv("matched (TP)", group_digits(m.tp as u64));
            ui.kv("false pos", group_digits(m.fp as u64));
            ui.kv("false neg", group_digits(m.fn_ as u64));

            ui.section("Scores");
            ui.kv("precision", format!("{:.4}", m.precision));
            ui.kv("recall", format!("{:.4}", m.recall));
            ui.kv("f1", format!("{:.4}", m.f1));
            ui.kv("mean IoU", format!("{:.4} (over matched)", m.mean_iou));

            for (label, sample, total) in [
                ("false positives", &m.sample_fp, m.fp),
                ("false negatives", &m.sample_fn, m.fn_),
            ] {
                if !sample.is_empty() {
                    ui.section(&format!("Sample {label} ({}/{total})", sample.len()));
                    for tuple in sample {
                        ui.kv("", format!("{tuple:?}"));
                    }
                }
            }
        })
    }
}
