//! `trql-cli detect`: run a detector+tracker over sub-dataset videos, align the
//! predicted track ids to ground truth, and score the result.
//!
//! The same pipeline the TUI's detect action drives — both call
//! [`trql_storage::detect`], [`trql_storage::align`] and [`trql_storage::eval`],
//! so a detection made here is the same detection. What this adds is batching: a
//! hundred DETRAC sequences is a shell loop rather than a hundred keystrokes, and
//! the command that produced a paper's numbers can be written down.
//!
//! Detection is not reproducible the way the rest of the benchmark is — it needs
//! a GPU, weights Ultralytics downloads on first use, and a Python environment
//! outside this workspace. So the work is a subprocess, and everything this
//! module does around it (choosing sub-datasets, laying out outputs, scoring,
//! reporting) stays on the Rust side where it can be tested and re-run.

use std::path::{Path, PathBuf};

use anyhow::Context as _;
use serde::Serialize;
use trql::storage::{
    align::{self, AlignStats},
    detect::{self, DetectConfig, DetectEvent, DetectJob, SourceInfo},
    eval::{self, Metrics},
};

/// One sub-dataset's detection, from video to score.
#[derive(Serialize)]
pub struct JobReport {
    /// Sub-dataset the video came from, relative to the tree root.
    pub source: String,
    pub out_dir: PathBuf,
    /// Tracks the detector produced, before alignment.
    pub tracks: u32,
    /// How many of them were matched to a ground-truth object.
    pub matched: Option<u64>,
    /// Detection quality against ground truth — localisation only; see
    /// [`trql_storage::eval`] for why classification is excluded.
    pub metrics: Option<Metrics>,
    /// Why this sub-dataset produced nothing, when it did not.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// What `detect run` did across every selected sub-dataset.
///
/// Every knob is recorded, not just the ones overridden: a report read months
/// later should say what was run without anyone decoding the config id or
/// knowing which preset's defaults were in force at the time.
#[derive(Serialize)]
pub struct DetectReport {
    pub config: String,
    pub model: String,
    pub tracker: String,
    pub conf: f64,
    pub imgsz: u32,
    pub half: bool,
    /// COCO class ids kept; empty means every class.
    pub classes: Vec<u32>,
    pub iou_threshold: f64,
    pub jobs: Vec<JobReport>,
}

/// Command-line overrides applied on top of a preset.
///
/// The presets are the combinations someone thought worth compiling in; finding
/// a good configuration means trying ones nobody did. Any field left `None`
/// keeps the preset's value.
#[derive(Default, Clone, Debug)]
pub struct Overrides {
    pub model: Option<String>,
    pub tracker: Option<String>,
    pub conf: Option<f64>,
    pub imgsz: Option<u32>,
    pub classes: Option<Vec<u32>>,
    /// `(from, to)` class renames applied to the predictions.
    pub aliases: Vec<(String, String)>,
}

impl Overrides {
    pub fn is_empty(&self) -> bool {
        self.model.is_none()
            && self.tracker.is_none()
            && self.conf.is_none()
            && self.imgsz.is_none()
            && self.classes.is_none()
            && self.aliases.is_empty()
    }

    /// `base` with the overrides applied, and an `id` that names the result.
    ///
    /// The id becomes the output directory, so it has to distinguish one variant
    /// from another — otherwise a sweep silently overwrites itself and every run
    /// but the last is lost.
    pub fn apply(&self, base: &DetectConfig) -> DetectConfig {
        let mut cfg = base.clone();
        let mut parts: Vec<String> = Vec::new();
        if let Some(m) = &self.model {
            parts.push(format!("m{}", m.trim_end_matches(".pt")));
            cfg.model = m.clone();
        }
        if let Some(t) = &self.tracker {
            parts.push(format!("t{}", t.trim_end_matches(".yaml")));
            cfg.tracker = t.clone();
        }
        if let Some(c) = self.conf {
            parts.push(format!("conf{c}"));
            cfg.conf = c;
        }
        if let Some(i) = self.imgsz {
            parts.push(format!("sz{i}"));
            cfg.imgsz = i;
        }
        for (from, to) in &self.aliases {
            parts.push(format!("as{from}.{to}"));
        }
        if let Some(cl) = &self.classes {
            parts.push(format!(
                "cls{}",
                cl.iter().map(u32::to_string).collect::<Vec<_>>().join(".")
            ));
            cfg.classes = cl.clone();
        }
        if !parts.is_empty() {
            cfg.id = format!("{}@{}", base.id, parts.join("-"));
            cfg.name = format!("{} [{}]", base.name, parts.join(" "));
        }
        cfg
    }
}

/// The preset named `id`, or an error listing what there is.
pub fn config_by_id(id: &str) -> anyhow::Result<DetectConfig> {
    let configs = detect::predefined_configs();
    configs
        .iter()
        .find(|c| c.id == id)
        .cloned()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "unknown detect config {id:?}; known: {}",
                configs
                    .iter()
                    .map(|c| c.id.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
}

/// Where a sub-dataset's predictions for `config` live.
///
/// One directory per (source, config) so configs coexist instead of overwriting
/// each other, and `predicted/` groups apart from the ground-truth families —
/// the same layout the TUI writes, so either front end can read the other's work.
pub fn out_dir(data_dir: &Path, rel_path: &Path, config: &DetectConfig) -> PathBuf {
    data_dir.join("predicted").join(rel_path).join(&config.id)
}

/// Build the job for one sub-dataset, or `None` when it has no single video to
/// run on (a merged sub-dataset has a `videos/` directory instead).
pub fn job_for(
    data_dir: &Path,
    entry: &trql::storage::datasets::SubDataset,
    config: &DetectConfig,
) -> Option<DetectJob> {
    let video = entry.dir().join("video.mp4");
    if !video.is_file() {
        return None;
    }
    let meta = entry.metadata.as_ref();
    Some(DetectJob {
        key: entry.duckdb_path.clone(),
        label: entry.rel_path.display().to_string(),
        video_path: video,
        out_dir: out_dir(data_dir, &entry.rel_path, config),
        // The sub-dataset's own database is the ground truth to score against.
        gt_db: entry.duckdb_path.clone(),
        source: SourceInfo {
            rel_path: entry.rel_path.display().to_string(),
            fps: meta.and_then(|m| m.fps),
            width: meta.and_then(|m| m.width),
            height: meta.and_then(|m| m.height),
            num_frames: meta.and_then(|m| m.num_frames),
            duration_s: meta.and_then(|m| m.duration_s),
        },
    })
}

/// Rename predicted classes, so a query can name one class and run on both sides.
///
/// The detector speaks COCO; a dataset speaks its own vocabulary. Where the two
/// happen to coincide (`car`) nothing is needed, but MOT17 calls COCO's `person`
/// a `pedestrian`, and without a rename no single query text runs against both
/// ground truth and predictions — which is the whole basis of the accuracy
/// comparison.
///
/// A rename, not a remap: it asserts the two labels denote the same thing, which
/// is the caller's claim to make and is recorded in the run's configuration id.
pub fn apply_class_aliases(pred_db: &Path, aliases: &[(String, String)]) -> anyhow::Result<()> {
    if aliases.is_empty() {
        return Ok(());
    }
    let conn = trql::storage::duckdb_meta::open_read_write(pred_db)
        .with_context(|| format!("opening {} to rename classes", pred_db.display()))?;
    for (from, to) in aliases {
        conn.execute("UPDATE objects SET class = ?2 WHERE class = ?1", [from, to])
            .with_context(|| format!("renaming class {from:?} to {to:?}"))?;
    }
    Ok(())
}

/// Detect, align and score one job.
///
/// Alignment runs before scoring because a tracker numbers its tracks
/// arbitrarily: without renumbering them onto the ground truth's ids, a
/// perfectly-placed box still belongs to the "wrong" object and every
/// per-object comparison downstream is meaningless.
pub fn run_job(
    python: &str,
    script: &Path,
    job: &DetectJob,
    config: &DetectConfig,
    iou: f64,
    aliases: &[(String, String)],
    progress: &mut dyn FnMut(DetectEvent),
) -> anyhow::Result<(u32, AlignStats, Metrics)> {
    let tracks = detect::run_detect_job(python, script, job, config, progress)?;
    let pred_db = job.out_dir.join("duckdb.db");
    apply_class_aliases(&pred_db, aliases)?;
    let stats = align::align_to_file(&job.gt_db, &pred_db, &job.out_dir, iou)
        .with_context(|| format!("aligning {} to ground truth", job.label))?;
    let metrics = eval::evaluate_to_file(&job.gt_db, &pred_db, &job.out_dir, iou)
        .with_context(|| format!("scoring {}", job.label))?;
    Ok((tracks, stats, metrics))
}
