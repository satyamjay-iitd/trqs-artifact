use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::mpsc,
    thread,
};

use trql::storage::merge::{self, MergeEvent, MergeSource};
use trql::storage::{
    datasets::{Datasets, SubDataset},
    duckdb_meta,
};

use crate::trql_stats::{self};
use trql::storage::{
    align,
    detect::{self, DetectConfig, DetectEvent, DetectJob, SourceInfo},
    eval,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
    List,
    Detail,
    Schema,
    Log,
    Peek,
    Help,
    DetectMenu,
}

/// Which column of the list view currently has keyboard focus.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pane {
    Dataset,
    Subdataset,
}

/// Build state of the `.trql` for the parameters the user currently has
/// selected. Derived from a sub-dataset's built artifacts, not stored on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrqlStatus {
    Missing,
    Generated,
    /// Older than the source `duckdb.db`, so it should be rebuilt.
    Stale,
}

impl TrqlStatus {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Missing => "missing",
            Self::Generated => "generated",
            Self::Stale => "stale",
        }
    }
}

/// Per-subdataset conversion status during a bulk run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvItemStatus {
    Pending,
    Running,
    Done,
    Failed,
}

/// A single conversion unit handed to the worker thread.
struct ConvJob {
    key: PathBuf,
    label: String,
    sub_dataset: SubDataset,
    max_gap: u32,
    max_error_px: f64,
}

/// What kind of background job a [`ConversionState`] is tracking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobKind {
    Convert,
    Merge,
    Detect,
}

impl JobKind {
    /// Present-tense verb for progress text ("Converting" / …).
    pub fn verb(&self) -> &'static str {
        match self {
            Self::Convert => "Converting",
            Self::Merge => "Merging",
            Self::Detect => "Detecting",
        }
    }

    /// Past-tense noun for the completion summary.
    pub fn noun(&self) -> &'static str {
        match self {
            Self::Convert => "Conversion",
            Self::Merge => "Merge",
            Self::Detect => "Detection",
        }
    }
}

/// Progress messages sent from the worker thread back to the UI.
enum ConvUpdate {
    Started {
        key: PathBuf,
        label: String,
    },
    Finished {
        key: PathBuf,
        label: String,
        result: Result<u32, String>,
    },
    /// Frame-level progress for the in-flight item (used by detection).
    Progress {
        current: String,
    },
    /// A free-form line to append to the log (used for merge summaries/errors).
    Log(String),
    Done,
}

/// Live state of an in-progress (or just-finished) background job.
pub struct ConversionState {
    rx: mpsc::Receiver<ConvUpdate>,
    pub kind: JobKind,
    pub total: usize,
    pub completed: usize,
    pub succeeded: usize,
    pub failed: usize,
    pub current: Option<String>,
    statuses: HashMap<PathBuf, ConvItemStatus>,
}

/// Run one conversion job (inspect + convert) on the worker thread. Always
/// rebuilds so a stale `.trql` is refreshed; the TUI reports progress per item
/// through its own [`ConvUpdate`] channel, so no per-phase callback is used.
fn run_job(job: &ConvJob) -> Result<u32, String> {
    job.sub_dataset
        .rebuild_trql(job.max_gap, job.max_error_px, &mut |_| {})
        .map_err(|e| format!("{e:#}"))
}

pub struct App {
    pub data_dir: PathBuf,
    pub max_gap: u32,
    pub max_error_px: f64,
    /// The dataset tree: sub-datasets plus the families grouping them.
    pub datasets: Datasets,
    /// Resolved flat index into `datasets.entries()` for the current selection.
    pub selected: usize,
    pub selected_group: usize,
    pub selected_sub: usize,
    pub focus: Pane,
    pub view: View,
    pub log: Vec<String>,
    /// Marked subdatasets for bulk actions, keyed by their `duckdb.db` path so
    /// the marks survive a `refresh()` that re-indexes `entries`.
    pub marked: HashSet<PathBuf>,
    /// In-progress bulk conversion, if any.
    pub conversion: Option<ConversionState>,
    /// Python interpreter used to run the detect pipeline.
    pub python: String,
    /// Path to the detect+track pipeline script.
    pub detect_script: PathBuf,
    /// Predefined detector+tracker configurations the user can pick from.
    pub detect_configs: Vec<DetectConfig>,
    /// Cursor into `detect_configs` while the detect menu is open.
    pub detect_menu_sel: usize,
}

impl App {
    pub fn load(
        data_dir: PathBuf,
        max_gap: u32,
        max_error_px: f64,
        python: String,
        detect_script: PathBuf,
    ) -> anyhow::Result<Self> {
        let mut app = Self {
            data_dir,
            max_gap,
            max_error_px,
            datasets: Datasets::default(),
            selected: 0,
            selected_group: 0,
            selected_sub: 0,
            focus: Pane::Dataset,
            view: View::List,
            log: Vec::new(),
            marked: HashSet::new(),
            conversion: None,
            python,
            detect_script,
            detect_configs: detect::predefined_configs(),
            detect_menu_sel: 0,
        };
        app.refresh()?;
        app.inspect_selected();
        Ok(app)
    }

    pub fn refresh(&mut self) -> anyhow::Result<()> {
        self.datasets = Datasets::load(&self.data_dir)?;
        self.clamp_selection();
        self.log
            .push(format!("Loaded {} dataset entries", self.entries().len()));
        Ok(())
    }

    /// All sub-datasets, flat and sorted by relative path.
    pub fn entries(&self) -> &[SubDataset] {
        self.datasets.entries()
    }

    /// Clamp group/sub cursors to valid ranges and recompute `selected`.
    fn clamp_selection(&mut self) {
        let groups = self.datasets.datasets();
        if groups.is_empty() {
            self.selected_group = 0;
            self.selected_sub = 0;
            self.selected = 0;
            return;
        }
        self.selected_group = self.selected_group.min(groups.len() - 1);
        let group = &groups[self.selected_group];
        self.selected_sub = self.selected_sub.min(group.entries.len().saturating_sub(1));
        self.selected = group.entries[self.selected_sub];
    }

    /// The entries belonging to the currently selected dataset group.
    pub fn current_group_entries(&self) -> &[usize] {
        self.datasets.dataset_entries(self.selected_group)
    }

    pub fn inspect_selected(&mut self) {
        self.inspect_index(self.selected);
    }

    /// Lazily load DuckDB metadata for the entry at `idx`, if not already done.
    fn inspect_index(&mut self, idx: usize) {
        let Some(entry) = self.datasets.entry_mut(idx) else {
            return;
        };
        if entry.duckdb.is_some() {
            return;
        }
        match duckdb_meta::inspect_duckdb(&entry.duckdb_path, entry.metadata.as_ref()) {
            Ok(meta) => {
                entry.duckdb = Some(meta);
                entry.error = None;
            }
            Err(err) => {
                entry.error = Some(format!("{err:#}"));
            }
        }
    }

    /// Whether the given entry is currently marked for bulk actions.
    pub fn is_marked(&self, entry: &SubDataset) -> bool {
        self.marked.contains(&entry.duckdb_path)
    }

    /// Number of subdatasets currently marked.
    pub fn marked_count(&self) -> usize {
        self.marked.len()
    }

    /// Toggle the mark on the currently selected subdataset.
    pub fn toggle_mark(&mut self) {
        let Some(entry) = self.datasets.entry(self.selected) else {
            return;
        };
        let key = entry.duckdb_path.clone();
        if !self.marked.remove(&key) {
            self.marked.insert(key);
        }
    }

    /// Toggle marks on every subdataset in the current group: if all are
    /// already marked, clear them; otherwise mark them all.
    pub fn toggle_mark_group(&mut self) {
        let keys: Vec<PathBuf> = self
            .current_group_entries()
            .iter()
            .map(|&idx| self.entries()[idx].duckdb_path.clone())
            .collect();
        if keys.is_empty() {
            return;
        }
        let all_marked = keys.iter().all(|k| self.marked.contains(k));
        for key in keys {
            if all_marked {
                self.marked.remove(&key);
            } else {
                self.marked.insert(key);
            }
        }
    }

    /// Clear all marks.
    pub fn clear_marks(&mut self) {
        self.marked.clear();
    }

    pub fn selected_entry(&self) -> Option<&SubDataset> {
        self.datasets.entry(self.selected)
    }

    pub fn select_next(&mut self) {
        match self.focus {
            Pane::Dataset => self.select_group(self.selected_group + 1),
            // Wrap around to the top once past the last sub-dataset.
            Pane::Subdataset => {
                let len = self.current_group_entries().len();
                if len > 0 {
                    self.select_sub((self.selected_sub + 1) % len);
                }
            }
        }
    }

    pub fn select_prev(&mut self) {
        match self.focus {
            Pane::Dataset => self.select_group(self.selected_group.saturating_sub(1)),
            // Wrap around to the bottom when stepping up past the first.
            Pane::Subdataset => {
                let len = self.current_group_entries().len();
                if len > 0 {
                    self.select_sub((self.selected_sub + len - 1) % len);
                }
            }
        }
    }

    /// Move focus to the dataset (left) pane.
    pub fn focus_dataset(&mut self) {
        self.focus = Pane::Dataset;
    }

    /// Move focus to the subdataset (right) pane, if the group is non-empty.
    pub fn focus_subdataset(&mut self) {
        if !self.current_group_entries().is_empty() {
            self.focus = Pane::Subdataset;
        }
    }

    /// Open the Schema view, computing `.trql` stats for the selection first.
    pub fn open_schema(&mut self) {
        self.ensure_trql_stats();
        self.view = View::Schema;
    }

    /// Load and cache `.trql` statistics for the selected entry, if a built
    /// `.trql` exists for the current parameters and it isn't already cached.
    fn ensure_trql_stats(&mut self) {
        let Some(entry) = self.datasets.entry(self.selected) else {
            return;
        };
        if entry.trql_stats.is_some() || self.trql_status(entry) == TrqlStatus::Missing {
            return;
        }
        let path = self.trql_path(entry);
        match trql_stats::compute(&path) {
            Ok(stats) => self.datasets.entries_mut()[self.selected].trql_stats = Some(stats),
            Err(err) => self.log.push(format!("trql stats failed: {err:#}")),
        }
    }

    /// The `.trql` filename for `entry` under the current parameters.
    pub fn trql_name(&self, entry: &SubDataset) -> String {
        entry.trql_name(self.max_gap, self.max_error_px)
    }

    /// Where the current parameters' `.trql` for `entry` lives (built or not).
    pub fn trql_path(&self, entry: &SubDataset) -> PathBuf {
        entry.trql_path(self.max_gap, self.max_error_px)
    }

    /// Build state of the current parameters' `.trql` for `entry`, read off the
    /// artifacts discovered at load time.
    pub fn trql_status(&self, entry: &SubDataset) -> TrqlStatus {
        match entry.trql(self.max_gap, self.max_error_px) {
            Some(p) if p.stale => TrqlStatus::Stale,
            Some(_) => TrqlStatus::Generated,
            None => TrqlStatus::Missing,
        }
    }

    /// Adjust `max_gap` by `delta` (clamped at 0) and re-point trql targets.
    pub fn adjust_max_gap(&mut self, delta: i64) {
        let new = (self.max_gap as i64 + delta).max(0) as u32;
        if new != self.max_gap {
            self.max_gap = new;
            self.drop_cached_stats();
        }
    }

    /// Adjust `max_error_px` by `delta` (clamped at 0) and re-point trql targets.
    pub fn adjust_max_error(&mut self, delta: f64) {
        // Round to 2 decimals to avoid float drift like 0.30000000000000004.
        let new = ((self.max_error_px + delta).max(0.0) * 100.0).round() / 100.0;
        if new != self.max_error_px {
            self.max_error_px = new;
            self.drop_cached_stats();
        }
    }

    /// The `.trql` target changed, so stats cached for the old one are stale.
    fn drop_cached_stats(&mut self) {
        for entry in self.datasets.entries_mut() {
            entry.trql_stats = None;
        }
    }

    fn select_group(&mut self, idx: usize) {
        let len = self.datasets.datasets().len();
        if len == 0 {
            return;
        }
        self.selected_group = idx.min(len - 1);
        self.selected_sub = 0;
        self.clamp_selection();
        self.inspect_selected();
    }

    fn select_sub(&mut self, idx: usize) {
        let len = self.current_group_entries().len();
        if len == 0 {
            return;
        }
        self.selected_sub = idx.min(len - 1);
        self.clamp_selection();
        self.inspect_selected();
    }

    /// Start converting just the highlighted subdataset.
    pub fn convert_selected(&mut self) {
        if let Some(job) = self.job_for(self.selected) {
            self.start_conversion(vec![job]);
        }
    }

    /// Start converting every marked subdataset; falls back to the current
    /// selection when nothing is marked.
    pub fn convert_marked(&mut self) {
        if self.marked.is_empty() {
            self.convert_selected();
            return;
        }
        let jobs: Vec<ConvJob> = (0..self.entries().len())
            .filter(|&idx| self.marked.contains(&self.entries()[idx].duckdb_path))
            .filter_map(|idx| self.job_for(idx))
            .collect();
        self.start_conversion(jobs);
    }

    /// Merge every marked subdataset into a new dataset under
    /// `<data_dir>/merged/<name>`, keeping `oid`/`fid` continuous and
    /// symlinking (not copying) each source video.
    pub fn merge_marked(&mut self) {
        let sources: Vec<MergeSource> = (0..self.entries().len())
            .filter(|&idx| self.marked.contains(&self.entries()[idx].duckdb_path))
            .map(|idx| self.merge_source_for(idx))
            .collect();

        if sources.len() < 2 {
            self.log
                .push("Merge needs at least 2 marked subdatasets".to_owned());
            self.view = View::Log;
            return;
        }
        if self.conversion.is_some() {
            self.log.push("A job is already in progress".to_owned());
            self.view = View::Log;
            return;
        }

        let out_dir = self.unique_merge_dir(&sources);
        let total = sources.len();
        let statuses = sources
            .iter()
            .map(|s| (s.key.clone(), ConvItemStatus::Pending))
            .collect();

        let (tx, rx) = mpsc::channel();
        let worker_out = out_dir.clone();
        thread::spawn(move || {
            let mut on_event = |ev: MergeEvent| {
                let _ = match ev {
                    MergeEvent::Started { key, label } => {
                        tx.send(ConvUpdate::Started { key, label })
                    }
                    MergeEvent::Finished { key, label, result } => {
                        tx.send(ConvUpdate::Finished { key, label, result })
                    }
                };
            };
            match merge::merge_datasets(&sources, &worker_out, &mut on_event) {
                Ok(summary) => {
                    let _ = tx.send(ConvUpdate::Log(format!(
                        "Merged {} datasets ({} objects, {} frames) into {}",
                        summary.num_sources,
                        summary.num_objs,
                        summary.total_frames,
                        summary.out_dir.display()
                    )));
                }
                Err(err) => {
                    let _ = tx.send(ConvUpdate::Log(format!("Merge failed: {err:#}")));
                }
            }
            let _ = tx.send(ConvUpdate::Done);
        });

        self.log.push(format!(
            "Started merge of {total} subdataset(s) into {}",
            out_dir.display()
        ));
        self.conversion = Some(ConversionState {
            rx,
            kind: JobKind::Merge,
            total,
            completed: 0,
            succeeded: 0,
            failed: 0,
            current: None,
            statuses,
        });
        self.view = View::Log;
    }

    fn merge_source_for(&self, idx: usize) -> MergeSource {
        let entry = &self.entries()[idx];
        let video = entry.dir().join("video.mp4");
        MergeSource {
            key: entry.duckdb_path.clone(),
            label: entry.rel_path.display().to_string(),
            duckdb_path: entry.duckdb_path.clone(),
            video_path: video.is_file().then_some(video),
            num_frames: entry.metadata.as_ref().and_then(|m| m.num_frames),
            metadata_path: entry.metadata_path.clone(),
        }
    }

    /// Pick a non-existing output directory `<data_dir>/merged/<name>`, where
    /// `name` is derived from the source names (deduplicated with a suffix).
    fn unique_merge_dir(&self, sources: &[MergeSource]) -> PathBuf {
        let base = self.data_dir.join("merged");
        let name = merge_dir_name(sources);
        let mut candidate = base.join(&name);
        let mut n = 2;
        while candidate.exists() {
            candidate = base.join(format!("{name}-{n}"));
            n += 1;
        }
        candidate
    }

    /// Open the detector-configuration menu, if there is anything to run on.
    /// Detection targets the marked subdatasets, falling back to the selection.
    pub fn open_detect_menu(&mut self) {
        if self.conversion.is_some() {
            self.log.push("A job is already in progress".to_owned());
            self.view = View::Log;
            return;
        }
        if self.detect_targets().is_empty() {
            self.log
                .push("No subdataset with a video.mp4 to detect on".to_owned());
            self.view = View::Log;
            return;
        }
        self.detect_menu_sel = self
            .detect_menu_sel
            .min(self.detect_configs.len().saturating_sub(1));
        self.view = View::DetectMenu;
    }

    /// Move the detect-menu cursor (wrapping), while the menu is open.
    pub fn detect_menu_move(&mut self, delta: i64) {
        let len = self.detect_configs.len();
        if len == 0 {
            return;
        }
        let cur = self.detect_menu_sel as i64;
        self.detect_menu_sel = (cur + delta).rem_euclid(len as i64) as usize;
    }

    /// Close the menu and start detection with the highlighted configuration.
    pub fn confirm_detect(&mut self) {
        let Some(config) = self.detect_configs.get(self.detect_menu_sel).cloned() else {
            return;
        };
        let jobs: Vec<DetectJob> = self
            .detect_targets()
            .into_iter()
            .filter_map(|idx| self.detect_job_for(idx, &config))
            .collect();
        self.start_detect(jobs, config);
    }

    /// How many subdatasets a detect run would target (for the menu header).
    pub fn detect_target_count(&self) -> usize {
        self.detect_targets().len()
    }

    /// Indices of subdatasets to run detection on: marked ones (with a video),
    /// or the current selection when nothing is marked.
    fn detect_targets(&self) -> Vec<usize> {
        let has_video = |idx: usize| self.entries()[idx].dir().join("video.mp4").is_file();
        let marked: Vec<usize> = (0..self.entries().len())
            .filter(|&idx| self.marked.contains(&self.entries()[idx].duckdb_path) && has_video(idx))
            .collect();
        if !marked.is_empty() {
            return marked;
        }
        match self.datasets.entry(self.selected) {
            Some(_) if has_video(self.selected) => vec![self.selected],
            _ => Vec::new(),
        }
    }

    fn detect_job_for(&self, idx: usize, config: &DetectConfig) -> Option<DetectJob> {
        let entry = self.datasets.entry(idx)?;
        let video = entry.dir().join("video.mp4");
        if !video.is_file() {
            return None;
        }
        let meta = entry.metadata.as_ref();
        Some(DetectJob {
            key: entry.duckdb_path.clone(),
            label: entry.rel_path.display().to_string(),
            video_path: video,
            // One directory per (source, config) so different configs coexist
            // instead of overwriting each other.
            out_dir: self
                .data_dir
                .join("predicted")
                .join(&entry.rel_path)
                .join(&config.id),
            // This dataset's own duckdb is the ground truth to score against.
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

    /// Spawn the worker that runs `jobs` sequentially under `config`.
    fn start_detect(&mut self, jobs: Vec<DetectJob>, config: DetectConfig) {
        if jobs.is_empty() {
            self.view = View::List;
            return;
        }
        if self.conversion.is_some() {
            self.log.push("A job is already in progress".to_owned());
            self.view = View::Log;
            return;
        }

        let total = jobs.len();
        let statuses = jobs
            .iter()
            .map(|j| (j.key.clone(), ConvItemStatus::Pending))
            .collect();

        let config_name = config.name.clone();
        let (tx, rx) = mpsc::channel();
        let python = self.python.clone();
        let script = self.detect_script.clone();
        thread::spawn(move || {
            for job in jobs {
                let _ = tx.send(ConvUpdate::Started {
                    key: job.key.clone(),
                    label: job.label.clone(),
                });
                let tx_progress = tx.clone();
                let label = job.label.clone();
                let mut on_event = |ev: DetectEvent| match ev {
                    DetectEvent::Progress { done, total } => {
                        let _ = tx_progress.send(ConvUpdate::Progress {
                            current: format!("{label}  {done}/{total} frames"),
                        });
                    }
                };
                let detect_res =
                    detect::run_detect_job(&python, &script, &job, &config, &mut on_event)
                        .map_err(|e| format!("{e:#}"));
                // On success: align predicted ids to the ground truth (renumber
                // continuously, store the pred->GT map), then score. Alignment
                // failure — e.g. non-continuous GT ids — fails the job.
                let final_res = match detect_res {
                    Ok(n) => {
                        let pred_db = job.out_dir.join("duckdb.db");
                        match align::align_to_file(&job.gt_db, &pred_db, &job.out_dir, 0.5) {
                            Ok(a) => {
                                let _ = tx.send(ConvUpdate::Log(format!(
                                    "{}: aligned ids — {}/{} tracks matched to GT (base {})",
                                    job.label, a.matched, a.pred_objects, a.base
                                )));
                                match eval::evaluate_to_file(
                                    &job.gt_db,
                                    &pred_db,
                                    &job.out_dir,
                                    0.5,
                                ) {
                                    Ok(m) => {
                                        let _ = tx.send(ConvUpdate::Log(format!(
                                            "{}: F1 {:.3}  P {:.3}  R {:.3}  (TP {} FP {} FN {})",
                                            job.label,
                                            m.f1,
                                            m.precision,
                                            m.recall,
                                            m.tp,
                                            m.fp,
                                            m.fn_
                                        )));
                                    }
                                    Err(e) => {
                                        let _ = tx.send(ConvUpdate::Log(format!(
                                            "{}: eval skipped: {e:#}",
                                            job.label
                                        )));
                                    }
                                }
                                Ok(n)
                            }
                            Err(e) => Err(format!("id alignment failed: {e:#}")),
                        }
                    }
                    Err(e) => Err(e),
                };
                let _ = tx.send(ConvUpdate::Finished {
                    key: job.key,
                    label: job.label,
                    result: final_res,
                });
            }
            let _ = tx.send(ConvUpdate::Done);
        });

        self.log.push(format!(
            "Started detection of {total} subdataset(s) with {config_name}"
        ));
        self.conversion = Some(ConversionState {
            rx,
            kind: JobKind::Detect,
            total,
            completed: 0,
            succeeded: 0,
            failed: 0,
            current: None,
            statuses,
        });
        self.view = View::Log;
    }

    fn job_for(&self, idx: usize) -> Option<ConvJob> {
        let entry = self.datasets.entry(idx)?;
        Some(ConvJob {
            key: entry.duckdb_path.clone(),
            label: entry.rel_path.display().to_string(),
            sub_dataset: entry.clone(),
            max_gap: self.max_gap,
            max_error_px: self.max_error_px,
        })
    }

    /// Spawn a worker thread to run `jobs`, tracking live progress.
    fn start_conversion(&mut self, jobs: Vec<ConvJob>) {
        if jobs.is_empty() {
            return;
        }
        if self.conversion.is_some() {
            self.log
                .push("A conversion is already in progress".to_owned());
            self.view = View::Log;
            return;
        }

        let total = jobs.len();
        let statuses = jobs
            .iter()
            .map(|j| (j.key.clone(), ConvItemStatus::Pending))
            .collect();

        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            for job in jobs {
                let _ = tx.send(ConvUpdate::Started {
                    key: job.key.clone(),
                    label: job.label.clone(),
                });
                let result = run_job(&job);
                let _ = tx.send(ConvUpdate::Finished {
                    key: job.key.clone(),
                    label: job.label.clone(),
                    result,
                });
            }
            let _ = tx.send(ConvUpdate::Done);
        });

        self.log
            .push(format!("Started conversion of {total} subdataset(s)"));
        self.conversion = Some(ConversionState {
            rx,
            kind: JobKind::Convert,
            total,
            completed: 0,
            succeeded: 0,
            failed: 0,
            current: None,
            statuses,
        });
        self.view = View::Log;
    }

    /// Drain progress messages from the worker (non-blocking). Call once per
    /// UI frame while a conversion is active.
    pub fn poll_conversion(&mut self) {
        // Collect pending updates without holding a mutable borrow of `self`.
        let updates: Vec<ConvUpdate> = match &self.conversion {
            Some(conv) => conv.rx.try_iter().collect(),
            None => return,
        };

        let mut finished = false;
        for update in updates {
            match update {
                ConvUpdate::Started { key, label } => {
                    if let Some(conv) = self.conversion.as_mut() {
                        conv.statuses.insert(key, ConvItemStatus::Running);
                        conv.current = Some(label);
                    }
                }
                ConvUpdate::Finished { key, label, result } => {
                    if let Some(conv) = self.conversion.as_mut() {
                        conv.completed += 1;
                        conv.current = None;
                        match &result {
                            Ok(_) => conv.succeeded += 1,
                            Err(_) => conv.failed += 1,
                        }
                        conv.statuses.insert(
                            key,
                            if result.is_ok() {
                                ConvItemStatus::Done
                            } else {
                                ConvItemStatus::Failed
                            },
                        );
                    }
                    match result {
                        Ok(n) => self.log.push(format!("\u{2713} {label}: {n} objects")),
                        Err(e) => self.log.push(format!("\u{2717} {label}: {e}")),
                    }
                }
                ConvUpdate::Progress { current } => {
                    if let Some(conv) = self.conversion.as_mut() {
                        conv.current = Some(current);
                    }
                }
                ConvUpdate::Log(line) => self.log.push(line),
                ConvUpdate::Done => finished = true,
            }
        }

        if finished {
            if let Some(conv) = &self.conversion {
                self.log.push(format!(
                    "{} finished: {} ok, {} failed",
                    conv.kind.noun(),
                    conv.succeeded,
                    conv.failed
                ));
            }
            self.conversion = None;
            if let Err(err) = self.refresh() {
                self.log.push(format!("Refresh failed: {err:#}"));
            }
        }
    }

    /// Conversion status of `entry` during an active run, if tracked.
    pub fn item_status(&self, entry: &SubDataset) -> Option<ConvItemStatus> {
        self.conversion
            .as_ref()
            .and_then(|c| c.statuses.get(&entry.duckdb_path).copied())
    }
}

/// A directory name for a merge of `sources`, derived from their leaf names.
/// Up to three are joined with `+`; beyond that it's `<first>+<n>more`.
fn merge_dir_name(sources: &[MergeSource]) -> String {
    let leaf = |s: &MergeSource| -> String {
        let raw = Path::new(&s.label)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| s.label.clone());
        sanitize_component(&raw)
    };
    let names: Vec<String> = sources.iter().map(leaf).collect();
    let name = if names.len() <= 3 {
        names.join("+")
    } else {
        format!("{}+{}more", names[0], names.len() - 1)
    };
    if name.is_empty() {
        "merged".to_owned()
    } else {
        name
    }
}

/// Restrict a string to filesystem-friendly characters for a path component.
fn sanitize_component(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect()
}
