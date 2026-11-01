//! The on-disk dataset tree: dataset families, their sub-datasets, and the
//! artifacts (`duckdb.db`, `video.mp4`, `.trql`) found in each.

use std::{
    fmt::Display,
    fs,
    path::{Path, PathBuf},
    time::SystemTime,
};

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use trql_types::prop::TypeDesc;

use crate::{db::Trqs, stats::TrqsStats};

/// The full dataset tree: a flat list of sub-datasets plus the dataset
/// families grouping them.
///
/// Sub-datasets are stored flat (and sorted by relative path) so callers can
/// refer to one by a stable index; [`Dataset`] holds the indices belonging to
/// each family.
#[derive(Clone, Debug, Default)]
pub struct Datasets {
    root: PathBuf,
    entries: Vec<SubDataset>,
    datasets: Vec<Dataset>,
}

/// A dataset family (e.g. `soccernet`) and the sub-datasets it contains.
#[derive(Clone, Debug)]
pub struct Dataset {
    pub name: String,
    /// Indices into [`Datasets::entries`] belonging to this family.
    pub entries: Vec<usize>,
}

/// One sub-dataset directory: everything discovered about a single
/// `duckdb.db` and the artifacts sitting next to it.
#[derive(Clone, Debug)]
pub struct SubDataset {
    pub name: String,
    /// Path relative to the tree root, e.g. `cityflow/train_S01_c001`.
    pub rel_path: PathBuf,
    pub duckdb_path: PathBuf,
    pub metadata_path: Option<PathBuf>,
    pub metadata: Option<DatasetMetadata>,
    pub video_info: VideoInfo,
    /// Every `<max_gap>_<max_error_px>.trql` already built here.
    pub trql_files: Vec<TrqlParams>,
    /// Human annotations rather than detector output.
    pub is_gt: bool,
    /// Lazily inspected source database; see `trql-tui`'s `duckdb_meta`.
    pub duckdb: Option<DuckDbMeta>,
    /// Last inspection error, if any.
    pub error: Option<String>,
    /// Lazily read stats for one of `trql_files`.
    pub trql_stats: Option<TrqsStats>,
}

/// Whether a sub-dataset has a video, and in what form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoInfo {
    /// A single `video.mp4`.
    Single,
    /// A `videos/` directory, as produced by a merge.
    Merged,
    Absent,
}

/// The parameters a `.trql` was built with, named
/// `<max_gap>_<max_error_px>[_<repeat>].trql`.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize)]
pub struct TrqlParams {
    pub max_gap: u32,
    pub max_error_px: f64,
    /// How many times the base database was tiled to grow it (see
    /// [`Trqs::repeat`](crate::db::Trqs::repeat)). `1` is a plain, un-repeated
    /// `.trql` and is omitted from the file name.
    pub repeat: u32,
    /// Older than the source `duckdb.db`, so it should be rebuilt.
    pub stale: bool,
}

impl TrqlParams {
    /// Parse a `.trql` file name of the form
    /// `<max_gap>_<max_error_px>[_<repeat>].trql`. Returns `None` for names that
    /// don't follow the convention (e.g. `custom.trql`).
    fn from_file_name(name: &str, stale: bool) -> Option<Self> {
        let stem = name.strip_suffix(".trql")?;
        let mut parts = stem.split('_');
        let max_gap = parts.next()?.parse().ok()?;
        let max_error_px = parts.next()?.parse().ok()?;
        // The optional third component is a repeat count; default 1.
        let repeat = match parts.next() {
            Some(r) => r.parse().ok()?,
            None => 1,
        };
        if parts.next().is_some() {
            return None; // more components than the convention allows
        }
        Some(Self {
            max_gap,
            max_error_px,
            repeat,
            stale,
        })
    }

    pub fn file_name(&self) -> String {
        trql_filename(self.max_gap, self.max_error_px, self.repeat)
    }
}

/// Metadata parsed from a sub-dataset's `metadata.json`.
#[derive(Clone, Debug, Deserialize)]
pub struct DatasetMetadata {
    pub dataset: Option<String>,
    pub source: Option<String>,
    #[serde(default, deserialize_with = "scalar_to_string")]
    pub scene: Option<String>,
    #[serde(default, deserialize_with = "scalar_to_string")]
    pub clip: Option<String>,
    #[serde(default, deserialize_with = "scalar_to_string")]
    pub video_id: Option<String>,
    pub fps: Option<f64>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub num_frames: Option<u32>,
    pub duration_s: Option<f64>,
    pub row_counts: Option<RowCounts>,
    #[serde(default)]
    pub obj_classes: Vec<String>,
    #[serde(default)]
    pub activity_classes: Vec<String>,
    pub notes: Option<String>,
    /// Set when the rows came from a detector rather than human annotation.
    pub detector: Option<String>,
    pub is_gt: Option<bool>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RowCounts {
    pub objects: Option<u64>,
    pub bbox: Option<u64>,
}

/// Deserialize a JSON scalar (string, integer, float, or bool) into an
/// `Option<String>`. Dataset metadata is inconsistent about whether fields
/// like `video_id` are quoted, so coerce numbers to their string form.
fn scalar_to_string<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error as _;
    match Option::<serde_json::Value>::deserialize(deserializer)? {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(s)) => Ok(Some(s)),
        Some(serde_json::Value::Number(n)) => Ok(Some(n.to_string())),
        Some(serde_json::Value::Bool(b)) => Ok(Some(b.to_string())),
        Some(other) => Err(D::Error::custom(format!(
            "expected a string or number, found {other}"
        ))),
    }
}

/// Shape of a sub-dataset's source `duckdb.db`.
///
/// Plain data: it is filled in by whoever holds a DuckDB connection (see
/// `trql-tui`'s `duckdb_meta`), so this crate needs no DuckDB dependency.
#[derive(Clone, Debug)]
pub struct DuckDbMeta {
    pub object_count: u64,
    pub bbox_count: u64,
    pub frame_min: Option<u32>,
    pub frame_max: Option<u32>,
    pub object_columns: Vec<TableColumn>,
    pub bbox_columns: Vec<TableColumn>,
    pub schema_fields: Vec<SchemaField>,
}

#[derive(Clone, Debug)]
pub struct TableColumn {
    pub name: String,
    pub ty: String,
    pub nullable: bool,
}

/// A source column mapped to a TrQL property.
#[derive(Clone, Debug)]
pub struct SchemaField {
    pub name: String,
    pub column: String,
    pub type_desc: TypeDesc,
}

/// Phases of building a `.trql` from a source `duckdb.db`, reported to the
/// callback of [`SubDataset::load_database_with_progress`] /
/// [`SubDataset::rebuild_trql`].
///
/// `Fitting` is the long phase and the only one with a known total; the others
/// are single transitions, since DuckDB gives no progress of its own.
#[derive(Clone, Copy, Debug)]
pub enum ConvertProgress {
    /// Inferring the property schema from the source database.
    Inspecting,
    /// Reading objects and their bounding boxes out of DuckDB.
    Reading,
    /// Fitting polynomials, `(completed, total)` trajectories.
    Fitting { done: usize, total: usize },
    /// Writing the `.trql` to disk.
    Saving,
}

impl Datasets {
    /// Walk the tree under `root`, collecting every directory that holds a
    /// `duckdb.db` and grouping them into dataset families.
    ///
    /// Every `.trql` already built in a sub-dataset is listed in its
    /// [`trql_files`](SubDataset::trql_files). No single one is selected: call
    /// [`retarget`](Self::retarget) to point each sub-dataset at the `.trql` for
    /// a particular parameter pair.
    pub fn load(root: &Path) -> anyhow::Result<Self> {
        let mut entries = Vec::new();

        visit_dirs(root, &mut |dir| {
            // Interior directories of the tree hold no dataset of their own.
            if !dir.join("duckdb.db").is_file() {
                return Ok(());
            }
            entries.push(SubDataset::load(root, dir)?);
            Ok(())
        })?;

        entries.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));

        let mut this = Self {
            root: root.to_path_buf(),
            entries,
            datasets: Vec::new(),
        };
        this.regroup();
        Ok(this)
    }

    /// Group the (already sorted) entries by dataset family, preserving the
    /// order in which each family first appears.
    fn regroup(&mut self) {
        let mut datasets: Vec<Dataset> = Vec::new();
        for (idx, entry) in self.entries.iter().enumerate() {
            let key = entry.group_key();
            match datasets.last_mut() {
                Some(dataset) if dataset.name == key => dataset.entries.push(idx),
                _ => datasets.push(Dataset {
                    name: key,
                    entries: vec![idx],
                }),
            }
        }
        self.datasets = datasets;
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// All sub-datasets, flat and sorted by relative path.
    pub fn entries(&self) -> &[SubDataset] {
        &self.entries
    }

    pub fn entries_mut(&mut self) -> &mut [SubDataset] {
        &mut self.entries
    }

    pub fn entry(&self, idx: usize) -> Option<&SubDataset> {
        self.entries.get(idx)
    }

    pub fn entry_mut(&mut self, idx: usize) -> Option<&mut SubDataset> {
        self.entries.get_mut(idx)
    }

    /// The dataset families, in first-appearance order.
    pub fn datasets(&self) -> &[Dataset] {
        &self.datasets
    }

    pub fn dataset(&self, idx: usize) -> Option<&Dataset> {
        self.datasets.get(idx)
    }

    /// The entry indices of the dataset family at `idx`.
    pub fn dataset_entries(&self, idx: usize) -> &[usize] {
        self.datasets
            .get(idx)
            .map(|d| d.entries.as_slice())
            .unwrap_or(&[])
    }

    /// Look up a family by name.
    pub fn get(&self, name: &str) -> Option<&Dataset> {
        self.datasets.iter().find(|d| d.name == name)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// The names a sub-dataset's metadata may be filed under, most specific first.
///
/// `metadata.json` is what every converter writes. `world-metadata.json` is the
/// exception: a dataset whose sub-datasets are *worlds* rather than clips
/// (nuScenes' 850 scenes, and the CARLA trees `trql-server` serves) files the
/// same document under that name, because the server looks for it there to find
/// the world's coordinate frame and map raster.
const METADATA_NAMES: [&str; 2] = ["metadata.json", "world-metadata.json"];

/// The sub-dataset's metadata file, whichever of [`METADATA_NAMES`] it uses.
///
/// Without the fallback, every nuScenes scene loads with `fps`, `width`,
/// `height` and `is_gt` silently absent — the clip still opens, so nothing
/// fails, it just reports `None` for everything metadata-derived.
fn metadata_file(dir: &Path) -> Option<PathBuf> {
    METADATA_NAMES
        .iter()
        .map(|name| dir.join(name))
        .find(|p| p.is_file())
}

impl SubDataset {
    pub fn load(root: &Path, dir: &Path) -> anyhow::Result<Self> {
        let duckdb_path = dir.join("duckdb.db");
        if !duckdb_path.is_file() {
            return Err(anyhow::anyhow!(
                "no duckdb.db in {} — not a sub-dataset directory",
                dir.display()
            ));
        }
        let rel_path = dir.strip_prefix(root).unwrap_or(dir).to_path_buf();
        let name = dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("dataset")
            .to_owned();

        let metadata_path = metadata_file(dir);
        let metadata: Option<DatasetMetadata> = match &metadata_path {
            Some(path) => {
                let raw = fs::read_to_string(path)
                    .with_context(|| format!("failed reading {}", path.display()))?;
                Some(
                    serde_json::from_str(&raw)
                        .with_context(|| format!("failed parsing metadata {}", path.display()))?,
                )
            }
            None => None,
        };

        let video_info = if dir.join("videos").is_dir() {
            VideoInfo::Merged
        } else if dir.join("video.mp4").exists() {
            VideoInfo::Single
        } else {
            VideoInfo::Absent
        };

        // The tree carries human annotations unless the metadata says the rows
        // came from a detector.
        let is_gt = match metadata.as_ref() {
            Some(m) => m.is_gt.unwrap_or(m.detector.is_none()),
            None => true,
        };

        let trql_files = built_trqls(&duckdb_path);

        Ok(Self {
            name,
            rel_path,
            duckdb_path,
            metadata_path,
            metadata,
            video_info,
            trql_files,
            is_gt,
            duckdb: None,
            error: None,
            trql_stats: None,
        })
    }

    /// The directory containing this sub-dataset's `duckdb.db`, `video.mp4`, etc.
    pub fn dir(&self) -> &Path {
        self.duckdb_path.parent().unwrap_or(Path::new("."))
    }

    /// The base (un-repeated) `.trql` file name for these parameters, e.g.
    /// `10_0.5.trql`.
    pub fn trql_name(&self, max_gap: u32, max_error_px: f64) -> String {
        trql_filename(max_gap, max_error_px, 1)
    }

    /// Absolute path where the base `.trql` for these parameters lives, whether
    /// or not it has been built.
    pub fn trql_path(&self, max_gap: u32, max_error_px: f64) -> PathBuf {
        self.dir().join(trql_filename(max_gap, max_error_px, 1))
    }

    /// The built base `.trql` for these parameters, if one exists.
    ///
    /// Reads the artifacts discovered at load time ([`trql_files`]), so it does
    /// not touch the disk.
    ///
    /// [`trql_files`]: SubDataset::trql_files
    pub fn trql(&self, max_gap: u32, max_error_px: f64) -> Option<&TrqlParams> {
        let name = trql_filename(max_gap, max_error_px, 1);
        self.trql_files.iter().find(|p| p.file_name() == name)
    }

    /// Whether the `.trql` for these parameters has been built.
    pub fn has_trql(&self, max_gap: u32, max_error_px: f64) -> bool {
        self.trql(max_gap, max_error_px).is_some()
    }

    /// The dataset family this sub-dataset belongs to: `metadata.dataset` when
    /// set, otherwise the first component of the relative path.
    pub fn group_key(&self) -> String {
        if let Some(ds) = self.metadata.as_ref().and_then(|m| m.dataset.as_deref())
            && !ds.is_empty()
        {
            return ds.to_owned();
        }
        self.rel_path
            .components()
            .next()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .unwrap_or_else(|| self.name.clone())
    }

    /// Load the `<max_gap>_<max_error_px>` database, optionally tiled `repeat`
    /// times to grow it for benchmarking (see [`Trqs::repeat`](crate::db::Trqs::repeat)).
    ///
    /// A persisted `<max_gap>_<max_error_px>_<repeat>.trql` artifact is loaded
    /// directly if present; otherwise the base database is tiled in memory.
    /// `repeat` of `None` (or `Some(1)`) loads the base database unchanged.
    ///
    /// If the base `.trql` has not been built yet, it is built on demand from the
    /// sub-dataset's source `duckdb.db` (requires the `duckdb` feature).
    pub fn load_database(
        &self,
        max_gap: u32,
        max_error_px: f64,
        repeat: Option<u32>,
    ) -> anyhow::Result<Trqs> {
        self.load_database_with_progress(max_gap, max_error_px, repeat, &mut |_| {})
    }

    /// Like [`load_database`](Self::load_database), but reports
    /// [`ConvertProgress`] while building the base `.trql` on demand.
    pub fn load_database_with_progress(
        &self,
        max_gap: u32,
        max_error_px: f64,
        repeat: Option<u32>,
        on_progress: &mut dyn FnMut(ConvertProgress),
    ) -> anyhow::Result<Trqs> {
        let repeat = repeat.unwrap_or(1);

        // A persisted artifact wins — the base when `repeat == 1`, or a stored
        // tiling produced by [`build_repeated_trql`](Self::build_repeated_trql).
        let path = self
            .dir()
            .join(trql_filename(max_gap, max_error_px, repeat));
        if path.is_file() {
            return Ok(Trqs::from_bin(&path)?);
        }

        // Otherwise tile the base database in memory, building it first if it
        // has not been created yet.
        let base_path = self.trql_path(max_gap, max_error_px);
        if !base_path.is_file() {
            self.ensure_base_built(max_gap, max_error_px, on_progress)?;
        }
        // `repeat` is >= 2 here (`repeat == 1` resolves to `base_path` above);
        // `Trqs::repeat` returns a plain clone for `<= 1` regardless.
        Ok(Trqs::from_bin(&base_path)?.repeat(repeat))
    }

    /// Build the base `<max_gap>_<max_error_px>.trql` from the source
    /// `duckdb.db`, reporting [`ConvertProgress`] as it goes.
    #[cfg(feature = "duckdb")]
    fn ensure_base_built(
        &self,
        max_gap: u32,
        max_error_px: f64,
        on_progress: &mut dyn FnMut(ConvertProgress),
    ) -> anyhow::Result<()> {
        crate::convert::build_trql_with_progress(
            &self.duckdb_path,
            &self.trql_path(max_gap, max_error_px),
            self.metadata.as_ref(),
            max_gap,
            max_error_px,
            on_progress,
        )?;
        Ok(())
    }

    #[cfg(not(feature = "duckdb"))]
    fn ensure_base_built(
        &self,
        max_gap: u32,
        max_error_px: f64,
        _on_progress: &mut dyn FnMut(ConvertProgress),
    ) -> anyhow::Result<()> {
        Err(anyhow::anyhow!(
            "No TRQL file found for max_gap={max_gap} max_error_px={max_error_px} \
             (building one requires the `duckdb` feature)"
        ))
    }

    /// Unconditionally (re)build the base `<max_gap>_<max_error_px>.trql` from
    /// the source `duckdb.db`, overwriting any existing artifact. Returns the
    /// number of objects built.
    ///
    /// Unlike [`load_database`](Self::load_database), which only builds when the
    /// `.trql` is missing, this always rebuilds — used to refresh a stale
    /// artifact.
    #[cfg(feature = "duckdb")]
    pub fn rebuild_trql(
        &self,
        max_gap: u32,
        max_error_px: f64,
        on_progress: &mut dyn FnMut(ConvertProgress),
    ) -> anyhow::Result<u32> {
        let summary = crate::convert::build_trql_with_progress(
            &self.duckdb_path,
            &self.trql_path(max_gap, max_error_px),
            self.metadata.as_ref(),
            max_gap,
            max_error_px,
            on_progress,
        )?;
        Ok(summary.num_objs)
    }

    /// Build a `repeat`-times tiled copy of the base `<max_gap>_<max_error_px>`
    /// database and save it alongside as `<max_gap>_<max_error_px>_<repeat>.trql`
    /// (see [`Trqs::repeat`](crate::db::Trqs::repeat)).
    ///
    /// Grows a database for benchmarking without re-fitting. `repeat` must be
    /// `>= 2`; the base `.trql` must already be built. Returns the path written.
    /// The in-memory tree is not refreshed — reload the [`Datasets`] to see the
    /// new artifact.
    pub fn build_repeated_trql(
        &self,
        max_gap: u32,
        max_error_px: f64,
        repeat: u32,
    ) -> anyhow::Result<PathBuf> {
        if repeat < 2 {
            return Err(anyhow::anyhow!("repeat must be >= 2, got {repeat}"));
        }
        let base = self.load_database(max_gap, max_error_px, None)?;
        let repeated = base.repeat(repeat);
        let out = self
            .dir()
            .join(trql_filename(max_gap, max_error_px, repeat));
        repeated.save(&out)?;
        Ok(out)
    }
}

/// All `.trql` files already built next to `duckdb_path`, sorted by parameters.
/// Names that don't follow the `<max_gap>_<max_error_px>.trql` convention are
/// skipped.
fn built_trqls(duckdb_path: &Path) -> Vec<TrqlParams> {
    let Some(dir) = duckdb_path.parent() else {
        return Vec::new();
    };
    let duckdb_mtime = modified(duckdb_path).ok();
    let Ok(read_dir) = fs::read_dir(dir) else {
        return Vec::new();
    };

    let mut out = Vec::new();
    for entry in read_dir.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("trql") {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let stale = match (
            duckdb_mtime,
            entry.metadata().ok().and_then(|m| m.modified().ok()),
        ) {
            (Some(d), Some(v)) => d > v,
            _ => false,
        };
        if let Some(params) = TrqlParams::from_file_name(name, stale) {
            out.push(params);
        }
    }
    out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// The output `.trql` filename for a given parameter pair, e.g. `10_0.5.trql`.
///
/// The `<max_gap>_<max_error_px>[_<repeat>].trql` naming convention is internal
/// to this module; callers reach it through [`SubDataset::trql_name`] /
/// [`SubDataset::trql_path`] and [`TrqlParams::file_name`]. `repeat == 1` (a
/// plain, un-repeated database) omits the suffix.
fn trql_filename(max_gap: u32, max_error_px: f64, repeat: u32) -> String {
    // `{}` on f64 drops trailing zeros: 0.0 -> "0", 0.5 -> "0.5".
    if repeat > 1 {
        format!("{max_gap}_{max_error_px}_{repeat}.trql")
    } else {
        format!("{max_gap}_{max_error_px}.trql")
    }
}

fn visit_dirs(
    dir: &Path,
    on_dir: &mut impl FnMut(&Path) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    on_dir(dir)?;
    for entry in fs::read_dir(dir).with_context(|| format!("failed reading {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            visit_dirs(&path, on_dir)?;
        }
    }
    Ok(())
}

fn modified(path: &Path) -> anyhow::Result<SystemTime> {
    Ok(fs::metadata(path)
        .with_context(|| format!("failed stat {}", path.display()))?
        .modified()?)
}

impl Display for Datasets {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.datasets.is_empty() {
            return write!(f, "(no datasets)");
        }

        for (i, dataset) in self.datasets.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            writeln!(
                f,
                "{} ({} sub-datasets)",
                dataset.name,
                dataset.entries.len()
            )?;
            for (j, &idx) in dataset.entries.iter().enumerate() {
                let last = j + 1 == dataset.entries.len();
                let branch = if last { "└──" } else { "├──" };
                writeln!(f, "{} {}", branch, self.entries[idx])?;
            }
        }
        Ok(())
    }
}

impl Display for SubDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)?;
        if self.is_gt {
            write!(f, " [gt]")?;
        }
        write!(f, " video={}", self.video_info)?;

        if self.trql_files.is_empty() {
            write!(f, ", trql=none")
        } else {
            write!(f, ", trql=[")?;
            for (i, params) in self.trql_files.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{params}")?;
            }
            write!(f, "]")
        }
    }
}

impl Display for TrqlParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "gap={}/err={}px", self.max_gap, self.max_error_px)?;
        if self.repeat > 1 {
            write!(f, " (x{})", self.repeat)?;
        }
        if self.stale {
            write!(f, " (stale)")?;
        }
        Ok(())
    }
}

impl Display for VideoInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            VideoInfo::Single => "single",
            VideoInfo::Merged => "merged",
            VideoInfo::Absent => "absent",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `<root>/<dataset>/<sub>` directory with the given files.
    fn make_sub(root: &Path, rel: &str, files: &[(&str, &str)]) {
        let dir = root.join(rel);
        fs::create_dir_all(&dir).unwrap();
        for (name, contents) in files {
            fs::write(dir.join(name), contents).unwrap();
        }
    }

    fn temp_root(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(name);
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn display_empty() {
        assert_eq!(Datasets::default().to_string(), "(no datasets)");
    }

    #[test]
    fn display_video_info() {
        assert_eq!(VideoInfo::Single.to_string(), "single");
        assert_eq!(VideoInfo::Merged.to_string(), "merged");
        assert_eq!(VideoInfo::Absent.to_string(), "absent");
    }

    #[test]
    fn display_trql_params() {
        let params = TrqlParams {
            max_gap: 10,
            max_error_px: 0.5,
            repeat: 1,
            stale: false,
        };
        assert_eq!(params.to_string(), "gap=10/err=0.5px");
        assert_eq!(params.file_name(), "10_0.5.trql");
        assert_eq!(
            TrqlParams {
                stale: true,
                ..params
            }
            .to_string(),
            "gap=10/err=0.5px (stale)"
        );

        // A repeated database carries its tile count in name and Display.
        let repeated = TrqlParams {
            repeat: 100,
            ..params
        };
        assert_eq!(repeated.file_name(), "10_0.5_100.trql");
        assert_eq!(repeated.to_string(), "gap=10/err=0.5px (x100)");
    }

    #[test]
    fn parse_trql_file_names() {
        let parsed = TrqlParams::from_file_name("10_0.5.trql", false).unwrap();
        assert_eq!(parsed.max_gap, 10);
        assert_eq!(parsed.max_error_px, 0.5);
        assert_eq!(parsed.repeat, 1);

        // The optional third component is a repeat count.
        let repeated = TrqlParams::from_file_name("10_0.5_100.trql", false).unwrap();
        assert_eq!(repeated.repeat, 100);
        assert_eq!(repeated.file_name(), "10_0.5_100.trql");

        assert!(TrqlParams::from_file_name("custom.trql", false).is_none());
        assert!(TrqlParams::from_file_name("10_0.txt", false).is_none());
        assert!(TrqlParams::from_file_name("duckdb.db", false).is_none());
        // Too many components.
        assert!(TrqlParams::from_file_name("10_0.5_100_7.trql", false).is_none());
    }

    #[test]
    fn load_groups_by_family_and_renders() {
        let root = temp_root("trql_datasets_load_test");
        make_sub(
            &root,
            "cityflow/train_S01_c001",
            &[
                ("duckdb.db", ""),
                ("metadata.json", r#"{"dataset": "cityflow", "fps": 10.0}"#),
                ("video.mp4", ""),
                ("10_0.trql", ""),
                ("custom.trql", ""),
            ],
        );
        make_sub(
            &root,
            "cityflow/train_S01_c002",
            &[
                ("duckdb.db", ""),
                ("metadata.json", r#"{"dataset": "cityflow"}"#),
            ],
        );
        // No metadata: the family falls back to the first path component.
        make_sub(
            &root,
            "ytstream/beach_0",
            &[("duckdb.db", ""), ("video.mp4", "")],
        );

        let datasets = Datasets::load(&root).unwrap();
        let rendered = datasets.to_string();
        let _ = fs::remove_dir_all(&root);

        assert_eq!(datasets.len(), 3);
        assert_eq!(datasets.datasets().len(), 2);
        assert_eq!(datasets.dataset_entries(0), &[0, 1]);
        assert_eq!(datasets.get("ytstream").unwrap().entries, vec![2]);
        assert_eq!(
            rendered,
            "cityflow (2 sub-datasets)\n\
             ├── train_S01_c001 [gt] video=single, trql=[gap=10/err=0px]\n\
             └── train_S01_c002 [gt] video=absent, trql=none\n\
             \n\
             ytstream (1 sub-datasets)\n\
             └── beach_0 [gt] video=single, trql=none\n"
        );
    }

    #[test]
    fn a_world_files_its_metadata_under_world_metadata_json() {
        // nuScenes' 850 scenes and the CARLA world trees carry
        // `world-metadata.json`, never `metadata.json`. Without the fallback the
        // clip still loads — metadata is optional — so the failure is silent:
        // every metadata-derived field reads back as `None`.
        let root = temp_root("trql_datasets_world_metadata_test");
        make_sub(
            &root,
            "nuscenes/scene-0001",
            &[
                ("duckdb.db", ""),
                (
                    "world-metadata.json",
                    r#"{"dataset": "nuscenes", "fps": 12.0, "width": 158560}"#,
                ),
            ],
        );
        // Both present: `metadata.json` wins, so nothing changes for the
        // converters that already write it.
        make_sub(
            &root,
            "nuscenes/scene-0002",
            &[
                ("duckdb.db", ""),
                ("metadata.json", r#"{"fps": 12.0}"#),
                ("world-metadata.json", r#"{"fps": 999.0}"#),
            ],
        );

        let datasets = Datasets::load(&root).unwrap();
        let _ = fs::remove_dir_all(&root);

        let scene1 = datasets.entry(0).unwrap();
        let meta = scene1.metadata.as_ref().expect("world metadata not read");
        assert_eq!(meta.fps, Some(12.0));
        assert_eq!(meta.width, Some(158560));
        assert!(
            scene1
                .metadata_path
                .as_ref()
                .unwrap()
                .ends_with("world-metadata.json")
        );

        let scene2 = datasets.entry(1).unwrap();
        assert_eq!(scene2.metadata.as_ref().unwrap().fps, Some(12.0));
    }

    #[test]
    fn detector_output_is_not_ground_truth() {
        let root = temp_root("trql_datasets_gt_test");
        make_sub(
            &root,
            "predicted/cam0",
            &[
                ("duckdb.db", ""),
                ("metadata.json", r#"{"detector": "rtdetr_x"}"#),
            ],
        );
        let datasets = Datasets::load(&root).unwrap();
        let _ = fs::remove_dir_all(&root);

        assert!(!datasets.entry(0).unwrap().is_gt);
    }

    #[test]
    fn lists_every_built_trql_sorted_by_parameters() {
        let root = temp_root("trql_datasets_artifacts_test");
        make_sub(
            &root,
            "cityflow/cam0",
            &[
                ("duckdb.db", ""),
                ("10_0.trql", ""),
                ("5_0.5.trql", ""),
                ("custom.trql", ""),
            ],
        );

        let datasets = Datasets::load(&root).unwrap();
        let _ = fs::remove_dir_all(&root);

        // Every parameter pair present, sorted, with the odd name skipped.
        let built = &datasets.entry(0).unwrap().trql_files;
        assert_eq!(built.len(), 2);
        assert_eq!((built[0].max_gap, built[0].max_error_px), (5, 0.5));
        assert_eq!((built[1].max_gap, built[1].max_error_px), (10, 0.0));
        assert_eq!(built[0].file_name(), "5_0.5.trql");
    }
}
