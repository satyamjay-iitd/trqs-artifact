use std::{
    fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use anyhow::Context as _;

/// A detector+tracker configuration.
///
/// Owned rather than `&'static` so a caller can start from a preset and override
/// it — sweeping a confidence threshold or a class filter is how a good
/// configuration gets found, and a borrowed struct can only ever hold the
/// combinations someone compiled in.
#[derive(Clone, Debug)]
pub struct DetectConfig {
    /// Short stable id, also recorded in the output metadata and used to name
    /// the output directory, so two configurations cannot overwrite each other.
    pub id: String,
    pub name: String,
    pub description: String,
    /// Model weights name/path; Ultralytics auto-downloads bare names.
    pub model: String,
    /// Tracker config (`bytetrack.yaml` / `botsort.yaml`).
    pub tracker: String,
    pub conf: f64,
    pub imgsz: u32,
    pub half: bool,
    /// COCO class ids to keep; empty means all classes.
    pub classes: Vec<u32>,
}

/// COCO ids for the road/scene classes shared across most of our datasets:
/// person, bicycle, car, motorcycle, bus, truck.
const TRAFFIC: &[u32] = &[0, 1, 2, 3, 5, 7];

/// The selectable configurations, in menu order. The balanced traffic preset
/// is first so it reads as the default.
pub fn predefined_configs() -> Vec<DetectConfig> {
    vec![
        DetectConfig {
            id: "yolo11l-bytetrack-traffic".into(),
            name: "YOLO11l · ByteTrack · traffic".into(),
            description: "Balanced default. People + vehicles, FP16.".into(),
            model: "yolo11l.pt".into(),
            tracker: "bytetrack.yaml".into(),
            conf: 0.25,
            imgsz: 640,
            half: true,
            classes: TRAFFIC.to_vec(),
        },
        DetectConfig {
            id: "yolo11n-bytetrack-traffic".into(),
            name: "YOLO11n · ByteTrack · traffic".into(),
            description: "Fastest. Lowest accuracy, good for a quick pass.".into(),
            model: "yolo11n.pt".into(),
            tracker: "bytetrack.yaml".into(),
            conf: 0.25,
            imgsz: 640,
            half: true,
            classes: TRAFFIC.to_vec(),
        },
        DetectConfig {
            id: "yolo11x-botsort-traffic".into(),
            name: "YOLO11x · BoT-SORT · traffic".into(),
            description: "Most accurate. BoT-SORT ReID for steadier track ids.".into(),
            model: "yolo11x.pt".into(),
            tracker: "botsort.yaml".into(),
            conf: 0.25,
            imgsz: 960,
            half: true,
            classes: TRAFFIC.to_vec(),
        },
        DetectConfig {
            id: "rtdetr-x-botsort-traffic".into(),
            name: "RT-DETR-X · BoT-SORT · traffic".into(),
            description: "Transformer detector, NMS-free. Strong accuracy, slower.".into(),
            model: "rtdetr-x.pt".into(),
            tracker: "botsort.yaml".into(),
            conf: 0.3,
            imgsz: 640,
            half: true,
            classes: TRAFFIC.to_vec(),
        },
        DetectConfig {
            id: "yolo11l-bytetrack-coco".into(),
            name: "YOLO11l · ByteTrack · all COCO".into(),
            description: "All 80 COCO classes (type stored as a string).".into(),
            model: "yolo11l.pt".into(),
            tracker: "bytetrack.yaml".into(),
            conf: 0.25,
            imgsz: 640,
            half: true,
            classes: Vec::new(),
        },
    ]
}

/// Source video metadata copied into the predicted dataset so peek/detail and
/// `.trql` conversion keep working.
#[derive(Clone, Debug, Default)]
pub struct SourceInfo {
    pub rel_path: String,
    pub fps: Option<f64>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub num_frames: Option<u32>,
    pub duration_s: Option<f64>,
}

/// One detection unit: run `config` over `video_path`, write `out_dir/duckdb.db`.
#[derive(Clone, Debug)]
pub struct DetectJob {
    /// Stable identity for UI status (the source `duckdb.db` path).
    pub key: PathBuf,
    pub label: String,
    pub video_path: PathBuf,
    pub out_dir: PathBuf,
    pub source: SourceInfo,
    /// The source dataset's `duckdb.db` (ground truth) to score against once
    /// detection finishes.
    pub gt_db: PathBuf,
}

/// Progress events emitted while running one job.
pub enum DetectEvent {
    /// Frame-level progress within the current video.
    Progress { done: u64, total: u64 },
}

/// Run one detection job end to end: prepare the output dataset directory
/// (symlink the video, write metadata), invoke the Python pipeline, and stream
/// its progress. Returns the number of objects (tracks) produced.
pub fn run_detect_job(
    python: &str,
    script: &Path,
    job: &DetectJob,
    config: &DetectConfig,
    progress: &mut dyn FnMut(DetectEvent),
) -> anyhow::Result<u32> {
    if !script.is_file() {
        anyhow::bail!("detect script not found: {}", script.display());
    }
    fs::create_dir_all(&job.out_dir)
        .with_context(|| format!("failed to create {}", job.out_dir.display()))?;
    link_video(&job.video_path, &job.out_dir)?;
    write_metadata(&job.out_dir, &job.source, config)?;

    let out_db = job.out_dir.join("duckdb.db");
    let log_path = job.out_dir.join("detect.log");
    let log_file = fs::File::create(&log_path)
        .with_context(|| format!("failed to create {}", log_path.display()))?;

    let mut cmd = Command::new(python);
    cmd.arg(script)
        .arg("--video")
        .arg(&job.video_path)
        .arg("--out")
        .arg(&out_db)
        .args(["--model", config.model.as_str()])
        .args(["--tracker", config.tracker.as_str()])
        .arg("--conf")
        .arg(config.conf.to_string())
        .arg("--imgsz")
        .arg(config.imgsz.to_string());
    if config.half {
        cmd.arg("--half");
    }
    if !config.classes.is_empty() {
        cmd.arg("--classes");
        cmd.args(config.classes.iter().map(|c| c.to_string()));
    }
    cmd.stdout(Stdio::piped())
        .stderr(log_file)
        .stdin(Stdio::null());

    let mut child = cmd
        .spawn()
        .with_context(|| format!("failed to launch `{python} {}`", script.display()))?;

    let stdout = child.stdout.take().expect("stdout piped above");
    let mut num_objs = 0u32;
    for line in BufReader::new(stdout).lines() {
        let line = line?;
        if let Some(rest) = line.strip_prefix("PROGRESS ") {
            if let Some((done, total)) = parse_pair(rest) {
                progress(DetectEvent::Progress { done, total });
            }
        } else if let Some(rest) = line.strip_prefix("DONE ") {
            num_objs = rest
                .split_whitespace()
                .next()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
        }
    }

    let status = child.wait().context("waiting for detect process")?;
    if !status.success() {
        let tail = log_tail(&log_path);
        anyhow::bail!("detect failed (see {}):\n{tail}", log_path.display());
    }
    Ok(num_objs)
}

fn parse_pair(s: &str) -> Option<(u64, u64)> {
    let mut it = s.split_whitespace();
    let a = it.next()?.parse().ok()?;
    let b = it.next()?.parse().ok()?;
    Some((a, b))
}

/// Symlink the source video as `out_dir/video.mp4` (peek/extract use it).
fn link_video(video: &Path, out_dir: &Path) -> anyhow::Result<()> {
    let target = fs::canonicalize(video).unwrap_or_else(|_| video.to_path_buf());
    let link = out_dir.join("video.mp4");
    if link.exists() || fs::symlink_metadata(&link).is_ok() {
        fs::remove_file(&link).ok();
    }
    std::os::unix::fs::symlink(&target, &link).with_context(|| {
        format!(
            "failed to symlink {} -> {}",
            link.display(),
            target.display()
        )
    })?;
    Ok(())
}

/// Write the predicted dataset's `metadata.json`. `dataset` is forced to
/// `"predicted"` so it groups separately from the ground-truth families.
fn write_metadata(out_dir: &Path, src: &SourceInfo, config: &DetectConfig) -> anyhow::Result<()> {
    let meta = serde_json::json!({
        "dataset": "predicted",
        // Machine-readable, unlike `notes`: downstream wants to record *which*
        // configuration produced a row, and parsing prose to find out is the
        // kind of coupling that breaks the first time the sentence is reworded.
        "detector": config.id,
        "is_gt": false,
        "source": src.rel_path,
        "fps": src.fps,
        "width": src.width,
        "height": src.height,
        "num_frames": src.num_frames,
        "duration_s": src.duration_s,
        "notes": format!("Predicted by {} ({})", config.name, config.id),
    });
    let path = out_dir.join("metadata.json");
    fs::write(&path, serde_json::to_vec_pretty(&meta)?)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

/// Last few lines of the detect log, for surfacing an error in the UI.
fn log_tail(path: &Path) -> String {
    let text = fs::read_to_string(path).unwrap_or_default();
    let tail: Vec<&str> = text.lines().rev().take(8).collect();
    tail.into_iter().rev().collect::<Vec<_>>().join("\n")
}
