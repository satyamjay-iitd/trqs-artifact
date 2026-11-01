use std::{
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Context as _;

/// A frame extracted from a dataset's video, ready to display.
#[derive(Clone, Debug)]
pub struct Frame {
    pub image_path: PathBuf,
    pub frame_index: Option<u32>,
    pub timestamp_s: f64,
    pub duration_s: f64,
}

/// Extract a random frame from a dataset's video and write it as a PNG in the
/// temp directory. A plain dataset has a single `video.mp4`; a merged dataset
/// instead has a `videos/` directory of `<start>_<end>.<ext>` symlinked clips,
/// from which one clip is picked at random. Uses ffmpeg; duration and fps come
/// from metadata when known, otherwise from ffprobe.
pub fn extract_random_frame(
    entry_dir: &Path,
    fps: Option<f64>,
    duration_s: Option<f64>,
) -> anyhow::Result<Frame> {
    let clip = resolve_video(entry_dir)?;

    // Metadata fps/duration only describe a single-video dataset; for a clip
    // pulled out of a merge we must probe the clip itself.
    let (fps, metadata_duration) = if clip.frame_offset == 0 {
        (fps, duration_s)
    } else {
        (None, None)
    };

    let duration = match metadata_duration {
        Some(d) if d > 0.0 => d,
        _ => ffprobe_duration(&clip.path)?,
    };
    // Fall back to probing fps so we can still report a frame index.
    let fps = fps.or_else(|| ffprobe_fps(&clip.path).ok());

    // Random timestamp, kept away from the very end where seeks may fail.
    let timestamp = random_unit() * duration * 0.97;

    let out = std::env::temp_dir().join(format!("trql-peek-{}.png", std::process::id()));
    // `-ss` before `-i` does a fast keyframe seek; one frame, high quality.
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-ss"])
        .arg(format!("{timestamp:.3}"))
        .arg("-i")
        .arg(&clip.path)
        .args(["-frames:v", "1", "-q:v", "2"])
        .arg(&out)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .context("failed to launch ffmpeg")?;
    if !status.success() {
        anyhow::bail!(
            "ffmpeg failed to extract a frame from {}",
            clip.path.display()
        );
    }
    if !out.exists() {
        anyhow::bail!("ffmpeg produced no output frame");
    }

    // Offset the local frame index by the clip's start fid in the merged timeline.
    let frame_index = fps.map(|f| clip.frame_offset + (timestamp * f).round() as u32);
    Ok(Frame {
        image_path: out,
        frame_index,
        timestamp_s: timestamp,
        duration_s: duration,
    })
}

/// A resolved video to peek into, plus the frame id its first frame maps to in
/// the (possibly merged) dataset timeline.
struct ResolvedClip {
    path: PathBuf,
    frame_offset: u32,
}

/// Find the video to peek: a plain `video.mp4`, or a random clip from a merged
/// dataset's `videos/` directory.
fn resolve_video(entry_dir: &Path) -> anyhow::Result<ResolvedClip> {
    let single = entry_dir.join("video.mp4");
    if single.exists() {
        return Ok(ResolvedClip {
            path: single,
            frame_offset: 0,
        });
    }

    let videos_dir = entry_dir.join(trql::storage::merge::VIDEO_DIR);
    if videos_dir.is_dir() {
        let mut clips: Vec<PathBuf> = std::fs::read_dir(&videos_dir)
            .with_context(|| format!("failed reading {}", videos_dir.display()))?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_file() || p.is_symlink())
            .collect();
        clips.sort();
        if clips.is_empty() {
            anyhow::bail!("no clips found in {}", videos_dir.display());
        }
        let pick = (random_unit() * clips.len() as f64) as usize;
        let path = clips[pick.min(clips.len() - 1)].clone();
        let frame_offset = clip_start_fid(&path).unwrap_or(0);
        return Ok(ResolvedClip { path, frame_offset });
    }

    anyhow::bail!(
        "no video.mp4 or {}/ directory found in {}",
        trql::storage::merge::VIDEO_DIR,
        entry_dir.display()
    )
}

/// Parse the leading `<start>` from a `<start>_<end>.<ext>` clip filename.
fn clip_start_fid(path: &Path) -> Option<u32> {
    path.file_stem()
        .and_then(|s| s.to_str())
        .and_then(|s| s.split('_').next())
        .and_then(|s| s.parse().ok())
}

fn ffprobe_duration(video: &Path) -> anyhow::Result<f64> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(video)
        .output()
        .context("failed to launch ffprobe")?;
    if !output.status.success() {
        anyhow::bail!("ffprobe failed for {}", video.display());
    }
    let text = String::from_utf8_lossy(&output.stdout);
    text.trim()
        .parse::<f64>()
        .with_context(|| format!("could not parse duration from ffprobe: {text:?}"))
        .and_then(|d| {
            if d > 0.0 {
                Ok(d)
            } else {
                anyhow::bail!("video reports non-positive duration")
            }
        })
}

/// Probe a video's average frame rate, parsing ffprobe's `num/den` form.
fn ffprobe_fps(video: &Path) -> anyhow::Result<f64> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(video)
        .output()
        .context("failed to launch ffprobe")?;
    if !output.status.success() {
        anyhow::bail!("ffprobe failed for {}", video.display());
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let text = text.trim();
    let fps = match text.split_once('/') {
        Some((num, den)) => {
            let den: f64 = den.parse().unwrap_or(0.0);
            if den == 0.0 {
                anyhow::bail!("ffprobe reported zero denominator fps: {text:?}");
            }
            num.parse::<f64>()
                .with_context(|| format!("could not parse fps numerator: {text:?}"))?
                / den
        }
        None => text
            .parse::<f64>()
            .with_context(|| format!("could not parse fps from ffprobe: {text:?}"))?,
    };
    if fps > 0.0 {
        Ok(fps)
    } else {
        anyhow::bail!("video reports non-positive fps")
    }
}

/// Cheap, dependency-free pseudo-random number in `[0, 1)`, seeded from the
/// system clock. Good enough for picking an arbitrary frame to preview.
fn random_unit() -> f64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    // Mix the bits a little so consecutive calls differ noticeably.
    let mut x = nanos as u64 ^ 0x9E37_79B9_7F4A_7C15;
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    (x % 1_000_000) as f64 / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_clip_start_fid() {
        assert_eq!(clip_start_fid(Path::new("0_961.mp4")), Some(0));
        assert_eq!(clip_start_fid(Path::new("/x/962_1500.mp4")), Some(962));
        assert_eq!(clip_start_fid(Path::new("video.mp4")), None);
    }

    #[test]
    fn resolves_single_video() {
        let dir = std::env::temp_dir().join(format!("trql-peek-single-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("video.mp4"), b"x").unwrap();

        let clip = resolve_video(&dir).unwrap();
        assert_eq!(clip.frame_offset, 0);
        assert!(clip.path.ends_with("video.mp4"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolves_merged_clip_with_offset() {
        let dir = std::env::temp_dir().join(format!("trql-peek-merged-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let videos = dir.join(trql::storage::merge::VIDEO_DIR);
        std::fs::create_dir_all(&videos).unwrap();
        std::fs::write(videos.join("0_961.mp4"), b"x").unwrap();
        std::fs::write(videos.join("962_1500.mp4"), b"x").unwrap();

        // Every resolution must land on one of the two clips with a matching offset.
        for _ in 0..20 {
            let clip = resolve_video(&dir).unwrap();
            let name = clip.path.file_name().unwrap().to_str().unwrap();
            match name {
                "0_961.mp4" => assert_eq!(clip.frame_offset, 0),
                "962_1500.mp4" => assert_eq!(clip.frame_offset, 962),
                other => panic!("unexpected clip {other}"),
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
