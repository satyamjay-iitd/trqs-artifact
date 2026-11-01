//! Scans the `--data-dir` into worlds and standalone cameras.
//!
//! The data directory is a mix of world directories and standalone-camera
//! datasets. A world holds a *single* `.trql` db covering every object in the
//! world; its cameras only contribute a viewpoint (video + `metadata.json`).
//! A standalone camera owns its own `.trql`:
//! ```text
//! data_dir/
//! ├── carla/                 # world: has an .xodr road network
//! │   ├── world.trql         # one db shared by every camera in the world
//! │   ├── map.xodr
//! │   ├── world-metadata.json
//! │   ├── cam0/
//! │   │   ├── video.mp4
//! │   │   └── metadata.json
//! │   └── cam1/
//! └── cityflow/              # standalone: no .xodr
//!     ├── cam1/              # each camera has its own .trql
//!     └── cam2/
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use trql::db::Trqs;

use crate::camera::{CameraModel, VideoMeta, WorldMeta};
use crate::roadmap::{camera_markers, parse_roadmap, view_roadmap};
use crate::state::{CameraData, CameraEntry, StandaloneEntry, WorldModeData, WorldName};

/// Name of the per-world coordinate-frame file, at a world's root beside the
/// `.xodr` and the `.trql`.
const WORLD_META_FILE: &str = "world-metadata.json";

/// Name given to the stand-in viewpoint of a world that ships no camera
/// directory. Requests address a world through one of its cameras, so a world
/// without any would be unreachable — including for the map, which needs none.
const SYNTHETIC_CAMERA: &str = "world";

/// Recognised video file extensions served by the `/video` endpoints.
pub fn is_video_ext(ext: &str) -> bool {
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "mp4" | "webm" | "mov" | "mkv" | "m4v"
    )
}

/// The name a world or camera is addressed by: its directory stem.
fn dir_name(dir: &Path) -> anyhow::Result<String> {
    Ok(dir
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow::anyhow!("invalid directory name: {}", dir.display()))?
        .to_string())
}

/// The first file directly inside `dir` with the given extension.
fn find_file_with_ext(dir: &Path, ext: &str) -> anyhow::Result<Option<PathBuf>> {
    Ok(std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.extension().and_then(|s| s.to_str()) == Some(ext)))
}

/// Picture formats a world may use as its map in place of an `.xodr`.
fn is_world_view_ext(ext: &str) -> bool {
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "svg" | "png" | "jpg" | "jpeg" | "webp"
    )
}

/// The world's map picture: whatever `world-metadata.json` names in
/// `world_view`, else the first picture sitting at the world root.
///
/// Naming it in the metadata is preferred — a world directory may hold other
/// images (a thumbnail, a legend) that are not the map.
fn find_world_view(world_dir: &Path, meta: &WorldMeta) -> Option<PathBuf> {
    if let Some(name) = meta.world_view.as_deref() {
        let named = world_dir.join(name);
        if named.is_file() {
            return Some(named);
        }
        eprintln!(
            "warn: world {} names world_view {name:?} which is not a readable file; \
             falling back to any picture at its root",
            world_dir.display()
        );
    }
    std::fs::read_dir(world_dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .find(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .is_some_and(is_world_view_ext)
        })
}

/// Reads a world's `world-metadata.json`: the coordinate frame (`coordinate_scale`,
/// `x_delta`, `y_delta`) its `.trql` coordinates are stored in. Required — without
/// it every stored coordinate would be un-shifted by the wrong origin, which
/// misplaces boxes and map positions silently rather than failing.
fn load_world_meta(world_dir: &Path) -> anyhow::Result<WorldMeta> {
    let path = world_dir.join(WORLD_META_FILE);
    let text = std::fs::read_to_string(&path).map_err(|e| {
        anyhow::anyhow!(
            "world {} has no readable {WORLD_META_FILE}: {e}. It must sit beside the .xodr and \
             carry the coordinate_scale, x_delta and y_delta its .trql was built with",
            world_dir.display()
        )
    })?;
    let meta: WorldMeta = serde_json::from_str(&text)
        .map_err(|e| anyhow::anyhow!("failed to parse {}: {e}", path.display()))?;
    if meta.coordinate_scale <= 0.0 {
        anyhow::bail!(
            "{} has a non-positive coordinate_scale ({})",
            path.display(),
            meta.coordinate_scale
        );
    }
    Ok(meta)
}

/// Picks the one video a camera directory is served from. Directories often hold
/// extra renders beside the source — `video_bb.mp4` with boxes burned in, a
/// `video.mpeg4.bak.mp4` kept from a transcode — and those are frequently in
/// codecs browsers cannot decode (MPEG-4 Part 2), so the choice cannot be left to
/// `read_dir` order. `video.<ext>` wins when present; otherwise the
/// lexicographically first candidate, so the pick is at least deterministic.
fn pick_source_video(mut videos: Vec<PathBuf>) -> Option<PathBuf> {
    videos.sort();
    videos
        .iter()
        .find(|p| p.file_stem().and_then(|s| s.to_str()) == Some("video"))
        .or_else(|| videos.first())
        .cloned()
}

/// Loads a camera directory: its video, parsed `metadata.json`, and the derived
/// pinhole camera model. The trajectory db is loaded separately — once per world
/// in world mode, per camera in standalone mode.
pub fn load_camera_entry_from_camera_dir(camera_dir: &Path) -> anyhow::Result<CameraEntry> {
    let camera_name = dir_name(camera_dir)?;

    let mut videos: Vec<PathBuf> = Vec::new();
    let mut meta: Option<VideoMeta> = None;

    for camera_dir_entry in std::fs::read_dir(camera_dir)? {
        let path = camera_dir_entry?.path();

        if path.file_name().and_then(|s| s.to_str()) == Some("metadata.json") {
            meta = std::fs::read_to_string(&path)
                .ok()
                .and_then(|t| serde_json::from_str::<VideoMeta>(&t).ok());
            continue;
        }

        if path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(is_video_ext)
        {
            videos.push(path);
        }
    }

    let video_path = pick_source_video(videos);

    // Resolved here rather than per request so a metadata file naming a picture
    // that is not there is reported once, at load, instead of 404ing later.
    // let view_path = meta
    //     .as_ref()
    //     .and_then(|m| m.scene_view.as_deref())
    //     .and_then(|name| {
    //         let path = camera_dir.join(name);
    //         if path.is_file() {
    //             Some(path)
    //         } else {
    //             eprintln!(
    //                 "warn: camera {} names scene_view {name:?} which is not a readable file",
    //                 camera_dir.display()
    //             );
    //             None
    //         }
    //     });

    let camera = meta.as_ref().and_then(CameraModel::from_meta);

    Ok(CameraEntry {
        camera_name,
        camera_data: CameraData {
            video_path,
            // view_path,
            video_meta: meta,
            camera,
        },
        synthetic: false,
    })
}

/// Loads a standalone camera: its camera assets plus the `.trql` db sitting in
/// the same directory.
fn load_standalone_entry_from_camera_dir(
    camera_dir: &Path,
    dataset: &str,
) -> anyhow::Result<StandaloneEntry> {
    let trqs_path = find_file_with_ext(camera_dir, "trql")?
        .ok_or_else(|| anyhow::anyhow!("no .trql file in {}", camera_dir.display()))?;
    let db = Trqs::from_bin(&trqs_path)?;
    let entry = load_camera_entry_from_camera_dir(camera_dir)?;

    // No world file here, so the scale comes from the camera's own
    // metadata.json, where the dataset generator stamps it. A standalone
    // camera's stored coordinates are not shifted, so the deltas stay zero.
    let meta = entry.camera_data.video_meta.clone().unwrap_or_default();
    let world_meta = WorldMeta {
        coordinate_scale: meta.coordinate_scale,
        ..Default::default()
    };

    Ok(StandaloneEntry {
        dataset: dataset.to_string(),
        camera_name: entry.camera_name,
        db,
        world_meta,
        camera_data: entry.camera_data,
    })
}

/// How far below `--data-dir` a dataset may sit.
///
/// Datasets are not all one directory deep. A converter that emits several
/// worlds for one source keeps them together — `processed/WorldPose/BRA_KOR`,
/// `processed/WorldPose/ARG_FRA` — and that grouping directory is neither a
/// world nor a camera. Rather than force a flat layout, an unrecognised
/// directory is descended into, up to this depth.
const MAX_NEST_DEPTH: usize = 3;

/// What a directory under `--data-dir` turns out to be.
enum DirKind {
    /// A world: it declares a map. Carries its `.xodr`, if that is the form.
    World(Option<PathBuf>),
    /// A standalone dataset: at least one subdirectory owns a `.trql`.
    Standalone,
    /// Neither — a grouping directory, to be looked inside.
    Container,
}

/// Classifies a directory by what it declares, not by where it sits.
///
/// The order matters: a world is recognised by its own map file, a standalone
/// dataset by its cameras carrying dbs, and anything else is assumed to be
/// holding datasets rather than being one. That last case is what lets a
/// converter group its output without the layout having to be flat.
fn classify_dir(dir: &Path) -> anyhow::Result<DirKind> {
    let xodr = find_file_with_ext(dir, "xodr")?;
    if xodr.is_some() || dir.join(WORLD_META_FILE).is_file() {
        return Ok(DirKind::World(xodr));
    }
    // A standalone dataset holds *cameras*, and a camera is a directory with a
    // `.trql` and no map of its own. The second half matters: a world root also
    // has a `.trql`, so testing for the db alone would read a directory full of
    // worlds as a dataset full of cameras and load every world as a camera.
    for entry in std::fs::read_dir(dir)? {
        let sub = entry?.path();
        if !sub.is_dir() {
            continue;
        }
        let declares_map = sub.join(WORLD_META_FILE).is_file()
            || find_file_with_ext(&sub, "xodr")?.is_some();
        if !declares_map && find_file_with_ext(&sub, "trql")?.is_some() {
            return Ok(DirKind::Standalone);
        }
    }
    Ok(DirKind::Container)
}

/// The name a nested dataset is addressed by: its path below `--data-dir`,
/// joined with `-`.
///
/// A top-level directory keeps its own name, so nothing that already loads is
/// renamed. Deeper ones are qualified — `WorldPose/BRA_KOR` becomes
/// `WorldPose-BRA_KOR` — which keeps names unique across grouping directories
/// and matches how the converters already name flattened worlds
/// (`alfheim-Tottenham`, `SoccerTrackv2-117092-1st`). `-` rather than `/`
/// because the name is a single path segment in every `/world/…` route.
fn qualified_name(prefix: &[String], leaf: &Path) -> anyhow::Result<String> {
    let leaf = dir_name(leaf)?;
    if prefix.is_empty() {
        return Ok(leaf);
    }
    let joined = prefix.join("-");
    // The converters already name their output after the dataset —
    // `SoccerTrackv2/SoccerTrackv2-117092-1st`, `carla/carla_1` — so qualifying
    // blindly would give `SoccerTrackv2-SoccerTrackv2-117092-1st`. Where the
    // directory has already said it, take it at its word.
    if leaf.starts_with(&joined) {
        return Ok(leaf);
    }
    Ok(format!("{joined}-{leaf}"))
}

/// Loads one world directory. Separated out so a world that cannot be loaded
/// can be reported and skipped rather than taking the scan down with it.
fn load_world(
    dataset_dir_path: &Path,
    world_name: &str,
    xodr_path: Option<PathBuf>,
) -> anyhow::Result<WorldModeData> {
    let trqs_path = find_file_with_ext(dataset_dir_path, "trql")?.ok_or_else(|| {
        anyhow::anyhow!(
            "no .trql file in world {}: a world needs a single .trql at its root",
            dataset_dir_path.display()
        )
    })?;
    let db = Trqs::from_bin(&trqs_path)?;
    let world_meta = load_world_meta(dataset_dir_path)?;

    // Built once here rather than per `/world/map` request: the bounds also fix
    // the origin queries are written against, so the map has to be readable for
    // the world to be usable.
    let (mut roadmap, map_path) = match xodr_path {
        Some(xodr) => {
            let r = parse_roadmap(&xodr).ok_or_else(|| {
                anyhow::anyhow!("failed to parse road network {}", xodr.display())
            })?;
            (r, xodr)
        }
        None => {
            let view = find_world_view(dataset_dir_path, &world_meta).ok_or_else(|| {
                anyhow::anyhow!(
                    "world {} has no .xodr and no world view picture beside its \
                     {WORLD_META_FILE}: it needs one of the two to draw a map",
                    dataset_dir_path.display()
                )
            })?;
            let mut r = view_roadmap(&view, &world_meta).ok_or_else(|| {
                anyhow::anyhow!(
                    "world {} declares the view {} but neither its {WORLD_META_FILE} \
                     nor the file itself states a drawing area",
                    dataset_dir_path.display(),
                    view.display()
                )
            })?;
            if let Some(v) = r.view.as_mut() {
                v.url = format!("/world/view/{world_name}");
            }
            (r, view)
        }
    };
    roadmap.scale = world_meta.coordinate_scale;

    let mut cameras: Vec<CameraEntry> = Vec::new();
    for dataset_dir_entry in std::fs::read_dir(dataset_dir_path)? {
        let camera_dir_path = dataset_dir_entry?.path();
        if !camera_dir_path.is_dir() {
            continue;
        }
        cameras.push(load_camera_entry_from_camera_dir(&camera_dir_path)?);
    }

    // A world with no camera directory at all — trajectories and a map, and no
    // viewpoint on them. That is a real dataset rather than a broken one:
    // WorldPose ships poses for footage it is not allowed to distribute. It
    // still needs *a* camera, because every request names one and the shared db
    // is reached through it, so one is synthesised from the world's own frame.
    //
    // It states no video, which the client already knows how to present — the
    // drawing area comes from `width`/`height` and playback runs on a clock
    // instead of a media element, as for any other video-less camera. It states
    // no pose either, so nothing is projected onto the blank panel and no
    // marker appears on the map, while map positions are unaffected.
    if cameras.is_empty() {
        let meta = VideoMeta {
            fps: world_meta.fps.unwrap_or(0.0),
            num_frames: world_meta.num_frames.unwrap_or(0),
            duration_s: world_meta.duration_s.unwrap_or(0.0),
            width: world_meta.width.unwrap_or(0.0) as u32,
            height: world_meta.height.unwrap_or(0.0) as u32,
            coordinate_scale: world_meta.coordinate_scale,
            ..Default::default()
        };
        cameras.push(CameraEntry {
            camera_name: SYNTHETIC_CAMERA.to_string(),
            camera_data: CameraData {
                video_path: None,
                video_meta: Some(meta),
                camera: None,
            },
            synthetic: true,
        });
    }

    // Camera markers close over the road network and the camera metadata, both
    // fixed at load, so the whole `/world/map` response is finished here and
    // served as a clone. Rasterizing the occlusion grid per request cost ~10s
    // on a map with a long ring road, whose lane bounding boxes span most of
    // the grid.
    roadmap.cameras = camera_markers(
        cameras
            .iter()
            .filter_map(|c| Some((c.camera_name.as_str(), c.camera_data.video_meta.as_ref()?))),
        &roadmap.lanes,
        roadmap.bounds,
    );

    Ok(WorldModeData {
        // Overwritten by the caller, which knows the grouping.
        dataset: String::new(),
        db,
        world_meta,
        cameras,
        roadmap,
        map_path,
    })
}

/// Walks `dir`, loading every world and standalone dataset at or below it.
///
/// `prefix` is the path walked so far below `--data-dir`, used to name what is
/// found; `depth` stops the descent at [`MAX_NEST_DEPTH`].
fn scan_dir(
    dir: &Path,
    prefix: &[String],
    depth: usize,
    world_data: &mut HashMap<WorldName, WorldModeData>,
    standalone_data: &mut Vec<StandaloneEntry>,
) -> anyhow::Result<()> {
    for root_dir_entry in std::fs::read_dir(dir)? {
        let dataset_dir_path = root_dir_entry?.path();
        if !dataset_dir_path.is_dir() {
            continue;
        }

        // The dataset is the first directory below `--data-dir` on the way to
        // this entry; an entry found at the top level is its own dataset.
        let dataset = match prefix.first() {
            Some(first) => first.clone(),
            None => dir_name(&dataset_dir_path)?,
        };

        match classify_dir(&dataset_dir_path)? {
            DirKind::World(xodr) => {
                let world_name: WorldName = qualified_name(prefix, &dataset_dir_path)?;
                // A world that will not load must not take the server down with
                // it — the same courtesy the camera loop below already extends.
                // A converter's output commonly lands before its `.trql` is
                // built, and one such directory should cost that world, not the
                // whole data directory.
                match load_world(&dataset_dir_path, &world_name, xodr) {
                    Ok(mut data) => {
                        data.dataset = dataset.clone();
                        if world_data.insert(world_name.clone(), data).is_some() {
                            eprintln!(
                                "warn: two worlds are both named {world_name}; the later one \
                                 ({}) wins",
                                dataset_dir_path.display()
                            );
                        }
                    }
                    Err(e) => eprintln!(
                        "warn: skipping world {}: {e}",
                        dataset_dir_path.display()
                    ),
                }
            }
            DirKind::Standalone => {
                for dataset_dir_entry in std::fs::read_dir(&dataset_dir_path)? {
                    let camera_dir_path = dataset_dir_entry?.path();
                    if !camera_dir_path.is_dir() {
                        continue;
                    }
                    // One unusable camera directory must not take the server down
                    // with it: a dataset can hold exports that never got a `.trql`
                    // (a bare `duckdb.db`, say) beside cameras that are perfectly
                    // loadable.
                    match load_standalone_entry_from_camera_dir(&camera_dir_path, &dataset) {
                        Ok(mut entry) => {
                            if !prefix.is_empty() {
                                entry.camera_name =
                                    format!("{}-{}", prefix.join("-"), entry.camera_name);
                            }
                            standalone_data.push(entry);
                        }
                        Err(e) => eprintln!(
                            "warning: skipping standalone camera {}: {e}",
                            camera_dir_path.display()
                        ),
                    }
                }
            }
            DirKind::Container => {
                if depth + 1 > MAX_NEST_DEPTH {
                    eprintln!(
                        "warn: {} is neither a world nor a dataset, and is too deeply \
                         nested to look inside",
                        dataset_dir_path.display()
                    );
                    continue;
                }
                let mut next = prefix.to_vec();
                next.push(dir_name(&dataset_dir_path)?);
                scan_dir(&dataset_dir_path, &next, depth + 1, world_data, standalone_data)?;
            }
        }
    }
    Ok(())
}

/// Scans `dir` into worlds and standalone cameras.
///
/// A subdirectory that declares a map (an `.xodr`, or a `world-metadata.json`
/// naming a picture) is a world, and its own subdirectories are its cameras.
/// One whose subdirectories carry `.trql` dbs is a standalone dataset, and
/// those subdirectories are flattened into the standalone camera list. Anything
/// else is taken to be a grouping directory and descended into, so a converter
/// may keep its worlds together under one name.
pub fn load_dbs_from_data_dir(
    dir: &Path,
) -> anyhow::Result<(HashMap<WorldName, WorldModeData>, Vec<StandaloneEntry>)> {
    let mut world_data = HashMap::new();
    let mut standalone_data = Vec::new();
    scan_dir(dir, &[], 0, &mut world_data, &mut standalone_data)?;
    Ok((world_data, standalone_data))
}
