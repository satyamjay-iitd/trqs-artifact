//! Shared application state: the loaded worlds and standalone cameras, plus the
//! lookup helpers the handlers use to resolve a request to a camera's data.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use trql::db::Trqs;

use crate::camera::{CameraModel, VideoMeta, WorldMeta};
use crate::query::QueryFrame;
use crate::roadmap::RoadmapResponse;

pub type WorldName = String;
pub type CameraName = String;

/// One camera's own assets: its source video, the parsed `metadata.json`, and
/// the derived pinhole camera model.
///
/// The trajectory db is deliberately *not* here: a world stores world-space
/// trajectories once, in [`WorldModeData::db`], and every camera in the world
/// projects that same db through its own camera model. Standalone cameras keep
/// their db alongside, in [`StandaloneEntry`].
pub struct CameraData {
    /// Path to the source video for this camera, if one was found in its
    /// directory.
    pub video_path: Option<PathBuf>,
    /// Backdrop to draw this camera's boxes over when it has no video: the
    /// picture its `metadata.json` names in `scene_view`, resolved against the
    /// camera directory. A trajectory-only dataset has no frame, and a blank
    /// one gives the boxes no spatial context; a scene drawing in the same
    /// coordinate space as the boxes does (a pitch, a floor plan).
    // pub view_path: Option<PathBuf>,
    /// Parsed `metadata.json` sitting beside the video, if present. Provides the
    /// frame rate needed to map frame intervals to video timestamps.
    pub video_meta: Option<VideoMeta>,
    /// Full pinhole camera model (intrinsics + extrinsics) built from
    /// `metadata.json`, used to project world-space boxes into pixel space.
    /// `None` for metadata without camera parameters.
    pub camera: Option<CameraModel>,
}

/// A named camera within a world.
pub struct CameraEntry {
    pub camera_name: CameraName,
    pub camera_data: CameraData,
    /// True for the stand-in viewpoint given to a world that ships no camera
    /// directory. It exists so requests have something to address — every world
    /// endpoint resolves through a camera — but it is not a viewpoint anyone
    /// can look through, so it is kept out of the listings the client builds
    /// its camera picker from. A world with no cameras should offer no choice
    /// of camera, not a choice of one fictional one.
    pub synthetic: bool,
}

/// A standalone camera: its own trajectory db plus its camera assets.
pub struct StandaloneEntry {
    /// The dataset this camera belongs to — see [`WorldModeData::dataset`].
    pub dataset: String,
    pub camera_name: CameraName,
    pub db: Trqs,
    /// Coordinate frame of `db`, read from the camera's own `metadata.json`
    /// (a standalone camera has no `world-metadata.json`).
    pub world_meta: WorldMeta,
    pub camera_data: CameraData,
}

/// A world (CARLA map): one trajectory db and one OpenDRIVE road network shared
/// by several cameras.
pub struct WorldModeData {
    /// The dataset this world belongs to: the first directory below
    /// `--data-dir` on the way to it, which is how a converter's output is
    /// grouped (`SoccerTrackv2`, `WorldPose`). A world sitting directly under
    /// the data directory is its own dataset.
    pub dataset: String,
    /// The world's single `.trql` db, holding world-space trajectories for every
    /// object in the world. Shared by all its cameras.
    pub db: Trqs,
    /// Coordinate frame of `db`, from the world's `world-metadata.json`.
    pub world_meta: WorldMeta,
    pub cameras: Vec<CameraEntry>,
    /// The parsed OpenDRIVE road network, built once at load — including
    /// `cameras` and their fields of view, so this is the complete
    /// `/world/map` response and is served as a clone. Its `bounds` also fix
    /// the origin queries are written against (see [`QueryFrame`]).
    pub roadmap: RoadmapResponse,
    /// Path the map came from — the `.xodr` that was parsed, or the picture a
    /// picture-world is drawn from. Kept for diagnostics and for re-reading the
    /// source file: `/world/view` serves the picture from here. `roadmap` is
    /// what map requests are served from.
    #[allow(dead_code)]
    pub map_path: PathBuf,
}

/// A trajectory db paired with the camera it is being viewed through — the
/// world's shared db in world mode, the camera's own db in standalone mode —
/// and the coordinate frames that db's stored coordinates live in.
#[derive(Clone, Copy)]
pub struct CameraView<'a> {
    pub db: &'a Trqs,
    /// Stored → CARLA metres, for projecting boxes and map positions.
    pub world: &'a WorldMeta,
    /// Query units → stored, for translating thresholds.
    pub frame: QueryFrame,
    pub cam: &'a CameraData,
    /// Whether `db` holds **world** coordinates rather than this camera's own
    /// pixels. A world's db is shared by every camera in it and is stored in
    /// metres, so a box only becomes pixels by projection; a standalone
    /// camera's db is already in its own image space and needs none. The
    /// difference decides what to do when the camera has no model: pass the box
    /// through, or produce nothing.
    pub world_space: bool,
}

/// Immutable after load — nothing mutates once the data directory is scanned.
pub struct AppState {
    pub world_mode: HashMap<WorldName, WorldModeData>,
    pub standalone_mode: Vec<StandaloneEntry>,
}

pub type SharedState = Arc<AppState>;

impl WorldModeData {
    /// Query units → stored coordinates for this world: map-grid metres measured
    /// from the road network's bottom-left corner.
    pub fn query_frame(&self) -> QueryFrame {
        QueryFrame::world(
            self.world_meta.coordinate_scale,
            self.world_meta.x_delta,
            self.world_meta.y_delta,
            self.roadmap.bounds,
        )
    }
}

impl AppState {
    /// The world with this name, if loaded.
    pub fn world(&self, world: &str) -> Option<&WorldModeData> {
        self.world_mode.get(world)
    }

    /// The world's shared db, viewed through one of its cameras.
    pub fn world_camera(&self, world: &str, camera: &str) -> Option<CameraView<'_>> {
        let w = self.world(world)?;
        // An unknown name falls back to the *synthetic* viewpoint only. A
        // client offered no camera to pick has no name to send, so it sends a
        // placeholder; but on a world that does have cameras a wrong name is a
        // mistake and should still be reported rather than quietly resolving to
        // whichever camera happens to be first.
        let cam = w
            .cameras
            .iter()
            .find(|c| c.camera_name == camera)
            .or_else(|| w.cameras.iter().find(|c| c.synthetic))
            .map(|c| &c.camera_data)?;
        Some(CameraView {
            db: &w.db,
            world: &w.world_meta,
            frame: w.query_frame(),
            cam,
            world_space: true,
        })
    }

    /// The world's shared db viewed through an arbitrary camera, for requests
    /// that need the world's metadata (coordinate scale, frame rate) but not a
    /// specific viewpoint. All cameras in a world share those, so the first one
    /// stands in for the world.
    pub fn world_repr(&self, world: &str) -> Option<CameraView<'_>> {
        let w = self.world(world)?;
        let cam = w.cameras.first().map(|c| &c.camera_data)?;
        Some(CameraView {
            db: &w.db,
            world: &w.world_meta,
            frame: w.query_frame(),
            cam,
            world_space: true,
        })
    }

    /// A standalone camera's own db and data, by name.
    pub fn standalone_camera(&self, camera: &str) -> Option<CameraView<'_>> {
        self.standalone_mode
            .iter()
            .find(|c| c.camera_name == camera)
            .map(|c| CameraView {
                db: &c.db,
                world: &c.world_meta,
                // No map, so no grid to measure against: the db's own axes.
                frame: QueryFrame::standalone(c.world_meta.coordinate_scale),
                cam: &c.camera_data,
                world_space: false,
            })
    }
}
