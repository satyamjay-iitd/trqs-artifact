//! Request/response bodies shared by the handlers (serde + OpenAPI schemas).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use trql_types::{ObjId, intervals::Interval, obj::BoundingBox};

// --- schema ----------------------------------------------------------------

#[derive(Serialize, ToSchema)]
pub struct ColInfo {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variants: Option<Vec<String>>,
}

#[derive(Serialize, ToSchema)]
pub struct SchemaResponse {
    pub num_objs: u32,
    pub columns: Vec<ColInfo>,
}

// --- query -----------------------------------------------------------------

/// Unit the temporal literals in a query are written in — the `duration(…, n)`
/// minimum length and the `THEN [gap]` window.
///
/// The engine counts in frames throughout. `Seconds` (the default) converts on
/// the way in using the camera's frame rate; `Frames` passes the numbers
/// through untouched, for queries written against frame counts directly.
#[derive(Deserialize, Serialize, ToSchema, Clone, Copy, PartialEq, Eq, Debug, Default)]
#[serde(rename_all = "lowercase")]
pub enum TimeUnit {
    #[default]
    Seconds,
    Frames,
}

#[derive(Deserialize, ToSchema)]
pub struct QueryRequest {
    /// In world mode, optionally the camera (within the path's world) whose
    /// metadata supplies the coordinate scale and frame rate; the db queried is
    /// the world's shared one either way, so the results do not depend on it.
    /// Ignored in standalone mode, where the camera comes from the path.
    #[serde(default)]
    pub camera: Option<String>,
    /// DSL query string: SELECT … WHERE …
    pub dsl: String,
    /// Unit the query's temporal literals are in. Defaults to seconds, which is
    /// how queries behaved before this was selectable.
    #[serde(default)]
    pub time_unit: TimeUnit,
}

#[derive(Serialize, ToSchema)]
pub struct QueryResult {
    #[schema(value_type = Vec<u32>)]
    pub tuple: Vec<ObjId>,
    /// Each interval is [start, end] (inclusive), in frames.
    #[schema(value_type = Vec<[u32; 2]>)]
    pub intervals: Vec<Interval>,
    /// The same intervals expressed in seconds (`frame / fps`), for display.
    pub intervals_s: Vec<[f64; 2]>,
}

/// A region the query tested against, in the coordinates the boxes are drawn in.
///
/// Returned so the dashboard can draw the region over the video, which the results alone
/// cannot support: `INSIDE` expands at parse time into comparisons that no longer name a
/// shape. Coordinates go through the same translation as the query's own thresholds, so a
/// vertex sits in the same space as a `/bboxes` box.
#[derive(Serialize, ToSchema)]
pub struct QueryRegionInfo {
    /// The name it was declared under in `REGIONS`, or absent for an inline literal.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The query variable whose box was tested against it.
    pub var: String,
    /// Whether the whole box had to be inside, or only its centre.
    pub whole_box: bool,
    /// The ring, implicitly closed.
    pub vertices: Vec<[f64; 2]>,
}

#[derive(Serialize, ToSchema)]
pub struct QueryResponse {
    pub count: usize,
    pub elapsed_ms: f64,
    pub results: Vec<QueryResult>,
    /// Regions the query tested against, drawable over this camera's video.
    ///
    /// Empty when the query has no `INSIDE`, and also when the regions cannot be placed on
    /// the video: in world mode they are map coordinates, and turning those into pixels
    /// needs the camera model that `draws_boxes` reports on.
    pub regions: Vec<QueryRegionInfo>,
}

// --- camera info -----------------------------------------------------------

/// Metadata about a single camera/video (frame rate, timeline, intrinsics).
#[derive(Serialize, ToSchema)]
pub struct CameraInfoResponse {
    /// Camera name (its directory stem).
    pub name: String,
    /// Whether a playable video exists for this camera.
    pub has_video: bool,
    /// Whether a scene picture exists to draw this camera's boxes over in place
    /// of a blank frame, served from `/standalone/view/{camera}`. It spans
    /// exactly the `width` x `height` drawing area below, so the client lays it
    /// under the box overlay without interpreting what it depicts.
    pub has_view: bool,
    /// Whether boxes can be drawn over this camera's video at all. False for a
    /// world camera with no camera model: the world stores map-space
    /// coordinates, and without a model there is no way to project them into
    /// this view, so `/bboxes` returns none for it. Standalone cameras are
    /// always true — their stored coordinates already are their own pixels.
    pub draws_boxes: bool,
    /// Frames per second, used to convert frame intervals to timestamps.
    pub fps: f64,
    pub num_frames: u32,
    pub duration_s: f64,
    /// Frame size in pixels (the resolution bounding boxes live in). For a
    /// video-less camera it is instead the drawing area fitted to the
    /// trajectories, measured from the origin, which the client scales boxes
    /// against.
    pub width: u32,
    pub height: u32,
    /// Horizontal field of view, in degrees.
    pub fov_deg: f64,
}

// --- bounding boxes --------------------------------------------------------

#[derive(Deserialize, ToSchema)]
pub struct BBrequest {
    /// In world mode, the camera (within the path's world) to project through:
    /// the objects come from the world's shared db, `camera` decides which
    /// viewpoint their boxes are rendered for. Required in world mode; ignored
    /// in standalone mode (camera comes from the path).
    #[serde(default)]
    pub camera: Option<String>,
    /// Object ids whose per-frame boxes are requested.
    pub oids: Vec<u32>,
    /// Frame range start (global frame counter).
    pub ts_start: u64,
    /// Frame range end (global frame counter).
    pub ts_end: u64,
}

/// Pixel bounding box, documentation mirror of [`BoundingBox`] for OpenAPI.
#[derive(ToSchema)]
#[allow(dead_code)]
pub struct CameraBoxSchema {
    pub x1: u32,
    pub y1: u32,
    pub x2: u32,
    pub y2: u32,
}

#[derive(Serialize, ToSchema)]
pub struct BBresponse {
    /// One frame-id → box map per requested object id, in the same order as
    /// `BBrequest::oids`.
    #[schema(value_type = Vec<std::collections::HashMap<String, CameraBoxSchema>>)]
    pub bbox_sequences: Vec<HashMap<u64, BoundingBox>>,
    /// Per object (same order as `oids`): frame-id → the vehicle's ground-centre
    /// position `[x, y]` in the **map** frame (metres), matching the worldmap
    /// coordinates so the frontend can plot each vehicle as a moving dot. Empty
    /// in standalone mode (no worldmap).
    #[schema(value_type = Vec<std::collections::HashMap<String, [f64; 2]>>)]
    pub world_positions: Vec<HashMap<u64, [f64; 2]>>,
    /// Whether boxes could be produced for this view at all.
    ///
    /// False when the db is in world coordinates but the camera has no pose or
    /// FOV to project them through — the dataset never surveyed it. The
    /// `bbox_sequences` are then empty by design rather than by accident, and a
    /// client can say so instead of showing an empty video overlay. Map
    /// positions are unaffected: they need no camera.
    pub boxes_available: bool,
}
