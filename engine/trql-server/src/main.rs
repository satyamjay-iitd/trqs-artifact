//! TrQL HTTP query server.
//!
//! Data is organised as **worlds** (a CARLA map: several cameras sharing one
//! OpenDRIVE road network and one world-space trajectory db) and **standalone**
//! cameras (no map, one db each). Every request path selects one of the two
//! modes; see `api.md` for the full surface.
//!
//! This file holds only the request handlers, the router, and `main`. The
//! projection maths, roadmap parsing, data loading, query scaling, DTOs, error
//! type and shared state live in the sibling modules below.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    Json, Router,
    body::Body,
    extract::{Path as AxumPath, State},
    http::{HeaderMap, Request, header},
    response::Response,
    routing::{get, post},
};
use clap::Parser;
use tower::util::ServiceExt;
use tower_http::services::ServeFile;

use trql::db::Trqs;
use trql::eval::AlgebraCtx;
use trql_dsl::parse_query_regions;
use trql_types::obj::BoundingBox;

use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

mod camera;
mod dto;
mod error;
mod loader;
mod query;
mod roadmap;
mod state;

use camera::{object_height, world_to_camera};
use dto::{
    BBrequest, BBresponse, CameraBoxSchema, CameraInfoResponse, ColInfo, QueryRegionInfo,
    QueryRequest, QueryResponse, QueryResult, SchemaResponse, TimeUnit,
};
use error::AppResult;
use query::{scale_query_time, schema_to_cols, translate_query_coords, translate_region};
use roadmap::{CameraMarker, LanePolygon, RoadmapResponse};
use state::{AppState, CameraData, CameraView, SharedState};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(about = "TrQL HTTP query server")]
struct Args {
    /// Address to listen on
    #[arg(long, default_value = "127.0.0.1:3000")]
    addr: String,

    /// Directory containing worlds and standalone-camera datasets
    #[arg(long)]
    data_dir: PathBuf,
}

// ---------------------------------------------------------------------------
// Shared request-handling helpers (the common core of the world/standalone
// handler pairs — kept here alongside the handlers they serve).
// ---------------------------------------------------------------------------

/// Camera/video metadata as an API response.
/// `world_space` distinguishes the two modes, as in [`bboxes_on`]: it decides
/// whether this camera can render boxes at all.
fn camera_info(name: &str, cam: &CameraData, world_space: bool) -> CameraInfoResponse {
    let meta = cam.video_meta.clone().unwrap_or_default();
    CameraInfoResponse {
        name: name.to_string(),
        has_video: cam.video_path.is_some(),
        has_view: false,//cam.view_path.is_some(),
        // Standalone coordinates are already this camera's pixels; world ones
        // have to be projected, which needs a camera model.
        draws_boxes: !world_space || cam.camera.is_some(),
        // Fall back to a sane default so the client can still map frames->time.
        fps: if meta.fps > 0.0 { meta.fps } else { 30.0 },
        num_frames: meta.num_frames,
        duration_s: meta.duration_s,
        width: meta.width,
        height: meta.height,
        fov_deg: meta.fov_deg,
    }
}

/// Streams a file honouring the client's `Range` header (206 Partial Content on
/// seek), used by both video endpoints.
async fn serve_video(path: PathBuf, headers: HeaderMap) -> AppResult<Response> {
    // Rebuild a bare request carrying only the Range header so ServeFile can
    // produce a 206 Partial Content response when the client seeks.
    let mut builder = Request::builder().uri("/");
    if let Some(range) = headers.get(header::RANGE) {
        builder = builder.header(header::RANGE, range);
    }
    let req = builder.body(Body::empty())?;

    // ServeFile's error type is Infallible, so this never actually fails.
    let res = ServeFile::new(path)
        .oneshot(req)
        .await
        .map_err(|e| anyhow::anyhow!("failed to serve video: {e}"))?;
    Ok(res.map(Body::new))
}

/// Executes a DSL query against a view's db (shared by both query endpoints).
/// In world mode that db covers the whole world, so the result set is the same
/// whichever camera the view was resolved through — the camera only supplies the
/// coordinate scale and frame rate, which every camera in a world shares.
///
/// Coordinate thresholds are entered in map-grid units — the numbers on the
/// dashboard map's axes — while the db stores CARLA metres × `coordinate_scale`
/// shifted by the world's deltas. Temporal parameters are entered in whichever
/// `time_unit` the caller asked for. Both are converted to the stored order
/// here.
fn run_query_on(
    view: CameraView,
    dsl: &str,
    time_unit: TimeUnit,
    world_space: bool,
) -> AppResult<Json<QueryResponse>> {
    let db = view.db;
    let (constraints, mut expr, used_regions) =
        parse_query_regions(dsl, db.schema()).map_err(|e| anyhow::anyhow!("{e}"))?;

    if view.frame.is_usable() {
        translate_query_coords(&mut expr, &view.frame);
    }

    // Regions are reported in the space the boxes are drawn in, so they are only useful
    // where that space is the db's own — standalone, whose stored coordinates *are* this
    // camera's pixels. A world stores map coordinates, and placing those on a video needs
    // the camera model that projects the boxes; until a region is projected the same way,
    // sending it would put a shape on the video that is not where the query looked.
    //
    // Regions follow the thresholds exactly, including when there is nothing to follow: a
    // db with no `coordinate_scale` leaves the query untranslated, and the regions are then
    // already in stored units and pass through as written.
    let regions = if world_space {
        Vec::new()
    } else {
        used_regions
            .iter()
            .map(|r| QueryRegionInfo {
                name: r.name.clone(),
                var: r.var.clone(),
                whole_box: r.whole_box,
                vertices: if view.frame.is_usable() {
                    translate_region(&r.vertices, &view.frame)
                } else {
                    r.vertices.clone()
                },
            })
            .collect()
    };

    // Frame rate: seconds→frames on the way in, frames→seconds on the way out.
    let fps = view
        .cam
        .video_meta
        .as_ref()
        .map(|m| m.fps)
        .filter(|f| *f > 0.0)
        .unwrap_or(1.0);
    // In `Frames` the literals are already in the engine's own unit, so they are
    // handed over untouched. `intervals_s` below is unaffected either way — it
    // is derived from the frame indices the engine returns, not from the input.
    if time_unit == TimeUnit::Seconds && fps != 1.0 {
        scale_query_time(&mut expr, fps);
    }

    let poly_trajs = db.poly_trajs();
    let ctx = AlgebraCtx {
        db,
        poly_trajs: Some(poly_trajs),
    };

    let t0 = Instant::now();
    let results: Vec<QueryResult> = ctx
        .run(&constraints, &expr)
        .into_iter()
        .map(|(tuple, intervals)| {
            let frames = intervals.maximal();
            let intervals_s = frames
                .iter()
                .map(|&(s, e)| [s as f64 / fps, e as f64 / fps])
                .collect();
            QueryResult {
                tuple: tuple.to_vec(),
                intervals: frames,
                intervals_s,
            }
        })
        .collect();
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    Ok(Json(QueryResponse {
        count: results.len(),
        elapsed_ms,
        results,
        regions,
    }))
}

/// Per-frame boxes (and, in world mode, map-frame ground positions) for a set of
/// objects over a frame range — shared by both bbox endpoints.
///
/// `world_space` says what the db's stored coordinates mean, which decides both
/// outputs:
///
///   * **world mode** (`true`) — coordinates are world-space, so they carry a
///     map position, and reaching the camera's pixels needs `view.cam`'s model.
///     Any camera in a world can render boxes for any object the world knows
///     about — but a camera with no model can render none, and returns none
///     rather than leaking map numbers into pixel space.
///   * **standalone mode** (`false`) — coordinates already *are* this camera's
///     pixels, so they pass through untouched and there is no map to place them
///     on.
fn bboxes_on(
    view: CameraView,
    oids: &[u32],
    ts_start: u64,
    ts_end: u64,
    world_space: bool,
) -> BBresponse {
    let cam = view.cam;
    let meta = cam.video_meta.clone().unwrap_or_default();
    let (lo, hi) = if ts_start <= ts_end {
        (ts_start, ts_end)
    } else {
        (ts_end, ts_start)
    };

    let mut bbox_sequences: Vec<HashMap<u64, BoundingBox>> = Vec::with_capacity(oids.len());
    let mut world_positions: Vec<HashMap<u64, [f64; 2]>> = Vec::with_capacity(oids.len());

    // Ground centre of a stored world box, in the map frame (metres). Un-shift /
    // un-scale through the world's coordinate frame (as `world_to_camera`), then
    // negate Y like the camera markers.
    let to_map_pos = |bb: &BoundingBox| -> [f64; 2] {
        let cx = view
            .world
            .x_to_metres((bb.x1 as i32 + bb.x2 as i32) as f64 * 0.5);
        let cy = view
            .world
            .y_to_metres((bb.y1 as i32 + bb.y2 as i32) as f64 * 0.5);
        [cx, -cy]
    };

    for &oid in oids {
        let mut bbox_sequence = HashMap::new();
        let mut pos_sequence = HashMap::new();

        if let Some(poly) = view.db.poly_trajs().get(oid as usize) {
            // Vehicle height (by class) used to lift the footprint.
            let height = object_height(view.db, oid);
            // Convert a stored box into the response box.
            //
            // A world's db is in metres, so it only becomes pixels by
            // projection — and projection needs the camera's pose and FOV. When
            // the dataset never surveyed the camera (Alfheim, SoccerTrack: a
            // fixed rig with no calibration published) there is no honest
            // answer, so no box is produced rather than passing metres off as
            // pixels. That fallback used to hand the client coordinates like
            // x=5810 for a 4096-wide frame.
            //
            // A standalone camera's db is already in its own image space, so it
            // needs no projection and no camera model.
            let to_box = |world_bb: &BoundingBox| -> Option<BoundingBox> {
                match &cam.camera {
                    Some(c) => Some(world_to_camera(world_bb, c, &meta, view.world, height)),
                    None if view.world_space => None,
                    None => Some(world_bb.clone()),
                }
            };

            // Frame ids are 0-based and contiguous, so walk each integer the
            // object spans within the requested range.
            for &(seg_start, seg_end) in poly.canonical_segments() {
                let start = (seg_start as u64).max(lo);
                let end = (seg_end as u64).min(hi);
                for fid in start..=end {
                    if let Some(world_bb) = poly.eval(fid as u32) {
                        // Map positions survive a missing camera model — they
                        // are the world's own frame, not a projection — so the
                        // map keeps its dots even when the video gets no boxes.
                        // println!("world_bb: {:?}", world_bb);
                        if let Some(bb) = to_box(&world_bb) {
                            bbox_sequence.insert(fid, bb);
                        }
                        if world_space {
                            pos_sequence.insert(fid, to_map_pos(&world_bb));
                        }
                    }
                }
            }
        }

        bbox_sequences.push(bbox_sequence);
        world_positions.push(pos_sequence);
    }


    // println!("world positions: {:?}", world_positions);

    BBresponse {
        boxes_available: cam.camera.is_some() || !view.world_space,
        bbox_sequences,
        world_positions,
    }
}

/// Schema + object count for a db.
fn schema_of(db: &Trqs) -> SchemaResponse {
    SchemaResponse {
        num_objs: db.num_objs(),
        columns: schema_to_cols(db),
    }
}

// ---------------------------------------------------------------------------
// Discovery handlers
// ---------------------------------------------------------------------------

/// GET /world/database — loaded worlds as `{ dataset: { world: [camera…] } }`.
#[utoipa::path(
    get,
    path = "/world/database",
    responses((status = 200, description = "dataset → world → camera-names tree"))
)]
async fn get_world_database(
    State(state): State<SharedState>,
) -> Json<HashMap<String, HashMap<String, Vec<String>>>> {
    // Grouped by dataset — the directory a converter wrote its output into —
    // so the client can offer that as a choice before the world. Flat, the list
    // is 116 worlds with no indication of what belongs with what.
    let mut out: HashMap<String, HashMap<String, Vec<String>>> = HashMap::new();
    for (world, data) in &state.world_mode {
        // Synthetic viewpoints are excluded: the client builds its camera
        // picker from this, and a world with no real camera should offer none.
        let mut cams: Vec<String> = data
            .cameras
            .iter()
            .filter(|c| !c.synthetic)
            .map(|c| c.camera_name.clone())
            .collect();
        cams.sort();
        out.entry(data.dataset.clone())
            .or_default()
            .insert(world.clone(), cams);
    }
    Json(out)
}

/// GET /standalone/database — standalone cameras as `{ dataset: [camera…] }`.
#[utoipa::path(
    get,
    path = "/standalone/database",
    responses((status = 200, description = "dataset → standalone camera names"))
)]
async fn get_standalone_database(State(state): State<SharedState>) -> Json<HashMap<String, Vec<String>>> {
    // Grouped by dataset, as the world listing is.
    let mut out: HashMap<String, Vec<String>> = HashMap::new();
    for c in &state.standalone_mode {
        out.entry(c.dataset.clone())
            .or_default()
            .push(c.camera_name.clone());
    }
    for cams in out.values_mut() {
        cams.sort();
    }
    Json(out)
}

/// GET /world/camera_info/{world} — camera metadata for every camera in a world.
#[utoipa::path(
    get,
    path = "/world/camera_info/{world}",
    params(("world" = String, Path, description = "world name")),
    responses(
        (status = 200, description = "camera info for each camera in the world", body = Vec<CameraInfoResponse>),
        (status = 400, description = "world not found")
    )
)]
async fn list_world_cameras(
    State(state): State<SharedState>,
    AxumPath(world): AxumPath<String>,
) -> AppResult<Json<Vec<CameraInfoResponse>>> {
    let w = state
        .world(&world)
        .ok_or_else(|| anyhow::anyhow!("world not found: {world}"))?;
    let infos = w
        .cameras
        .iter()
        .filter(|c| !c.synthetic)
        .map(|c| camera_info(&c.camera_name, &c.camera_data, true))
        .collect();
    Ok(Json(infos))
}

/// GET /world/camera_info/{world}/{camera} — metadata for one world camera.
#[utoipa::path(
    get,
    path = "/world/camera_info/{world}/{camera}",
    params(
        ("world" = String, Path, description = "world name"),
        ("camera" = String, Path, description = "camera name")
    ),
    responses(
        (status = 200, description = "camera metadata", body = CameraInfoResponse),
        (status = 400, description = "camera not found")
    )
)]
async fn get_world_camera_info(
    State(state): State<SharedState>,
    AxumPath((world, camera)): AxumPath<(String, String)>,
) -> AppResult<Json<CameraInfoResponse>> {
    let view = state
        .world_camera(&world, &camera)
        .ok_or_else(|| anyhow::anyhow!("camera not found: {world}/{camera}"))?;
    Ok(Json(camera_info(&camera, view.cam, true)))
}

/// GET /standalone/camera_info/{camera} — metadata for one standalone camera.
#[utoipa::path(
    get,
    path = "/standalone/camera_info/{camera}",
    params(("camera" = String, Path, description = "camera name")),
    responses(
        (status = 200, description = "camera metadata", body = CameraInfoResponse),
        (status = 400, description = "camera not found")
    )
)]
async fn get_standalone_camera_info(
    State(state): State<SharedState>,
    AxumPath(camera): AxumPath<String>,
) -> AppResult<Json<CameraInfoResponse>> {
    let view = state
        .standalone_camera(&camera)
        .ok_or_else(|| anyhow::anyhow!("camera not found: {camera}"))?;
    Ok(Json(camera_info(&camera, view.cam, false)))
}

// ---------------------------------------------------------------------------
// Video handlers
// ---------------------------------------------------------------------------

/// GET /world/video/{world}/{camera} — stream a world camera's video.
#[utoipa::path(
    get,
    path = "/world/video/{world}/{camera}",
    params(
        ("world" = String, Path, description = "world name"),
        ("camera" = String, Path, description = "camera name")
    ),
    responses(
        (status = 200, description = "video stream (supports Range requests)", content_type = "video/mp4"),
        (status = 400, description = "camera or video not found")
    )
)]
async fn get_world_video(
    State(state): State<SharedState>,
    AxumPath((world, camera)): AxumPath<(String, String)>,
    headers: HeaderMap,
) -> AppResult<Response> {
    let path = state
        .world_camera(&world, &camera)
        .ok_or_else(|| anyhow::anyhow!("camera not found: {world}/{camera}"))?
        .cam
        .video_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no video for camera: {world}/{camera}"))?;
    serve_video(path, headers).await
}

/// GET /standalone/video/{camera} — stream a standalone camera's video.
#[utoipa::path(
    get,
    path = "/standalone/video/{camera}",
    params(("camera" = String, Path, description = "camera name")),
    responses(
        (status = 200, description = "video stream (supports Range requests)", content_type = "video/mp4"),
        (status = 400, description = "camera or video not found")
    )
)]
async fn get_standalone_video(
    State(state): State<SharedState>,
    AxumPath(camera): AxumPath<String>,
    headers: HeaderMap,
) -> AppResult<Response> {
    let path = state
        .standalone_camera(&camera)
        .ok_or_else(|| anyhow::anyhow!("camera not found: {camera}"))?
        .cam
        .video_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no video for camera: {camera}"))?;
    serve_video(path, headers).await
}

/// GET /standalone/view/{camera} — the picture a video-less camera's boxes are
/// drawn over, in place of a blank frame.
///
/// Served as a file rather than inlined in `/camera_info` so the browser can
/// cache it: a pitch or floor plan is static for the life of the dataset.
// #[utoipa::path(
//     get,
//     path = "/standalone/view/{camera}",
//     params(("camera" = String, Path, description = "camera name")),
//     responses(
//         (status = 200, description = "The camera's scene picture", content_type = "image/svg+xml"),
//         (status = 400, description = "camera not found, or it declares no scene view")
//     )
// )]
// async fn get_standalone_view(
//     State(state): State<SharedState>,
//     AxumPath(camera): AxumPath<String>,
//     headers: HeaderMap,
// ) -> AppResult<Response> {
//     // let path = state
//     //     .standalone_camera(&camera)
//     //     .ok_or_else(|| anyhow::anyhow!("camera not found: {camera}"))?
//     //     .cam
//     //     .view_path
//     //     .clone()
//     //     .ok_or_else(|| anyhow::anyhow!("no scene view for camera: {camera}"))?;
//     serve_video(path, headers).await
// }

// ---------------------------------------------------------------------------
// Worldmap handler
// ---------------------------------------------------------------------------

/// GET /world/map/{world} — OpenDRIVE road network + cameras for a world.
#[utoipa::path(
    get,
    path = "/world/map/{world}",
    params(("world" = String, Path, description = "world name")),
    responses(
        (status = 200, description = "OpenDRIVE road network, cameras and fields of view", body = RoadmapResponse),
        (status = 400, description = "no map available for world")
    )
)]
async fn get_worldmap(
    State(state): State<SharedState>,
    AxumPath(world): AxumPath<String>,
) -> AppResult<Json<RoadmapResponse>> {
    let w = state
        .world(&world)
        .ok_or_else(|| anyhow::anyhow!("world not found: {world}"))?;

    // Fully built at load — lanes, bounds, scale, cameras and their fields of
    // view are all fixed once the data directory is scanned, so serving is a
    // clone.
    Ok(Json(w.roadmap.clone()))
}

/// GET /world/view/{world} — the picture a non-`.xodr` world is drawn from.
///
/// Served as a file rather than embedded in `/world/map` so the browser can
/// cache it: a stadium plan or floor plan is static for the life of the dataset,
/// while the map response carries per-request camera state.
#[utoipa::path(
    get,
    path = "/world/view/{world}",
    params(("world" = String, Path, description = "world name")),
    responses(
        (status = 200, description = "The world's map picture (SVG or raster)", content_type = "image/svg+xml"),
        (status = 400, description = "world not found, or its map is an .xodr road network")
    )
)]
async fn get_world_view(
    State(state): State<SharedState>,
    AxumPath(world): AxumPath<String>,
    headers: HeaderMap,
) -> AppResult<Response> {
    let w = state
        .world(&world)
        .ok_or_else(|| anyhow::anyhow!("world not found: {world}"))?;

    let view = w.roadmap.view.as_ref().ok_or_else(|| {
        anyhow::anyhow!("world {world} is drawn from a road network, not a picture")
    })?;
    let content_type = view.content_type.clone();
    let path = w.map_path.clone();

    let mut res = serve_video(path, headers).await?;
    // `ServeFile` guesses from the extension; state what the loader decided so
    // the two can never disagree.
    if let Ok(value) = content_type.parse() {
        res.headers_mut().insert(header::CONTENT_TYPE, value);
    }
    Ok(res)
}

// ---------------------------------------------------------------------------
// Query handlers
// ---------------------------------------------------------------------------

/// POST /world/query/{world} — run a DSL query against the world's shared db.
/// Body: `{ "dsl": "SELECT … WHERE …" }`; the optional `camera` field no longer
/// selects a db (there is one per world) and only picks whose metadata supplies
/// the coordinate scale and frame rate.
#[utoipa::path(
    post,
    path = "/world/query/{world}",
    params(("world" = String, Path, description = "world name")),
    request_body = QueryRequest,
    responses(
        (status = 200, description = "query results", body = QueryResponse),
        (status = 400, description = "invalid query, or camera not found")
    )
)]
async fn run_world_query(
    State(state): State<SharedState>,
    AxumPath(world): AxumPath<String>,
    Json(req): Json<QueryRequest>,
) -> AppResult<Json<QueryResponse>> {
    let view = match req.camera.as_deref() {
        Some(camera) => state
            .world_camera(&world, camera)
            .ok_or_else(|| anyhow::anyhow!("camera not found: {world}/{camera}"))?,
        None => state
            .world_repr(&world)
            .ok_or_else(|| anyhow::anyhow!("world not found or has no cameras: {world}"))?,
    };
    run_query_on(view, &req.dsl, req.time_unit, true)
}

/// POST /standalone/query/{camera} — run a DSL query against a standalone camera.
/// Body: `{ "dsl": "SELECT … WHERE …" }`.
#[utoipa::path(
    post,
    path = "/standalone/query/{camera}",
    params(("camera" = String, Path, description = "camera name")),
    request_body = QueryRequest,
    responses(
        (status = 200, description = "query results", body = QueryResponse),
        (status = 400, description = "invalid query, or camera not found")
    )
)]
async fn run_standalone_query(
    State(state): State<SharedState>,
    AxumPath(camera): AxumPath<String>,
    Json(req): Json<QueryRequest>,
) -> AppResult<Json<QueryResponse>> {
    let view = state
        .standalone_camera(&camera)
        .ok_or_else(|| anyhow::anyhow!("camera not found: {camera}"))?;
    run_query_on(view, &req.dsl, req.time_unit, false)
}

// ---------------------------------------------------------------------------
// Bounding-box handlers
// ---------------------------------------------------------------------------

/// POST /world/bboxes/{world} — per-frame camera boxes + map positions.
/// Body: `{ "camera": "…", "oids": [..], "ts_start": .., "ts_end": .. }`.
#[utoipa::path(
    post,
    path = "/world/bboxes/{world}",
    params(("world" = String, Path, description = "world name")),
    request_body = BBrequest,
    responses(
        (status = 200, description = "per-frame camera boxes and map positions", body = BBresponse),
        (status = 400, description = "camera not found")
    )
)]
async fn get_world_bboxes(
    State(state): State<SharedState>,
    AxumPath(world): AxumPath<String>,
    Json(req): Json<BBrequest>,
) -> AppResult<Json<BBresponse>> {
    let camera = req
        .camera
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("world bboxes require a 'camera' field"))?;
    let view = state
        .world_camera(&world, camera)
        .ok_or_else(|| anyhow::anyhow!("camera not found: {world}/{camera}"))?;
    Ok(Json(bboxes_on(
        view,
        &req.oids,
        req.ts_start,
        req.ts_end,
        true,
    )))
}

/// POST /standalone/bboxes/{camera} — per-frame camera boxes (no map positions).
/// Body: `{ "oids": [..], "ts_start": .., "ts_end": .. }`.
#[utoipa::path(
    post,
    path = "/standalone/bboxes/{camera}",
    params(("camera" = String, Path, description = "camera name")),
    request_body = BBrequest,
    responses(
        (status = 200, description = "per-frame camera boxes", body = BBresponse),
        (status = 400, description = "camera not found")
    )
)]
async fn get_standalone_bboxes(
    State(state): State<SharedState>,
    AxumPath(camera): AxumPath<String>,
    Json(req): Json<BBrequest>,
) -> AppResult<Json<BBresponse>> {
    let view = state
        .standalone_camera(&camera)
        .ok_or_else(|| anyhow::anyhow!("camera not found: {camera}"))?;
    Ok(Json(bboxes_on(
        view,
        &req.oids,
        req.ts_start,
        req.ts_end,
        false,
    )))
}

// ---------------------------------------------------------------------------
// Schema handlers
// ---------------------------------------------------------------------------

/// POST /world/schema/{world} — schema of the world's single shared db.
#[utoipa::path(
    post,
    path = "/world/schema/{world}",
    params(("world" = String, Path, description = "world name")),
    responses(
        (status = 200, description = "schema and object count", body = SchemaResponse),
        (status = 400, description = "world not found")
    )
)]
async fn get_world_schema(
    State(state): State<SharedState>,
    AxumPath(world): AxumPath<String>,
) -> AppResult<Json<SchemaResponse>> {
    let w = state
        .world(&world)
        .ok_or_else(|| anyhow::anyhow!("world not found: {world}"))?;
    Ok(Json(schema_of(&w.db)))
}

/// POST /standalone/schema/{camera} — schema for a standalone camera.
#[utoipa::path(
    post,
    path = "/standalone/schema/{camera}",
    params(("camera" = String, Path, description = "camera name")),
    responses(
        (status = 200, description = "schema and object count", body = SchemaResponse),
        (status = 400, description = "camera not found")
    )
)]
async fn get_standalone_schema(
    State(state): State<SharedState>,
    AxumPath(camera): AxumPath<String>,
) -> AppResult<Json<SchemaResponse>> {
    let view = state
        .standalone_camera(&camera)
        .ok_or_else(|| anyhow::anyhow!("camera not found: {camera}"))?;
    Ok(Json(schema_of(view.db)))
}

// ---------------------------------------------------------------------------
// OpenAPI
// ---------------------------------------------------------------------------

#[derive(OpenApi)]
#[openapi(
    paths(
        get_world_database,
        get_standalone_database,
        list_world_cameras,
        get_world_camera_info,
        get_standalone_camera_info,
        get_world_video,
        get_standalone_video,
        get_worldmap,
        get_world_view,
        // get_standalone_view,
        run_world_query,
        run_standalone_query,
        get_world_bboxes,
        get_standalone_bboxes,
        get_world_schema,
        get_standalone_schema,
    ),
    components(schemas(
        CameraInfoResponse,
        QueryRequest,
        QueryResponse,
        QueryResult,
        QueryRegionInfo,
        BBrequest,
        BBresponse,
        CameraBoxSchema,
        SchemaResponse,
        ColInfo,
        RoadmapResponse,
        CameraMarker,
        LanePolygon,
    ))
)]
struct ApiDoc;

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if !args.data_dir.is_dir() {
        panic!("data_dir {:?} is not a directory", args.data_dir);
    }

    let (world_data, standalone_data) = loader::load_dbs_from_data_dir(&args.data_dir)?;
    let n_worlds = world_data.len();
    let n_standalone = standalone_data.len();

    let state: SharedState = Arc::new(AppState {
        world_mode: world_data,
        standalone_mode: standalone_data,
    });

    let app = Router::new()
        .route("/world/database", get(get_world_database))
        .route("/world/camera_info/{world}", get(list_world_cameras))
        .route(
            "/world/camera_info/{world}/{camera}",
            get(get_world_camera_info),
        )
        .route("/world/video/{world}/{camera}", get(get_world_video))
        .route("/world/map/{world}", get(get_worldmap))
        .route("/world/view/{world}", get(get_world_view))
        .route("/world/query/{world}", post(run_world_query))
        .route("/world/bboxes/{world}", post(get_world_bboxes))
        .route("/world/schema/{world}", post(get_world_schema))
        .route("/standalone/database", get(get_standalone_database))
        .route("/standalone/video/{camera}", get(get_standalone_video))
        // .route("/standalone/view/{camera}", get(get_standalone_view))
        .route(
            "/standalone/camera_info/{camera}",
            get(get_standalone_camera_info),
        )
        .route("/standalone/query/{camera}", post(run_standalone_query))
        .route("/standalone/bboxes/{camera}", post(get_standalone_bboxes))
        .route("/standalone/schema/{camera}", post(get_standalone_schema))
        .with_state(state);
    let app = app.merge(SwaggerUi::new("/docs").url("/api-doc/openapi.json", ApiDoc::openapi()));

    let listener = tokio::net::TcpListener::bind(&args.addr).await?;
    println!(
        "listening on http://{} ({} worlds, {} standalone cameras)",
        args.addr, n_worlds, n_standalone
    );
    axum::serve(listener, app).await?;
    Ok(())
}
