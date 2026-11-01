//! Pinhole camera model and world→pixel projection, plus the `metadata.json`
//! structs it is built from.

use serde::{Deserialize, Serialize};

use trql::db::Trqs;
use trql_types::{
    obj::BoundingBox,
    prop::{TypeDesc, Value},
};

/// Assumed vehicle height (metres) used to lift the stored ground footprint
/// edge into a 3D box before projecting.
pub const VEHICLE_HEIGHT_M: f64 = 1.5;

/// A 3D point / camera location (CARLA world coordinates, metres).
#[derive(Deserialize, Serialize, Clone, Copy, Default)]
pub struct Vec3 {
    #[serde(default)]
    pub x: f64,
    #[serde(default)]
    pub y: f64,
    #[serde(default)]
    pub z: f64,
}

/// Camera orientation in CARLA Euler angles (degrees).
#[derive(Deserialize, Serialize, Clone, Copy, Default)]
pub struct Rot3 {
    #[serde(default)]
    pub pitch: f64,
    #[serde(default)]
    pub yaw: f64,
    #[serde(default)]
    pub roll: f64,
}

/// `world-metadata.json`, sitting at a world's root beside the `.xodr` and the
/// `.trql`: the transform between the integer coordinates stored in the db and
/// CARLA world metres.
///
/// Stored coordinates are `round(metres × coordinate_scale) + delta`, the shift
/// putting the map's bottom-left corner at the origin, so metres come back as
/// `(stored − delta) / coordinate_scale`. The deltas are already in scaled
/// units. This frame belongs to the world, not to any one camera — every camera
/// in a world reads the same db and so shares it.
#[derive(Deserialize, Serialize, Clone, Default)]
pub struct WorldMeta {
    #[serde(default, alias = "scale")]
    pub coordinate_scale: f64,
    #[serde(default)]
    pub x_delta: f64,
    #[serde(default)]
    pub y_delta: f64,
    /// File name of a picture of the world, beside `world-metadata.json` — a
    /// stadium pitch, a floor plan, a satellite crop. An alternative to an
    /// `.xodr` for worlds that are not road networks: there is no geometry to
    /// parse, so it is served as-is and drawn as the map's backdrop.
    #[serde(default)]
    pub world_view: Option<String>,
    /// Drawing area the world's coordinates span, in **stored** units. Together
    /// with `coordinate_scale` this places `world_view` in the map frame; when
    /// absent the view's own `viewBox` is used instead.
    #[serde(default)]
    pub width: Option<f64>,
    #[serde(default)]
    pub height: Option<f64>,
    /// The world's own timeline. Only read for a world that ships no camera
    /// directory at all, where there is no `metadata.json` to state it and the
    /// client still needs a frame rate and a length to scrub against.
    #[serde(default)]
    pub fps: Option<f64>,
    #[serde(default)]
    pub num_frames: Option<u32>,
    #[serde(default)]
    pub duration_s: Option<f64>,
}

impl WorldMeta {
    /// Stored X → CARLA world metres.
    pub fn x_to_metres(&self, stored: f64) -> f64 {
        (stored - self.x_delta) / self.coordinate_scale
    }

    /// Stored Y → CARLA world metres.
    pub fn y_to_metres(&self, stored: f64) -> f64 {
        (stored - self.y_delta) / self.coordinate_scale
    }
}

/// Subset of `metadata.json` we care about. All fields are optional so partial
/// or differently-shaped metadata still parses.
#[derive(Deserialize, Serialize, Clone, Default)]
pub struct VideoMeta {
    #[serde(default)]
    pub fps: f64,
    #[serde(default)]
    pub num_frames: u32,
    #[serde(default)]
    pub duration_s: f64,
    /// Frame width/height in pixels, used to build intrinsics and clamp boxes.
    ///
    /// Also the drawing area for a video-less camera, which has no frame: there
    /// the file must state an area, measured from the origin, that contains the
    /// trajectories.
    #[serde(default)]
    pub width: u32,
    #[serde(default)]
    pub height: u32,
    /// Picture to draw this camera's boxes over when it has no video, named
    /// relative to the camera directory (e.g. `pitch.svg`). It must be drawn in
    /// the coordinate space the boxes are stored in, spanning exactly the
    /// `width` x `height` drawing area above, so the client can lay it under
    /// them without knowing anything about the scene it depicts.
    #[serde(default)]
    pub scene_view: Option<String>,
    /// Scale of the world's coordinate frame, stamped into each camera's
    /// `metadata.json` by the dataset generator. Authoritatively it lives in the
    /// world's `world-metadata.json`; this copy is what standalone cameras
    /// (which have no world file) are read from. Older data used `scale`.
    ///
    /// A standalone camera's coordinates carry no origin shift, so unlike
    /// [`WorldMeta`] there are no deltas here.
    #[serde(default, alias = "scale")]
    pub coordinate_scale: f64,
    /// Camera intrinsics/extrinsics for the full pinhole projection.
    #[serde(default)]
    pub fov_deg: f64,
    /// Where the camera stands in the world, when the dataset says. `None` for
    /// footage that was never surveyed — most fixed-rig datasets ship no pose —
    /// and distinct from a camera genuinely standing at the origin, which is
    /// why this is an `Option` rather than a defaulted `Vec3`: a missing pose
    /// used to read as (0, 0, 0) and put a phantom marker on the map.
    #[serde(default)]
    pub location: Option<Vec3>,
    #[serde(default)]
    pub rotation: Option<Rot3>,
}

impl VideoMeta {
    /// Whether the dataset places this camera in the world. Only a camera with
    /// a pose can be drawn on the map or projected through.
    pub fn has_pose(&self) -> bool {
        self.location.is_some()
    }

    /// The camera's pose, falling back to the origin for the maths that runs
    /// before [`has_pose`] is consulted.
    pub fn pose(&self) -> (Vec3, Rot3) {
        (
            self.location.unwrap_or_default(),
            self.rotation.unwrap_or_default(),
        )
    }
}

/// Pinhole camera model precomputed from `metadata.json`.
///
/// `km = K · M_PERM` folds the intrinsics with the CARLA→OpenCV axis
/// permutation; `w2c` is the world→camera transform `[Rᵀ | −Rᵀt]` (first three
/// rows). A world point projects as `img = km · (w2c · [x,y,z,1])`.
#[derive(Clone)]
pub struct CameraModel {
    km: [[f64; 3]; 3],
    w2c: [[f64; 4]; 3],
}

impl CameraModel {
    /// Builds the camera model from metadata. Returns `None` when the required
    /// intrinsics (frame size + FOV) are missing.
    pub fn from_meta(meta: &VideoMeta) -> Option<CameraModel> {
        if meta.width == 0 || meta.height == 0 || meta.fov_deg <= 0.0 {
            return None;
        }
        let w = meta.width as f64;
        let h = meta.height as f64;
        let f = w / (2.0 * (meta.fov_deg.to_radians() / 2.0).tan());
        let k = [[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]];
        // CARLA camera (x-fwd, y-right, z-up) → OpenCV (x-right, y-down, z-fwd).
        let m_perm = [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]];
        let mut km = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                km[i][j] = k[i][0] * m_perm[0][j] + k[i][1] * m_perm[1][j] + k[i][2] * m_perm[2][j];
            }
        }
        // CARLA rotation matrix R (camera→world) from yaw/pitch/roll.
        let (location, rotation) = meta.pose();
        let (cy, sy) = (
            rotation.yaw.to_radians().cos(),
            rotation.yaw.to_radians().sin(),
        );
        let (cr, sr) = (
            rotation.roll.to_radians().cos(),
            rotation.roll.to_radians().sin(),
        );
        let (cp, sp) = (
            rotation.pitch.to_radians().cos(),
            rotation.pitch.to_radians().sin(),
        );
        let r = [
            [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
            [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
            [sp, -cp * sr, cp * cr],
        ];
        let t = [location.x, location.y, location.z];
        // world→camera = [Rᵀ | −Rᵀt].
        let mut w2c = [[0.0; 4]; 3];
        for i in 0..3 {
            for j in 0..3 {
                w2c[i][j] = r[j][i];
            }
            w2c[i][3] = -(r[0][i] * t[0] + r[1][i] * t[1] + r[2][i] * t[2]);
        }
        Some(CameraModel { km, w2c })
    }

    /// Projects a world point (metres) to pixel `(u, v)`. Returns `None` when
    /// the point is behind the camera.
    fn project(&self, x: f64, y: f64, z: f64) -> Option<(f64, f64)> {
        let pc0 = self.w2c[0][0] * x + self.w2c[0][1] * y + self.w2c[0][2] * z + self.w2c[0][3];
        let pc1 = self.w2c[1][0] * x + self.w2c[1][1] * y + self.w2c[1][2] * z + self.w2c[1][3];
        let pc2 = self.w2c[2][0] * x + self.w2c[2][1] * y + self.w2c[2][2] * z + self.w2c[2][3];
        if pc0 < 0.1 {
            return None;
        }
        let iu = self.km[0][0] * pc0 + self.km[0][1] * pc1 + self.km[0][2] * pc2;
        let iv = self.km[1][0] * pc0 + self.km[1][1] * pc1 + self.km[1][2] * pc2;
        let iw = self.km[2][0] * pc0 + self.km[2][1] * pc1 + self.km[2][2] * pc2;
        if iw.abs() < 1e-9 {
            return None;
        }
        Some((iu / iw, iv / iw))
    }
}

/// Assumed vehicle height in metres for lifting the footprint, by class.
fn class_height(class: &str) -> f64 {
    match class {
        "bus" => 3.2,
        "truck" => 2.6,
        "car" => 1.5,
        "motorcycle" | "bicycle" => 1.6,
        _ => VEHICLE_HEIGHT_M,
    }
}

/// Looks up an object's height from its enum class column (e.g. `class`),
/// defaulting to [`VEHICLE_HEIGHT_M`] when no class column/value is present.
pub fn object_height(db: &Trqs, oid: u32) -> f64 {
    let schema = db.schema();
    for (name, ty) in schema.col_iter() {
        if let TypeDesc::Enum(variants) = ty {
            if let Some(col_idx) = schema.get_col_idx(name) {
                if let Value::Enum(vi) = db.prop_idx_to_prop[col_idx].get(oid) {
                    if let Some(cls) = variants.get(*vi) {
                        return class_height(cls);
                    }
                }
            }
        }
    }
    VEHICLE_HEIGHT_M
}

/// Projects a world-space bounding box into camera pixel space.
///
/// `(x1,y1)` and `(x2,y2)` are two opposite corners (top-left / bottom-right) of
/// the vehicle's axis-aligned ground footprint, stored in the world's coordinate
/// frame and un-shifted / un-scaled back to CARLA world metres through `world`.
/// All four footprint corners are reconstructed, each lifted from the ground
/// (z=0) to `height`, and the resulting eight 3D points are projected through
/// the full pinhole model. The axis-aligned pixel bounds are returned, clamped
/// to the frame. A degenerate box is returned if every point falls behind the
/// camera.
pub fn world_to_camera(
    bb: &BoundingBox,
    cam: &CameraModel,
    meta: &VideoMeta,
    world: &WorldMeta,
    height: f64,
) -> BoundingBox {
    let x1 = world.x_to_metres(bb.x1 as i32 as f64);
    let y1 = world.y_to_metres(bb.y1 as i32 as f64);
    let x2 = world.x_to_metres(bb.x2 as i32 as f64);
    let y2 = world.y_to_metres(bb.y2 as i32 as f64);
    // Four corners of the footprint rectangle from the two opposite corners.
    let corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)];

    let mut min_u = f64::INFINITY;
    let mut min_v = f64::INFINITY;
    let mut max_u = f64::NEG_INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    let mut any = false;
    for (wx, wy) in corners {
        for z in [0.0, height] {
            if let Some((u, v)) = cam.project(wx, wy, z) {
                any = true;
                min_u = min_u.min(u);
                min_v = min_v.min(v);
                max_u = max_u.max(u);
                max_v = max_v.max(v);
            }
        }
    }
    if !any {
        return BoundingBox {
            x1: 0,
            y1: 0,
            x2: 0,
            y2: 0,
        };
    }

    let max_w = meta.width as f64;
    let max_h = meta.height as f64;
    let clamp = |val: f64, hi: f64| val.max(0.0).min(hi).round() as i64;

    BoundingBox {
        x1: clamp(min_u, max_w) as u32,
        y1: clamp(min_v, max_h) as u32,
        x2: clamp(max_u, max_w) as u32,
        y2: clamp(max_v, max_h) as u32,
    }
}

/// Ground polygon (CARLA world XY, metres) visible by a camera, from the four
/// image corners projected onto the z=0 plane. Rays that miss the ground (look
/// above the horizon) are clamped to `max_dist` metres so the region is bounded.
pub fn fov_footprint(meta: &VideoMeta, max_dist: f64) -> Vec<[f64; 2]> {
    let w = meta.width as f64;
    let h = meta.height as f64;
    if w <= 0.0 || h <= 0.0 || meta.fov_deg <= 0.0 {
        return Vec::new();
    }
    let f = w / (2.0 * (meta.fov_deg.to_radians() / 2.0).tan());
    let (cx, cy) = (w / 2.0, h / 2.0);
    // R (camera→world), as in CameraModel.
    let (location, rotation) = meta.pose();
    let (yw_c, yw_s) = (
        rotation.yaw.to_radians().cos(),
        rotation.yaw.to_radians().sin(),
    );
    let (r_c, r_s) = (
        rotation.roll.to_radians().cos(),
        rotation.roll.to_radians().sin(),
    );
    let (p_c, p_s) = (
        rotation.pitch.to_radians().cos(),
        rotation.pitch.to_radians().sin(),
    );
    let r = [
        [
            p_c * yw_c,
            yw_c * p_s * r_s - yw_s * r_c,
            -yw_c * p_s * r_c - yw_s * r_s,
        ],
        [
            p_c * yw_s,
            yw_s * p_s * r_s + yw_c * r_c,
            -yw_s * p_s * r_c + yw_c * r_s,
        ],
        [p_s, -p_c * r_s, p_c * r_c],
    ];
    let t = [location.x, location.y, location.z];
    // Image corners in a consistent ring: bottom-left, bottom-right, top-right, top-left.
    let corners = [[0.0, h], [w, h], [w, 0.0], [0.0, 0.0]];
    let mut out = Vec::with_capacity(4);
    for [u, v] in corners {
        // K⁻¹·[u,v,1] (OpenCV frame) → CARLA cam frame via M_PERMᵀ → world via R.
        let d_cv = [(u - cx) / f, (v - cy) / f, 1.0];
        let d_carla = [d_cv[2], d_cv[0], -d_cv[1]];
        let wr = [
            r[0][0] * d_carla[0] + r[0][1] * d_carla[1] + r[0][2] * d_carla[2],
            r[1][0] * d_carla[0] + r[1][1] * d_carla[1] + r[1][2] * d_carla[2],
            r[2][0] * d_carla[0] + r[2][1] * d_carla[1] + r[2][2] * d_carla[2],
        ];
        let hyp = (wr[0] * wr[0] + wr[1] * wr[1]).sqrt();
        // Intersect z=0: t.z + s·wr.z = 0. Clamp horizontal distance to max_dist.
        let s = if wr[2] < -1e-6 {
            -t[2] / wr[2]
        } else {
            f64::INFINITY
        };
        let (gx, gy) = if s > 0.0 && s.is_finite() && s * hyp <= max_dist {
            (t[0] + s * wr[0], t[1] + s * wr[1])
        } else if hyp > 1e-9 {
            let k = max_dist / hyp;
            (t[0] + wr[0] * k, t[1] + wr[1] * k)
        } else {
            (t[0], t[1])
        };
        out.push([gx, gy]);
    }
    out
}
