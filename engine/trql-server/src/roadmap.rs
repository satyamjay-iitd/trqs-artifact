//! OpenDRIVE (`.xodr`) parsing into filled lane polygons, plus camera
//! field-of-view visibility against an occlusion grid.

use std::path::Path;

use serde::Serialize;
use utoipa::ToSchema;

use crate::camera::{VideoMeta, WorldMeta, fov_footprint};

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

/// A single lane rendered as a filled world-space polygon.
#[derive(Serialize, ToSchema, Clone)]
pub struct LanePolygon {
    /// OpenDRIVE lane type (`driving`, `sidewalk`, `shoulder`, `median`, …).
    #[serde(rename = "type")]
    pub lane_type: String,
    pub polygon: Vec<[f64; 2]>,
}

/// Camera position/orientation + ground field-of-view, in the **map** frame
/// (metres; CARLA Y already negated to match the OpenDRIVE map).
#[derive(Serialize, ToSchema, Clone)]
pub struct CameraMarker {
    /// Display label (the camera/video name).
    pub label: String,
    pub x: f64,
    pub y: f64,
    /// Heading on the ground plane, in degrees (map frame).
    pub yaw: f64,
    /// Horizontal field of view, in degrees.
    pub fov_deg: f64,
    /// Ground polygon the camera can see (map-frame metres), or empty if unknown.
    pub fov_area: Vec<[f64; 2]>,
}

/// A picture standing in for parsed geometry, for worlds that are not road
/// networks. The client fetches `url` and draws it across `bounds`; there are no
/// lanes to render.
#[derive(Serialize, ToSchema, Clone)]
pub struct WorldView {
    /// MIME type of the file served at `url` (`image/svg+xml`, `image/png`, …).
    pub content_type: String,
    /// Endpoint serving the picture, relative to the API root.
    pub url: String,
    /// Whether the picture's own y axis points down (SVG, raster) and so must
    /// be flipped to sit in the map frame, whose y points up.
    pub flip_y: bool,
}

#[derive(Serialize, ToSchema, Clone)]
pub struct RoadmapResponse {
    /// `[min_x, min_y, max_x, max_y]` in world metres.
    pub bounds: [f64; 4],
    /// Fixed-point scale applied to stored object coordinates (metres × scale).
    /// Map geometry is in metres; multiply by this to match query coordinates.
    pub scale: f64,
    /// Every camera belonging to this map (all videos in the map).
    pub cameras: Vec<CameraMarker>,
    /// One filled polygon per lane across all roads. Empty for a world drawn
    /// from a `view` instead of an `.xodr`.
    pub lanes: Vec<LanePolygon>,
    /// Set when the world's map is a picture rather than a road network.
    /// Mutually exclusive with a non-empty `lanes` in practice.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub view: Option<WorldView>,
}

// ---------------------------------------------------------------------------
// OpenDRIVE geometry
// ---------------------------------------------------------------------------

/// A single `<planView>` geometry record (line or arc).
struct Geom {
    s: f64,
    x: f64,
    y: f64,
    hdg: f64,
    /// Arc curvature; 0 for a straight line.
    k: f64,
}

/// A lane `<width>` record: cubic polynomial in the section-relative offset.
struct WidthRec {
    s_offset: f64,
    c: [f64; 4],
}

struct Lane {
    lane_type: String,
    widths: Vec<WidthRec>,
}

/// Evaluates a cubic `a + b·ds + c·ds² + d·ds³`.
fn poly3(c: &[f64; 4], ds: f64) -> f64 {
    c[0] + c[1] * ds + c[2] * ds * ds + c[3] * ds * ds * ds
}

/// Numeric attribute reader for an XML node.
fn attr_f(n: roxmltree::Node, a: &str) -> Option<f64> {
    n.attribute(a).and_then(|v| v.trim().parse::<f64>().ok())
}

/// Reference-line pose `(x, y, hdg)` at arc-length `s` along a road.
fn pose(geoms: &[Geom], s: f64) -> (f64, f64, f64) {
    let mut g = &geoms[0];
    for gg in geoms {
        if gg.s <= s + 1e-9 {
            g = gg;
        }
    }
    let ds = s - g.s;
    if g.k.abs() < 1e-9 {
        (g.x + ds * g.hdg.cos(), g.y + ds * g.hdg.sin(), g.hdg)
    } else {
        let k = g.k;
        (
            g.x + ((g.hdg + k * ds).sin() - g.hdg.sin()) / k,
            g.y - ((g.hdg + k * ds).cos() - g.hdg.cos()) / k,
            g.hdg + k * ds,
        )
    }
}

/// Lateral shift of the `t=0` line from the `<laneOffset>` records.
fn lane_offset(offs: &[(f64, [f64; 4])], s: f64) -> f64 {
    let mut rec = offs.first().copied().unwrap_or((0.0, [0.0; 4]));
    for r in offs {
        if r.0 <= s + 1e-9 {
            rec = *r;
        }
    }
    poly3(&rec.1, s - rec.0)
}

/// Lane width at a section-relative arc-length.
fn width_at(lane: &Lane, s_sec: f64) -> f64 {
    if lane.widths.is_empty() {
        return 0.0;
    }
    let mut rec = &lane.widths[0];
    for w in &lane.widths {
        if w.s_offset <= s_sec + 1e-9 {
            rec = w;
        }
    }
    poly3(&rec.c, s_sec - rec.s_offset).max(0.0)
}

/// Parses an OpenDRIVE (`.xodr`) file into filled lane polygons.
///
/// For each road the reference line (`<line>`/`<arc>` primitives) is sampled and
/// offset laterally by the cumulative lane widths (from `<laneSection>` /
/// `<width>`, plus `<laneOffset>`), producing one polygon per lane in world XY.
pub fn parse_roadmap(path: &Path) -> Option<RoadmapResponse> {
    let text = std::fs::read_to_string(path).ok()?;
    let doc = roxmltree::Document::parse(&text).ok()?;
    let mut lanes_out: Vec<LanePolygon> = Vec::new();

    for road in doc.descendants().filter(|n| n.has_tag_name("road")) {
        let road_len = attr_f(road, "length").unwrap_or(0.0);

        // planView reference geometry.
        let mut geoms: Vec<Geom> = Vec::new();
        if let Some(pv) = road.children().find(|n| n.has_tag_name("planView")) {
            for g in pv.children().filter(|n| n.has_tag_name("geometry")) {
                let (Some(s), Some(x), Some(y), Some(hdg)) = (
                    attr_f(g, "s"),
                    attr_f(g, "x"),
                    attr_f(g, "y"),
                    attr_f(g, "hdg"),
                ) else {
                    continue;
                };
                let k = g
                    .children()
                    .find(|n| n.has_tag_name("arc"))
                    .and_then(|a| attr_f(a, "curvature"))
                    .unwrap_or(0.0);
                geoms.push(Geom { s, x, y, hdg, k });
            }
        }
        if geoms.is_empty() {
            continue;
        }
        geoms.sort_by(|a, b| a.s.total_cmp(&b.s));

        let lanes_el = match road.children().find(|n| n.has_tag_name("lanes")) {
            Some(l) => l,
            None => continue,
        };

        // laneOffset records.
        let mut offs: Vec<(f64, [f64; 4])> = Vec::new();
        for lo in lanes_el.children().filter(|n| n.has_tag_name("laneOffset")) {
            offs.push((
                attr_f(lo, "s").unwrap_or(0.0),
                [
                    attr_f(lo, "a").unwrap_or(0.0),
                    attr_f(lo, "b").unwrap_or(0.0),
                    attr_f(lo, "c").unwrap_or(0.0),
                    attr_f(lo, "d").unwrap_or(0.0),
                ],
            ));
        }
        offs.sort_by(|a, b| a.0.total_cmp(&b.0));

        // Lane sections: (s0, left lanes ascending, right lanes by |id| ascending).
        let parse_side = |ls: roxmltree::Node, side: &str| -> Vec<(i32, Lane)> {
            let mut v = Vec::new();
            if let Some(sd) = ls.children().find(|n| n.has_tag_name(side)) {
                for lane in sd.children().filter(|n| n.has_tag_name("lane")) {
                    let id = lane
                        .attribute("id")
                        .and_then(|s| s.parse::<i32>().ok())
                        .unwrap_or(0);
                    let lane_type = lane.attribute("type").unwrap_or("driving").to_string();
                    let mut widths: Vec<WidthRec> = lane
                        .children()
                        .filter(|n| n.has_tag_name("width"))
                        .map(|w| WidthRec {
                            s_offset: attr_f(w, "sOffset").unwrap_or(0.0),
                            c: [
                                attr_f(w, "a").unwrap_or(0.0),
                                attr_f(w, "b").unwrap_or(0.0),
                                attr_f(w, "c").unwrap_or(0.0),
                                attr_f(w, "d").unwrap_or(0.0),
                            ],
                        })
                        .collect();
                    widths.sort_by(|a, b| a.s_offset.total_cmp(&b.s_offset));
                    v.push((id, Lane { lane_type, widths }));
                }
            }
            v
        };

        let mut sections: Vec<(f64, Vec<Lane>, Vec<Lane>)> = Vec::new();
        for ls in lanes_el
            .children()
            .filter(|n| n.has_tag_name("laneSection"))
        {
            let s0 = attr_f(ls, "s").unwrap_or(0.0);
            let mut left = parse_side(ls, "left");
            left.sort_by_key(|(id, _)| *id); // 1, 2, 3, … (inner → outer)
            let mut right = parse_side(ls, "right");
            right.sort_by_key(|(id, _)| -*id); // -1, -2, -3, …
            sections.push((
                s0,
                left.into_iter().map(|(_, l)| l).collect(),
                right.into_iter().map(|(_, l)| l).collect(),
            ));
        }
        sections.sort_by(|a, b| a.0.total_cmp(&b.0));

        for si in 0..sections.len() {
            let (s0, left, right) = &sections[si];
            let s1 = if si + 1 < sections.len() {
                sections[si + 1].0
            } else {
                road_len
            };
            if s1 <= *s0 {
                continue;
            }
            let n = (((s1 - s0) / 1.0).ceil() as usize).max(2);
            let ss: Vec<f64> = (0..=n)
                .map(|i| s0 + (s1 - s0) * i as f64 / n as f64)
                .collect();

            for (lanes, sign) in [(right, -1.0f64), (left, 1.0f64)] {
                for (li, lane) in lanes.iter().enumerate() {
                    let mut inner = Vec::with_capacity(ss.len());
                    let mut outer = Vec::with_capacity(ss.len());
                    for &s in &ss {
                        let (x, y, h) = pose(&geoms, s);
                        let (nx, ny) = (-h.sin(), h.cos());
                        let t0 = lane_offset(&offs, s);
                        let s_sec = s - s0;
                        let mut cin = 0.0;
                        for l in &lanes[..li] {
                            cin += width_at(l, s_sec);
                        }
                        let wj = width_at(lane, s_sec);
                        let ti = t0 + sign * cin;
                        let to = t0 + sign * (cin + wj);
                        inner.push([x + ti * nx, y + ti * ny]);
                        outer.push([x + to * nx, y + to * ny]);
                    }
                    let mut poly = inner;
                    poly.extend(outer.into_iter().rev());
                    if poly.len() >= 3 {
                        lanes_out.push(LanePolygon {
                            lane_type: lane.lane_type.clone(),
                            polygon: poly,
                        });
                    }
                }
            }
        }
    }

    if lanes_out.is_empty() {
        return None;
    }
    let mut bounds = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for lp in &lanes_out {
        for p in &lp.polygon {
            bounds[0] = bounds[0].min(p[0]);
            bounds[1] = bounds[1].min(p[1]);
            bounds[2] = bounds[2].max(p[0]);
            bounds[3] = bounds[3].max(p[1]);
        }
    }
    // `scale` and `cameras` are filled in by the handler from db metadata.
    Some(RoadmapResponse {
        bounds,
        scale: 10_000.0,
        cameras: Vec::new(),
        lanes: lanes_out,
        view: None,
    })
}

/// Builds the map response for a world whose map is a picture, not a road
/// network.
///
/// There is no geometry to parse: the picture is served as-is and the client
/// stretches it across `bounds`. Those bounds come from the world's declared
/// drawing area (`width`/`height` in stored units, divided by
/// `coordinate_scale`), falling back to the SVG's own `viewBox` — which the
/// dataset generators write in the same stored units, so both agree.
pub fn view_roadmap(view_path: &Path, meta: &WorldMeta) -> Option<RoadmapResponse> {
    let scale = if meta.coordinate_scale > 0.0 {
        meta.coordinate_scale
    } else {
        1.0
    };

    let (w, h) = match (meta.width, meta.height) {
        (Some(w), Some(h)) if w > 0.0 && h > 0.0 => (w, h),
        // The metadata is the authority — it states the drawing area in stored
        // units. Failing that, the picture is asked how big it is: an SVG's
        // viewBox is already in those units (the generators author it that
        // way), while a raster only knows pixels, so falling back to those
        // treats one pixel as one stored unit. That is right for a map drawn at
        // stored-unit resolution and wrong otherwise, which is why it is a last
        // resort rather than the first choice.
        _ => svg_viewbox(view_path).or_else(|| png_size(view_path))?,
    };

    let content_type = match view_path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .as_deref()
    {
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        _ => "application/octet-stream",
    };

    // The bounds must be the extent of the *plotted objects*, because that is
    // the frame the client draws everything else in. An object's map position is
    // `((stored - delta) / scale)` with Y negated (see `to_map_pos`), so the
    // picture — which spans stored 0..width and 0..height — has to be mapped
    // through exactly the same transform. Taking the stored extent at face value
    // instead leaves the picture shifted by the deltas and mirrored in Y, which
    // looks like a scale bug but is a frame bug.
    let x0 = -meta.x_delta / scale;
    let x1 = (w - meta.x_delta) / scale;
    let y0 = -(h - meta.y_delta) / scale;
    let y1 = meta.y_delta / scale;

    Some(RoadmapResponse {
        bounds: [x0, y0, x1, y1],
        scale,
        cameras: Vec::new(),
        lanes: Vec::new(),
        view: Some(WorldView {
            content_type: content_type.to_string(),
            url: String::new(), // filled in by the loader, which knows the name
            // Stored Y grows downward and `to_map_pos` negates it, so by the
            // time the client re-negates for display the two cancel: the
            // picture's own top-down Y already matches the screen. Mirroring it
            // as well would stand the map on its head.
            flip_y: false,
        }),
    })
}

/// Pixel `width`/`height` from a PNG's IHDR — the first chunk, at a fixed
/// offset, so no image decoder is needed. `None` for anything that is not a
/// PNG, including the other raster formats a world view may use, whose sizes
/// are not worth a decoder to recover when the metadata should state them.
fn png_size(path: &Path) -> Option<(f64, f64)> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < 24 || &bytes[..8] != b"\x89PNG\r\n\x1a\n" {
        return None;
    }
    let w = u32::from_be_bytes(bytes[16..20].try_into().ok()?);
    let h = u32::from_be_bytes(bytes[20..24].try_into().ok()?);
    (w > 0 && h > 0).then(|| (w as f64, h as f64))
}

/// `width`/`height` of an SVG's `viewBox`, used when the world metadata does not
/// state a drawing area. Only the two size components are needed; a `viewBox`
/// with a non-zero origin still describes the same extent.
fn svg_viewbox(path: &Path) -> Option<(f64, f64)> {
    // The attribute is in the root element, so only the head of the file is
    // read — these can carry a whole stadium's worth of vector detail.
    let text = std::fs::read_to_string(path).ok()?;
    let head = &text[..text.len().min(4096)];
    let at = head.find("viewBox")?;
    let rest = &head[at + "viewBox".len()..];
    let open = rest.find('"').or_else(|| rest.find('\''))?;
    let body = &rest[open + 1..];
    let close = body.find(['"', '\'']).unwrap_or(body.len());
    let nums: Vec<f64> = body[..close]
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse().ok())
        .collect();
    match nums.as_slice() {
        [.., w, h] if *w > 0.0 && *h > 0.0 => Some((*w, *h)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Camera visibility
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box `[min_x, min_y, max_x, max_y]` of a polygon.
fn poly_bbox(poly: &[[f64; 2]]) -> [f64; 4] {
    let mut b = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
    for p in poly {
        b[0] = b[0].min(p[0]);
        b[1] = b[1].min(p[1]);
        b[2] = b[2].max(p[0]);
        b[3] = b[3].max(p[1]);
    }
    b
}

/// Even-odd ray-cast point-in-polygon test.
fn point_in_poly(p: [f64; 2], poly: &[[f64; 2]]) -> bool {
    let mut inside = false;
    let n = poly.len();
    let mut j = n - 1;
    for i in 0..n {
        let (a, b) = (poly[i], poly[j]);
        if (a[1] > p[1]) != (b[1] > p[1]) {
            let t = (p[1] - a[1]) / (b[1] - a[1]);
            if p[0] < a[0] + t * (b[0] - a[0]) {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// Rasterized occlusion map. Cells inside a large enclosed non-road region (a
/// building block) are occluders; small non-road patches — sampling seams,
/// medians, open ground — are not, so line of sight passes over them.
pub struct OccGrid {
    min_x: f64,
    min_y: f64,
    cell: f64,
    nx: usize,
    ny: usize,
    building: Vec<bool>,
}

impl OccGrid {
    fn is_building(&self, x: f64, y: f64) -> bool {
        if x < self.min_x || y < self.min_y {
            return false;
        }
        let i = ((x - self.min_x) / self.cell) as usize;
        let j = ((y - self.min_y) / self.cell) as usize;
        if i >= self.nx || j >= self.ny {
            return false;
        }
        self.building[j * self.nx + i]
    }
}

/// Build an occlusion grid from lane polygons. Cells not covered by any lane are
/// grouped into 4-connected components; a component of at least `min_building_m2`
/// is treated as a building block (only these obstruct visibility). Buildings
/// here are cleanly separated by area (≥500 m²) from parsing seams (≤50 m²).
pub fn build_occ_grid(lanes: &[LanePolygon], bounds: [f64; 4], min_building_m2: f64) -> OccGrid {
    let cell = 1.5;
    let [minx, miny, maxx, maxy] = bounds;
    let nx = ((maxx - minx) / cell).ceil() as usize + 2;
    let ny = ((maxy - miny) / cell).ceil() as usize + 2;
    let mut covered = vec![false; nx * ny];
    let boxes: Vec<[f64; 4]> = lanes.iter().map(|l| poly_bbox(&l.polygon)).collect();
    for (lane, b) in lanes.iter().zip(&boxes) {
        let i0 = ((b[0] - minx) / cell).floor().max(0.0) as usize;
        let i1 = (((b[2] - minx) / cell).ceil() as usize).min(nx - 1);
        let j0 = ((b[1] - miny) / cell).floor().max(0.0) as usize;
        let j1 = (((b[3] - miny) / cell).ceil() as usize).min(ny - 1);
        for j in j0..=j1 {
            for i in i0..=i1 {
                let idx = j * nx + i;
                if covered[idx] {
                    continue;
                }
                let p = [
                    minx + (i as f64 + 0.5) * cell,
                    miny + (j as f64 + 0.5) * cell,
                ];
                if point_in_poly(p, &lane.polygon) {
                    covered[idx] = true;
                }
            }
        }
    }
    // Flood-fill non-covered cells into components; keep the large ones.
    let min_cells = (min_building_m2 / (cell * cell)).ceil() as usize;
    let mut building = vec![false; nx * ny];
    let mut visited = vec![false; nx * ny];
    let mut queue: Vec<usize> = Vec::new();
    let mut comp: Vec<usize> = Vec::new();
    for start in 0..nx * ny {
        if covered[start] || visited[start] {
            continue;
        }
        queue.clear();
        comp.clear();
        queue.push(start);
        visited[start] = true;
        while let Some(idx) = queue.pop() {
            comp.push(idx);
            let (i, j) = (idx % nx, idx / nx);
            let mut nbrs = [0usize; 4];
            let mut k = 0;
            if i > 0 {
                nbrs[k] = idx - 1;
                k += 1;
            }
            if i + 1 < nx {
                nbrs[k] = idx + 1;
                k += 1;
            }
            if j > 0 {
                nbrs[k] = idx - nx;
                k += 1;
            }
            if j + 1 < ny {
                nbrs[k] = idx + nx;
                k += 1;
            }
            for &n in &nbrs[..k] {
                if !covered[n] && !visited[n] {
                    visited[n] = true;
                    queue.push(n);
                }
            }
        }
        if comp.len() >= min_cells {
            for &idx in &comp {
                building[idx] = true;
            }
        }
    }
    OccGrid {
        min_x: minx,
        min_y: miny,
        cell,
        nx,
        ny,
        building,
    }
}

/// Ground area a camera can see, in the **map** frame. Rays are cast across the
/// horizontal FOV and each extends until it enters a building block (`grid`);
/// non-building ground (roads, medians, open areas) is seen across freely.
pub fn fov_visibility(meta: &VideoMeta, grid: &OccGrid, max_dist: f64) -> Vec<[f64; 2]> {
    // Camera pose in the map frame (CARLA Y/yaw negated).
    let (location, rotation) = meta.pose();
    let cam = [location.x, -location.y];
    let yaw = -rotation.yaw.to_radians();
    let hfov = meta.fov_deg.to_radians();
    let half = hfov / 2.0;
    // Near cutoff: where the bottom image edge first meets the ground (from pitch).
    let vfov = 2.0 * ((hfov / 2.0).tan() * (meta.height as f64) / (meta.width as f64)).atan();
    let down = -rotation.pitch.to_radians() + vfov / 2.0;
    let near = if down > 0.02 {
        (location.z / down.tan()).clamp(1.0, max_dist * 0.9)
    } else {
        1.0
    };

    let rays = 96usize;
    let ds = 0.5; // march step (metres)
    let mut outer = Vec::with_capacity(rays + 1);
    let mut inner = Vec::with_capacity(rays + 1);
    for i in 0..=rays {
        let a = yaw - half + hfov * (i as f64 / rays as f64);
        let dir = [a.cos(), a.sin()];
        let mut rvis = near;
        let mut r = near;
        while r <= max_dist {
            let p = [cam[0] + r * dir[0], cam[1] + r * dir[1]];
            if grid.is_building(p[0], p[1]) {
                break;
            }
            rvis = r;
            r += ds;
        }
        inner.push([cam[0] + near * dir[0], cam[1] + near * dir[1]]);
        outer.push([cam[0] + rvis * dir[0], cam[1] + rvis * dir[1]]);
    }
    // Outer (far) boundary across the fan, then inner (near) boundary back.
    let mut poly = outer;
    poly.extend(inner.into_iter().rev());
    poly
}

/// Builds the map-frame marker for every camera that has metadata.
///
/// Visibility is occlusion-limited against building blocks only (derived from
/// the map surface); open ground and roads are seen across freely. The CARLA
/// world is left-handed vs the OpenDRIVE map, so Y and yaw are negated per
/// camera.
///
/// The occlusion grid is rasterized once for the whole camera set rather than
/// per camera — on a map with long ring roads it is by far the dominant cost
/// here, so it is built at load and never recomputed per request.
pub fn camera_markers<'a>(
    cameras: impl IntoIterator<Item = (&'a str, &'a VideoMeta)>,
    lanes: &[LanePolygon],
    bounds: [f64; 4],
) -> Vec<CameraMarker> {
    let grid = (!lanes.is_empty()).then(|| build_occ_grid(lanes, bounds, 300.0));
    cameras
        .into_iter()
        // A camera the dataset never placed gets no marker. Its pose would
        // default to the origin, which is a real point on the map, so it would
        // be drawn as a camera genuinely standing in the map's bottom-left
        // corner facing east rather than as the unknown it is.
        .filter(|(_, m)| m.has_pose())
        .map(|(label, m)| {
            let (location, rotation) = m.pose();
            let fov_area = match &grid {
                Some(g) => fov_visibility(m, g, 120.0),
                None => fov_footprint(m, 120.0)
                    .into_iter()
                    .map(|[x, y]| [x, -y])
                    .collect(),
            };
            CameraMarker {
                label: label.to_string(),
                x: location.x,
                y: -location.y,
                yaw: -rotation.yaw,
                fov_deg: m.fov_deg,
                fov_area,
            }
        })
        .collect()
}
