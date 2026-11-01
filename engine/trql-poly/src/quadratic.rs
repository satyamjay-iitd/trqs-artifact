use bincode::{Decode, Encode};
use trql_types::{
    intervals::Interval,
    obj::{BoundingBox, ExactTraj, Trajectory},
};

#[derive(Clone, Debug, Encode, Decode)]
pub struct PolySegment<const S: usize> {
    pub start: u32,
    pub end: u32,
    pub cx1: ScalarSegmentData2<S>,
    pub cy1: ScalarSegmentData2<S>,
    pub cx2: ScalarSegmentData2<S>,
    pub cy2: ScalarSegmentData2<S>,
}

#[derive(Clone, Debug, Encode, Decode)]
pub enum ScalarSegmentData2<const S: usize> {
    Poly([f64; 3]),
    // Values sampled at integer frames t = 0..len-1 where frame_id = start_frame + t.
    Direct { values: [u32; S] },
}

impl<const S: usize> ScalarSegmentData2<S> {
    pub fn eval_local(&self, t: u32) -> f64 {
        match &self {
            ScalarSegmentData2::Poly(coeffs) => {
                let tf = t as f64;
                coeffs[0] * tf * tf + coeffs[1] * tf + coeffs[2]
            }
            ScalarSegmentData2::Direct { values } => {
                if t < values.len() as u32 {
                    values[t as usize] as f64
                } else {
                    // `values` is fixed at S samples, so this means the segment
                    // was asked for a frame past its own capacity — an upstream
                    // invariant break, not something a query can cause. Hold the
                    // last sample and let debug builds trip on it; this used to
                    // print to stdout, which corrupted `trql-cli --format json`.
                    debug_assert!(
                        false,
                        "direct segment of {S} samples evaluated at local frame {t}"
                    );
                    values[S - 1] as f64
                }
            }
        }
    }
}
#[derive(Clone, Debug, Encode, Decode)]
pub struct PolyTraj {
    segments: Vec<PolySegment<4>>,
    canonical_segments: Vec<Interval>,
}

impl PolyTraj {
    pub fn eval(&self, t: u32) -> Option<BoundingBox> {
        let idx = self.segments.partition_point(|s| s.end < t);
        let seg = self.segments.get(idx)?;
        if t < seg.start || t > seg.end {
            return None;
        }
        let t = t - seg.start;
        let x1 = seg.cx1.eval_local(t).round().max(0.0) as u32;
        let x2 = seg.cx2.eval_local(t).round().max(0.0) as u32;
        let y1 = seg.cy1.eval_local(t).round().max(0.0) as u32;
        let y2 = seg.cy2.eval_local(t).round().max(0.0) as u32;
        // println!("seg.cx1.eval_local: {:?}",seg.cx1.eval_local(t));
        Some(BoundingBox { x1, y1, x2, y2 })
    }

    pub fn segments(&self) -> &[PolySegment<4>] {
        &self.segments
    }

    pub fn canonical_segments(&self) -> &[Interval] {
        &self.canonical_segments
    }

    /// The last frame this trajectory covers, or `None` if it has no segments.
    pub fn end_frame(&self) -> Option<u32> {
        self.segments.last().map(|s| s.end)
    }

    /// A copy of this trajectory with every frame index shifted forward by
    /// `offset`.
    ///
    /// Polynomial and direct channels are evaluated in *local* frame
    /// (`frame - start`), so shifting each segment's `start`/`end` translates
    /// the trajectory in time without touching the fitted data. Used to tile a
    /// database when repeating it (see `Trqs::repeated`).
    pub fn shifted(&self, offset: u32) -> PolyTraj {
        PolyTraj {
            segments: self
                .segments
                .iter()
                .map(|s| PolySegment {
                    start: s.start + offset,
                    end: s.end + offset,
                    cx1: s.cx1.clone(),
                    cy1: s.cy1.clone(),
                    cx2: s.cx2.clone(),
                    cy2: s.cy2.clone(),
                })
                .collect(),
            canonical_segments: self
                .canonical_segments
                .iter()
                .map(|&(a, b)| (a + offset, b + offset))
                .collect(),
        }
    }
}

/// Selects a bounding-box coordinate for polynomial operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coord {
    X1,
    Y1,
    X2,
    Y2,
    CenterX,
    CenterY,
}

/// Selects a coordinate from a [`BoundingBox`] for unary comparisons.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryCoord {
    X1,
    Y1,
    X2,
    Y2,
    CenterX,
    CenterY,
}

/// Comparison operator applied to a unary coordinate threshold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryCmp {
    Gt(u32),
    Gte(u32),
    Lt(u32),
    Lte(u32),
}

impl UnaryCoord {
    /// Extracts the selected coordinate value from a bounding box.
    pub fn eval(&self, bbox: &BoundingBox) -> u32 {
        match self {
            UnaryCoord::X1 => bbox.x1,
            UnaryCoord::Y1 => bbox.y1,
            UnaryCoord::X2 => bbox.x2,
            UnaryCoord::Y2 => bbox.y2,
            UnaryCoord::CenterX => bbox.center_x(),
            UnaryCoord::CenterY => bbox.center_y(),
        }
    }
}

impl UnaryCmp {
    /// Returns `true` if `value` satisfies this comparison.
    pub fn eval(&self, value: u32) -> bool {
        match self {
            UnaryCmp::Gt(rhs) => value > *rhs,
            UnaryCmp::Gte(rhs) => value >= *rhs,
            UnaryCmp::Lt(rhs) => value < *rhs,
            UnaryCmp::Lte(rhs) => value <= *rhs,
        }
    }
}

/// Full trajectory as four independent ordered lists of non-overlapping polynomial segments.
///
/// Each coordinate (`x1`, `y1`, `x2`, `y2`) has its own set of breakpoints, allowing
/// independent fitting error budgets. All four lists cover the same overall frame range
/// (they are fitted from the same trajectory runs), but internal breakpoints may differ.
// #[derive(Clone, Debug, Encode, Decode)]
// pub struct PolyTraj {
//     pub x1: Vec<ScalarSegment>,
//     pub y1: Vec<ScalarSegment>,
//     pub x2: Vec<ScalarSegment>,
//     pub y2: Vec<ScalarSegment>,
//     pub appear_intervals: Vec<(u32, u32)>,
// }

// #[derive(Clone, Debug)]
// pub struct PolyTrajIter<'a> {
//     traj: &'a PolyTraj,
//     segment_idx: usize,
//     next_frame: Option<u32>,
// }

#[derive(Clone, Debug)]
pub struct PolyTrajIter<'a> {
    traj: &'a PolyTraj,
    segment_idx: usize,
    next_frame: Option<u32>,
}

impl<'a> PolyTrajIter<'a> {
    fn new(traj: &'a PolyTraj) -> Self {
        Self {
            traj,
            segment_idx: 0,
            next_frame: traj.segments.first().map(|seg| seg.start),
        }
    }
}

impl Iterator for PolyTrajIter<'_> {
    type Item = (u32, BoundingBox);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let frame_id = self.next_frame?;
            let seg = self.traj.segments.get(self.segment_idx)?;

            if frame_id < seg.start {
                self.next_frame = Some(seg.start);
                continue;
            }
            if frame_id > seg.end {
                self.segment_idx += 1;
                self.next_frame = self.traj.segments.get(self.segment_idx).map(|s| s.start);
                continue;
            }

            let bbox = self.traj.eval(frame_id)?;
            self.next_frame = if frame_id < seg.end {
                Some(frame_id + 1)
            } else {
                self.segment_idx += 1;
                self.traj.segments.get(self.segment_idx).map(|s| s.start)
            };
            return Some((frame_id, bbox));
        }
    }
}

impl<'a> IntoIterator for &'a PolyTraj {
    type Item = (u32, BoundingBox);
    type IntoIter = PolyTrajIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        PolyTrajIter::new(self)
    }
}

fn dot(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum()
}

fn fit_linear_in_z(zs: &[f64], ys: &[f64]) -> [f64; 3] {
    let n = ys.len() as f64;
    if n == 0.0 {
        return [0.0, 0.0, 0.0];
    }

    let mean_y = ys.iter().sum::<f64>() / n;
    let denom = zs.iter().map(|z| z * z).sum::<f64>();
    if denom.abs() < 1e-12 {
        return [0.0, 0.0, mean_y];
    }

    let slope = zs.iter().zip(ys.iter()).map(|(z, y)| z * y).sum::<f64>() / denom;
    [0.0, slope, mean_y]
}

fn normalized_to_raw_t(coeffs_z: [f64; 3], mu: f64, sigma: f64) -> [f64; 3] {
    let az = coeffs_z[0];
    let bz = coeffs_z[1];
    let cz = coeffs_z[2];

    let inv_sigma = 1.0 / sigma;
    let inv_sigma2 = inv_sigma * inv_sigma;

    let a = az * inv_sigma2;
    let b = -2.0 * mu * az * inv_sigma2 + bz * inv_sigma;
    let c = az * mu * mu * inv_sigma2 - bz * mu * inv_sigma + cz;
    [a, b, c]
}

fn fit_quadratic_qr_in_z(zs: &[f64], ys: &[f64]) -> Option<[f64; 3]> {
    let n = ys.len();
    if n < 3 {
        return None;
    }

    let mut q0: Vec<f64> = zs.iter().map(|z| z * z).collect();
    let mut q1: Vec<f64> = zs.to_vec();
    let mut q2: Vec<f64> = vec![1.0; n];

    let mut r = [[0.0_f64; 3]; 3];

    let r00 = dot(&q0, &q0).sqrt();
    if r00 < 1e-12 {
        return None;
    }
    r[0][0] = r00;
    for v in &mut q0 {
        *v /= r00;
    }

    r[0][1] = dot(&q0, &q1);
    for i in 0..n {
        q1[i] -= r[0][1] * q0[i];
    }
    let r11 = dot(&q1, &q1).sqrt();
    if r11 < 1e-12 {
        return None;
    }
    r[1][1] = r11;
    for v in &mut q1 {
        *v /= r11;
    }

    r[0][2] = dot(&q0, &q2);
    for i in 0..n {
        q2[i] -= r[0][2] * q0[i];
    }
    r[1][2] = dot(&q1, &q2);
    for i in 0..n {
        q2[i] -= r[1][2] * q1[i];
    }
    let r22 = dot(&q2, &q2).sqrt();
    if r22 < 1e-12 {
        return None;
    }
    r[2][2] = r22;
    for v in &mut q2 {
        *v /= r22;
    }

    let qty0 = dot(&q0, ys);
    let qty1 = dot(&q1, ys);
    let qty2 = dot(&q2, ys);

    let z2 = qty2 / r[2][2];
    let z1 = (qty1 - r[1][2] * z2) / r[1][1];
    let z0 = (qty0 - r[0][1] * z1 - r[0][2] * z2) / r[0][0];

    Some([z0, z1, z2])
}

/// Fit a degree-2 polynomial to `points` (frame_offset, value) via least squares.
/// Returns [a, b, c] so that value(t) ≈ a*t^2 + b*t + c
/// where t = frame_id - start_frame (i.e. points\[0\].0 should already be offset-zeroed).
pub fn fit_polynomial(points: &[(u32, f64)]) -> [f64; 3] {
    let n = points.len();
    if n == 0 {
        return [0.0; 3];
    }
    if n == 1 {
        return [0.0, 0.0, points[0].1];
    }
    if n == 2 {
        // Exact linear fit through two points
        let t1 = points[0].0 as f64;
        let t2 = points[1].0 as f64;
        let y1 = points[0].1;
        let y2 = points[1].1;
        let dt = t2 - t1;
        let b = if dt.abs() > 1e-12 {
            (y2 - y1) / dt
        } else {
            0.0
        };
        let c = y1 - b * t1;
        return [0.0, b, c];
    }

    // Numerically stable path: center/scale t and solve least-squares in z-space via QR.
    let ts: Vec<f64> = points.iter().map(|&(t, _)| t as f64).collect();
    let ys: Vec<f64> = points.iter().map(|&(_, y)| y).collect();

    let mu = ts.iter().sum::<f64>() / n as f64;
    let var = ts
        .iter()
        .map(|t| {
            let d = *t - mu;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    let sigma = var.sqrt().max(1.0);
    let zs: Vec<f64> = ts.iter().map(|t| (*t - mu) / sigma).collect();

    if let Some(coeffs_z) = fit_quadratic_qr_in_z(&zs, &ys) {
        return normalized_to_raw_t(coeffs_z, mu, sigma);
    }

    let coeffs_z = fit_linear_in_z(&zs, &ys);
    normalized_to_raw_t(coeffs_z, mu, sigma)
}

/// Max absolute residual between the polynomial and the actual points.
pub fn max_residual(points: &[(u32, f64)], coeffs: &[f64; 3]) -> f64 {
    points
        .iter()
        .map(|&(t, y)| {
            let t = t as f64;
            let y_pred = coeffs[0] * t * t + coeffs[1] * t + coeffs[2];
            (y - y_pred).abs()
        })
        .fold(0.0_f64, f64::max)
}

/// Coupled fitting: split `run` until the max error across all four coordinates is ≤
/// `max_error_px`, then push one [`ScalarSegment`] per coordinate with shared breakpoints.
///
/// `end` is the last frame this piece must cover. It is the run's last detection at the
/// top level, but when a run is split it is the frame *before* the right half starts, so
/// a missed detection that falls on a split point stays covered by the left piece
/// instead of belonging to neither.
fn fit_run_bounded(
    run: &[(u32, BoundingBox)],
    end: u32,
    max_error_px: f64,
    segments: &mut Vec<PolySegment<4>>,
) {
    const FIT_EPS: f64 = 1e-9;
    if run.is_empty() {
        return;
    }
    let start = run[0].0;
    // `Direct` stores one sample per frame from `start`, so it is only exact when the
    // piece covers exactly its detections, with no missed frame inside or after them.
    // Anything else takes the polynomial path, which places points at their real offsets.
    let consecutive = (end - start + 1) as usize == run.len();
    if run.len() < 5 && consecutive {
        let val_extractor = |val_getter: fn(&BoundingBox) -> u32| match run.len() {
            1 => [val_getter(&run[0].1), 0, 0, 0],
            2 => [val_getter(&run[0].1), val_getter(&run[1].1), 0, 0],
            3 => [
                val_getter(&run[0].1),
                val_getter(&run[1].1),
                val_getter(&run[2].1),
                0,
            ],
            4 => [
                val_getter(&run[0].1),
                val_getter(&run[1].1),
                val_getter(&run[2].1),
                val_getter(&run[3].1),
            ],
            _ => panic!(),
        };

        // let len = run.len() as u8;
        segments.push(PolySegment {
            start,
            end,
            cx1: ScalarSegmentData2::Direct {
                values: val_extractor(|b| b.x1),
            },
            cx2: ScalarSegmentData2::Direct {
                values: val_extractor(|b| b.x2),
            },
            cy1: ScalarSegmentData2::Direct {
                values: val_extractor(|b| b.y1),
            },
            cy2: ScalarSegmentData2::Direct {
                values: val_extractor(|b| b.y2),
            },
        });
        return;
    }

    let make_pts = |coord: fn(&BoundingBox) -> f64| -> Vec<(u32, f64)> {
        run.iter().map(|(f, b)| (f - start, coord(b))).collect()
    };
    let pts_x1 = make_pts(|b| b.x1 as f64);
    let pts_y1 = make_pts(|b| b.y1 as f64);
    let pts_x2 = make_pts(|b| b.x2 as f64);
    let pts_y2 = make_pts(|b| b.y2 as f64);

    let cx1 = fit_polynomial(&pts_x1);
    let cy1 = fit_polynomial(&pts_y1);
    let cx2 = fit_polynomial(&pts_x2);
    let cy2 = fit_polynomial(&pts_y2);

    let err = max_residual(&pts_x1, &cx1)
        .max(max_residual(&pts_y1, &cy1))
        .max(max_residual(&pts_x2, &cx2))
        .max(max_residual(&pts_y2, &cy2));

    if err <= max_error_px + FIT_EPS || run.len() <= 3 {
        segments.push(PolySegment {
            start,
            end,
            cx1: ScalarSegmentData2::Poly(cx1),
            cy1: ScalarSegmentData2::Poly(cy1),
            cx2: ScalarSegmentData2::Poly(cx2),
            cy2: ScalarSegmentData2::Poly(cy2),
        });
    } else {
        let mid = run.len() / 2;
        fit_run_bounded(&run[..mid], run[mid].0 - 1, max_error_px, segments);
        fit_run_bounded(&run[mid..], end, max_error_px, segments);
    }
}

/// Split a flat list of (frame_id, BoundingBox) pairs into contiguous runs
/// (no gap larger than `gap_threshold` frames between consecutive points).
fn split_into_runs(
    points: Vec<(u32, BoundingBox)>,
    gap_threshold: u32,
) -> Vec<Vec<(u32, BoundingBox)>> {
    if points.is_empty() {
        return vec![];
    }
    let mut runs: Vec<Vec<(u32, BoundingBox)>> = vec![];
    let mut current: Vec<(u32, BoundingBox)> = vec![points[0].clone()];
    for i in 1..points.len() {
        if points[i].0 - points[i - 1].0 > gap_threshold {
            runs.push(std::mem::take(&mut current));
        }
        current.push(points[i].clone());
    }
    runs.push(current);
    runs
}

/// Returns sub-intervals of [lo, hi] (inclusive integer ranges) where A*f^2 + B*f + C > 0.
///
/// Finds real roots of the polynomial, then for each sign-constant sub-interval checks the
/// sign at the start frame. Handles integer roots correctly (eval = 0 → not included).
pub fn positive_integer_intervals(a: f64, b: f64, c: f64, lo: u32, hi: u32) -> Vec<Interval> {
    let _g = trql_instrument::scope(trql_instrument::Bucket::RootFinding);
    if lo > hi {
        return vec![];
    }

    let eval = |f: u32| -> f64 {
        let fv = f as f64;
        a * fv * fv + b * fv + c
    };

    // For each real root r, the sign can change between floor(r) and floor(r)+1.
    // Adding both as breakpoints correctly handles both non-integer and integer roots.
    let mut breaks: Vec<u32> = vec![lo, hi.saturating_add(1)];

    let mut add_root = |r: f64| {
        let k = r.floor() as i64;
        for p in [k, k + 1] {
            if p >= lo as i64 && p <= hi as i64 + 1 {
                breaks.push(p as u32);
            }
        }
    };

    if a.abs() < 1e-10 {
        if b.abs() > 1e-10 {
            add_root(-c / b);
        }
        // else constant — no roots
    } else {
        let disc = b * b - 4.0 * a * c;
        if disc >= 0.0 {
            let sq = disc.sqrt();
            add_root((-b - sq) / (2.0 * a));
            add_root((-b + sq) / (2.0 * a));
        }
        // else no real roots — one sign everywhere
    }

    breaks.sort_unstable();
    breaks.dedup();

    let mut result = Vec::new();
    for w in breaks.windows(2) {
        let start = w[0];
        let end = w[1] - 1; // inclusive
        if start <= end && eval(start) > 0.0 {
            result.push((start, end));
        }
    }
    result
}

/// Pushes the integer frames at which `a·f² + b·f + c` can change sign into `breaks`.
///
/// The same breakpoint convention [`positive_integer_intervals`] uses internally, exposed
/// for callers that need the sign pattern of *several* polynomials at once — a polygon
/// membership test collects the breaks of every edge and then samples once per resulting
/// sub-range. Roots outside `[lo, hi + 1]` are dropped, so a caller can hand over every
/// edge of a large region and pay only for the ones the curve actually approaches.
pub fn push_sign_change_breaks(coeffs: [f64; 3], lo: u32, hi: u32, breaks: &mut Vec<u32>) {
    let [a, b, c] = coeffs;
    let mut add_root = |r: f64| {
        if !r.is_finite() {
            return;
        }
        let k = r.floor() as i64;
        for p in [k, k + 1] {
            if p >= lo as i64 && p <= hi as i64 + 1 {
                breaks.push(p as u32);
            }
        }
    };

    if a.abs() < 1e-10 {
        if b.abs() > 1e-10 {
            add_root(-c / b);
        }
        // else constant — no sign change anywhere.
    } else {
        let disc = b * b - 4.0 * a * c;
        if disc >= 0.0 {
            let sq = disc.sqrt();
            add_root((-b - sq) / (2.0 * a));
            add_root((-b + sq) / (2.0 * a));
        }
    }
}

/// Like [`positive_integer_intervals`] but requires **both** `p1 > 0` and `p2 > 0`
/// simultaneously. Collects all roots of both polynomials, merges them into a single
/// sorted list of breakpoints, then checks both signs at each sub-interval midpoint.
/// This avoids two separate root-finds plus an interval intersection.
pub fn both_positive_integer_intervals(
    a1: f64,
    b1: f64,
    c1: f64,
    a2: f64,
    b2: f64,
    c2: f64,
    lo: u32,
    hi: u32,
) -> Vec<Interval> {
    let _g = trql_instrument::scope(trql_instrument::Bucket::RootFinding);
    if lo > hi {
        return vec![];
    }

    let eval1 = |f: u32| -> f64 {
        let fv = f as f64;
        a1 * fv * fv + b1 * fv + c1
    };
    let eval2 = |f: u32| -> f64 {
        let fv = f as f64;
        a2 * fv * fv + b2 * fv + c2
    };

    let mut breaks: Vec<u32> = vec![lo, hi.saturating_add(1)];

    let mut add_roots = |a: f64, b: f64, c: f64| {
        if a.abs() < 1e-10 {
            if b.abs() > 1e-10 {
                let r = -c / b;
                let k = r.floor() as i64;
                for p in [k, k + 1] {
                    if p >= lo as i64 && p <= hi as i64 + 1 {
                        breaks.push(p as u32);
                    }
                }
            }
        } else {
            let disc = b * b - 4.0 * a * c;
            if disc >= 0.0 {
                let sq = disc.sqrt();
                for r in [(-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)] {
                    let k = r.floor() as i64;
                    for p in [k, k + 1] {
                        if p >= lo as i64 && p <= hi as i64 + 1 {
                            breaks.push(p as u32);
                        }
                    }
                }
            }
        }
    };

    add_roots(a1, b1, c1);
    add_roots(a2, b2, c2);

    breaks.sort_unstable();
    breaks.dedup();

    let mut result = Vec::new();
    for w in breaks.windows(2) {
        let start = w[0];
        let end = w[1] - 1;
        if start <= end && eval1(start) > 0.0 && eval2(start) > 0.0 {
            result.push((start, end));
        }
    }
    result
}

impl PolyTraj {
    /// Builds a polynomial trajectory from an [`ExactTraj`], splitting on gaps larger than
    /// `gap_threshold`. All four coordinates share the same breakpoints (coupled fitting).
    pub fn from_traj_with_bboxes(
        traj: &ExactTraj,
        gap_threshold: u32,
        max_error_px: f64,
    ) -> PolyTraj {
        let points: Vec<(u32, BoundingBox)> = traj
            .frames
            .iter()
            .copied()
            .zip(traj.bboxes.iter().cloned())
            .collect();

        if points.is_empty() {
            return PolyTraj {
                segments: vec![],
                canonical_segments: vec![],
            };
        }

        let mut segments = Vec::new();

        let mut canonical_segments = Vec::new();
        for run in split_into_runs(points, gap_threshold) {
            let (start, end) = (run[0].0, run.last().unwrap().0);
            canonical_segments.push((start, end));
            fit_run_bounded(&run, end, max_error_px, &mut segments);
        }
        PolyTraj {
            segments,
            canonical_segments,
        }
    }

    pub fn from_traj_with_bboxes_independent(
        traj: &ExactTraj,
        gap_threshold: u32,
        max_error_px: f64,
    ) -> PolyTraj {
        // Coordinate breakpoints are now intentionally coupled.
        Self::from_traj_with_bboxes(traj, gap_threshold, max_error_px)
    }

    /// Like [`Self::from_traj_with_bboxes`], but each coordinate picks its own breakpoints
    /// independently, minimising total segments for each coordinate separately.
    // pub fn from_traj_with_bboxes_independent(
    //     traj: &ExactTraj,
    //     gap_threshold: u32,
    //     max_error_px: f64,
    // ) -> PolyTraj {
    // let points: Vec<(u32, BoundingBox)> = traj
    //     .frames
    //     .iter()
    //     .copied()
    //     .zip(traj.bboxes.iter().cloned())
    //     .collect();

    // if points.is_empty() {
    //     return PolyTraj {
    //         x1: vec![],
    //         y1: vec![],
    //         x2: vec![],
    //         y2: vec![],
    //         appear_intervals: vec![],
    //     };
    // }

    // let mut x1 = Vec::new();
    // let mut y1 = Vec::new();
    // let mut x2 = Vec::new();
    // let mut y2 = Vec::new();

    // let mut appear_intervals: Vec<(u32, u32)> = Vec::new();
    // for run in split_into_runs(points, gap_threshold) {
    //     let appear_interval = (run[0].0, run.last().unwrap().0);
    //     appear_intervals.push(appear_interval);
    //     fit_run_independent_bounded(&run, max_error_px, &mut x1, &mut y1, &mut x2, &mut y2);
    // }
    // PolyTraj {
    //     x1,
    //     y1,
    //     x2,
    //     y2,
    //     appear_intervals,
    // }
    // }

    /// Returns frame intervals (inclusive) where `self.coord relation other.coord`,
    /// computed analytically by finding the roots of the difference polynomial.
    /// Only considers frames covered by overlapping windows of both trajectories.
    // pub fn intervals_where_coord_cmp(
    //     &self,
    //     coord: Coord,
    //     relation: Ordering,
    //     other: &PolyTraj,
    // ) -> Vec<(u32, u32)> {
    //     let wins_self = self.coverage_windows();
    //     let wins_other = other.coverage_windows();
    //     let mut result = Vec::new();
    //     let mut i = 0;
    //     let mut j = 0;

    //     while i < wins_self.len() && j < wins_other.len() {
    //         let w1 = &wins_self[i];
    //         let w2 = &wins_other[j];

    //         let lo = w1.start_frame.max(w2.start_frame);
    //         let hi = w1.end_frame.min(w2.end_frame);

    //         if lo <= hi {
    //             let lhs = w1.global_coeffs(coord);
    //             let rhs = w2.global_coeffs(coord);
    //             let (da, db, dc) = match relation {
    //                 Ordering::Greater => (lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]),
    //                 Ordering::Less => (rhs[0] - lhs[0], rhs[1] - lhs[1], rhs[2] - lhs[2]),
    //                 Ordering::Equal => {
    //                     if w1.end_frame < w2.end_frame {
    //                         i += 1;
    //                     } else if w2.end_frame < w1.end_frame {
    //                         j += 1;
    //                     } else {
    //                         i += 1;
    //                         j += 1;
    //                     }
    //                     continue;
    //                 }
    //             };
    //             result.extend(positive_integer_intervals(da, db, dc, lo, hi));
    //         }

    //         if w1.end_frame < w2.end_frame {
    //             i += 1;
    //         } else if w2.end_frame < w1.end_frame {
    //             j += 1;
    //         } else {
    //             i += 1;
    //             j += 1;
    //         }
    //     }

    //     result
    // }

    /// Returns frame intervals where a unary coordinate comparison holds analytically.
    // pub fn intervals_where_unary_coord_cmp(
    //     &self,
    //     coord: UnaryCoord,
    //     cmp: UnaryCmp,
    // ) -> Vec<(u32, u32)> {
    //     let poly_coord = match coord {
    //         UnaryCoord::X1 => Coord::X1,
    //         UnaryCoord::Y1 => Coord::Y1,
    //         UnaryCoord::X2 => Coord::X2,
    //         UnaryCoord::Y2 => Coord::Y2,
    //         UnaryCoord::CenterX => Coord::CenterX,
    //         UnaryCoord::CenterY => Coord::CenterY,
    //     };
    //     let threshold = match cmp {
    //         UnaryCmp::Gt(v) | UnaryCmp::Gte(v) | UnaryCmp::Lt(v) | UnaryCmp::Lte(v) => v as f64,
    //     };
    //     const EPS: f64 = 1e-9;

    //     let mut result = Vec::new();
    //     for win in self.coverage_windows() {
    //         let [ga, gb, gc] = win.global_coeffs(poly_coord);
    //         let intervals = match cmp {
    //             UnaryCmp::Gt(_) => positive_integer_intervals(
    //                 ga,
    //                 gb,
    //                 gc - threshold,
    //                 win.start_frame,
    //                 win.end_frame,
    //             ),
    //             UnaryCmp::Gte(_) => positive_integer_intervals(
    //                 ga,
    //                 gb,
    //                 gc - threshold + EPS,
    //                 win.start_frame,
    //                 win.end_frame,
    //             ),
    //             UnaryCmp::Lt(_) => positive_integer_intervals(
    //                 -ga,
    //                 -gb,
    //                 threshold - gc,
    //                 win.start_frame,
    //                 win.end_frame,
    //             ),
    //             UnaryCmp::Lte(_) => positive_integer_intervals(
    //                 -ga,
    //                 -gb,
    //                 threshold - gc + EPS,
    //                 win.start_frame,
    //                 win.end_frame,
    //             ),
    //         };
    //         result.extend(intervals);
    //     }

    //     result
    // }

    /// Returns frame intervals (inclusive) where self.x1(f) > other.x1(f).
    // pub fn intervals_where_x1_greater(&self, other: &PolyTraj) -> Vec<(u32, u32)> {
    //     self.intervals_where_coord_cmp(Coord::X1, Ordering::Greater, other)
    // }

    /// Returns frame intervals where the normalised Euclidean distance between
    /// the two trajectories' centres is less than `theta` (i.e. `Near(theta)`).
    // pub fn intervals_where_near(&self, theta: f64, other: &PolyTraj) -> Vec<(u32, u32)> {
    //     self.near_far_intervals(theta, true, other)
    // }

    /// Returns frame intervals where the normalised Euclidean distance is greater
    // /// than `theta` (i.e. `Far(theta)`).
    // pub fn intervals_where_far(&self, theta: f64, other: &PolyTraj) -> Vec<(u32, u32)> {
    //     self.near_far_intervals(theta, false, other)
    // }

    // fn near_far_intervals(&self, theta: f64, near: bool, other: &PolyTraj) -> Vec<(u32, u32)> {
    //     let wins_self = self.coverage_windows();
    //     let wins_other = other.coverage_windows();
    //     let mut result = Vec::new();
    //     let mut i = 0;
    //     let mut j = 0;

    //     while i < wins_self.len() && j < wins_other.len() {
    //         let w1 = &wins_self[i];
    //         let w2 = &wins_other[j];

    //         let lo = w1.start_frame.max(w2.start_frame);
    //         let hi = w1.end_frame.min(w2.end_frame);

    //         if lo <= hi {
    //             // Center-x/y polynomials in global frame coords.
    //             let cx1 = w1.global_coeffs(Coord::CenterX);
    //             let cy1 = w1.global_coeffs(Coord::CenterY);
    //             let cx2 = w2.global_coeffs(Coord::CenterX);
    //             let cy2 = w2.global_coeffs(Coord::CenterY);

    //             // dx = cx1 - cx2, dy = cy1 - cy2  (degree-2)
    //             let dx = [cx1[0] - cx2[0], cx1[1] - cx2[1], cx1[2] - cx2[2]];
    //             let dy = [cy1[0] - cy2[0], cy1[1] - cy2[1], cy1[2] - cy2[2]];

    //             // avg_w = ((x2_1 - x1_1) + (x2_2 - x1_2)) / 2  (degree-2)
    //             let x2_1g = w1.global_coeffs(Coord::X2);
    //             let x1_1g = w1.global_coeffs(Coord::X1);
    //             let x2_2g = w2.global_coeffs(Coord::X2);
    //             let x1_2g = w2.global_coeffs(Coord::X1);
    //             let w1g = [
    //                 x2_1g[0] - x1_1g[0],
    //                 x2_1g[1] - x1_1g[1],
    //                 x2_1g[2] - x1_1g[2],
    //             ];
    //             let w2g = [
    //                 x2_2g[0] - x1_2g[0],
    //                 x2_2g[1] - x1_2g[1],
    //                 x2_2g[2] - x1_2g[2],
    //             ];
    //             let avg_w = [
    //                 (w1g[0] + w2g[0]) / 2.0,
    //                 (w1g[1] + w2g[1]) / 2.0,
    //                 (w1g[2] + w2g[2]) / 2.0,
    //             ];

    //             // P(f) = dx² + dy² - theta² * avg_w²
    //             let dx2 = poly2_mul(dx, dx);
    //             let dy2 = poly2_mul(dy, dy);
    //             let aw2 = poly2_mul(avg_w, avg_w);
    //             let dist2 = poly4_add(dx2, dy2);
    //             let thresh2 = poly4_scale(aw2, theta * theta);
    //             let p = poly4_add(dist2, poly4_scale(thresh2, -1.0));
    //             let query_coeffs = if near { poly4_scale(p, -1.0) } else { p };

    //             result.extend(positive_integer_intervals_deg4(query_coeffs, lo, hi));
    //         }

    //         if w1.end_frame < w2.end_frame {
    //             i += 1;
    //         } else if w2.end_frame < w1.end_frame {
    //             j += 1;
    //         } else {
    //             i += 1;
    //             j += 1;
    //         }
    //     }

    //     result
    // }

    /// Max reconstruction error (in pixels) across all frames and coordinates.
    pub fn max_reconstruction_error(&self, traj: &Trajectory, bboxes: &[BoundingBox]) -> f64 {
        let mut max_err = 0.0_f64;
        for (&fid, &bbid) in traj.frames.iter().zip(&traj.bboxes) {
            if let Some(approx) = self.eval(fid) {
                let real = &bboxes[bbid as usize];
                let err = (approx.x1 as f64 - real.x1 as f64)
                    .abs()
                    .max((approx.y1 as f64 - real.y1 as f64).abs())
                    .max((approx.x2 as f64 - real.x2 as f64).abs())
                    .max((approx.y2 as f64 - real.y2 as f64).abs());
                max_err = max_err.max(err);
            }
        }
        max_err
    }

    /// Returns the number of segments for the x1 coordinate (representative when coupled).
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    pub fn poly_bounds(&self) -> Option<Interval> {
        Some((self.segments.first()?.start, self.segments.last()?.end))
    }

    pub fn poly_overlaps(&self, rhs: &Self) -> bool {
        let mut i = 0;
        let mut j = 0;

        while i < self.num_segments() && j < rhs.num_segments() {
            let a = &self.segments[i];
            let b = &rhs.segments[j];
            if a.start <= b.end && b.start <= a.end {
                return true;
            }

            if a.end < b.end {
                i += 1;
            } else {
                j += 1;
            }
        }

        false
    }
}

// impl<'a> PolyTrajIter<'a> {
//     fn new(traj: &'a PolyTraj) -> Self {
//         Self {
//             traj,
//             segment_idx: 0,
//             next_frame: traj.x1.first().map(|seg| seg.start_frame),
//         }
//     }
// }

// impl Iterator for PolyTrajIter<'_> {
//     type Item = (u32, BoundingBox);

//     fn next(&mut self) -> Option<Self::Item> {
//         loop {
//             let frame_id = self.next_frame?;
//             let seg = self.traj.x1.get(self.segment_idx)?;

//             if frame_id > seg.end_frame {
//                 self.segment_idx += 1;
//                 self.next_frame = self
//                     .traj
//                     .x1
//                     .get(self.segment_idx)
//                     .map(|next_seg| next_seg.start_frame);
//                 continue;
//             }

//             let bbox = self.traj.eval(frame_id)?;
//             self.next_frame = if frame_id < seg.end_frame {
//                 Some(frame_id + 1)
//             } else {
//                 self.segment_idx += 1;
//                 self.traj
//                     .x1
//                     .get(self.segment_idx)
//                     .map(|next_seg| next_seg.start_frame)
//             };
//             return Some((frame_id, bbox));
//         }
//     }
// }

// impl<'a> IntoIterator for &'a PolyTraj {
//     type Item = (u32, BoundingBox);
//     type IntoIter = PolyTrajIter<'a>;

//     fn into_iter(self) -> Self::IntoIter {
//         PolyTrajIter::new(self)
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    fn bbox(x1: u32, y1: u32, x2: u32, y2: u32) -> BoundingBox {
        BoundingBox { x1, y1, x2, y2 }
    }

    // --- fit_polynomial ---

    #[test]
    fn fit_poly_empty() {
        let c = fit_polynomial(&[]);
        assert_eq!(c, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn fit_poly_single_point() {
        let c = fit_polynomial(&[(0, 42.0)]);
        assert_eq!(c, [0.0, 0.0, 42.0]);
    }

    #[test]
    fn fit_poly_triple() {
        let c = fit_polynomial(&[(0, 70.0), (1, 70.0), (2, 71.0)]);
        let d = fit_polynomial(&[(0, 114.0), (1, 114.0), (2, 115.0)]);

        // With three points and degree-2 fit, the curve should interpolate exactly.
        let err_c = max_residual(&[(0, 70.0), (1, 70.0), (2, 71.0)], &c);
        let err_d = max_residual(&[(0, 114.0), (1, 114.0), (2, 115.0)], &d);
        assert!(
            err_c < 1e-9,
            "fit should interpolate triple C, got coeffs {c:?}"
        );
        assert!(
            err_d < 1e-9,
            "fit should interpolate triple D, got coeffs {d:?}"
        );
    }

    #[test]
    fn fit_poly_linear() {
        // y = 2t + 10
        let pts: Vec<(u32, f64)> = (0..5).map(|t| (t, 2.0 * t as f64 + 10.0)).collect();
        let c = fit_polynomial(&pts);
        assert!(c[0].abs() < 1e-6, "a should be ~0, got {}", c[0]);
        assert!((c[1] - 2.0).abs() < 1e-6, "b should be ~2, got {}", c[1]);
        assert!((c[2] - 10.0).abs() < 1e-6, "c should be ~10, got {}", c[2]);
    }

    #[test]
    fn fit_poly_quadratic() {
        // y = 3t^2 + 1
        let pts: Vec<(u32, f64)> = (0..10)
            .map(|t| (t, 3.0 * (t as f64).powi(2) + 1.0))
            .collect();
        let c = fit_polynomial(&pts);
        assert!((c[0] - 3.0).abs() < 1e-6, "a should be ~3, got {}", c[0]);
        assert!(c[1].abs() < 1e-6, "b should be ~0, got {}", c[1]);
        assert!((c[2] - 1.0).abs() < 1e-6, "c should be ~1, got {}", c[2]);
    }

    // --- max_residual ---

    #[test]
    fn max_residual_exact_fit() {
        let pts = vec![(0, 5.0), (1, 5.0), (2, 5.0)];
        let coeffs = [0.0, 0.0, 5.0];
        assert!(max_residual(&pts, &coeffs) < 1e-10);
    }

    #[test]
    fn max_residual_with_error() {
        let pts = vec![(0, 5.0), (1, 8.0)]; // actual: 5, 8
        let coeffs = [0.0, 0.0, 5.0]; // predicts: 5, 5
        assert!((max_residual(&pts, &coeffs) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn from_traj_with_bboxes_independent_valid_bbox_ordering() {
        let traj = ExactTraj {
            frames: vec![0, 1, 2, 3, 4, 5],
            bboxes: vec![
                bbox(10, 10, 20, 20),
                bbox(10, 10, 20, 20),
                bbox(25, 10, 30, 20),
                bbox(25, 35, 30, 40),
                bbox(10, 35, 20, 40),
                bbox(10, 10, 20, 20),
            ],
        };
        let poly = PolyTraj::from_traj_with_bboxes_independent(&traj, 5, 0.0);
        for f in 0..=5 {
            let bb = poly.eval(f).expect("frame should evaluate");
            assert!(bb.x1 <= bb.x2, "x-order invalid at frame {f}: {bb:?}");
            assert!(bb.y1 <= bb.y2, "y-order invalid at frame {f}: {bb:?}");
        }
    }

    #[test]
    fn missed_frame_at_split_point_stays_covered() {
        // Jittery boxes force splits at 0 px; frame 5 is a missed detection that the
        // midpoint split lands on. Every frame of the run must still evaluate.
        let frames: Vec<u32> = (0..12).filter(|&f| f != 5).collect();
        let bboxes = frames
            .iter()
            .map(|&f| {
                let j = if f % 2 == 0 { 0 } else { 3 };
                bbox(10 + f * 2 + j, 10, 40 + f * 2 + j, 30)
            })
            .collect();
        let poly = PolyTraj::from_traj_with_bboxes(&ExactTraj { frames, bboxes }, 10, 0.0);
        for f in 0..12 {
            assert!(poly.eval(f).is_some(), "frame {f} not covered");
        }
    }

    #[test]
    fn short_piece_spanning_a_missed_frame_keeps_detections_in_place() {
        // Three detections with a hole (0, 1, 4): each must evaluate to its own box,
        // not to its neighbour's, and the hole must not collapse to the origin.
        let traj = ExactTraj {
            frames: vec![0, 1, 4],
            bboxes: vec![bbox(10, 10, 20, 20), bbox(12, 10, 22, 20), bbox(18, 10, 28, 20)],
        };
        let poly = PolyTraj::from_traj_with_bboxes(&traj, 10, 0.0);
        assert_eq!(poly.eval(0).unwrap().x1, 10);
        assert_eq!(poly.eval(1).unwrap().x1, 12);
        assert_eq!(poly.eval(4).unwrap().x1, 18);
        for f in 2..=3 {
            assert!(poly.eval(f).unwrap().x1 >= 12, "hole frame {f} collapsed");
        }
    }

    // --- UnaryCoord ---

    #[test]
    fn unary_coord_extracts_values() {
        let b = bbox(10, 20, 30, 40);
        assert_eq!(UnaryCoord::X1.eval(&b), 10);
        assert_eq!(UnaryCoord::Y1.eval(&b), 20);
        assert_eq!(UnaryCoord::X2.eval(&b), 30);
        assert_eq!(UnaryCoord::Y2.eval(&b), 40);
        assert_eq!(UnaryCoord::CenterX.eval(&b), 20);
        assert_eq!(UnaryCoord::CenterY.eval(&b), 30);
    }

    // --- UnaryCmp ---

    #[test]
    fn unary_cmp_gt() {
        assert!(UnaryCmp::Gt(10).eval(11));
        assert!(!UnaryCmp::Gt(10).eval(10));
        assert!(!UnaryCmp::Gt(10).eval(9));
    }

    #[test]
    fn unary_cmp_gte() {
        assert!(UnaryCmp::Gte(10).eval(11));
        assert!(UnaryCmp::Gte(10).eval(10));
        assert!(!UnaryCmp::Gte(10).eval(9));
    }

    #[test]
    fn unary_cmp_lt() {
        assert!(UnaryCmp::Lt(10).eval(9));
        assert!(!UnaryCmp::Lt(10).eval(10));
        assert!(!UnaryCmp::Lt(10).eval(11));
    }

    #[test]
    fn unary_cmp_lte() {
        assert!(UnaryCmp::Lte(10).eval(9));
        assert!(UnaryCmp::Lte(10).eval(10));
        assert!(!UnaryCmp::Lte(10).eval(11));
    }
}
