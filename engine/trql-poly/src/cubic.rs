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
    // coord(t) = coeffs[0]*t^3 + coeffs[1]*t^2 + coeffs[2]*t + coeffs[3]
    Poly([f64; 4]),
    // Values sampled at integer frames t = 0..len-1 where frame_id = start_frame + t.
    Direct { values: [u32; S] },
}

impl<const S: usize> ScalarSegmentData2<S> {
    pub fn eval_local(&self, t: u32) -> f64 {
        match &self {
            ScalarSegmentData2::Poly(coeffs) => {
                let tf = t as f64;
                ((coeffs[0] * tf + coeffs[1]) * tf + coeffs[2]) * tf + coeffs[3]
            }
            ScalarSegmentData2::Direct { values } => {
                if t < values.len() as u32 {
                    values[t as usize] as f64
                } else {
                    // `values` is fixed at S samples, so this means the segment
                    // was asked for a frame past its own capacity — an upstream
                    // invariant break, not something a query can cause. Hold the
                    // last sample and let debug builds trip on it.
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

/// One cubic polynomial segment for a single coordinate.
/// coord(t) = coeffs[0]*(t - start_frame)^3 + coeffs[1]*(t - start_frame)^2
///          + coeffs[2]*(t - start_frame) + coeffs[3]
#[derive(Clone, Debug, Encode, Decode)]
pub struct ScalarSegment {
    pub start_frame: u32,
    pub end_frame: u32,
    data: ScalarSegmentData,
}

#[derive(Clone, Debug, Encode, Decode)]
enum ScalarSegmentData {
    Poly([f64; 4]),
    // Values sampled at integer frames t = 0..len-1 where frame_id = start_frame + t.
    Direct { len: u8, values: [u32; 4] },
}

impl ScalarSegment {
    pub fn from_coeffs(start_frame: u32, end_frame: u32, coeffs: [f64; 4]) -> Self {
        Self {
            start_frame,
            end_frame,
            data: ScalarSegmentData::Poly(coeffs),
        }
    }

    #[allow(dead_code)]
    fn from_values(run: &[(u32, f64)]) -> Self {
        let values = match run.len() {
            1 => [run[0].1 as u32, 0, 0, 0],
            2 => [run[0].1 as u32, run[1].1 as u32, 0, 0],
            3 => [run[0].1 as u32, run[1].1 as u32, run[2].1 as u32, 0],
            4 => [
                run[0].1 as u32,
                run[1].1 as u32,
                run[2].1 as u32,
                run[3].1 as u32,
            ],
            _ => panic!("Shouldn't happen"),
        };
        ScalarSegment {
            start_frame: run[0].0,
            end_frame: run.last().unwrap().0,
            data: ScalarSegmentData::Direct {
                len: run.len() as u8,
                values: values,
            },
        }
    }

    pub fn is_direct(&self) -> bool {
        matches!(self.data, ScalarSegmentData::Direct { .. })
    }

    pub fn local_coeffs(&self) -> [f64; 4] {
        match &self.data {
            ScalarSegmentData::Poly(coeffs) => *coeffs,
            ScalarSegmentData::Direct { .. } => panic!("Shouldn't do this"),
        }
    }

    pub fn eval_local(&self, t: u32) -> Option<f64> {
        match &self.data {
            ScalarSegmentData::Poly(coeffs) => {
                let tf = t as f64;
                Some(((coeffs[0] * tf + coeffs[1]) * tf + coeffs[2]) * tf + coeffs[3])
            }
            ScalarSegmentData::Direct { len, values } => {
                if t < *len as u32 {
                    Some(values[t as usize] as f64)
                } else {
                    None
                }
            }
        }
    }

    ///// Expand local polynomial to global frame coordinates.
    ///// Returns (A, B, C) such that coord(f) = A*f^2 + B*f + C.
    // pub fn global_coeffs(&self) -> [f64; 3] {
    //     let coeffs = self.local_coeffs();
    //     let (a, b, c) = (coeffs[0], coeffs[1], coeffs[2]);
    //     let sf = self.start_frame as f64;
    //     [a, b - 2.0 * a * sf, a * sf * sf - b * sf + c]
    // }
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
//     pub appear_intervals: Vec<Interval>,
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

/// Solve the small dense linear system `m·x = rhs` (size `n ≤ 4`) via Gaussian
/// elimination with partial pivoting. Returns `None` if the matrix is singular.
fn solve_linear_system(mut m: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    let n = rhs.len();
    for col in 0..n {
        // Partial pivot.
        let mut pivot = col;
        for r in (col + 1)..n {
            if m[r][col].abs() > m[pivot][col].abs() {
                pivot = r;
            }
        }
        if m[pivot][col].abs() < 1e-12 {
            return None;
        }
        m.swap(col, pivot);
        rhs.swap(col, pivot);

        for r in (col + 1)..n {
            let factor = m[r][col] / m[col][col];
            for c in col..n {
                m[r][c] -= factor * m[col][c];
            }
            rhs[r] -= factor * rhs[col];
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut acc = rhs[i];
        for c in (i + 1)..n {
            acc -= m[i][c] * x[c];
        }
        x[i] = acc / m[i][i];
    }
    Some(x)
}

/// Least-squares fit of a polynomial of `degree` to `(z, y)` samples, returning
/// coefficients in ascending power order (`out[i]` multiplies `z^i`).
fn fit_ls_in_z(zs: &[f64], ys: &[f64], degree: usize) -> Option<Vec<f64>> {
    let cols = degree + 1;
    // Powers of z up to 2*degree, summed across all samples (normal equations).
    let mut power_sums = vec![0.0_f64; 2 * degree + 1];
    for &z in zs {
        let mut zp = 1.0;
        for ps in power_sums.iter_mut() {
            *ps += zp;
            zp *= z;
        }
    }
    let mut rhs = vec![0.0_f64; cols];
    for (&z, &y) in zs.iter().zip(ys) {
        let mut zp = 1.0;
        for r in rhs.iter_mut() {
            *r += y * zp;
            zp *= z;
        }
    }
    let m: Vec<Vec<f64>> = (0..cols)
        .map(|i| (0..cols).map(|j| power_sums[i + j]).collect())
        .collect();
    solve_linear_system(m, rhs)
}

/// Re-express a polynomial given in ascending `z`-coefficients (`z = (t-mu)/sigma`)
/// as ascending coefficients in raw `t`.
fn z_coeffs_to_raw_t(coeffs_z: &[f64], mu: f64, sigma: f64) -> Vec<f64> {
    // z = p*t + q
    let p = 1.0 / sigma;
    let q = -mu / sigma;
    // Horner over the z-coefficients (highest degree first), composing with (p*t + q).
    let mut acc = vec![*coeffs_z.last().unwrap()];
    for &cz in coeffs_z.iter().rev().skip(1) {
        // acc = acc * (p*t + q) + cz
        let mut next = vec![0.0_f64; acc.len() + 1];
        for (i, &a) in acc.iter().enumerate() {
            next[i] += a * q;
            next[i + 1] += a * p;
        }
        next[0] += cz;
        acc = next;
    }
    acc
}

/// Fit a degree-3 polynomial to `points` (frame_offset, value) via least squares.
/// Returns [a, b, c, d] so that value(t) ≈ a*t^3 + b*t^2 + c*t + d
/// where t = frame_id - start_frame (i.e. points\[0\].0 should already be offset-zeroed).
pub fn fit_polynomial(points: &[(u32, f64)]) -> [f64; 4] {
    let n = points.len();
    if n == 0 {
        return [0.0; 4];
    }
    if n == 1 {
        return [0.0, 0.0, 0.0, points[0].1];
    }

    // Center/scale t for conditioning, then fit the highest degree the data supports.
    let degree = (n - 1).min(3);
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

    // Fall back to progressively lower degrees if the normal equations are singular.
    let mut coeffs_t_asc: Vec<f64> = vec![0.0];
    for deg in (1..=degree).rev() {
        if let Some(coeffs_z) = fit_ls_in_z(&zs, &ys, deg) {
            coeffs_t_asc = z_coeffs_to_raw_t(&coeffs_z, mu, sigma);
            break;
        }
    }
    if coeffs_t_asc.len() == 1 {
        // Degenerate (e.g. all frames equal): use the mean value as a constant.
        coeffs_t_asc = vec![ys.iter().sum::<f64>() / n as f64];
    }

    // Pack ascending t-coefficients into [a, b, c, d] (descending, t^3..t^0).
    let get = |i: usize| coeffs_t_asc.get(i).copied().unwrap_or(0.0);
    [get(3), get(2), get(1), get(0)]
}

/// Max absolute residual between the polynomial and the actual points.
pub fn max_residual(points: &[(u32, f64)], coeffs: &[f64; 4]) -> f64 {
    points
        .iter()
        .map(|&(t, y)| {
            let t = t as f64;
            let y_pred = ((coeffs[0] * t + coeffs[1]) * t + coeffs[2]) * t + coeffs[3];
            (y - y_pred).abs()
        })
        .fold(0.0_f64, f64::max)
}

/// Recursively split `run` (single coordinate) until reconstruction error ≤ `max_error_px`,
/// then push the fitted [`ScalarSegment`](s).
#[allow(dead_code)]
fn fit_scalar_run_bounded(run: &[(u32, f64)], max_error_px: f64, out: &mut Vec<ScalarSegment>) {
    const FIT_EPS: f64 = 1e-9;
    if run.is_empty() {
        return;
    }
    let start_frame = run[0].0;
    let span_len = run.len();
    if span_len < 5 {
        out.push(ScalarSegment::from_values(run));
        return;
    }
    let pts: Vec<(u32, f64)> = run.iter().map(|(f, v)| (f - start_frame, *v)).collect();
    let coeffs = fit_polynomial(&pts);
    let err = max_residual(&pts, &coeffs);

    if err <= max_error_px + FIT_EPS || run.len() <= 3 {
        out.push(ScalarSegment::from_coeffs(
            start_frame,
            run[run.len() - 1].0,
            coeffs,
        ));
    } else {
        let mid = run.len() / 2;
        fit_scalar_run_bounded(&run[..mid], max_error_px, out);
        fit_scalar_run_bounded(&run[mid..], max_error_px, out);
    }
}

/// Coupled fitting: split `run` until the max error across all four coordinates is ≤
/// `max_error_px`, then push one [`ScalarSegment`] per coordinate with shared breakpoints.
fn fit_run_bounded(
    run: &[(u32, BoundingBox)],
    max_error_px: f64,
    segments: &mut Vec<PolySegment<4>>,
) {
    const FIT_EPS: f64 = 1e-9;
    if run.is_empty() {
        return;
    }
    let start = run[0].0;
    let end = run[run.len() - 1].0;
    if run.len() < 5 {
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
        fit_run_bounded(&run[..mid], max_error_px, segments);
        fit_run_bounded(&run[mid..], max_error_px, segments);
    }
}

/// Independent fitting: each coordinate gets its own optimal breakpoints.
#[allow(dead_code)]
fn fit_run_independent_bounded(
    run: &[(u32, BoundingBox)],
    max_error_px: f64,
    x1_out: &mut Vec<ScalarSegment>,
    y1_out: &mut Vec<ScalarSegment>,
    x2_out: &mut Vec<ScalarSegment>,
    y2_out: &mut Vec<ScalarSegment>,
) {
    if run.is_empty() {
        return;
    }
    let x1_vals: Vec<(u32, f64)> = run.iter().map(|(f, b)| (*f, b.x1 as f64)).collect();
    let y1_vals: Vec<(u32, f64)> = run.iter().map(|(f, b)| (*f, b.y1 as f64)).collect();
    let x2_vals: Vec<(u32, f64)> = run.iter().map(|(f, b)| (*f, b.x2 as f64)).collect();
    let y2_vals: Vec<(u32, f64)> = run.iter().map(|(f, b)| (*f, b.y2 as f64)).collect();

    fit_scalar_run_bounded(&x1_vals, max_error_px, x1_out);
    fit_scalar_run_bounded(&y1_vals, max_error_px, y1_out);
    fit_scalar_run_bounded(&x2_vals, max_error_px, x2_out);
    fit_scalar_run_bounded(&y2_vals, max_error_px, y2_out);
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
pub fn positive_integer_intervals(
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    lo: u32,
    hi: u32,
) -> Vec<Interval> {
    let _g = trql_instrument::scope(trql_instrument::Bucket::RootFinding);
    if lo > hi {
        return vec![];
    }

    let eval = |f: u32| -> f64 {
        let fv = f as f64;
        ((a * fv + b) * fv + c) * fv + d
    };

    // For each real root r, the sign can change between floor(r) and floor(r)+1.
    // Adding both as breakpoints correctly handles both non-integer and integer roots.
    let mut breaks: Vec<u32> = vec![lo, hi.saturating_add(1)];
    push_root_breaks(&real_roots(a, b, c, d), lo, hi, &mut breaks);

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

/// Like [`positive_integer_intervals`] but requires **both** `p1 > 0` and `p2 > 0`
/// simultaneously. Collects all roots of both polynomials, merges them into a single
/// sorted list of breakpoints, then checks both signs at each sub-interval midpoint.
/// This avoids two separate root-finds plus an interval intersection.
pub fn both_positive_integer_intervals(
    a1: f64,
    b1: f64,
    c1: f64,
    d1: f64,
    a2: f64,
    b2: f64,
    c2: f64,
    d2: f64,
    lo: u32,
    hi: u32,
) -> Vec<Interval> {
    let _g = trql_instrument::scope(trql_instrument::Bucket::RootFinding);
    if lo > hi {
        return vec![];
    }

    let eval1 = |f: u32| -> f64 {
        let fv = f as f64;
        ((a1 * fv + b1) * fv + c1) * fv + d1
    };
    let eval2 = |f: u32| -> f64 {
        let fv = f as f64;
        ((a2 * fv + b2) * fv + c2) * fv + d2
    };

    let mut breaks: Vec<u32> = vec![lo, hi.saturating_add(1)];
    push_root_breaks(&real_roots(a1, b1, c1, d1), lo, hi, &mut breaks);
    push_root_breaks(&real_roots(a2, b2, c2, d2), lo, hi, &mut breaks);

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

/// Pushes the integer frames at which `a·f³ + b·f² + c·f + d` can change sign into `breaks`.
///
/// The same breakpoint convention [`positive_integer_intervals`] uses internally, exposed
/// for callers that need the sign pattern of *several* polynomials at once — a polygon
/// membership test collects the breaks of every edge and then samples once per resulting
/// sub-range. Roots outside `[lo, hi + 1]` are dropped, so a caller can hand over every
/// edge of a large region and pay only for the ones the curve actually approaches.
pub fn push_sign_change_breaks(coeffs: [f64; 4], lo: u32, hi: u32, breaks: &mut Vec<u32>) {
    let [a, b, c, d] = coeffs;
    push_root_breaks(&real_roots(a, b, c, d), lo, hi, breaks);
}

/// For each real root `r`, the polynomial's sign can only change between `floor(r)` and
/// `floor(r) + 1`; push both (clamped to `[lo, hi+1]`) as candidate breakpoints.
fn push_root_breaks(roots: &[f64], lo: u32, hi: u32, breaks: &mut Vec<u32>) {
    for &r in roots {
        if !r.is_finite() {
            continue;
        }
        let k = r.floor() as i64;
        for p in [k, k + 1] {
            if p >= lo as i64 && p <= hi as i64 + 1 {
                breaks.push(p as u32);
            }
        }
    }
}

/// Real roots of `a*x^3 + b*x^2 + c*x + d`, handling all lower-degree degenerate cases.
fn real_roots(a: f64, b: f64, c: f64, d: f64) -> Vec<f64> {
    const EPS: f64 = 1e-10;

    if a.abs() < EPS {
        // Quadratic b*x^2 + c*x + d.
        if b.abs() < EPS {
            if c.abs() < EPS {
                return vec![];
            }
            return vec![-d / c];
        }
        let disc = c * c - 4.0 * b * d;
        if disc < 0.0 {
            return vec![];
        }
        let sq = disc.sqrt();
        return vec![(-c - sq) / (2.0 * b), (-c + sq) / (2.0 * b)];
    }

    // Monic form x^3 + b1 x^2 + c1 x + d1, then depress with x = t - b1/3.
    let b1 = b / a;
    let c1 = c / a;
    let d1 = d / a;
    let shift = b1 / 3.0;
    let p = c1 - b1 * b1 / 3.0;
    let q = 2.0 * b1 * b1 * b1 / 27.0 - b1 * c1 / 3.0 + d1;

    if p.abs() < EPS && q.abs() < EPS {
        return vec![-shift]; // triple root
    }

    let disc = q * q / 4.0 + p * p * p / 27.0;
    if disc > EPS {
        // One real root (Cardano).
        let sq = disc.sqrt();
        let u = (-q / 2.0 + sq).cbrt();
        let v = (-q / 2.0 - sq).cbrt();
        vec![u + v - shift]
    } else if disc.abs() <= EPS {
        // Multiple roots, all real.
        let u = (-q / 2.0).cbrt();
        vec![2.0 * u - shift, -u - shift]
    } else {
        // Three distinct real roots (trigonometric method; here p < 0).
        let neg_p_over_3 = -p / 3.0;
        let m = 2.0 * neg_p_over_3.sqrt();
        let arg = (-q / (2.0 * neg_p_over_3.powf(1.5))).clamp(-1.0, 1.0);
        let phi = arg.acos();
        (0..3)
            .map(|k| {
                let angle = phi / 3.0 - 2.0 * std::f64::consts::PI * k as f64 / 3.0;
                m * angle.cos() - shift
            })
            .collect()
    }
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
            fit_run_bounded(&run, max_error_px, &mut segments);
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

    // let mut appear_intervals: Vec<Interval> = Vec::new();
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
    // ) -> Vec<Interval> {
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
    // ) -> Vec<Interval> {
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
    // pub fn intervals_where_x1_greater(&self, other: &PolyTraj) -> Vec<Interval> {
    //     self.intervals_where_coord_cmp(Coord::X1, Ordering::Greater, other)
    // }

    /// Returns frame intervals where the normalised Euclidean distance between
    /// the two trajectories' centres is less than `theta` (i.e. `Near(theta)`).
    // pub fn intervals_where_near(&self, theta: f64, other: &PolyTraj) -> Vec<Interval> {
    //     self.near_far_intervals(theta, true, other)
    // }

    /// Returns frame intervals where the normalised Euclidean distance is greater
    // /// than `theta` (i.e. `Far(theta)`).
    // pub fn intervals_where_far(&self, theta: f64, other: &PolyTraj) -> Vec<Interval> {
    //     self.near_far_intervals(theta, false, other)
    // }

    // fn near_far_intervals(&self, theta: f64, near: bool, other: &PolyTraj) -> Vec<Interval> {
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

    fn scalar_seg(start: u32, end: u32, coeffs: [f64; 4]) -> ScalarSegment {
        ScalarSegment::from_coeffs(start, end, coeffs)
    }

    // --- fit_polynomial ---

    #[test]
    fn fit_poly_empty() {
        let c = fit_polynomial(&[]);
        assert_eq!(c, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn fit_poly_single_point() {
        let c = fit_polynomial(&[(0, 42.0)]);
        assert_eq!(c, [0.0, 0.0, 0.0, 42.0]);
    }

    #[test]
    fn fit_poly_triple() {
        let c = fit_polynomial(&[(0, 70.0), (1, 70.0), (2, 71.0)]);
        let d = fit_polynomial(&[(0, 114.0), (1, 114.0), (2, 115.0)]);

        // Three points are fit with a degree-2 curve, which interpolates exactly.
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
        // y = 2t + 10 -> [a, b, c, d] = [0, 0, 2, 10]
        let pts: Vec<(u32, f64)> = (0..5).map(|t| (t, 2.0 * t as f64 + 10.0)).collect();
        let c = fit_polynomial(&pts);
        assert!(c[0].abs() < 1e-6, "t^3 coeff should be ~0, got {}", c[0]);
        assert!(c[1].abs() < 1e-6, "t^2 coeff should be ~0, got {}", c[1]);
        assert!(
            (c[2] - 2.0).abs() < 1e-6,
            "t coeff should be ~2, got {}",
            c[2]
        );
        assert!(
            (c[3] - 10.0).abs() < 1e-6,
            "const should be ~10, got {}",
            c[3]
        );
    }

    #[test]
    fn fit_poly_quadratic() {
        // y = 3t^2 + 1 -> [a, b, c, d] = [0, 3, 0, 1]
        let pts: Vec<(u32, f64)> = (0..10)
            .map(|t| (t, 3.0 * (t as f64).powi(2) + 1.0))
            .collect();
        let c = fit_polynomial(&pts);
        assert!(c[0].abs() < 1e-6, "t^3 coeff should be ~0, got {}", c[0]);
        assert!(
            (c[1] - 3.0).abs() < 1e-6,
            "t^2 coeff should be ~3, got {}",
            c[1]
        );
        assert!(c[2].abs() < 1e-6, "t coeff should be ~0, got {}", c[2]);
        assert!(
            (c[3] - 1.0).abs() < 1e-6,
            "const should be ~1, got {}",
            c[3]
        );
    }

    #[test]
    fn fit_poly_cubic() {
        // y = 2t^3 - t^2 + 4 -> [a, b, c, d] = [2, -1, 0, 4]
        let pts: Vec<(u32, f64)> = (0..10)
            .map(|t| {
                let tf = t as f64;
                (t, 2.0 * tf.powi(3) - tf.powi(2) + 4.0)
            })
            .collect();
        let c = fit_polynomial(&pts);
        assert!(
            (c[0] - 2.0).abs() < 1e-6,
            "t^3 coeff should be ~2, got {}",
            c[0]
        );
        assert!(
            (c[1] + 1.0).abs() < 1e-6,
            "t^2 coeff should be ~-1, got {}",
            c[1]
        );
        assert!(c[2].abs() < 1e-6, "t coeff should be ~0, got {}", c[2]);
        assert!(
            (c[3] - 4.0).abs() < 1e-6,
            "const should be ~4, got {}",
            c[3]
        );
    }

    // --- max_residual ---

    #[test]
    fn max_residual_exact_fit() {
        let pts = vec![(0, 5.0), (1, 5.0), (2, 5.0)];
        let coeffs = [0.0, 0.0, 0.0, 5.0];
        assert!(max_residual(&pts, &coeffs) < 1e-10);
    }

    #[test]
    fn max_residual_with_error() {
        let pts = vec![(0, 5.0), (1, 8.0)]; // actual: 5, 8
        let coeffs = [0.0, 0.0, 0.0, 5.0]; // predicts: 5, 5
        assert!((max_residual(&pts, &coeffs) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn cubic_roots_three_real() {
        // (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
        let mut roots = real_roots(1.0, -6.0, 11.0, -6.0);
        roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(roots.len(), 3);
        for (got, want) in roots.iter().zip([1.0, 2.0, 3.0]) {
            assert!((got - want).abs() < 1e-6, "root {got} != {want}");
        }
    }

    #[test]
    fn cubic_positive_intervals() {
        // (x-1)(x-2)(x-3) > 0 on (1,2) and (3,inf); over [0,5] integers: {} between 1&2,
        // and {4,5}. At x just above 1 it's positive until 2, integer frame 2..1 empty;
        // positive again past 3 -> frames 4,5.
        let got = positive_integer_intervals(1.0, -6.0, 11.0, -6.0, 0, 5);
        // Sign: x=0 -> -6 (neg); the cubic is >0 on (1,2) and (3,5].
        // Integers strictly giving >0: none in (1,2) other than none (no integer), 4 and 5.
        let frames: Vec<u32> = got.iter().flat_map(|&(l, h)| l..=h).collect();
        for f in [4u32, 5] {
            assert!(frames.contains(&f), "frame {f} should be positive: {got:?}");
        }
        for f in [0u32, 2, 3] {
            assert!(
                !frames.contains(&f),
                "frame {f} should be non-positive: {got:?}"
            );
        }
    }

    // --- PolyTraj::from_traj_with_bboxes ---

    // #[test]
    // fn from_traj_with_bboxes_contiguous() {
    //     let traj = ExactTraj {
    //         frames: vec![0, 1, 2, 3, 4],
    //         bboxes: vec![
    //             bbox(10, 20, 30, 40),
    //             bbox(12, 22, 32, 42),
    //             bbox(14, 24, 34, 44),
    //             bbox(16, 26, 36, 46),
    //             bbox(18, 28, 38, 48),
    //         ],
    //     };
    //     let poly = PolyTraj::from_traj_with_bboxes(&traj, 5, 0.0);
    //     assert_eq!(poly.num_segments(), 1.0);
    //     assert_eq!(poly.x1[0].start_frame, 0);
    //     assert_eq!(poly.x1[0].end_frame, 4);
    // }

    // #[test]
    // fn from_traj_with_bboxes_gap_splits() {
    //     let traj = ExactTraj {
    //         frames: vec![0, 1, 2, 20, 21, 22],
    //         bboxes: vec![
    //             bbox(10, 10, 20, 20),
    //             bbox(10, 10, 20, 20),
    //             bbox(10, 10, 20, 20),
    //             bbox(50, 50, 60, 60),
    //             bbox(50, 50, 60, 60),
    //             bbox(50, 50, 60, 60),
    //         ],
    //     };
    //     let poly = PolyTraj::from_traj_with_bboxes(&traj, 5, 0.0);
    //     assert_eq!(poly.num_segments(), 2.0);
    // }

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

    // --- PolyTraj::eval ---

    // #[test]
    // fn eval_in_segment() {
    //     let traj = ExactTraj {
    //         frames: vec![0, 1, 2],
    //         bboxes: vec![
    //             bbox(10, 10, 20, 20),
    //             bbox(10, 10, 20, 20),
    //             bbox(10, 10, 20, 20),
    //         ],
    //     };
    //     let poly = PolyTraj::from_traj_with_bboxes(&traj, 5, 0.0);
    //     let result = poly.eval(1).expect("should be within segment");
    //     assert_eq!(result.x1, 10);
    //     assert_eq!(result.y1, 10);
    // }

    // #[test]
    // fn eval_in_gap_returns_none() {
    //     let traj = ExactTraj {
    //         frames: vec![0, 1, 20, 21],
    //         bboxes: vec![
    //             bbox(10, 10, 20, 20),
    //             bbox(10, 10, 20, 20),
    //             bbox(50, 50, 60, 60),
    //             bbox(50, 50, 60, 60),
    //         ],
    //     };
    //     let poly = PolyTraj::from_traj_with_bboxes(&traj, 5, 0.0);
    //     assert!(poly.eval(10).is_none());
    // }

    // #[test]
    // fn eval_out_of_range_returns_none() {
    //     let traj = ExactTraj {
    //         frames: vec![5, 6, 7],
    //         bboxes: vec![bbox(10, 10, 20, 20); 3],
    //     };
    //     let poly = PolyTraj::from_traj_with_bboxes(&traj, 5, 0.0);
    //     assert!(poly.eval(100).is_none());
    // }

    // #[test]
    // fn eval_reorders_crossed_edges() {
    //     // x1 > x2 and y1 > y2 in stored coefficients — eval should swap them.
    //     let poly = PolyTraj {
    //         x1: vec![scalar_seg(0, 0, [0.0, 0.0, 30.0])],
    //         y1: vec![scalar_seg(0, 0, [0.0, 0.0, 40.0])],
    //         x2: vec![scalar_seg(0, 0, [0.0, 0.0, 10.0])],
    //         y2: vec![scalar_seg(0, 0, [0.0, 0.0, 20.0])],
    //         appear_intervals: vec![],
    //     };

    //     let bb = poly.eval(0).expect("frame should evaluate");
    //     assert_eq!(bb, bbox(10, 20, 30, 40));
    // }

    // #[test]
    // fn iterates_all_frames_in_segment() {
    //     let traj = ExactTraj {
    //         frames: vec![5, 6, 7],
    //         bboxes: vec![bbox(10, 10, 20, 20); 3],
    //     };
    //     let poly = PolyTraj::from_traj_with_bboxes(&traj, 5, 0.0);

    //     let items: Vec<_> = (&poly).into_iter().collect();
    //     assert_eq!(
    //         items,
    //         vec![
    //             (5, bbox(10, 10, 20, 20)),
    //             (6, bbox(10, 10, 20, 20)),
    //             (7, bbox(10, 10, 20, 20)),
    //         ]
    //     );
    // }

    // #[test]
    // fn iterator_skips_gaps_between_segments() {
    //     let traj = ExactTraj {
    //         frames: vec![0, 1, 10, 11],
    //         bboxes: vec![
    //             bbox(10, 10, 20, 20),
    //             bbox(12, 12, 22, 22),
    //             bbox(50, 50, 60, 60),
    //             bbox(52, 52, 62, 62),
    //         ],
    //     };
    //     let poly = PolyTraj::from_traj_with_bboxes(&traj, 5, 0.0);

    //     let frames: Vec<_> = (&poly).into_iter().map(|(fid, _)| fid).collect();
    //     assert_eq!(frames, vec![0, 1, 10, 11]);
    // }

    // --- PolyTraj::intervals_where_coord_cmp ---

    // #[test]
    // fn intervals_where_x1_greater() {
    //     // Object A at x1=100, object B at x1=50 — A.x1 > B.x1 always
    //     let traj_a = ExactTraj {
    //         frames: vec![0, 1, 2, 3, 4],
    //         bboxes: vec![bbox(100, 0, 110, 10); 5],
    //     };
    //     let traj_b = ExactTraj {
    //         frames: vec![0, 1, 2, 3, 4],
    //         bboxes: vec![bbox(50, 0, 60, 10); 5],
    //     };
    //     let pa = PolyTraj::from_traj_with_bboxes(&traj_a, 10, 0.0);
    //     let pb = PolyTraj::from_traj_with_bboxes(&traj_b, 10, 0.0);

    //     let result = pa.intervals_where_x1_greater(&pb);
    //     assert_eq!(result, vec![(0, 4)]);

    //     // Reverse should be empty
    //     let result_rev = pb.intervals_where_x1_greater(&pa);
    //     assert!(result_rev.is_empty());
    // }

    // #[test]
    // fn intervals_where_coord_cmp_crossing() {
    //     // A starts left, crosses over B
    //     let traj_a = ExactTraj {
    //         frames: vec![0, 1, 2, 3, 4],
    //         bboxes: vec![
    //             bbox(10, 0, 20, 10),
    //             bbox(30, 0, 40, 10),
    //             bbox(50, 0, 60, 10),
    //             bbox(70, 0, 80, 10),
    //             bbox(90, 0, 100, 10),
    //         ],
    //     };
    //     let traj_b = ExactTraj {
    //         frames: vec![0, 1, 2, 3, 4],
    //         bboxes: vec![bbox(50, 0, 60, 10); 5],
    //     };
    //     let pa = PolyTraj::from_traj_with_bboxes(&traj_a, 10, 0.0);
    //     let pb = PolyTraj::from_traj_with_bboxes(&traj_b, 10, 0.0);

    //     let result = pa.intervals_where_x1_greater(&pb);
    //     // A.x1 > B.x1(=50) for frames 3,4 (x1=70,90)
    //     assert!(!result.is_empty());
    //     // Should not include frame 0 or 1
    //     for &(s, _) in &result {
    //         assert!(s >= 2, "should not include early frames, got start={s}");
    //     }
    // }

    // --- PolyTraj::intervals_where_unary_coord_cmp ---

    // #[test]
    // fn unary_coord_cmp_gt() {
    //     let traj = ExactTraj {
    //         frames: vec![0, 1, 2, 3, 4],
    //         bboxes: vec![
    //             bbox(10, 0, 20, 10),
    //             bbox(30, 0, 40, 10),
    //             bbox(50, 0, 60, 10),
    //             bbox(70, 0, 80, 10),
    //             bbox(90, 0, 100, 10),
    //         ],
    //     };
    //     let poly = PolyTraj::from_traj_with_bboxes(&traj, 10, 0.0);
    //     let result = poly.intervals_where_unary_coord_cmp(UnaryCoord::X1, UnaryCmp::Gt(60));
    //     // x1 at frames: 10, 30, 50, 70, 90 — > 60 at frames 3, 4
    //     assert!(!result.is_empty());
    // }

    // --- CoverageWindow::global_coeffs ---

    // #[test]
    // fn coverage_window_global_coeffs_center() {
    //     // x1 segment: constant 10, x2 segment: constant 30 → CenterX = 20
    //     let x1_seg = scalar_seg(0, 10, [0.0, 0.0, 10.0]);
    //     let x2_seg = scalar_seg(0, 10, [0.0, 0.0, 30.0]);
    //     let y1_seg = scalar_seg(0, 10, [0.0, 0.0, 5.0]);
    //     let y2_seg = scalar_seg(0, 10, [0.0, 0.0, 15.0]);
    //     let win = CoverageWindow {
    //         start_frame: 0,
    //         end_frame: 10,
    //         x1: &x1_seg,
    //         y1: &y1_seg,
    //         x2: &x2_seg,
    //         y2: &y2_seg,
    //     };
    //     let [a, b, c] = win.global_coeffs(Coord::CenterX);
    //     assert!(a.abs() < 1e-10);
    //     assert!(b.abs() < 1e-10);
    //     assert!((c - 20.0).abs() < 1e-10, "CenterX should be 20, got {c}");
    // }

    // --- memory_bytes / num_segments ---

    // #[test]
    // fn memory_and_segment_count() {
    //     let traj = ExactTraj {
    //         frames: vec![0, 1, 2],
    //         bboxes: vec![bbox(0, 0, 10, 10); 3],
    //     };
    //     let poly = PolyTraj::from_traj_with_bboxes(&traj, 5, 0.0);
    //     assert_eq!(poly.num_segments(), 1.0);
    //     assert_eq!(
    //         poly.memory_bytes(),
    //         4 * std::mem::size_of::<ScalarSegment>()
    //     );
    // }

    // --- intervals_where_near / intervals_where_far (degree-4 analytic solver) ---
    //
    // Setup (genuinely degree-4 P):
    //   Object A: center_x(f) = 0.1·f² + 50  (quadratic — accelerates rightward)
    //             width = 20  →  x1 = 0.1f²+40,  x2 = 0.1f²+60
    //             center_y = 100  (fixed)
    //   Object B: center_x = 100, width = 20, center_y = 100  (stationary)
    //
    //   dx(f) = (0.1f²+50) − 100 = 0.1f² − 50   (degree-2)
    //   P(f) = dx² − theta²·avg_w² = (0.1f²−50)² − 400·theta²
    //        = 0.01f⁴ − 10f² + (2500 − 400·theta²)
    //
    // Using theta = 1.05 (CLEVRER default):
    //   P(f) = 0.01f⁴ − 10f² + (2500 − 441) = 0.01f⁴ − 10f² + 2059
    //   P = 0  →  0.01u² − 10u + 2059 = 0  (u = f²)
    //          →  u = (10 ± sqrt(100 − 4·0.01·2059)) / 0.02
    //                = (10 ± sqrt(17.64)) / 0.02
    //                = (10 ± 4.2) / 0.02
    //          →  u₁ = 290  (f = √290 ≈ 17.03)
    //             u₂ = 710  (f = √710 ≈ 26.64)
    //
    //   Near(1.05): P < 0  →  f ∈ (17.03, 26.64)  →  integer frames {18..26}
    //   Far(1.05):  P > 0  →  f ∈ [0, 17.03) ∪ (26.64, 30]  →  integer frames {0..17}∪{27..30}
    //
    // Both roots are non-integer, so there are no exact-boundary ambiguity frames.
    // This makes the test fully deterministic.

    // fn make_near_far_polys() -> (PolyTraj, PolyTraj) {
    //     // Object A: quadratic center_x = 0.1t² + 50, width=20, center_y=100.
    //     // Local coefficients: x1=[0.1,0,40], x2=[0.1,0,60], y1=[0,0,90], y2=[0,0,110]
    //     let traj_a = PolyTraj {
    //         x1: vec![scalar_seg(0, 30, [0.1, 0.0, 40.0])],
    //         x2: vec![scalar_seg(0, 30, [0.1, 0.0, 60.0])],
    //         y1: vec![scalar_seg(0, 30, [0.0, 0.0, 90.0])],
    //         y2: vec![scalar_seg(0, 30, [0.0, 0.0, 110.0])],
    //         appear_intervals: vec![],
    //     };
    //     // Object B: stationary, center_x=100, width=20, center_y=100.
    //     let traj_b = PolyTraj {
    //         x1: vec![scalar_seg(0, 30, [0.0, 0.0, 90.0])],
    //         x2: vec![scalar_seg(0, 30, [0.0, 0.0, 110.0])],
    //         y1: vec![scalar_seg(0, 30, [0.0, 0.0, 90.0])],
    //         y2: vec![scalar_seg(0, 30, [0.0, 0.0, 110.0])],
    //         appear_intervals: vec![],
    //     };
    //     (traj_a, traj_b)
    // }

    const THETA: f64 = 1.05;

    /// Brute-force: collect every integer frame in [0, 30] where the condition holds,
    /// using exact float arithmetic directly on the polynomial (no rounding to integer bbox).
    fn near_frames_brute() -> Vec<u32> {
        (0u32..=30)
            .filter(|&f| {
                let dx = 0.1 * f as f64 * f as f64 - 50.0; // cx_A - cx_B
                dx.abs() / 20.0 < THETA
            })
            .collect()
    }

    fn far_frames_brute() -> Vec<u32> {
        (0u32..=30)
            .filter(|&f| {
                let dx = 0.1 * f as f64 * f as f64 - 50.0;
                dx.abs() / 20.0 > THETA
            })
            .collect()
    }

    /// Flatten a list of (lo, hi) intervals into the set of integer frames.
    fn expand_intervals(intervals: &[Interval]) -> Vec<u32> {
        intervals.iter().flat_map(|&(lo, hi)| lo..=hi).collect()
    }

    // #[test]
    // fn intervals_where_near_deg4() {
    //     let (traj_a, traj_b) = make_near_far_polys();
    //     let analytic = traj_a.intervals_where_near(THETA, &traj_b);
    //     let expected = near_frames_brute();

    //     println!("Near({THETA}) analytic : {analytic:?}");
    //     println!("Near({THETA}) brute    : {expected:?}");

    //     assert_eq!(
    //         expand_intervals(&analytic),
    //         expected,
    //         "analytic Near intervals do not match brute-force frame scan"
    //     );
    //     // Roots at √290≈17.03 and √710≈26.64 → integer frames 18..26
    //     assert_eq!(analytic, vec![(18, 26)]);
    // }

    // #[test]
    // fn intervals_where_far_deg4() {
    //     let (traj_a, traj_b) = make_near_far_polys();
    //     let analytic = traj_a.intervals_where_far(THETA, &traj_b);
    //     let expected = far_frames_brute();

    //     println!("Far({THETA}) analytic  : {analytic:?}");
    //     println!("Far({THETA}) brute     : {expected:?}");

    //     assert_eq!(
    //         expand_intervals(&analytic),
    //         expected,
    //         "analytic Far intervals do not match brute-force frame scan"
    //     );
    //     // Complement of Near over [0,30]: frames 0..17 and 27..30
    //     assert_eq!(analytic, vec![(0, 17), (27, 30)]);
    // }
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
