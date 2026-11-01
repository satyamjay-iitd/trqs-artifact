//! Analytical compiler for structured bounding-box predicates.
//!
//! Defines [`BBoxExpr`] — a linear combination of bounding-box coordinates — and
//! [`UnaryPred`] / [`BinaryPred`] predicate trees built from it. Each predicate compiles
//! analytically to frame intervals by evaluating the underlying quadratic polynomials
//! from a [`PolyTraj`], never falling back to per-frame sampling.
//!
//! # Design
//!
//! A [`PolyTraj`] stores each coordinate (`x1`, `y1`, `x2`, `y2`) as a piecewise
//! quadratic `coord(f) = a·f² + b·f + c`. Any *linear combination* of those coordinates
//! is therefore also a piecewise quadratic, and finding the integer frame-ranges where it
//! exceeds a threshold reduces to solving a degree-2 polynomial — handled exactly by
//! [`positive_integer_intervals`].
//!
//! The expressible predicates include:
//! - Individual coordinates: `x1`, `y1`, `x2`, `y2`, `center_x`, `center_y`
//! - Derived scalars: `width = x2 − x1`, `height = y2 − y1`
//! - Aspect-ratio comparisons: `width − k·height > 0` (linear in the base coords)
//! - Any affine combination: `a·x1 + b·x2 + c·y1 + d·y2 + e`
//! - Region containment: `x1 ≥ lx1 ∧ y1 ≥ ly1 ∧ x2 ≤ lx2 ∧ y2 ≤ ly2`
//! - Convex polygon containment: one half-plane comparison per edge (see [`Polygon`])
//! - Non-convex polygon containment, via [`AnalyticalPredKind::InPolygon`]
//! - Boolean combinations of the above
//!
//! `Area = width·height` is degree 4 and is intentionally excluded; it would require a
//! quartic root solver. All other natural predicates on a bounding box are linear in the
//! base coordinates and are covered.

use std::collections::HashMap;
use std::sync::Arc;

use trql_poly::{
    PolySegment, PolyTraj, ScalarSegmentData2, both_positive_integer_intervals,
    positive_integer_intervals, push_sign_change_breaks,
};
use trql_types::{Slot, intervals::Interval};

use crate::polygon::{Polygon, corner_extreme_weights};

// ── Scalar expression ────────────────────────────────────────────────────────

/// A linear combination of the four base bounding-box coordinates plus a constant offset.
///
/// Evaluates as: `x1_w·x1 + y1_w·y1 + x2_w·x2 + y2_w·y2 + offset`.
///
/// Because each coordinate is a quadratic polynomial in frame `f`, this expression is
/// also a quadratic polynomial and can be compared to a threshold analytically.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BBoxExpr {
    pub x1_w: f64,
    pub y1_w: f64,
    pub x2_w: f64,
    pub y2_w: f64,
    pub offset: f64,
    pub used_coords: (bool, bool, bool, bool),
}

impl BBoxExpr {
    /// Builds an expression from its weights, deriving `used_coords` — a
    /// coordinate is used exactly when its weight is non-zero. Every
    /// constructor and operator goes through here, so the cached flags can
    /// never drift from the weights they summarize (evaluation reads only the
    /// flags, so a stale `false` silently drops that coordinate's term).
    pub fn from_weights(x1_w: f64, y1_w: f64, x2_w: f64, y2_w: f64, offset: f64) -> Self {
        BBoxExpr {
            x1_w,
            y1_w,
            x2_w,
            y2_w,
            offset,
            used_coords: (x1_w != 0.0, y1_w != 0.0, x2_w != 0.0, y2_w != 0.0),
        }
    }

    /// Which of the four bbox coordinates this expression depends on.
    pub fn used_coords(&self) -> (bool, bool, bool, bool) {
        self.used_coords
    }

    pub fn x1() -> Self {
        Self::from_weights(1.0, 0.0, 0.0, 0.0, 0.0)
    }
    pub fn y1() -> Self {
        Self::from_weights(0.0, 1.0, 0.0, 0.0, 0.0)
    }
    pub fn x2() -> Self {
        Self::from_weights(0.0, 0.0, 1.0, 0.0, 0.0)
    }
    pub fn y2() -> Self {
        Self::from_weights(0.0, 0.0, 0.0, 1.0, 0.0)
    }
    pub fn center_x() -> Self {
        Self::from_weights(0.5, 0.0, 0.5, 0.0, 0.0)
    }
    pub fn center_y() -> Self {
        Self::from_weights(0.0, 0.5, 0.0, 0.5, 0.0)
    }
    pub fn width() -> Self {
        Self::from_weights(-1.0, 0.0, 1.0, 0.0, 0.0)
    }
    pub fn height() -> Self {
        Self::from_weights(0.0, -1.0, 0.0, 1.0, 0.0)
    }
    pub fn constant(v: f64) -> Self {
        Self::from_weights(0.0, 0.0, 0.0, 0.0, v)
    }

    /// The four `(x, y)` expression pairs naming the corners of the box.
    ///
    /// A convex region contains the box exactly when it contains all four, which is what
    /// turns whole-box containment into more of the same half-plane comparisons.
    pub fn corners() -> [(Self, Self); 4] {
        [
            (Self::x1(), Self::y1()),
            (Self::x2(), Self::y1()),
            (Self::x2(), Self::y2()),
            (Self::x1(), Self::y2()),
        ]
    }

    /// `n_x·x + n_y·y + d` — the signed distance to a half-plane boundary, where `x` and `y`
    /// are the expressions that pick the point out of the box (a corner pair, or
    /// [`center_x`](Self::center_x)/[`center_y`](Self::center_y)).
    ///
    /// This is the whole of convex polygon containment. Each coordinate is a polynomial in
    /// the frame, so this affine combination is one too, of the same degree — meaning
    /// `≥ 0` on every edge of a convex ring is "inside", expressed entirely in comparisons
    /// the evaluator already solves. Pair with [`Polygon::half_planes`].
    pub fn half_plane(x: &Self, y: &Self, (nx, ny, d): (f64, f64, f64)) -> Self {
        x.clone() * nx + y.clone() * ny + Self::constant(d)
    }

    /// Expands this expression's polynomial for `seg` into a local coordinate
    /// `k = f - origin`, returning `(B, C)` such that `expr(f) = B·k + C`.
    ///
    /// This is the `linear` variant: each coordinate is `coord(t) = b·t + c` in
    /// `t = f - seg.start`. Substituting `t = k + delta` gives `b·k + (b·delta + c)`.
    fn local_poly(&self, seg: &PolySegment<4>, origin: u32) -> Option<(f64, f64)> {
        let delta = origin as f64 - seg.start as f64;
        let expand = |coeffs: [f64; 2]| -> (f64, f64) {
            let (b, c) = (coeffs[0], coeffs[1]);
            (b, b * delta + c)
        };
        let mut gb = 0.0f64;
        let mut gc = self.offset;
        let mut add = |w: f64, data: &ScalarSegmentData2<4>| -> Option<()> {
            let ScalarSegmentData2::Poly(coeffs) = data else {
                return None;
            };
            let (b, c) = expand(*coeffs);
            gb += w * b;
            gc += w * c;
            Some(())
        };
        if self.x1_w != 0.0 {
            add(self.x1_w, &seg.cx1)?;
        }
        if self.y1_w != 0.0 {
            add(self.y1_w, &seg.cy1)?;
        }
        if self.x2_w != 0.0 {
            add(self.x2_w, &seg.cx2)?;
        }
        if self.y2_w != 0.0 {
            add(self.y2_w, &seg.cy2)?;
        }
        Some((gb, gc))
    }

    fn eval_at(&self, seg: &PolySegment<4>, frame: u32) -> f64 {
        let t = frame - seg.start;
        let mut out = self.offset;
        if self.used_coords.0 {
            out += self.x1_w * seg.cx1.eval_local(t);
        }
        if self.used_coords.1 {
            out += self.y1_w * seg.cy1.eval_local(t);
        }
        if self.used_coords.2 {
            out += self.x2_w * seg.cx2.eval_local(t);
        }
        if self.used_coords.3 {
            out += self.y2_w * seg.cy2.eval_local(t);
        }
        out
    }
}

impl std::ops::Add for BBoxExpr {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::from_weights(
            self.x1_w + rhs.x1_w,
            self.y1_w + rhs.y1_w,
            self.x2_w + rhs.x2_w,
            self.y2_w + rhs.y2_w,
            self.offset + rhs.offset,
        )
    }
}

impl std::ops::Sub for BBoxExpr {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::from_weights(
            self.x1_w - rhs.x1_w,
            self.y1_w - rhs.y1_w,
            self.x2_w - rhs.x2_w,
            self.y2_w - rhs.y2_w,
            self.offset - rhs.offset,
        )
    }
}

impl std::ops::Mul<f64> for BBoxExpr {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        Self::from_weights(
            self.x1_w * s,
            self.y1_w * s,
            self.x2_w * s,
            self.y2_w * s,
            self.offset * s,
        )
    }
}

impl std::ops::Neg for BBoxExpr {
    type Output = Self;
    fn neg(self) -> Self {
        self * -1.0
    }
}

// ── Comparison operator ───────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    Gt,
    Gte,
    Lt,
    Lte,
}

// ── Analytical predicate (unified, slots in leaves) ──────────────────────────

/// The variants of an analytical predicate tree.
///
/// Wrapped by [`AnalyticalPred`], which additionally carries a `group` annotation used
/// at evaluation time to look up co-appearance intervals for segment-level clipping.
#[derive(Clone, Debug, PartialEq)]
pub enum AnalyticalPredKind {
    /// `expr(tuple[slot])  op  threshold` — single-object comparison.
    Unary {
        expr: BBoxExpr,
        op: CmpOp,
        threshold: f64,
        slot: Slot,
    },
    /// `lhs_expr(tuple[slots[0]]) − rhs_expr(tuple[slots[1]])  op  threshold`
    Binary {
        expr: BinaryBBoxExpr,
        op: CmpOp,
        threshold: f64,
        slots: [Slot; 2],
    },
    /// `expr1(slot[0], slot[1]) op1 threshold1 AND expr2(slot[1], slot[2]) op2 threshold2`
    /// Evaluates both conditions in a single 3-way window sweep, computing the shared
    /// middle object's windows once and finding all polynomial roots in one merged pass.
    Ternary {
        expr1: BinaryBBoxExpr,
        op1: CmpOp,
        threshold1: f64,
        expr2: BinaryBBoxExpr,
        op2: CmpOp,
        threshold2: f64,
        slots: [Slot; 3],
        /// Precomputed union of expr1.rhs_used and expr2.lhs_used (coords used from middle object).
        b_used: (bool, bool, bool, bool),
    },
    /// `(x(f), y(f)) ∈ poly` — the point picked out of `slot`'s box lies in a **non-convex**
    /// polygon.
    ///
    /// Convex regions never reach here: [`AnalyticalPred::point_in_polygon`] expands those
    /// into a conjunction of half-plane [`Unary`](Self::Unary) comparisons, which every
    /// downstream consumer already understands. This node exists only for rings that have no
    /// conjunctive form, and it is still solved analytically — see [`polygon_intervals`].
    ///
    /// `x` and `y` are coordinate selectors with no constant offset; that invariant is what
    /// lets a coordinate-frame change move the polygon instead of the expressions.
    InPolygon {
        x: BBoxExpr,
        y: BBoxExpr,
        poly: Arc<Polygon>,
        slot: Slot,
    },
    And(Box<AnalyticalPred>, Box<AnalyticalPred>),
    Or(Box<AnalyticalPred>, Box<AnalyticalPred>),
    Not(Box<AnalyticalPred>),
    True,
}

/// A unified analytical predicate tree annotated with its selector group index.
///
/// The `group` field is set at query-planning time (by `AlgebraCtx::annotate_groups`)
/// and indexes into the per-tuple co-appearance interval array so that polynomial
/// evaluation is clipped to the frames where the referenced objects actually co-appear.
///
/// Example: `a.x1 > b.x1 AND b.x1 > c.x1` becomes
/// `And(Binary{slots:[0,1],...}, Binary{slots:[1,2],...})` with `group` pointing to
/// whichever selector group contains all referenced slots.
#[derive(Clone, Debug, PartialEq)]
pub struct AnalyticalPred {
    /// Index into the per-tuple `group_coappears` slice. Set by `annotate_groups` before
    /// evaluation; defaults to `usize::MAX` (no clip).
    pub group: usize,
    pub kind: AnalyticalPredKind,
}

impl AnalyticalPred {
    fn new(kind: AnalyticalPredKind) -> Self {
        Self {
            group: usize::MAX,
            kind,
        }
    }

    pub fn unary_gt(expr: BBoxExpr, threshold: f64, slot: Slot) -> Self {
        Self::new(AnalyticalPredKind::Unary {
            expr,
            op: CmpOp::Gt,
            threshold,
            slot,
        })
    }
    pub fn unary_gte(expr: BBoxExpr, threshold: f64, slot: Slot) -> Self {
        Self::new(AnalyticalPredKind::Unary {
            expr,
            op: CmpOp::Gte,
            threshold,
            slot,
        })
    }
    pub fn unary_lt(expr: BBoxExpr, threshold: f64, slot: Slot) -> Self {
        Self::new(AnalyticalPredKind::Unary {
            expr,
            op: CmpOp::Lt,
            threshold,
            slot,
        })
    }
    pub fn unary_lte(expr: BBoxExpr, threshold: f64, slot: Slot) -> Self {
        Self::new(AnalyticalPredKind::Unary {
            expr,
            op: CmpOp::Lte,
            threshold,
            slot,
        })
    }

    pub fn true_pred() -> Self {
        Self::new(AnalyticalPredKind::True)
    }

    /// `coord(tuple[slots[0]])  op  coord(tuple[slots[1]])` — compare a coordinate of two objects.
    pub fn coord_cmp(lhs: BBoxExpr, op: CmpOp, rhs: BBoxExpr, slots: [Slot; 2]) -> Self {
        Self::new(AnalyticalPredKind::Binary {
            expr: BinaryBBoxExpr::new(lhs, rhs),
            op,
            threshold: 0.0,
            slots,
        })
    }

    /// `(x, y) ∈ poly` for the object in `slot`, where `x`/`y` select the point from its box.
    ///
    /// A **convex** polygon compiles to a conjunction of half-plane comparisons — ordinary
    /// [`Unary`](AnalyticalPredKind::Unary) nodes, so the evaluator, the optimizer, and the
    /// SQL backend need to know nothing about polygons. Only a non-convex ring, which has no
    /// such conjunctive form, becomes an [`InPolygon`](AnalyticalPredKind::InPolygon) node.
    pub fn point_in_polygon(x: BBoxExpr, y: BBoxExpr, poly: Arc<Polygon>, slot: Slot) -> Self {
        if poly.is_convex() {
            return poly
                .half_planes()
                .map(|hp| Self::unary_gte(BBoxExpr::half_plane(&x, &y, hp), 0.0, slot))
                .reduce(Self::and)
                // `Polygon` guarantees three edges, so the fold never comes back empty.
                .unwrap_or_else(Self::true_pred);
        }
        Self::new(AnalyticalPredKind::InPolygon { x, y, poly, slot })
    }

    /// Every point of `slot`'s box lies in `poly` — exact whole-box containment.
    ///
    /// For a **convex** region the four corners settle it: the region is the intersection of
    /// its half-planes, and a half-plane holding all four corners holds their hull.
    ///
    /// A **concave** region needs one condition more, because it can hold all four corners of
    /// a box whose middle crosses a notch. The box is contained exactly when its corners are
    /// inside *and* no edge of the region passes through its interior — an edge that did would
    /// put the box astride the boundary, and a box astride the boundary has points outside.
    ///
    /// That second condition is still affine, which is what makes this worth doing rather
    /// than refusing the query: edge-versus-box disjointness is a separating axis among three
    /// candidates, and on each one the box's extreme corner is selected by the *sign* of the
    /// axis — known when the region is compiled. So the result is a boolean combination of
    /// ordinary coordinate comparisons, plus one `InPolygon` per corner.
    pub fn box_in_polygon(poly: Arc<Polygon>, slot: Slot) -> Self {
        let corners_inside = BBoxExpr::corners()
            .into_iter()
            .map(|(x, y)| Self::point_in_polygon(x, y, poly.clone(), slot))
            .reduce(Self::and)
            // `BBoxExpr::corners` is a fixed four, so the fold never comes back empty.
            .unwrap_or_else(Self::true_pred);

        if poly.is_convex() {
            return corners_inside;
        }

        // Edge clearance first: `And` returns early once a side comes back empty, and an
        // edge's bounding box usually settles the question without solving the corner tests.
        match poly
            .edges()
            .zip(poly.half_planes())
            .map(|(edge, half_plane)| edge_clear_of_box(edge, half_plane, slot))
            .reduce(Self::and)
        {
            Some(edges_clear) => edges_clear.and(corners_inside),
            None => corners_inside,
        }
    }

    pub fn and(self, other: Self) -> Self {
        Self::new(AnalyticalPredKind::And(Box::new(self), Box::new(other)))
    }
    pub fn or(self, other: Self) -> Self {
        Self::new(AnalyticalPredKind::Or(Box::new(self), Box::new(other)))
    }
    pub fn not(self) -> Self {
        Self::new(AnalyticalPredKind::Not(Box::new(self)))
    }
}

// ── Binary predicate (two objects) ───────────────────────────────────────────

/// A linear expression spanning two objects' bounding boxes.
///
/// Evaluates as: `lhs_expr(obj_A) − rhs_expr(obj_B)`.
/// Use with [`BinaryPred::Cmp`] to express e.g. `obj_A.x1 > obj_B.x2`.
#[derive(Clone, Debug, PartialEq)]
pub struct BinaryBBoxExpr {
    pub lhs: BBoxExpr,
    pub rhs: BBoxExpr,
    pub lhs_used: (bool, bool, bool, bool),
    pub rhs_used: (bool, bool, bool, bool),
}

impl BinaryBBoxExpr {
    pub fn new(lhs: BBoxExpr, rhs: BBoxExpr) -> Self {
        let lhs_used = lhs.used_coords();
        let rhs_used = rhs.used_coords();
        Self {
            lhs,
            rhs,
            lhs_used,
            rhs_used,
        }
    }

    /// `lhs_expr(A) − rhs_expr(B)`, evaluated on a pair of overlapping windows.
    // #[allow(dead_code)]
    // fn global_poly(&self, win_a: &CoverageWindow, win_b: &CoverageWindow) -> (f64, f64, f64) {
    //     let (al, bl, cl) = self.lhs.global_poly(win_a);
    //     let (ar, br, cr) = self.rhs.global_poly(win_b);
    //     (al - ar, bl - br, cl - cr)
    // }

    /// `lhs_expr(A) − rhs_expr(B)`, evaluated in local coordinate `k = f - origin`.
    fn local_poly(
        &self,
        seg_a: &PolySegment<4>,
        seg_b: &PolySegment<4>,
        origin: u32,
    ) -> Option<(f64, f64)> {
        let (bl, cl) = self.lhs.local_poly(seg_a, origin)?;
        let (br, cr) = self.rhs.local_poly(seg_b, origin)?;
        Some((bl - br, cl - cr))
    }

    fn eval_at(&self, seg_a: &PolySegment<4>, seg_b: &PolySegment<4>, frame: u32) -> f64 {
        self.lhs.eval_at(seg_a, frame) - self.rhs.eval_at(seg_b, frame)
    }
}

// ── Compilation / evaluation ──────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct SegmentSpan<'a> {
    start_frame: u32,
    end_frame: u32,
    segment: &'a PolySegment<4>,
}

/// Intersects segment coverage with `clip` on the fly without allocating.
struct SegmentSpanIter<'s> {
    segments: &'s [PolySegment<4>],
    clip: Interval,
    seg_idx: usize,
}

impl<'s> SegmentSpanIter<'s> {
    fn new(segments: &'s [PolySegment<4>], clip: Interval) -> Self {
        // Seek to the first segment the clip can touch rather than walking to
        // it one segment at a time: a clip is usually a co-appearance window
        // opening partway through the trajectory, and at `max_error_px = 0`
        // there are ~108 segments per object. The `first()` check short-circuits
        // the search when the answer is 0, which is the common case.
        let seg_idx = match segments.first() {
            Some(first) if first.end >= clip.0 => 0,
            _ => segments.partition_point(|s| s.end < clip.0),
        };
        Self {
            segments,
            clip,
            seg_idx,
        }
    }
}

impl<'s> Iterator for SegmentSpanIter<'s> {
    type Item = SegmentSpan<'s>;

    fn next(&mut self) -> Option<Self::Item> {
        let (clip_lo, clip_hi) = self.clip;

        while let Some(seg) = self.segments.get(self.seg_idx) {
            self.seg_idx += 1;
            if clip_hi < seg.start {
                return None;
            }
            if clip_lo > seg.end {
                continue;
            }

            let lo = seg.start.max(clip_lo);
            let hi = seg.end.min(clip_hi);
            return Some(SegmentSpan {
                start_frame: lo,
                end_frame: hi,
                segment: seg,
            });
        }

        None
    }
}

/// Analytically evaluates `pred` on a tuple of object IDs against the trajectory array,
/// returning merged frame intervals where the predicate holds. No per-frame sampling.
///
/// `tuple[slot]` gives the object ID for each slot; `trajs[oid]` gives its trajectory.

/// Evaluates `pred` over `tuple` using polynomial trajectories `trajs`.
///
/// `clip` restricts evaluation to a frame interval (e.g. one group co-appearance
/// window). Polynomials are only solved within the intersection of their segment
/// coverage and the clip domain, avoiding work outside valid co-presence windows.
pub fn eval_analytical<'a>(
    pred: &AnalyticalPred,
    tuple: &[u32],
    trajs: &'a [PolyTraj],
    clip: Interval,
) -> Vec<Interval> {
    match &pred.kind {
        AnalyticalPredKind::True => vec![clip],
        AnalyticalPredKind::Unary {
            expr,
            op,
            threshold,
            slot,
        } => {
            let _g = trql_instrument::scope(trql_instrument::Bucket::SegmentIter);
            let oid = tuple[*slot] as usize;
            let segments = trajs[oid].segments();
            let mut result = Vec::new();
            for span in SegmentSpanIter::new(segments, clip) {
                let lo = span.start_frame;
                let hi = span.end_frame;
                if let Some((b, c)) = expr.local_poly(span.segment, lo) {
                    result.extend(
                        poly_cmp_intervals(b, c, *op, *threshold, 0, hi - lo)
                            .into_iter()
                            .map(|(s, e)| (s + lo, e + lo)),
                    );
                } else {
                    result.extend(scan_cmp_intervals(lo, hi, |f| {
                        cmp_value(expr.eval_at(span.segment, f), *op, *threshold)
                    }));
                }
            }
            result
        }
        AnalyticalPredKind::Binary {
            expr,
            op,
            threshold,
            slots: [s0, s1],
        } => {
            let _g = trql_instrument::scope(trql_instrument::Bucket::SegmentIter);
            let lhs_oid = tuple[*s0] as usize;
            let rhs_oid = tuple[*s1] as usize;
            let lhs_segments = trajs[lhs_oid].segments();
            let rhs_segments = trajs[rhs_oid].segments();
            let eval = |sa: &SegmentSpan<'_>, sb: &SegmentSpan<'_>, lo: u32, hi: u32| {
                if let Some((b, c)) = expr.local_poly(sa.segment, sb.segment, lo) {
                    poly_cmp_intervals(b, c, *op, *threshold, 0, hi - lo)
                        .into_iter()
                        .map(|(s, e)| (s + lo, e + lo))
                        .collect()
                } else {
                    scan_cmp_intervals(lo, hi, |f| {
                        cmp_value(expr.eval_at(sa.segment, sb.segment, f), *op, *threshold)
                    })
                }
            };
            sweep_segments(
                SegmentSpanIter::new(lhs_segments, clip),
                SegmentSpanIter::new(rhs_segments, clip),
                eval,
            )
        }
        AnalyticalPredKind::Ternary {
            expr1,
            op1,
            threshold1,
            expr2,
            op2,
            threshold2,
            slots,
            b_used,
        } => {
            let _g = trql_instrument::scope(trql_instrument::Bucket::SegmentIter);
            const EPS: f64 = 1e-9;
            const STRICT_EPS: f64 = 1e-6;
            // Normalize (b,c) and threshold to "poly > 0" form, matching poly_cmp_intervals.
            let normalize = |op: &CmpOp, t: f64, b: f64, c: f64| match op {
                CmpOp::Gt => (b, c - t - STRICT_EPS),
                CmpOp::Gte => (b, c - t + EPS),
                CmpOp::Lt => (-b, -c + t - STRICT_EPS),
                CmpOp::Lte => (-b, -c + t + EPS),
            };
            let a_oid = tuple[slots[0]] as usize;
            let b_oid = tuple[slots[1]] as usize;
            let c_oid = tuple[slots[2]] as usize;
            let _ = b_used;
            let a_segments = trajs[a_oid].segments();
            let b_segments = trajs[b_oid].segments();
            let c_segments = trajs[c_oid].segments();
            let (op1, threshold1, op2, threshold2) = (op1, *threshold1, op2, *threshold2);
            let expr1 = expr1;
            let expr2 = expr2;
            let eval = |sa: &SegmentSpan<'_>,
                        sb: &SegmentSpan<'_>,
                        sc: &SegmentSpan<'_>,
                        lo: u32,
                        hi: u32| {
                if let (Some((rb1, rc1)), Some((rb2, rc2))) = (
                    expr1.local_poly(sa.segment, sb.segment, lo),
                    expr2.local_poly(sb.segment, sc.segment, lo),
                ) {
                    let (b1, c1) = normalize(op1, threshold1, rb1, rc1);
                    let (b2, c2) = normalize(op2, threshold2, rb2, rc2);
                    both_positive_integer_intervals(b1, c1, b2, c2, 0, hi - lo)
                        .into_iter()
                        .map(|(s, e)| (s + lo, e + lo))
                        .collect()
                } else {
                    scan_cmp_intervals(lo, hi, |f| {
                        cmp_value(expr1.eval_at(sa.segment, sb.segment, f), *op1, threshold1)
                            && cmp_value(expr2.eval_at(sb.segment, sc.segment, f), *op2, threshold2)
                    })
                }
            };
            sweep_segments_3(
                SegmentSpanIter::new(a_segments, clip),
                SegmentSpanIter::new(b_segments, clip),
                SegmentSpanIter::new(c_segments, clip),
                eval,
            )
        }
        AnalyticalPredKind::InPolygon { x, y, poly, slot } => {
            let _g = trql_instrument::scope(trql_instrument::Bucket::SegmentIter);
            let oid = tuple[*slot] as usize;
            let segments = trajs[oid].segments();
            let mut result = Vec::new();
            for span in SegmentSpanIter::new(segments, clip) {
                let (lo, hi) = (span.start_frame, span.end_frame);
                // Both coordinates must expand about the *same* origin, or the two
                // polynomials describe points at different frames.
                match (
                    x.local_poly(span.segment, lo),
                    y.local_poly(span.segment, lo),
                ) {
                    (Some(px), Some(py)) => result.extend(
                        polygon_intervals(px, py, poly, 0, hi - lo)
                            .into_iter()
                            .map(|(s, e)| (s + lo, e + lo)),
                    ),
                    // A segment stored as sampled values rather than coefficients has no
                    // polynomial to solve; fall back to scanning, as every other arm does.
                    _ => result.extend(scan_cmp_intervals(lo, hi, |f| {
                        poly.contains(x.eval_at(span.segment, f), y.eval_at(span.segment, f))
                    })),
                }
            }
            result
        }
        AnalyticalPredKind::And(l, r) => {
            let lhs = eval_analytical(l, tuple, trajs, clip);
            if lhs.is_empty() {
                return lhs;
            }
            let rhs: Vec<Interval> = lhs
                .iter()
                .flat_map(|&clip| eval_analytical(r, tuple, trajs, clip))
                .collect();
            trql_types::intervals::intersect(&lhs, &rhs)
        }
        AnalyticalPredKind::Or(l, r) => {
            let lhs = eval_analytical(l, tuple, trajs, clip);
            let rhs = eval_analytical(r, tuple, trajs, clip);
            trql_types::intervals::union(lhs, rhs)
        }
        AnalyticalPredKind::Not(inner) => {
            let domain = analytical_domain(inner, tuple, trajs);
            let effective_domain = intersect_with_clip(&domain, clip);
            let inner_eval = eval_analytical(inner, tuple, trajs, clip);
            trql_types::intervals::complement_in_domain(&inner_eval, &effective_domain)
        }
    }
}

/// Computes the co-presence domain (intersection of trajectory coverages) for all
/// slots referenced within `pred`. Used as the complement domain for `Not`.
fn analytical_domain(pred: &AnalyticalPred, tuple: &[u32], trajs: &[PolyTraj]) -> Vec<Interval> {
    let mut slots: Vec<Slot> = Vec::new();
    collect_pred_slots(pred, &mut slots);
    slots.sort_unstable();
    slots.dedup();

    if slots.is_empty() {
        return vec![(0, u32::MAX)];
    }

    let mut domain: Vec<Interval> = trajs[tuple[slots[0]] as usize]
        .segments()
        .iter()
        .map(|s| (s.start, s.end))
        .collect();
    for &slot in &slots[1..] {
        let cov: Vec<Interval> = trajs[tuple[slot] as usize]
            .segments()
            .iter()
            .map(|s| (s.start, s.end))
            .collect();
        domain = trql_types::intervals::intersect(&domain, &cov);
    }
    domain
}

fn intersect_with_clip(intervals: &[Interval], clip: Interval) -> Vec<Interval> {
    let (clip_lo, clip_hi) = clip;
    intervals
        .iter()
        .filter_map(|&(lo, hi)| {
            let lo = lo.max(clip_lo);
            let hi = hi.min(clip_hi);
            (lo <= hi).then_some((lo, hi))
        })
        .collect()
}

pub fn collect_pred_slots(pred: &AnalyticalPred, out: &mut Vec<Slot>) {
    match &pred.kind {
        AnalyticalPredKind::Unary { slot, .. } => out.push(*slot),
        AnalyticalPredKind::InPolygon { slot, .. } => out.push(*slot),
        AnalyticalPredKind::Binary { slots, .. } => out.extend_from_slice(slots),
        AnalyticalPredKind::Ternary { slots, .. } => out.extend_from_slice(slots),
        AnalyticalPredKind::And(l, r) | AnalyticalPredKind::Or(l, r) => {
            collect_pred_slots(l, out);
            collect_pred_slots(r, out);
        }
        AnalyticalPredKind::Not(inner) => collect_pred_slots(inner, out),
        AnalyticalPredKind::True => {}
    }
}

/// `slot`'s box does not meet the interior of the polygon edge `p → q`.
///
/// Separating-axis test: a segment and a box are disjoint exactly when their projections
/// fail to overlap on one of the box's two axes or on the edge's own normal. Checking the
/// edge's *line* on that third axis rather than the segment is what keeps every disjunct
/// affine — and it only ever reports "not clear" more often, which is the safe direction for
/// a containment test.
///
/// The comparisons are non-strict because it is the box's *open* interior that has to stay
/// clear: an edge lying flat along the box's border leaves no part of the box outside the
/// region, so a rectangle written as a polygon still contains itself.
fn edge_clear_of_box(
    (p, q): ([f64; 2], [f64; 2]),
    (nx, ny, d): (f64, f64, f64),
    slot: Slot,
) -> AnalyticalPred {
    // `d` is `−n·p`, so these read as "the extreme corner is on the interior/exterior side".
    let nearest = weighted(corner_extreme_weights(nx, ny, false), d);
    let farthest = weighted(corner_extreme_weights(nx, ny, true), d);

    AnalyticalPred::unary_gte(BBoxExpr::x1(), p[0].max(q[0]), slot)
        .or(AnalyticalPred::unary_lte(
            BBoxExpr::x2(),
            p[0].min(q[0]),
            slot,
        ))
        .or(AnalyticalPred::unary_gte(
            BBoxExpr::y1(),
            p[1].max(q[1]),
            slot,
        ))
        .or(AnalyticalPred::unary_lte(
            BBoxExpr::y2(),
            p[1].min(q[1]),
            slot,
        ))
        // Whole box on the interior side of the edge's line, or whole box on the exterior
        // side: either way the line misses the interior, so the edge on it does too.
        .or(AnalyticalPred::unary_gte(nearest, 0.0, slot))
        .or(AnalyticalPred::unary_lte(farthest, 0.0, slot))
}

/// Builds `x1_w·x1 + y1_w·y1 + x2_w·x2 + y2_w·y2 + offset` from the public constructors, so
/// whatever bookkeeping `BBoxExpr` keeps alongside its weights stays consistent.
fn weighted([x1_w, y1_w, x2_w, y2_w]: [f64; 4], offset: f64) -> BBoxExpr {
    BBoxExpr::x1() * x1_w
        + BBoxExpr::y1() * y1_w
        + BBoxExpr::x2() * x2_w
        + BBoxExpr::y2() * y2_w
        + BBoxExpr::constant(offset)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Finds frame intervals where `B·f + C  op  threshold`.
fn poly_cmp_intervals(
    b: f64,
    c: f64,
    op: CmpOp,
    threshold: f64,
    lo: u32,
    hi: u32,
) -> Vec<Interval> {
    // Reduce all cases to "polynomial > 0" by algebraic rearrangement.
    // For non-strict comparisons we add a small epsilon so that boundary frames
    // where the polynomial is exactly zero are included (matching integer semantics).
    const EPS: f64 = 1e-9;
    // For strict inequalities, require a tiny margin away from equality to
    // suppress near-zero floating-point jitter at polynomial boundaries.
    const STRICT_EPS: f64 = 1e-6;
    match op {
        CmpOp::Gt => positive_integer_intervals(b, c - threshold - STRICT_EPS, lo, hi),
        CmpOp::Gte => positive_integer_intervals(b, c - threshold + EPS, lo, hi),
        CmpOp::Lt => positive_integer_intervals(-b, -c + threshold - STRICT_EPS, lo, hi),
        CmpOp::Lte => positive_integer_intervals(-b, -c + threshold + EPS, lo, hi),
    }
}

/// Finds the frame ranges in `[lo, hi]` where the parametric point `(x(k), y(k))` lies
/// inside `poly`, where each coordinate is linear in the segment-local frame `k`.
///
/// The parametric form is what makes this exact and cheap. Substituting `(x(k), y(k))` into
/// an edge's implicit line `n·(P − p) = 0` yields one more *polynomial in k* of the same
/// degree — there is no per-edge parameter to solve for and no distinction between an edge
/// and the line carrying it. Membership can only flip where the curve crosses one of those
/// lines, so the roots of all of them are a superset of the flip frames, and between
/// consecutive breaks the answer is constant. One [`Polygon::contains`] test per sub-range
/// therefore settles it — the trick [`both_positive_integer_intervals`] uses for two
/// polynomials, generalized to one per edge.
///
/// Using the edge *lines* rather than the segments is deliberate: a break where the curve
/// crosses a line outside the polygon costs one extra containment test and changes no answer,
/// which is a better trade than clamping every root to its edge's extent.
///
/// Sampling at each sub-range's *start* rather than its midpoint matches
/// [`positive_integer_intervals`], which is what keeps boundary frames — where the curve sits
/// exactly on an edge — agreeing with every other predicate in the compiler.
fn polygon_intervals(
    (xb, xc): (f64, f64),
    (yb, yc): (f64, f64),
    poly: &Polygon,
    lo: u32,
    hi: u32,
) -> Vec<Interval> {
    if lo > hi {
        return vec![];
    }

    let mut breaks: Vec<u32> = vec![lo, hi.saturating_add(1)];
    for (nx, ny, d) in poly.half_planes() {
        push_sign_change_breaks(
            [nx * xb + ny * yb, nx * xc + ny * yc + d],
            lo,
            hi,
            &mut breaks,
        );
    }
    breaks.sort_unstable();
    breaks.dedup();

    let mut out: Vec<Interval> = Vec::new();
    for w in breaks.windows(2) {
        let (start, end) = (w[0], w[1] - 1);
        if start > end {
            continue;
        }
        let k = start as f64;
        if !poly.contains(xb * k + xc, yb * k + yc) {
            continue;
        }
        // Adjacent breaks are common — most edges contribute roots the curve never
        // actually crosses — so stitch the runs back together rather than emitting a
        // fragmented interval list for what is one continuous visit.
        match out.last_mut() {
            Some(last) if last.1 + 1 == start => last.1 = end,
            _ => out.push((start, end)),
        }
    }
    out
}

#[inline]
fn cmp_value(value: f64, op: CmpOp, threshold: f64) -> bool {
    const EPS: f64 = 1e-9;
    const STRICT_EPS: f64 = 1e-6;
    match op {
        CmpOp::Gt => value > threshold + STRICT_EPS,
        CmpOp::Gte => value >= threshold - EPS,
        CmpOp::Lt => value < threshold - STRICT_EPS,
        CmpOp::Lte => value <= threshold + EPS,
    }
}

fn scan_cmp_intervals<F>(lo: u32, hi: u32, mut pred: F) -> Vec<Interval>
where
    F: FnMut(u32) -> bool,
{
    if lo > hi {
        return vec![];
    }
    let mut out = Vec::new();
    let mut current_start: Option<u32> = None;
    for f in lo..=hi {
        if pred(f) {
            if current_start.is_none() {
                current_start = Some(f);
            }
        } else if let Some(s) = current_start.take() {
            out.push((s, f - 1));
        }
    }
    if let Some(s) = current_start {
        out.push((s, hi));
    }
    out
}

/// Sweep overlapping segment spans from two trajectories and collect intervals produced by `f`.
fn sweep_segments<'a, 'b, IL, IR, F>(mut lhs_wins: IL, mut rhs_wins: IR, mut f: F) -> Vec<Interval>
where
    IL: Iterator<Item = SegmentSpan<'a>>,
    IR: Iterator<Item = SegmentSpan<'b>>,
    F: FnMut(&SegmentSpan<'a>, &SegmentSpan<'b>, u32, u32) -> Vec<Interval>,
{
    let mut result = Vec::new();
    let mut lhs_next = lhs_wins.next();
    let mut rhs_next = rhs_wins.next();

    while let (Some(wa), Some(wb)) = (&lhs_next, &rhs_next) {
        let lo = wa.start_frame.max(wb.start_frame);
        let hi = wa.end_frame.min(wb.end_frame);
        if lo <= hi {
            result.extend(f(wa, wb, lo, hi));
        }

        let wa_end = wa.end_frame;
        let wb_end = wb.end_frame;
        if wa_end <= wb_end {
            lhs_next = lhs_wins.next();
        }
        if wb_end <= wa_end {
            rhs_next = rhs_wins.next();
        }
    }
    result
}

fn sweep_segments_3<'a, 'b, 'c, IA, IB, IC, F>(
    mut aw: IA,
    mut bw: IB,
    mut cw: IC,
    mut f: F,
) -> Vec<Interval>
where
    IA: Iterator<Item = SegmentSpan<'a>>,
    IB: Iterator<Item = SegmentSpan<'b>>,
    IC: Iterator<Item = SegmentSpan<'c>>,
    F: FnMut(&SegmentSpan<'a>, &SegmentSpan<'b>, &SegmentSpan<'c>, u32, u32) -> Vec<Interval>,
{
    let mut result = Vec::new();
    let mut a_next = aw.next();
    let mut b_next = bw.next();
    let mut c_next = cw.next();

    while let (Some(wa), Some(wb), Some(wc)) = (&a_next, &b_next, &c_next) {
        let lo = wa.start_frame.max(wb.start_frame).max(wc.start_frame);
        let hi = wa.end_frame.min(wb.end_frame).min(wc.end_frame);
        if lo <= hi {
            result.extend(f(wa, wb, wc, lo, hi));
        }

        let end = wa.end_frame.min(wb.end_frame).min(wc.end_frame);
        if wa.end_frame == end {
            a_next = aw.next();
        }
        if wb.end_frame == end {
            b_next = bw.next();
        }
        if wc.end_frame == end {
            c_next = cw.next();
        }
    }
    result
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    /// Whole-box containment, the shape `trql-dsl` desugars `INSIDE` into. Kept here so
    /// the evaluator still has coverage for that conjunction after the DSL took over
    /// building it.
    fn inside(lx1: f64, ly1: f64, lx2: f64, ly2: f64, slot: Slot) -> AnalyticalPred {
        AnalyticalPred::unary_gte(BBoxExpr::x1(), lx1, slot)
            .and(AnalyticalPred::unary_gte(BBoxExpr::y1(), ly1, slot))
            .and(AnalyticalPred::unary_lte(BBoxExpr::x2(), lx2, slot))
            .and(AnalyticalPred::unary_lte(BBoxExpr::y2(), ly2, slot))
    }

    /// The objects these tests are written against: `a`, `b`, `c` in tuple order.
    fn a() -> Slot {
        0
    }
    fn b() -> Slot {
        1
    }
    fn c() -> Slot {
        2
    }
    use super::*;

    // fn traj_const(start: u32, end: u32, x1: f64, y1: f64, x2: f64, y2: f64) -> PolyTraj {
    //     // Constant bbox over [start, end]: all coefficients are [0, 0, coord].
    //     PolyTraj {
    //         x1: vec![scalar_seg(start, end, 0.0, 0.0, x1)],
    //         y1: vec![scalar_seg(start, end, 0.0, 0.0, y1)],
    //         x2: vec![scalar_seg(start, end, 0.0, 0.0, x2)],
    //         y2: vec![scalar_seg(start, end, 0.0, 0.0, y2)],
    //         appear_intervals: vec![],
    //     }
    // }

    // fn traj_linear_x1(start: u32, end: u32, x1_start: f64, x1_end: f64) -> PolyTraj {
    //     // x1(f) = x1_start + (x1_end - x1_start) / (end - start) * (f - start)
    //     // as local polynomial [0, slope, x1_start] with offset = start
    //     let slope = (x1_end - x1_start) / (end - start) as f64;
    //     // ScalarSegment stores local coefficients [a,b,c] where t = f - start_frame.
    //     // So [0, slope, x1_start] means x1(t) = slope·t + x1_start, and
    //     // global_poly() will expand this to global frame coordinates correctly.
    //     PolyTraj {
    //         x1: vec![scalar_seg(start, end, 0.0, slope, x1_start)],
    //         y1: vec![scalar_seg(start, end, 0.0, 0.0, 0.0)],
    //         x2: vec![scalar_seg(start, end, 0.0, 0.0, x1_start + 10.0)],
    //         y2: vec![scalar_seg(start, end, 0.0, 0.0, 10.0)],
    //         appear_intervals: vec![],
    //     }
    // }

    // use trql_types::obj::BoundingBox;
    // use trql_types::obj::ExactTraj;
    // fn bbox(x1: u32, y1: u32, x2: u32, y2: u32) -> BoundingBox {
    //     BoundingBox { x1, y1, x2, y2 }
    // }
    // #[test]
    // fn from_traj_with_bboxes_fid_x1() {
    //     let csv = include_str!("../../baselines/duckdb_eval/src/clevrer/database/fid_x1_zero.csv");
    //     let mut frames = Vec::new();
    //     let mut bboxes = Vec::new();
    //     for line in csv.lines().skip(1) {
    //         let mut parts = line.splitn(2, ',');
    //         let fid: u32 = parts.next().unwrap().trim().parse::<u32>().unwrap() - 28558u32;
    //         let x1: u32 = parts.next().unwrap().trim().parse::<f64>().unwrap() as u32;
    //         frames.push(fid);
    //         bboxes.push(bbox(x1, 0, 0, 0));
    //     }
    //     let traj = ExactTraj { frames, bboxes };
    //     let poly = PolyTraj::from_traj_with_bboxes(&traj, 1, 0.0);

    //     let pred = UnaryPred::gt(BBoxExpr::x1(), 0.0);
    //     let intervals = eval_unary(&pred, &poly);
    //     let num_frames: u32 = intervals.iter().map(|(s, e)| e - s + 1).sum();
    //     println!(
    //         "x1 > 0: {} frames across {} intervals",
    //         num_frames,
    //         intervals.len()
    //     );
    // }

    // ── BBoxExpr ──────────────────────────────────────────────────────────────

    #[test]
    fn bbox_expr_add_sub() {
        let w = BBoxExpr::width();
        let h = BBoxExpr::height();
        let sum = w.clone() + h.clone();
        assert_eq!(sum.x1_w, -1.0);
        assert_eq!(sum.y1_w, -1.0);
        assert_eq!(sum.x2_w, 1.0);
        assert_eq!(sum.y2_w, 1.0);

        let diff = w - h;
        assert_eq!(diff.x1_w, -1.0);
        assert_eq!(diff.y1_w, 1.0);
        assert_eq!(diff.x2_w, 1.0);
        assert_eq!(diff.y2_w, -1.0);
    }

    #[test]
    fn bbox_expr_scale() {
        let cx = BBoxExpr::center_x() * 2.0;
        assert_eq!(cx.x1_w, 1.0);
        assert_eq!(cx.x2_w, 1.0);
    }

    // ── AnalyticalPred — constant trajectories ───────────────────────────────

    // Helper: evaluate a unary pred on a single traj at slot 0.
    fn eval1(pred: &AnalyticalPred, traj: PolyTraj) -> Vec<Interval> {
        eval_analytical(pred, &[0], &[traj], (0, u32::MAX))
    }
    // Helper: evaluate a binary pred on two trajs at slots [0,1].
    fn eval2(pred: &AnalyticalPred, a: PolyTraj, b: PolyTraj) -> Vec<Interval> {
        eval_analytical(pred, &[0, 1], &[a, b], (0, u32::MAX))
    }

    #[test]
    fn unary_gt_all_frames() {
        let traj = traj_const(0, 10, 100.0, 0.0, 200.0, 50.0);
        let pred = AnalyticalPred::unary_gt(BBoxExpr::x1(), 50.0, a());
        assert_eq!(eval1(&pred, traj), vec![(0, 10)]);
    }

    #[test]
    fn unary_gt_no_frames() {
        let traj = traj_const(0, 10, 10.0, 0.0, 20.0, 5.0);
        let pred = AnalyticalPred::unary_gt(BBoxExpr::x1(), 50.0, a());
        assert!(eval1(&pred, traj).is_empty());
    }

    #[test]
    fn unary_width_gt() {
        let traj = traj_const(0, 5, 100.0, 0.0, 200.0, 50.0);
        let pred = AnalyticalPred::unary_gt(BBoxExpr::width(), 80.0, a());
        assert_eq!(eval1(&pred, traj), vec![(0, 5)]);
    }

    #[test]
    fn unary_width_not_gt() {
        let traj = traj_const(0, 5, 10.0, 0.0, 30.0, 20.0);
        let pred = AnalyticalPred::unary_gt(BBoxExpr::width(), 50.0, a());
        assert!(eval1(&pred, traj).is_empty());
    }

    #[test]
    fn unary_inside() {
        let traj = traj_const(0, 5, 50.0, 50.0, 150.0, 150.0);
        let pred = inside(0.0, 0.0, 200.0, 200.0, a());
        assert_eq!(eval1(&pred, traj), vec![(0, 5)]);
    }

    #[test]
    fn unary_inside_partially_outside() {
        let traj = traj_const(0, 5, 50.0, 50.0, 250.0, 150.0);
        let pred = inside(0.0, 0.0, 200.0, 200.0, a());
        assert!(eval1(&pred, traj).is_empty());
    }

    #[test]
    fn unary_and() {
        let traj = traj_const(0, 10, 100.0, 0.0, 200.0, 50.0);
        let pred = AnalyticalPred::unary_gt(BBoxExpr::x1(), 50.0, a())
            .and(AnalyticalPred::unary_gt(BBoxExpr::width(), 80.0, a()));
        assert_eq!(eval1(&pred, traj), vec![(0, 10)]);
    }

    #[test]
    fn unary_or_one_side_false() {
        let traj = traj_const(0, 10, 100.0, 0.0, 200.0, 50.0);
        let pred = AnalyticalPred::unary_gt(BBoxExpr::x1(), 150.0, a())
            .or(AnalyticalPred::unary_gt(BBoxExpr::x1(), 50.0, a()));
        assert_eq!(eval1(&pred, traj), vec![(0, 10)]);
    }

    #[test]
    fn unary_not() {
        let traj = traj_const(0, 10, 100.0, 0.0, 200.0, 50.0);
        // NOT(x1 > 150) — x1=100 so always false inside, complement = whole segment
        let pred = AnalyticalPred::unary_gt(BBoxExpr::x1(), 150.0, a()).not();
        assert_eq!(eval1(&pred, traj), vec![(0, 10)]);
    }

    #[test]
    fn unary_linear_moving_x1() {
        let traj = traj_linear_x1(0, 10, 0.0, 100.0);
        let pred = AnalyticalPred::unary_gt(BBoxExpr::x1(), 50.0, a());
        let ivs = eval1(&pred, traj);
        assert!(!ivs.is_empty());
        for &(lo, hi) in &ivs {
            assert!(lo >= 6, "lo={lo}");
            assert!(hi <= 10, "hi={hi}");
        }
    }

    // ── Binary ────────────────────────────────────────────────────────────────

    #[test]
    fn binary_true_is_co_presence() {
        let a = traj_const(0, 10, 0.0, 0.0, 10.0, 10.0);
        let b = traj_const(5, 15, 50.0, 0.0, 60.0, 10.0);
        // True intersected with co-presence: expect (5,10)
        let pred = AnalyticalPred::true_pred();
        let ivs = eval_analytical(&pred, &[0, 1], &[a, b], (0, u32::MAX));
        // True returns (0, MAX); we only assert both trajectories exist in overlapping region
        // The actual co-presence clipping happens at a higher level; just check not empty here.
        assert!(!ivs.is_empty());
    }

    #[test]
    fn binary_x1_greater() {
        let a = traj_const(0, 5, 100.0, 0.0, 110.0, 10.0);
        let b = traj_const(0, 5, 50.0, 0.0, 60.0, 10.0);
        let pred = AnalyticalPred::coord_cmp(BBoxExpr::x1(), CmpOp::Gt, BBoxExpr::x1(), [a(), b()]);
        assert_eq!(eval2(&pred, a, b), vec![(0, 5)]);
    }

    #[test]
    fn binary_x1_not_greater_when_reversed() {
        let a = traj_const(0, 5, 50.0, 0.0, 60.0, 10.0);
        let b = traj_const(0, 5, 100.0, 0.0, 110.0, 10.0);
        let pred = AnalyticalPred::coord_cmp(BBoxExpr::x1(), CmpOp::Gt, BBoxExpr::x1(), [a(), b()]);
        assert!(eval2(&pred, a, b).is_empty());
    }

    #[test]
    fn binary_x2_of_a_lt_x1_of_b_separation() {
        let a = traj_const(0, 5, 0.0, 0.0, 60.0, 10.0);
        let b = traj_const(0, 5, 100.0, 0.0, 160.0, 10.0);
        let pred = AnalyticalPred {
            group: usize::MAX,
            kind: AnalyticalPredKind::Binary {
                expr: BinaryBBoxExpr::new(BBoxExpr::x2(), BBoxExpr::x1()),
                op: CmpOp::Lt,
                threshold: 0.0,
                slots: [a(), b()],
            },
        };
        assert_eq!(eval2(&pred, a, b), vec![(0, 5)]);
    }

    #[test]
    fn binary_and() {
        let a = traj_const(0, 5, 100.0, 0.0, 200.0, 10.0);
        let b = traj_const(0, 5, 50.0, 0.0, 60.0, 10.0);
        let p1 = AnalyticalPred::coord_cmp(BBoxExpr::x1(), CmpOp::Gt, BBoxExpr::x1(), [a(), b()]);
        let p2 = AnalyticalPred::coord_cmp(BBoxExpr::x2(), CmpOp::Gt, BBoxExpr::x2(), [a(), b()]);
        assert_eq!(eval2(&p1.and(p2), a, b), vec![(0, 5)]);
    }

    #[test]
    fn cross_slot_and() {
        // a.x1 > b.x1 AND b.x1 > c.x1 — slots [0,1] and [1,2]
        let a = traj_const(0, 5, 300.0, 0.0, 310.0, 10.0); // x1=300
        let b = traj_const(0, 5, 200.0, 0.0, 210.0, 10.0); // x1=200
        let c = traj_const(0, 5, 100.0, 0.0, 110.0, 10.0); // x1=100
        let p_ab = AnalyticalPred::coord_cmp(BBoxExpr::x1(), CmpOp::Gt, BBoxExpr::x1(), [a(), b()]);
        let p_bc = AnalyticalPred::coord_cmp(BBoxExpr::x1(), CmpOp::Gt, BBoxExpr::x1(), [b(), c()]);
        let pred = p_ab.and(p_bc);
        let ivs = eval_analytical(&pred, &[0, 1, 2], &[a, b, c], (0, u32::MAX));
        assert_eq!(ivs, vec![(0, 5)]);
    }

    // ── Interval helpers ──────────────────────────────────────────────────────

    // #[test]
    // fn test_xyz() {
    //     let poly1 = PolyTraj {
    //         x1: vec![scalar_seg(313598, 313599, 0.0, 0.0, 199.0)],
    //         y1: vec![scalar_seg(313598, 313599, 0.0, 0.0, 70.0)],
    //         x2: vec![scalar_seg(313598, 313599, 0.0, 0.0, 235.0)],
    //         y2: vec![scalar_seg(313598, 313599, 0.0, -1.0, 114.0)],
    //         appear_intervals: vec![],
    //     };
    //     let poly2 = PolyTraj {
    //         x1: vec![scalar_seg(
    //             313597,
    //             313599,
    //             -0.49999999999986627,
    //             0.49999999999970618,
    //             235.00000000000009,
    //         )],
    //         y1: vec![scalar_seg(
    //             313597,
    //             313599,
    //             0.4999999999999638,
    //             -0.4999999999999286,
    //             70.0,
    //         )],
    //         x2: vec![scalar_seg(
    //             313597,
    //             313599,
    //             -1.8724890918854404E-13,
    //             -1.9999999999996301,
    //             274.99999999999994,
    //         )],
    //         y2: vec![scalar_seg(
    //             313597,
    //             313599,
    //             0.4999999999999437,
    //             -0.4999999999998935,
    //             114.0,
    //         )],
    //         appear_intervals: vec![],
    //     };
    //     let pred = AnalyticalPred {
    //         group: usize::MAX,
    //         kind: AnalyticalPredKind::Binary {
    //             expr: BinaryBBoxExpr::new(
    //                 BBoxExpr::y1() + BBoxExpr::y2(),
    //                 BBoxExpr::y1() + BBoxExpr::y2(),
    //             ),
    //             op: CmpOp::Lt,
    //             threshold: 0.0,
    //             slots: [a(), b()],
    //         },
    //     };
    //     let out = eval_analytical(&pred, &[0, 1], &[poly1, poly2], (0, u32::MAX));
    //     println!("{:?}", out);
    //     assert!(!out.is_empty());
    // }

    #[test]
    fn ternary_matches_and_of_two_binaries() {
        // a.x1=300, b.x1=200, c.x1=100: a > b > c always, expect full range
        let a = traj_const(0, 10, 300.0, 0.0, 400.0, 50.0);
        let b = traj_const(0, 10, 200.0, 0.0, 300.0, 50.0);
        let c = traj_const(0, 10, 100.0, 0.0, 200.0, 50.0);

        let and_pred =
            AnalyticalPred::coord_cmp(BBoxExpr::x1(), CmpOp::Gt, BBoxExpr::x1(), [a(), b()]).and(
                AnalyticalPred::coord_cmp(BBoxExpr::x1(), CmpOp::Gt, BBoxExpr::x1(), [b(), c()]),
            );
        let and_result = eval_analytical(
            &and_pred,
            &[0, 1, 2],
            &[a.clone(), b.clone(), c.clone()],
            (0, u32::MAX),
        );

        // Ternary variant (what the optimizer produces)
        let b_used = (true, false, false, false);
        let ternary_pred = AnalyticalPred {
            group: usize::MAX,
            kind: AnalyticalPredKind::Ternary {
                expr1: BinaryBBoxExpr::new(BBoxExpr::x1(), BBoxExpr::x1()),
                op1: CmpOp::Gt,
                threshold1: 0.0,
                expr2: BinaryBBoxExpr::new(BBoxExpr::x1(), BBoxExpr::x1()),
                op2: CmpOp::Gt,
                threshold2: 0.0,
                slots: [a(), b(), c()],
                b_used,
            },
        };
        let ternary_result = eval_analytical(&ternary_pred, &[0, 1, 2], &[a, b, c], (0, u32::MAX));

        assert_eq!(and_result, ternary_result);
        assert_eq!(ternary_result, vec![(0, 10)]);
    }

    #[test]
    fn ternary_partial_overlap() {
        // a.x1=150, b.x1=100, c.x1=200: a>b always but b<c always, so no result
        let a = traj_const(0, 10, 150.0, 0.0, 250.0, 50.0);
        let b = traj_const(0, 10, 100.0, 0.0, 200.0, 50.0);
        let c = traj_const(0, 10, 200.0, 0.0, 300.0, 50.0);

        let b_used = (true, false, false, false);
        let ternary_pred = AnalyticalPred {
            group: usize::MAX,
            kind: AnalyticalPredKind::Ternary {
                expr1: BinaryBBoxExpr::new(BBoxExpr::x1(), BBoxExpr::x1()),
                op1: CmpOp::Gt,
                threshold1: 0.0,
                expr2: BinaryBBoxExpr::new(BBoxExpr::x1(), BBoxExpr::x1()),
                op2: CmpOp::Gt,
                threshold2: 0.0,
                slots: [a(), b(), c()],
                b_used,
            },
        };
        let result = eval_analytical(&ternary_pred, &[0, 1, 2], &[a, b, c], (0, u32::MAX));
        assert!(result.is_empty());
    }
}
