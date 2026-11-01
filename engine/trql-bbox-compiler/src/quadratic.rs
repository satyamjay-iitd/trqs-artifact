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
    ///
    /// Reads the cached flags rather than recomputing them. It exists so that
    /// code shared across degrees can ask the same question of every
    /// `BBoxExpr`: the linear and cubic modules derive this on demand and have
    /// no such field.
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

    /// Expands this expression's polynomial for `win` into a local coordinate
    /// `k = f - origin`, returning `(A, B, C)` such that `expr(f) = A·k² + B·k + C`.
    fn local_poly(&self, seg: &PolySegment<4>, origin: u32) -> Option<(f64, f64, f64)> {
        let expand = |coeffs: [f64; 3]| -> (f64, f64, f64) {
            let delta = origin as f64 - seg.start as f64;
            let (a, b, c) = (coeffs[0], coeffs[1], coeffs[2]);
            (a, 2.0 * a * delta + b, a * delta * delta + b * delta + c)
        };
        let mut ga = 0.0f64;
        let mut gb = 0.0f64;
        let mut gc = self.offset;
        if self.used_coords.0 {
            let ScalarSegmentData2::Poly(coeffs) = &seg.cx1 else {
                return None;
            };
            let (a, b, c) = expand(*coeffs);
            ga += self.x1_w * a;
            gb += self.x1_w * b;
            gc += self.x1_w * c;
        }
        if self.used_coords.1 {
            let ScalarSegmentData2::Poly(coeffs) = &seg.cy1 else {
                return None;
            };
            let (a, b, c) = expand(*coeffs);
            ga += self.y1_w * a;
            gb += self.y1_w * b;
            gc += self.y1_w * c;
        }
        if self.used_coords.2 {
            let ScalarSegmentData2::Poly(coeffs) = &seg.cx2 else {
                return None;
            };
            let (a, b, c) = expand(*coeffs);
            ga += self.x2_w * a;
            gb += self.x2_w * b;
            gc += self.x2_w * c;
        }
        if self.used_coords.3 {
            let ScalarSegmentData2::Poly(coeffs) = &seg.cy2 else {
                return None;
            };
            let (a, b, c) = expand(*coeffs);
            ga += self.y2_w * a;
            gb += self.y2_w * b;
            gc += self.y2_w * c;
        }
        Some((ga, gb, gc))
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
}

impl BinaryBBoxExpr {
    pub fn new(lhs: BBoxExpr, rhs: BBoxExpr) -> Self {
        Self { lhs, rhs }
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
    ) -> Option<(f64, f64, f64)> {
        let (al, bl, cl) = self.lhs.local_poly(seg_a, origin)?;
        let (ar, br, cr) = self.rhs.local_poly(seg_b, origin)?;
        Some((al - ar, bl - br, cl - cr))
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
        // Seek on `end`, not `start`: the first segment that can overlap the clip
        // is the first one ending at or after `clip.0`. Seeking on `start` skips
        // the segment `clip.0` falls *inside* — the common case, since a clip is
        // usually a co-appearance window opening partway through a segment.
        //
        // The `first()` check short-circuits that seek when the answer is 0,
        // which it usually is: a clip is typically the object's own appearance
        // window, so it starts at or before the first segment's end. The binary
        // search is O(log n) but far from free here — it runs once per predicate
        // per tuple over a cache-cold segment array (108 entries per object at
        // `max_error_px = 0`), and a profile charged 38% of `eval_analytical` to
        // this one line. `seek_from_start_is_the_common_case` pins the
        // equivalence.
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
            let mut result = Vec::new();
            for span in SegmentSpanIter::new(trajs[oid].segments(), clip) {
                let lo = span.start_frame;
                let hi = span.end_frame;
                if let Some((a, b, c)) = expr.local_poly(span.segment, lo) {
                    result.extend(
                        poly_cmp_intervals(a, b, c, *op, *threshold, 0, hi - lo)
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
            let eval = |sa: &SegmentSpan<'_>, sb: &SegmentSpan<'_>, lo: u32, hi: u32| {
                if let Some((a, b, c)) = expr.local_poly(sa.segment, sb.segment, lo) {
                    poly_cmp_intervals(a, b, c, *op, *threshold, 0, hi - lo)
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
                SegmentSpanIter::new(trajs[lhs_oid].segments(), clip),
                SegmentSpanIter::new(trajs[rhs_oid].segments(), clip),
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
            // Normalize (a,b,c) and threshold to "poly > 0" form, matching poly_cmp_intervals.
            let normalize = |op: &CmpOp, t: f64, a: f64, b: f64, c: f64| match op {
                CmpOp::Gt => (a, b, c - t - STRICT_EPS),
                CmpOp::Gte => (a, b, c - t + EPS),
                CmpOp::Lt => (-a, -b, -c + t - STRICT_EPS),
                CmpOp::Lte => (-a, -b, -c + t + EPS),
            };
            let a_oid = tuple[slots[0]] as usize;
            let b_oid = tuple[slots[1]] as usize;
            let c_oid = tuple[slots[2]] as usize;
            let _ = b_used;
            let (op1, threshold1, op2, threshold2) = (op1, *threshold1, op2, *threshold2);
            let expr1 = expr1;
            let expr2 = expr2;
            let eval = |sa: &SegmentSpan<'_>,
                        sb: &SegmentSpan<'_>,
                        sc: &SegmentSpan<'_>,
                        lo: u32,
                        hi: u32| {
                if let (Some((ra1, rb1, rc1)), Some((ra2, rb2, rc2))) = (
                    expr1.local_poly(sa.segment, sb.segment, lo),
                    expr2.local_poly(sb.segment, sc.segment, lo),
                ) {
                    let (a1, b1, c1) = normalize(op1, threshold1, ra1, rb1, rc1);
                    let (a2, b2, c2) = normalize(op2, threshold2, ra2, rb2, rc2);
                    both_positive_integer_intervals(a1, b1, c1, a2, b2, c2, 0, hi - lo)
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
                SegmentSpanIter::new(trajs[a_oid].segments(), clip),
                SegmentSpanIter::new(trajs[b_oid].segments(), clip),
                SegmentSpanIter::new(trajs[c_oid].segments(), clip),
                eval,
            )
        }
        AnalyticalPredKind::InPolygon { x, y, poly, slot } => {
            let _g = trql_instrument::scope(trql_instrument::Bucket::SegmentIter);
            let oid = tuple[*slot] as usize;
            let mut result = Vec::new();
            for span in SegmentSpanIter::new(trajs[oid].segments(), clip) {
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

/// Finds frame intervals where `A·f² + B·f + C  op  threshold`.
fn poly_cmp_intervals(
    a: f64,
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
        CmpOp::Gt => positive_integer_intervals(a, b, c - threshold - STRICT_EPS, lo, hi),
        CmpOp::Gte => positive_integer_intervals(a, b, c - threshold + EPS, lo, hi),
        CmpOp::Lt => positive_integer_intervals(-a, -b, -c + threshold - STRICT_EPS, lo, hi),
        CmpOp::Lte => positive_integer_intervals(-a, -b, -c + threshold + EPS, lo, hi),
    }
}

/// Finds the frame ranges in `[lo, hi]` where the parametric point `(x(k), y(k))` lies
/// inside `poly`, where each coordinate is a quadratic in the segment-local frame `k`.
///
/// The parametric form is what makes this exact and cheap. Substituting `(x(k), y(k))` into
/// an edge's implicit line `n·(P − p) = 0` yields one more *quadratic in k* — there is no
/// per-edge parameter to solve for and no distinction between an edge and the line carrying
/// it. Membership can only flip where the curve crosses one of those lines, so the roots of
/// all of them are a superset of the flip frames, and between consecutive breaks the answer
/// is constant. One [`Polygon::contains`] test per sub-range therefore settles it — the trick
/// [`both_positive_integer_intervals`] uses for two polynomials, generalized to one per edge.
///
/// Using the edge *lines* rather than the segments is deliberate: a break where the curve
/// crosses a line outside the polygon costs one extra containment test and changes no answer,
/// which is a better trade than clamping every root to its edge's extent.
///
/// Sampling at each sub-range's *start* rather than its midpoint matches
/// [`positive_integer_intervals`], which is what keeps boundary frames — where the curve sits
/// exactly on an edge — agreeing with every other predicate in the compiler.
fn polygon_intervals(
    (xa, xb, xc): (f64, f64, f64),
    (ya, yb, yc): (f64, f64, f64),
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
            [nx * xa + ny * ya, nx * xb + ny * yb, nx * xc + ny * yc + d],
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
        if !poly.contains(xa * k * k + xb * k + xc, ya * k * k + yb * k + yc) {
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
    use super::*;

    /// The `first()` short-circuit in [`SegmentSpanIter::new`] must agree with the
    /// binary search it replaces, for every clip position relative to every
    /// segment boundary — including the empty-array and clip-before-everything
    /// cases where the fast path must *not* fire.
    #[test]
    fn seek_from_start_is_the_common_case() {
        fn seg(start: u32, end: u32) -> PolySegment<4> {
            let z = ScalarSegmentData2::Poly([0.0, 0.0, 0.0]);
            PolySegment {
                start,
                end,
                cx1: z.clone(),
                cy1: z.clone(),
                cx2: z.clone(),
                cy2: z,
            }
        }
        // Gaps between segments so a clip can fall strictly between two of them.
        let segments: Vec<PolySegment<4>> = (0..6).map(|i| seg(i * 10, i * 10 + 4)).collect();

        for lo in 0..70u32 {
            let clip = (lo, lo + 3);
            let want = segments.partition_point(|s| s.end < clip.0);
            let got = SegmentSpanIter::new(&segments, clip).seg_idx;
            assert_eq!(got, want, "clip {clip:?}");
            // The fast path is only allowed to fire when the answer really is 0.
            if lo <= segments[0].end {
                assert_eq!(got, 0, "clip {clip:?} should short-circuit to 0");
            }
        }

        // Empty array: no first segment, so the search runs and yields 0.
        assert_eq!(SegmentSpanIter::new(&[], (5, 9)).seg_idx, 0);
        // Clip beyond every segment: the answer is `len`, never 0.
        let past = (1000, 1004);
        assert_eq!(
            SegmentSpanIter::new(&segments, past).seg_idx,
            segments.len()
        );
    }

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

    /// Evaluation reads `used_coords` alone, so a flag that disagrees with its
    /// weight silently drops that coordinate — `a.x1 > 0` became `0 > 0` and
    /// matched nothing. Every way of building an expression must keep the two
    /// in sync, including operators, which have to look at the *result's*
    /// weights rather than the left operand's.
    #[test]
    fn used_coords_always_matches_the_weights() {
        let check = |e: &BBoxExpr| {
            assert_eq!(
                e.used_coords,
                (e.x1_w != 0.0, e.y1_w != 0.0, e.x2_w != 0.0, e.y2_w != 0.0),
                "flags disagree with weights: {e:?}"
            );
        };

        for e in [
            BBoxExpr::x1(),
            BBoxExpr::y1(),
            BBoxExpr::x2(),
            BBoxExpr::y2(),
            BBoxExpr::center_x(),
            BBoxExpr::center_y(),
            BBoxExpr::width(),
            BBoxExpr::height(),
            BBoxExpr::constant(5.0),
            BBoxExpr::default(),
        ] {
            check(&e);
        }

        // `x2 - x1` picks up an x1 term the left operand never had.
        check(&(BBoxExpr::x2() - BBoxExpr::x1()));
        check(&(BBoxExpr::x1() + BBoxExpr::y2()));
        check(&(BBoxExpr::width() * 2.0));
        check(&(-BBoxExpr::x1()));
        // Cancellation goes the other way: the term is gone, so is the flag.
        check(&(BBoxExpr::x1() - BBoxExpr::x1()));
        check(&(BBoxExpr::x1() * 0.0));
    }

    // ── Segment iteration ─────────────────────────────────────────────────────

    fn const_seg(start: u32, end: u32, x1: f64) -> PolySegment<4> {
        let zero = ScalarSegmentData2::Poly([0.0, 0.0, 0.0]);
        PolySegment {
            start,
            end,
            cx1: ScalarSegmentData2::Poly([0.0, 0.0, x1]),
            cy1: zero.clone(),
            cx2: zero.clone(),
            cy2: zero,
        }
    }

    /// A clip opening partway through a segment — what a co-appearance window
    /// normally does — must still yield that segment. Seeking on `start`
    /// skipped it, so every predicate clipped to a group window that began
    /// mid-segment evaluated to nothing.
    #[test]
    fn segment_span_iter_keeps_the_segment_a_clip_opens_inside() {
        let segs = vec![const_seg(0, 15, 10.0), const_seg(20, 30, 10.0)];

        let spans: Vec<_> = SegmentSpanIter::new(&segs, (5, 25))
            .map(|s| (s.start_frame, s.end_frame))
            .collect();
        assert_eq!(spans, vec![(5, 15), (20, 25)]);

        // Clip inside one segment only, and clips landing on the boundaries.
        let spans: Vec<_> = SegmentSpanIter::new(&segs, (7, 9))
            .map(|s| (s.start_frame, s.end_frame))
            .collect();
        assert_eq!(spans, vec![(7, 9)]);

        let spans: Vec<_> = SegmentSpanIter::new(&segs, (15, 20))
            .map(|s| (s.start_frame, s.end_frame))
            .collect();
        assert_eq!(spans, vec![(15, 15), (20, 20)]);

        // Gaps between segments and clips past the end yield nothing.
        assert_eq!(SegmentSpanIter::new(&segs, (16, 19)).count(), 0);
        assert_eq!(SegmentSpanIter::new(&segs, (40, 50)).count(), 0);
    }

    // ── Polygon containment ───────────────────────────────────────────────────

    fn square() -> Arc<Polygon> {
        Arc::new(Polygon::from_rect([0.0, 0.0, 10.0, 10.0]).unwrap())
    }

    /// A U opening upwards: the notch `3 < x < 7, y > 3` is *outside*, so a horizontal
    /// sweep across it leaves and re-enters.
    fn u_shape() -> Arc<Polygon> {
        Arc::new(
            Polygon::new([
                [0.0, 0.0],
                [10.0, 0.0],
                [10.0, 10.0],
                [7.0, 10.0],
                [7.0, 3.0],
                [3.0, 3.0],
                [3.0, 10.0],
                [0.0, 10.0],
            ])
            .unwrap(),
        )
    }

    /// A chevron with a notch biting down to (5, 4). Unlike [`u_shape`] its reflex edges are
    /// diagonal, so the separating-axis test has to use the *edge normal* — on a rectilinear
    /// region that third axis only ever repeats what the two box axes already said.
    fn chevron() -> Arc<Polygon> {
        Arc::new(
            Polygon::new([
                [0.0, 0.0],
                [10.0, 0.0],
                [10.0, 10.0],
                [5.0, 4.0],
                [0.0, 10.0],
            ])
            .unwrap(),
        )
    }

    /// Evaluates a predicate tree against one concrete bounding box, so a compiled
    /// predicate can be compared against the geometry it is supposed to mean without
    /// building a whole `PolyTraj`.
    fn eval_at_bbox(pred: &AnalyticalPred, [x1, y1, x2, y2]: [f64; 4]) -> bool {
        match &pred.kind {
            AnalyticalPredKind::True => true,
            AnalyticalPredKind::Unary {
                expr,
                op,
                threshold,
                ..
            } => {
                let v =
                    expr.x1_w * x1 + expr.y1_w * y1 + expr.x2_w * x2 + expr.y2_w * y2 + expr.offset;
                cmp_value(v, *op, *threshold)
            }
            AnalyticalPredKind::InPolygon { x, y, poly, .. } => {
                let at =
                    |e: &BBoxExpr| e.x1_w * x1 + e.y1_w * y1 + e.x2_w * x2 + e.y2_w * y2 + e.offset;
                poly.contains(at(x), at(y))
            }
            AnalyticalPredKind::And(l, r) => {
                eval_at_bbox(l, [x1, y1, x2, y2]) && eval_at_bbox(r, [x1, y1, x2, y2])
            }
            AnalyticalPredKind::Or(l, r) => {
                eval_at_bbox(l, [x1, y1, x2, y2]) || eval_at_bbox(r, [x1, y1, x2, y2])
            }
            AnalyticalPredKind::Not(i) => !eval_at_bbox(i, [x1, y1, x2, y2]),
            other => panic!("unexpected kind in test: {other:?}"),
        }
    }

    fn has_in_polygon(pred: &AnalyticalPred) -> bool {
        match &pred.kind {
            AnalyticalPredKind::InPolygon { .. } => true,
            AnalyticalPredKind::And(l, r) | AnalyticalPredKind::Or(l, r) => {
                has_in_polygon(l) || has_in_polygon(r)
            }
            AnalyticalPredKind::Not(i) => has_in_polygon(i),
            _ => false,
        }
    }

    /// A convex region must compile to plain comparisons — that is the whole point of
    /// detecting convexity, and it is what lets the optimizer and SQL backend stay ignorant
    /// of polygons. If this regresses, those two silently start meeting an unknown node.
    #[test]
    fn convex_polygon_compiles_to_half_planes_only() {
        let pred = AnalyticalPred::point_in_polygon(
            BBoxExpr::center_x(),
            BBoxExpr::center_y(),
            square(),
            0,
        );
        assert!(!has_in_polygon(&pred));
        assert!(!has_in_polygon(&AnalyticalPred::box_in_polygon(
            square(),
            0
        )));
    }

    /// The half-plane expansion of a square must mean exactly what the equivalent
    /// rectangle `INSIDE` means, for every box — that equivalence is what makes the
    /// generalization safe to route existing queries through.
    #[test]
    fn convex_expansion_matches_the_rectangle_it_came_from() {
        let pred = AnalyticalPred::box_in_polygon(square(), 0);
        for x1 in [-3.0, 0.0, 1.0, 6.0, 9.0, 12.0] {
            for y1 in [-3.0, 0.0, 1.0, 6.0, 9.0, 12.0] {
                for (w, h) in [(1.0, 1.0), (4.0, 2.0), (20.0, 20.0)] {
                    let (x2, y2) = (x1 + w, y1 + h);
                    let expected = x1 >= 0.0 && y1 >= 0.0 && x2 <= 10.0 && y2 <= 10.0;
                    assert_eq!(
                        eval_at_bbox(&pred, [x1, y1, x2, y2]),
                        expected,
                        "box [{x1},{y1},{x2},{y2}]"
                    );
                }
            }
        }
    }

    /// The centre form of a concave region is a point test, and the point that matters is
    /// the centre — not the box.
    #[test]
    fn concave_centre_tests_the_centre_point() {
        let pred = AnalyticalPred::point_in_polygon(
            BBoxExpr::center_x(),
            BBoxExpr::center_y(),
            u_shape(),
            0,
        );
        assert!(has_in_polygon(&pred));
        // Centre at (5, 6) sits in the notch — inside the bounding box, outside the U.
        assert!(!eval_at_bbox(&pred, [4.0, 5.0, 6.0, 7.0]));
        // Centre at (1, 6) is in the left arm.
        assert!(eval_at_bbox(&pred, [0.0, 5.0, 2.0, 7.0]));
    }

    /// Whole-box containment in a concave region is exactly "corners inside, and no edge
    /// through the interior". This is the case the corner test alone gets wrong, so it is
    /// checked against a brute-force oracle rather than against hand-picked expectations.
    ///
    /// Box origins are offset by a half unit and their extents are chosen so that `x1 + w`
    /// is off the grid too — no box edge ever lands on a polygon edge, where ray-crossing
    /// membership is arbitrary by nature and neither answer is "correct".
    #[test]
    fn concave_whole_box_matches_a_brute_force_oracle() {
        for poly in [u_shape(), chevron()] {
            let pred = AnalyticalPred::box_in_polygon(poly.clone(), 0);

            // Sampling a box densely enough that no violation at these polygons' scale can
            // hide between samples; every feature of both shapes is at least 3 units across.
            let all_inside = |[x1, y1, x2, y2]: [f64; 4]| {
                let step = 0.1;
                let mut y = y1;
                while y <= y2 + 1e-9 {
                    let mut x = x1;
                    while x <= x2 + 1e-9 {
                        if !poly.contains(x, y) {
                            return false;
                        }
                        x += step;
                    }
                    y += step;
                }
                true
            };

            let mut contained = 0;
            let mut checked = 0;
            for i in -2..=11 {
                for j in -2..=11 {
                    for (w, h) in [(1.2, 1.2), (3.2, 1.2), (1.2, 3.2), (5.2, 2.2)] {
                        let b = [
                            i as f64 + 0.5,
                            j as f64 + 0.5,
                            i as f64 + 0.5 + w,
                            j as f64 + 0.5 + h,
                        ];
                        let want = all_inside(b);
                        contained += want as u32;
                        checked += 1;
                        assert_eq!(eval_at_bbox(&pred, b), want, "box {b:?} in {poly:?}");
                    }
                }
            }
            // Guard against a vacuous pass: both answers must actually occur.
            assert!(
                contained > 0 && contained < checked,
                "{contained}/{checked}"
            );
        }
    }

    /// The condition the corner test alone misses: a box spanning the notch of the U holds
    /// all four corners inside while its middle pokes out. Called out separately because it
    /// is the entire reason the edge-clearance conjunction exists.
    #[test]
    fn concave_whole_box_rejects_a_box_straddling_the_notch() {
        let pred = AnalyticalPred::box_in_polygon(u_shape(), 0);
        // Corners (1,4), (9,4), (9,8), (1,8) are all inside the U's two arms, but the box
        // spans the notch between them.
        let straddling = [1.0, 4.0, 9.0, 8.0];
        for (x, y) in [(1.0, 4.0), (9.0, 4.0), (9.0, 8.0), (1.0, 8.0)] {
            assert!(
                u_shape().contains(x, y),
                "corner ({x},{y}) should be inside"
            );
        }
        assert!(!u_shape().contains(5.0, 6.0), "the notch should be outside");
        assert!(!eval_at_bbox(&pred, straddling));

        // The same box shrunk into one arm is contained.
        assert!(eval_at_bbox(&pred, [0.5, 4.0, 2.5, 8.0]));
    }

    /// `collect_pred_slots` feeds group annotation and the `Not` domain; an `InPolygon`
    /// that reported no slots would be evaluated against an unclipped domain.
    #[test]
    fn in_polygon_reports_its_slot() {
        let pred = AnalyticalPred::point_in_polygon(
            BBoxExpr::center_x(),
            BBoxExpr::center_y(),
            u_shape(),
            2,
        );
        let mut slots = Vec::new();
        collect_pred_slots(&pred, &mut slots);
        assert_eq!(slots, vec![2]);
    }

    // ── polygon_intervals ─────────────────────────────────────────────────────

    /// A straight sweep across the U leaves through one wall of the notch and re-enters
    /// through the other, so the answer is two intervals — the case a single half-plane
    /// conjunction cannot express at all.
    #[test]
    fn polygon_intervals_splits_at_a_concave_notch() {
        // x(k) = k + 0.5, y(k) = 7. The half-offset keeps every sampled frame off the
        // polygon boundary, where ray-crossing membership is arbitrary either way.
        let got = polygon_intervals((0.0, 1.0, 0.5), (0.0, 0.0, 7.0), &u_shape(), 0, 9);
        assert_eq!(got, vec![(0, 2), (7, 9)]);
    }

    /// The point of solving the *quadratic*: an arc can leave a convex region and come
    /// back, which no straight path through a convex region can do. A linear solver would
    /// report one interval here, or none.
    #[test]
    fn polygon_intervals_follows_an_arc_out_of_a_convex_region_and_back() {
        // y(k) = −0.5·(k − 4.5)² + 12 pokes above y = 10 between k = 2.5 and k = 6.5,
        // while x(k) = k + 0.5 stays within the square throughout.
        let got = polygon_intervals((0.0, 1.0, 0.5), (-0.5, 4.5, 1.875), &square(), 0, 9);
        assert_eq!(got, vec![(0, 2), (7, 9)]);
    }

    /// Degenerate spans: wholly inside collapses to one interval rather than one per
    /// spurious edge root, and wholly outside yields nothing.
    #[test]
    fn polygon_intervals_handles_constant_membership() {
        let inside = polygon_intervals((0.0, 0.0, 5.0), (0.0, 0.0, 5.0), &square(), 0, 20);
        assert_eq!(inside, vec![(0, 20)]);

        let outside = polygon_intervals((0.0, 0.0, 50.0), (0.0, 0.0, 50.0), &square(), 0, 20);
        assert!(outside.is_empty());

        assert!(polygon_intervals((0.0, 1.0, 0.5), (0.0, 0.0, 5.0), &square(), 5, 4).is_empty());
    }

    /// Every frame's membership must match a direct `contains` test on the same point —
    /// the breakpoint-plus-sample argument is only valid if no flip hides between breaks.
    #[test]
    fn polygon_intervals_agrees_with_pointwise_containment() {
        let poly = u_shape();
        let (x, y) = ((0.02, 0.3, 0.4), (-0.06, 1.1, 0.7));
        let got = polygon_intervals(x, y, &poly, 0, 30);

        for f in 0..=30u32 {
            let k = f as f64;
            let point = (x.0 * k * k + x.1 * k + x.2, y.0 * k * k + y.1 * k + y.2);
            let covered = got.iter().any(|&(s, e)| f >= s && f <= e);
            assert_eq!(
                covered,
                poly.contains(point.0, point.1),
                "frame {f} at {point:?}"
            );
        }
        // Guard against the test passing vacuously if the coefficients ever drift.
        assert!(got.len() >= 2, "expected several visits, got {got:?}");
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
    //             slots: [0, 1],
    //         },
    //     };
    //     let out = eval_analytical(&pred, &[0, 1], &[poly1, poly2], (0, u32::MAX));
    //     println!("{:?}", out);
    //     assert!(!out.is_empty());
    // }
}
