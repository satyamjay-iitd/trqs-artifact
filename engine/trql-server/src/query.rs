//! Query-side helpers: schema introspection and DSL coordinate/time scaling.

use trql::db::Trqs;
use trql::expr::TrajExpr;
use trql_bbox_compiler::{AnalyticalPredKind, BBoxExpr};
use trql_types::prop::TypeDesc;

use crate::dto::ColInfo;

/// Converts a database's schema into the API's column descriptors.
pub fn schema_to_cols(db: &Trqs) -> Vec<ColInfo> {
    db.schema()
        .col_iter()
        .map(|(col_name, type_desc)| match type_desc {
            TypeDesc::Enum(variants) => ColInfo {
                name: col_name.clone(),
                ty: "Enum",
                variants: Some(variants.clone()),
            },
            TypeDesc::UInt => ColInfo {
                name: col_name.clone(),
                ty: "UInt",
                variants: None,
            },
            TypeDesc::Int => ColInfo {
                name: col_name.clone(),
                ty: "Int",
                variants: None,
            },
            TypeDesc::Bool => ColInfo {
                name: col_name.clone(),
                ty: "Bool",
                variants: None,
            },
            TypeDesc::String => ColInfo {
                name: col_name.clone(),
                ty: "String",
                variants: None,
            },
        })
        .collect()
}

/// The affine map from the coordinates a query is *written* in to the integer
/// coordinates the database *stores*.
///
/// Queries are written in map-grid units — the numbers on the dashboard map's
/// axes: metres from the map's bottom-left corner, Y running up the screen. The
/// db stores `metres × coordinate_scale + delta` in CARLA's frame, whose Y runs
/// the other way. Per axis:
///
/// ```text
/// grid_x = (stored_x − ax) / scale            ax =  min_x·scale + x_delta
/// grid_y = y_sign·(stored_y − ay) / scale     ay = −min_y·scale + y_delta
/// ```
///
/// `y_sign` is −1 in world mode, where the map defines "up". A standalone
/// camera has no map and its coordinates carry no origin shift, so it keeps the
/// db's own axes (`y_sign = 1`, `ax = ay = 0`) and the translation reduces to
/// the plain metres → stored-units conversion.
#[derive(Clone, Copy)]
pub struct QueryFrame {
    pub scale: f64,
    pub ax: f64,
    pub ay: f64,
    pub y_sign: f64,
}

impl QueryFrame {
    /// Frame for a world: grid units measured from the map's bottom-left corner
    /// (`bounds` = `[min_x, min_y, max_x, max_y]` of the road network), Y up.
    pub fn world(scale: f64, x_delta: f64, y_delta: f64, bounds: [f64; 4]) -> Self {
        QueryFrame {
            scale,
            ax: bounds[0] * scale + x_delta,
            ay: -bounds[1] * scale + y_delta,
            y_sign: -1.0,
        }
    }

    /// Frame for a standalone camera: the db's own axes, unshifted.
    pub fn standalone(scale: f64) -> Self {
        QueryFrame {
            scale,
            ax: 0.0,
            ay: 0.0,
            y_sign: 1.0,
        }
    }

    /// Whether the frame carries a usable scale. A db with no `coordinate_scale`
    /// leaves queries untranslated rather than multiplying every threshold by 0.
    pub fn is_usable(&self) -> bool {
        self.scale > 0.0
    }
}

/// Rewrites the coordinate thresholds (and expression offsets) of a parsed query
/// from the units it was written in into the database's stored units, per
/// [`QueryFrame`].
///
/// Substituting `stored = grid·scale + a` into a predicate
/// `Σ wᵢ·gridᵢ + offset  op  threshold` gives
/// `Σ wᵢ'·storedᵢ + offset·scale  op  threshold·scale + shift`, where the Y
/// weights pick up `y_sign` and `shift` is the constant the substitution leaves
/// behind. Relative comparisons need no shift and get none: in `a.x1 < b.x1` the
/// two shifts cancel, and in a width like `a.x2 − a.x1` the ±1 weights do.
///
/// Temporal parameters are untouched — `Then` gaps and `Duration` lengths are
/// frame counts, not distances.
pub fn translate_query_coords(expr: &mut TrajExpr, frame: &QueryFrame) {
    match expr {
        TrajExpr::AnalyticPred(pred) => translate_pred_kind(&mut pred.kind, frame),
        TrajExpr::Then { lhs, rhs, .. }
        | TrajExpr::Contained { lhs, rhs }
        | TrajExpr::Overlap { lhs, rhs } => {
            translate_query_coords(lhs, frame);
            translate_query_coords(rhs, frame);
        }
        TrajExpr::Duration { inner, .. } => translate_query_coords(inner, frame),
    }
}

/// The constant a coordinate expression contributes to the translated threshold:
/// what `Σ wᵢ·(storedᵢ − aᵢ)` leaves behind once the stored terms are kept.
fn shift_of(e: &BBoxExpr, f: &QueryFrame) -> f64 {
    (e.x1_w + e.x2_w) * f.ax + f.y_sign * (e.y1_w + e.y2_w) * f.ay
}

/// Rewrites one coordinate expression into stored units: Y may run the other way
/// in the db, so its weights take `y_sign`; the offset is a plain distance and
/// only scales. Weights stay non-zero, so the cached `used_coords` flags on the
/// enclosing binary expression remain valid.
fn rewrite_expr(e: &mut BBoxExpr, f: &QueryFrame) {
    e.y1_w *= f.y_sign;
    e.y2_w *= f.y_sign;
    swap_y_slots_if_flipped(e, f);
    e.offset *= f.scale;
}

/// Moves the Y weights between the `y1` and `y2` columns when the frame reverses Y.
///
/// Negating the weights is not enough on its own. A query says `y1` meaning the box's
/// *lower* edge in grid units, but the columns are named for the db's own axis, where
/// `y1 ≤ y2` always holds — so when grid Y runs the other way, the grid-lower edge lives in
/// the `y2` column. Without the swap, `a.height` comes out negative and `a INSIDE region`
/// compares the wrong two edges (accepting boxes that hang out of the region).
///
/// Expressions that weight both columns equally, like `center_y`, are unaffected, which is
/// why the bug hid: the centre form of `INSIDE` was always right.
fn swap_y_slots_if_flipped(e: &mut BBoxExpr, f: &QueryFrame) {
    if f.y_sign < 0.0 {
        std::mem::swap(&mut e.y1_w, &mut e.y2_w);
    }
}

/// Maps a region's vertices from the units the query was written in into the db's stored
/// coordinates.
///
/// This is the inverse of the `stored → grid` map in [`QueryFrame`]:
///
/// ```text
/// stored_x = grid_x·scale + ax
/// stored_y = y_sign·grid_y·scale + ay
/// ```
///
/// It is the same substitution [`translate_query_coords`] performs on the thresholds, so a
/// translated region lands exactly where the translated predicate accepts — which is what
/// makes drawing it over the boxes a check on the query rather than a decoration.
///
/// Winding is not renormalised here: this is only a description of where the region is, and
/// a `y_sign` of −1 mirrors the ring without moving it.
pub fn translate_region(vertices: &[[f64; 2]], f: &QueryFrame) -> Vec<[f64; 2]> {
    vertices
        .iter()
        .map(|[gx, gy]| [gx * f.scale + f.ax, f.y_sign * gy * f.scale + f.ay])
        .collect()
}

/// Converts the temporal parameters of a parsed query from seconds to frames by
/// multiplying by `fps`. These are the `Then` gap window (`min_gap`/`max_gap`)
/// and `Duration` minimum length (`min_frames`). Coordinate predicates carry no
/// temporal values and are left alone.
pub fn scale_query_time(expr: &mut TrajExpr, fps: f64) {
    let to_frames = |secs: u32| (secs as f64 * fps).round() as u32;
    match expr {
        TrajExpr::AnalyticPred(_) => {}
        TrajExpr::Then {
            lhs,
            rhs,
            min_gap,
            max_gap,
        } => {
            *min_gap = to_frames(*min_gap);
            if let Some(mg) = max_gap {
                *mg = to_frames(*mg);
            }
            scale_query_time(lhs, fps);
            scale_query_time(rhs, fps);
        }
        TrajExpr::Contained { lhs, rhs } | TrajExpr::Overlap { lhs, rhs } => {
            scale_query_time(lhs, fps);
            scale_query_time(rhs, fps);
        }
        TrajExpr::Duration { inner, min_frames } => {
            *min_frames = to_frames(*min_frames);
            scale_query_time(inner, fps);
        }
    }
}

fn translate_pred_kind(kind: &mut AnalyticalPredKind, f: &QueryFrame) {
    match kind {
        AnalyticalPredKind::Unary {
            expr, threshold, ..
        } => {
            *threshold = *threshold * f.scale + shift_of(expr, f);
            rewrite_expr(expr, f);
        }
        // `lhs(A) − rhs(B)`, so the right-hand shift subtracts.
        AnalyticalPredKind::Binary {
            expr, threshold, ..
        } => {
            *threshold = *threshold * f.scale + shift_of(&expr.lhs, f) - shift_of(&expr.rhs, f);
            rewrite_expr(&mut expr.lhs, f);
            rewrite_expr(&mut expr.rhs, f);
        }
        AnalyticalPredKind::Ternary {
            expr1,
            threshold1,
            expr2,
            threshold2,
            ..
        } => {
            *threshold1 = *threshold1 * f.scale + shift_of(&expr1.lhs, f) - shift_of(&expr1.rhs, f);
            *threshold2 = *threshold2 * f.scale + shift_of(&expr2.lhs, f) - shift_of(&expr2.rhs, f);
            rewrite_expr(&mut expr1.lhs, f);
            rewrite_expr(&mut expr1.rhs, f);
            rewrite_expr(&mut expr2.lhs, f);
            rewrite_expr(&mut expr2.rhs, f);
        }
        // A polygon has no threshold to move, so the *region* is what gets translated —
        // an affine bijection, so membership is preserved. The expressions are then left
        // selecting stored coordinates directly.
        //
        // Each vertex maps into the space of `Σ wᵢ·storedᵢ`, which is what the evaluator
        // compares against: solving `Σ wᵢ·gridᵢ + offset = v` for that sum gives
        // `(v − offset)·scale + aᵢ·Σwᵢ`, with Y picking up `y_sign` as everywhere else.
        // The offsets are folded in here, hence zeroed afterwards.
        //
        // The weights are only *swapped*, never negated: unlike a threshold comparison, the
        // `y_sign` factor has already been absorbed into the vertex above, so negating here
        // would apply it twice.
        AnalyticalPredKind::InPolygon { x, y, poly, .. } => {
            let (wx, wy) = (x.x1_w + x.x2_w, y.y1_w + y.y2_w);
            let (ox, oy) = (x.offset, y.offset);
            let (scale, ax, ay, y_sign) = (f.scale, f.ax, f.ay, f.y_sign);
            // `Polygon::map` re-normalizes winding, which matters here: `y_sign = -1`
            // mirrors the region and would otherwise leave it wound clockwise.
            match poly.map(|[vx, vy]| {
                [
                    (vx - ox) * scale + ax * wx,
                    y_sign * (vy - oy) * scale + ay * wy,
                ]
            }) {
                Ok(mapped) => {
                    *poly = std::sync::Arc::new(mapped);
                    swap_y_slots_if_flipped(x, f);
                    swap_y_slots_if_flipped(y, f);
                    x.offset = 0.0;
                    y.offset = 0.0;
                }
                // An affine map of a valid polygon is valid unless `scale` collapses it,
                // and `is_usable` already gates that. Leave the query untranslated rather
                // than silently substituting a different region.
                Err(e) => eprintln!("leaving polygon untranslated: {e}"),
            }
        }
        AnalyticalPredKind::And(l, r) | AnalyticalPredKind::Or(l, r) => {
            translate_pred_kind(&mut l.kind, f);
            translate_pred_kind(&mut r.kind, f);
        }
        AnalyticalPredKind::Not(inner) => translate_pred_kind(&mut inner.kind, f),
        AnalyticalPredKind::True => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trql_bbox_compiler::{AnalyticalPred, Polygon};
    use trql_types::prop::Schema;

    /// Evaluates a translated predicate against one concrete *stored* bounding box.
    fn eval(pred: &AnalyticalPred, [x1, y1, x2, y2]: [f64; 4]) -> bool {
        let at = |e: &BBoxExpr| e.x1_w * x1 + e.y1_w * y1 + e.x2_w * x2 + e.y2_w * y2 + e.offset;
        match &pred.kind {
            AnalyticalPredKind::True => true,
            AnalyticalPredKind::Unary {
                expr,
                op,
                threshold,
                ..
            } => {
                let v = at(expr);
                match op {
                    trql_bbox_compiler::CmpOp::Gt => v > *threshold,
                    trql_bbox_compiler::CmpOp::Gte => v >= *threshold - 1e-9,
                    trql_bbox_compiler::CmpOp::Lt => v < *threshold,
                    trql_bbox_compiler::CmpOp::Lte => v <= *threshold + 1e-9,
                }
            }
            AnalyticalPredKind::InPolygon { x, y, poly, .. } => poly.contains(at(x), at(y)),
            AnalyticalPredKind::And(l, r) => eval(l, [x1, y1, x2, y2]) && eval(r, [x1, y1, x2, y2]),
            AnalyticalPredKind::Or(l, r) => eval(l, [x1, y1, x2, y2]) || eval(r, [x1, y1, x2, y2]),
            AnalyticalPredKind::Not(i) => !eval(i, [x1, y1, x2, y2]),
            other => panic!("unexpected kind: {other:?}"),
        }
    }

    /// A frame with a genuine Y flip and a non-trivial origin shift and scale, so a term
    /// dropped or mis-signed anywhere in the substitution shows up.
    fn flipped() -> QueryFrame {
        QueryFrame::world(4.0, 30.0, -20.0, [5.0, 7.0, 100.0, 100.0])
    }

    /// Maps a grid-space box to the stored box that denotes it, keeping the db's own
    /// invariant that `y1 ≤ y2` — which is exactly why the Y weights have to change columns.
    fn to_stored([gx1, gy1, gx2, gy2]: [f64; 4], f: &QueryFrame) -> [f64; 4] {
        let sx = |g: f64| g * f.scale + f.ax;
        let sy = |g: f64| f.y_sign * g * f.scale + f.ay;
        let (a, b) = (sy(gy1), sy(gy2));
        [sx(gx1), a.min(b), sx(gx2), a.max(b)]
    }

    fn parse(q: &str) -> TrajExpr {
        let (_, expr) = trql_dsl::parse_query(q, &Schema::new(vec![])).expect("parses");
        expr
    }

    /// Translating a query must preserve what it means. For every predicate form, the
    /// translated predicate on the stored box has to agree with the original on the grid box.
    #[test]
    fn translation_preserves_meaning_under_a_y_flip() {
        // Boxes that hang out of the region on *one* Y side are the ones that matter: a
        // predicate testing the wrong Y column still answers correctly for a box that is
        // wholly inside or wholly outside, so a box set without these passes either way.
        let boxes = [
            [10.0, 10.0, 20.0, 20.0],
            [0.0, 0.0, 60.0, 60.0],
            [22.0, 31.0, 27.0, 39.0],
            [-5.0, 12.0, 8.0, 44.0],
            [30.0, 5.0, 55.0, 48.0],
            [10.0, 2.0, 20.0, 20.0],  // below the region only
            [10.0, 20.0, 20.0, 58.0], // above the region only
            [10.0, 1.0, 20.0, 58.0],  // out on both Y sides
            [2.0, 10.0, 20.0, 20.0],  // out on X, for symmetry
            [12.0, 21.0, 48.0, 40.0], // spans the concave notch
        ];
        let queries = [
            // Rectangle INSIDE, both forms — the whole-box form is what the column swap fixes.
            "SELECT (a:) WHERE a INSIDE [5, 5, 50, 50]",
            "SELECT (a:) WHERE a.center INSIDE [5, 5, 50, 50]",
            // A convex polygon, expanded to half-planes.
            "SELECT (a:) WHERE a INSIDE [[5,5],[50,5],[50,50],[5,50]]",
            // A concave polygon: `InPolygon` corner tests plus edge clearance.
            "SELECT (a:) WHERE a INSIDE [[0,0],[60,0],[60,60],[30,24],[0,60]]",
            "SELECT (a:) WHERE a.center INSIDE [[0,0],[60,0],[60,60],[30,24],[0,60]]",
            // Derived scalars and relative comparisons also cross the Y columns.
            "SELECT (a:) WHERE a.y2 - a.y1 > 8",
            "SELECT (a:) WHERE a.y1 < 30",
        ];

        for q in queries {
            let TrajExpr::AnalyticPred(reference) = parse(q) else {
                panic!("expected AnalyticPred for {q}")
            };
            let mut translated_expr = parse(q);
            translate_query_coords(&mut translated_expr, &flipped());
            let TrajExpr::AnalyticPred(translated) = translated_expr else {
                panic!("expected AnalyticPred")
            };

            for b in boxes {
                assert_eq!(
                    eval(&translated, to_stored(b, &flipped())),
                    eval(&reference, b),
                    "{q} disagrees on grid box {b:?}"
                );
            }
        }
    }

    /// Even-odd ray casting over a closed ring, written here so the check below does not
    /// lean on the same containment code the predicate uses.
    fn ray_cast(ring: &[[f64; 2]], px: f64, py: f64) -> bool {
        let n = ring.len();
        let mut inside = false;
        for i in 0..n {
            let ([ax, ay], [bx, by]) = (ring[i], ring[(i + 1) % n]);
            if (ay > py) != (by > py) && px < (bx - ax) * (py - ay) / (by - ay) + ax {
                inside = !inside;
            }
        }
        inside
    }

    /// A reported region has to land where the translated query actually looks.
    ///
    /// This is the property the dashboard overlay depends on: the drawn shape is only
    /// meaningful if a box the predicate accepts is a box drawn inside the shape. Checked
    /// under the flipped frame, where the region and the thresholds move differently — the
    /// region mirrors about `ay` while the predicate's Y weights change columns.
    #[test]
    fn a_reported_region_matches_where_the_translated_query_looks() {
        for q in [
            "SELECT (a:) WHERE a.center INSIDE [5, 5, 50, 50]",
            "SELECT (a:) WHERE a.center INSIDE [[5,5],[50,5],[50,50],[5,50]]",
            "SELECT (a:) WHERE a.center INSIDE [[0,0],[60,0],[60,60],[30,24],[0,60]]",
        ] {
            let (_, mut expr, used) =
                trql_dsl::parse_query_regions(q, &Schema::new(vec![])).expect("parses");
            translate_query_coords(&mut expr, &flipped());
            let TrajExpr::AnalyticPred(pred) = expr else {
                panic!("expected AnalyticPred for {q}")
            };
            assert_eq!(used.len(), 1, "{q} should report exactly one region");
            let ring = translate_region(&used[0].vertices, &flipped());

            let (mut inside, mut checked) = (0usize, 0usize);
            for gx in -10..70 {
                for gy in -10..70 {
                    // Off the integer grid, so no sample sits on a boundary the two
                    // conventions may legitimately disagree about.
                    let (gx, gy) = (gx as f64 + 0.5, gy as f64 + 0.5);
                    let [sx, sy, _, _] = to_stored([gx, gy, gx, gy], &flipped());
                    let want = eval(&pred, [sx, sy, sx, sy]);
                    assert_eq!(
                        ray_cast(&ring, sx, sy),
                        want,
                        "{q}: region and predicate disagree at grid ({gx}, {gy})"
                    );
                    checked += 1;
                    inside += usize::from(want);
                }
            }
            assert!(
                inside > 0 && inside < checked,
                "{q}: {inside}/{checked} inside — the sweep never crosses the boundary"
            );
        }
    }

    /// Rectangles are reported as rings too, so the dashboard needs one drawing path.
    /// Their corners must come back in order, or the ring self-intersects into a bow tie.
    #[test]
    fn a_rectangle_is_reported_as_its_four_corners_in_order() {
        let (_, _, used) = trql_dsl::parse_query_regions(
            "SELECT (a:) WHERE a INSIDE [5, 10, 50, 40]",
            &Schema::new(vec![]),
        )
        .expect("parses");
        assert_eq!(used.len(), 1);
        assert_eq!(
            used[0].vertices,
            vec![[5.0, 10.0], [50.0, 10.0], [50.0, 40.0], [5.0, 40.0]],
        );
        assert!(used[0].whole_box);
        assert_eq!(used[0].var, "a");
        assert_eq!(used[0].name, None);
    }

    /// Named regions keep their name, and every `INSIDE` is reported — including the same
    /// region used twice, since the two mentions test different objects.
    #[test]
    fn every_inside_is_reported_with_its_name_and_variable() {
        let (_, _, used) = trql_dsl::parse_query_regions(
            "REGIONS (kerb = [0, 0, 10, 10]) \
             SELECT (a:, b:) WHERE a.center INSIDE kerb THEN b INSIDE kerb",
            &Schema::new(vec![]),
        )
        .expect("parses");
        let seen: Vec<_> = used
            .iter()
            .map(|r| (r.name.as_deref(), r.var.as_str(), r.whole_box))
            .collect();
        assert_eq!(
            seen,
            vec![(Some("kerb"), "a", false), (Some("kerb"), "b", true)],
        );
    }

    /// A polygon node whose Y expression names *one* column instead of averaging both — the
    /// only shape where the column swap is observable at all.
    ///
    /// The compiler's own whole-box expansion tests all four corners, and that set is closed
    /// under swapping the Y columns, so it cannot notice; `center_y` weights both columns
    /// equally, so it cannot either. Neither fact makes the swap optional, since nothing
    /// stops a caller from building this node directly.
    #[test]
    fn in_polygon_translation_handles_a_single_column_corner() {
        let poly = std::sync::Arc::new(
            Polygon::new([
                [0.0, 0.0],
                [60.0, 0.0],
                [60.0, 60.0],
                [30.0, 24.0],
                [0.0, 60.0],
            ])
            .unwrap(),
        );
        // The grid bottom-left corner, which under a Y flip lives in the `y2` column.
        let reference = AnalyticalPred::point_in_polygon(BBoxExpr::x1(), BBoxExpr::y1(), poly, 0);
        assert!(
            matches!(reference.kind, AnalyticalPredKind::InPolygon { .. }),
            "the region must be concave, or this tests the half-plane path instead"
        );

        let mut translated = reference.clone();
        translate_pred_kind(&mut translated.kind, &flipped());

        for b in [
            [10.0, 10.0, 20.0, 20.0],
            [10.0, 2.0, 20.0, 20.0],
            [28.0, 26.0, 34.0, 40.0],
            [-5.0, 30.0, 5.0, 50.0],
            [25.0, 40.0, 35.0, 55.0],
        ] {
            assert_eq!(
                eval(&translated, to_stored(b, &flipped())),
                eval(&reference, b),
                "grid box {b:?}"
            );
        }
    }

    /// `height` is `y2 − y1`, a distance, so it must stay positive however the axes run.
    /// Negating the weights without moving them between columns made it negative.
    #[test]
    fn height_stays_positive_under_a_y_flip() {
        let mut e = BBoxExpr::height();
        rewrite_expr(&mut e, &flipped());
        let f = flipped();
        let stored = to_stored([10.0, 10.0, 20.0, 30.0], &f);
        let v = e.x1_w * stored[0] + e.y1_w * stored[1] + e.x2_w * stored[2] + e.y2_w * stored[3];
        assert!(v > 0.0, "height came out {v}");
        // 20 grid units tall, and the db stores metres × scale.
        assert!((v - 20.0 * f.scale).abs() < 1e-9, "height was {v}");
    }

    /// A frame that does not flip Y must be left exactly as it was.
    #[test]
    fn standalone_frame_does_not_swap_columns() {
        let mut e = BBoxExpr::height();
        rewrite_expr(&mut e, &QueryFrame::standalone(2.0));
        assert_eq!((e.y1_w, e.y2_w), (-1.0, 1.0));
    }
}
