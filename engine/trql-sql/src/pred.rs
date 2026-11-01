//! Rendering [`AnalyticalPred`] trees as SQL boolean expressions.
//!
//! Every predicate the DSL can build is *pointwise in the frame*: a comparison between
//! affine combinations of bounding-box coordinates, closed under `AND`/`OR`/`NOT`. So the
//! whole predicate tree collapses to one SQL boolean over the columns of a group's frame
//! relation — no interval reasoning happens here.

use trql_bbox_compiler::{AnalyticalPred, AnalyticalPredKind, BBoxExpr, CmpOp};
use trql_types::Slot;

use crate::{coord_col, num};

/// Renders `pred` as a SQL boolean over the coordinate columns of a frame relation.
///
/// Coordinate columns are named by [`coord_col`], so the caller only has to guarantee that
/// every slot the predicate mentions is projected by the relation being filtered.
pub(crate) fn render(pred: &AnalyticalPred) -> String {
    match &pred.kind {
        AnalyticalPredKind::True => "TRUE".to_owned(),

        AnalyticalPredKind::Unary {
            expr,
            op,
            threshold,
            slot,
        } => format!("({} {} {})", affine(expr, *slot), cmp(*op), num(*threshold)),

        // `lhs(A) − rhs(B) op threshold` — the difference is what the analytical
        // evaluator solves for, so the SQL mirrors it rather than moving terms across.
        AnalyticalPredKind::Binary {
            expr,
            op,
            threshold,
            slots,
        } => format!(
            "({} - {} {} {})",
            affine(&expr.lhs, slots[0]),
            affine(&expr.rhs, slots[1]),
            cmp(*op),
            num(*threshold)
        ),

        // Ternary is a fused two-comparison form the optimizer produces to share a
        // window sweep; as a boolean it is just the conjunction.
        AnalyticalPredKind::Ternary {
            expr1,
            op1,
            threshold1,
            expr2,
            op2,
            threshold2,
            slots,
            ..
        } => format!(
            "(({} - {} {} {}) AND ({} - {} {} {}))",
            affine(&expr1.lhs, slots[0]),
            affine(&expr1.rhs, slots[1]),
            cmp(*op1),
            num(*threshold1),
            affine(&expr2.lhs, slots[1]),
            affine(&expr2.rhs, slots[2]),
            cmp(*op2),
            num(*threshold2),
        ),

        // Only non-convex regions reach here — a convex polygon is expanded into half-plane
        // `Unary` comparisons before anything downstream sees it. With no conjunctive form
        // available, the SQL states point-in-polygon directly: ray-crossing parity, one
        // `CASE` per edge, summed and taken mod 2.
        AnalyticalPredKind::InPolygon { x, y, poly, slot } => {
            let (px, py) = (affine(x, *slot), affine(y, *slot));
            let crossings: Vec<String> = poly
                .edges()
                // A horizontal ray never crosses a horizontal edge, and its slope term
                // would divide by zero. The `(p_y > y) <> (q_y > y)` test excludes these
                // anyway; filtering here just keeps the literal out of the SQL.
                .filter(|(p, q)| p[1] != q[1])
                .map(|(p, q)| {
                    format!(
                        "CASE WHEN (({} > {py}) <> ({} > {py})) AND ({px} < {} * ({py} - {}) + {}) \
                         THEN 1 ELSE 0 END",
                        num(p[1]),
                        num(q[1]),
                        num((q[0] - p[0]) / (q[1] - p[1])),
                        num(p[1]),
                        num(p[0]),
                    )
                })
                .collect();
            // `Polygon` rejects zero-area rings, so this cannot be empty — but an empty
            // join would emit syntactically broken SQL rather than fail loudly, so give it
            // the answer an all-horizontal ring would deserve instead.
            if crossings.is_empty() {
                return "FALSE".to_owned();
            }
            format!("((({}) % 2) = 1)", crossings.join(" + "))
        }

        AnalyticalPredKind::And(l, r) => format!("({} AND {})", render(l), render(r)),
        AnalyticalPredKind::Or(l, r) => format!("({} OR {})", render(l), render(r)),
        AnalyticalPredKind::Not(inner) => format!("(NOT {})", render(inner)),
    }
}

/// Renders `x1_w·x1 + y1_w·y1 + x2_w·x2 + y2_w·y2 + offset` for one slot.
///
/// Zero-weight coordinates are dropped so the SQL stays readable; an expression that is
/// all zeros still has to render as something, hence the `0` fallback.
fn affine(expr: &BBoxExpr, slot: Slot) -> String {
    let mut terms: Vec<String> = Vec::new();
    for (w, coord) in [
        (expr.x1_w, "x1"),
        (expr.y1_w, "y1"),
        (expr.x2_w, "x2"),
        (expr.y2_w, "y2"),
    ] {
        if w == 0.0 {
            continue;
        }
        let col = coord_col(slot, coord);
        if w == 1.0 {
            terms.push(col);
        } else {
            terms.push(format!("{} * {}", num(w), col));
        }
    }
    if expr.offset != 0.0 || terms.is_empty() {
        terms.push(num(expr.offset));
    }
    format!("({})", terms.join(" + "))
}

fn cmp(op: CmpOp) -> &'static str {
    match op {
        CmpOp::Gt => ">",
        CmpOp::Gte => ">=",
        CmpOp::Lt => "<",
        CmpOp::Lte => "<=",
    }
}
