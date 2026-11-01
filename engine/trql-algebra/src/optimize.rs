use trql_bbox_compiler::{AnalyticalPred, AnalyticalPredKind};

use crate::expr::TrajExpr;

/// Optimizes a [`TrajExpr`] by merging nested `Duration` nodes (taking the larger
/// minimum length) and rewriting predicate trees via [`optimize_pred`].
pub fn optimize(expr: TrajExpr) -> TrajExpr {
    match expr {
        TrajExpr::Then {
            lhs,
            rhs,
            min_gap,
            max_gap,
        } => TrajExpr::Then {
            lhs: Box::new(optimize(*lhs)),
            rhs: Box::new(optimize(*rhs)),
            min_gap,
            max_gap,
        },
        TrajExpr::Duration { inner, min_frames } => {
            let inner = optimize(*inner);
            match inner {
                TrajExpr::Duration {
                    inner,
                    min_frames: inner_min,
                } => TrajExpr::Duration {
                    inner,
                    min_frames: min_frames.max(inner_min),
                },
                other => TrajExpr::Duration {
                    inner: Box::new(other),
                    min_frames,
                },
            }
        }
        TrajExpr::AnalyticPred(pred) => TrajExpr::AnalyticPred(optimize_pred(pred)),
        leaf => leaf,
    }
}

/// Rewrites `AnalyticalPred` trees. Currently detects:
/// `And(Binary[s0, s1], Binary[s1, s2])` → `Ternary[s0, s1, s2]`
fn optimize_pred(pred: AnalyticalPred) -> AnalyticalPred {
    match pred.kind {
        AnalyticalPredKind::And(l, r) => {
            let l = optimize_pred(*l);
            let r = optimize_pred(*r);
            // Check for shared-middle-slot binary chain: Binary[s0,s1] AND Binary[s1,s2]
            if let (
                AnalyticalPredKind::Binary {
                    expr: e1,
                    op: op1,
                    threshold: t1,
                    slots: [s0, s1a],
                },
                AnalyticalPredKind::Binary {
                    expr: e2,
                    op: op2,
                    threshold: t2,
                    slots: [s1b, s2],
                },
            ) = (&l.kind, &r.kind)
            {
                if s1a == s1b {
                    // Method rather than field access: the three degree
                    // modules agree on `used_coords()` but only the quadratic
                    // one caches it in a field.
                    let (r1, l2) = (e1.rhs.used_coords(), e2.lhs.used_coords());
                    let b_used = (
                        r1.0 || l2.0,
                        r1.1 || l2.1,
                        r1.2 || l2.2,
                        r1.3 || l2.3,
                    );
                    return AnalyticalPred {
                        group: usize::MAX,
                        kind: AnalyticalPredKind::Ternary {
                            expr1: e1.clone(),
                            op1: *op1,
                            threshold1: *t1,
                            expr2: e2.clone(),
                            op2: *op2,
                            threshold2: *t2,
                            slots: [*s0, *s1a, *s2],
                            b_used,
                        },
                    };
                }
            }
            AnalyticalPred {
                group: usize::MAX,
                kind: AnalyticalPredKind::And(Box::new(l), Box::new(r)),
            }
        }
        other => AnalyticalPred {
            kind: other,
            group: pred.group,
        },
    }
}
