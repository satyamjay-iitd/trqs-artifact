use trql_bbox_compiler::AnalyticalPred;

/// Trajectory query expression tree.
///
/// Leaf nodes (`AnalyticPred`) carry a unified [`AnalyticalPred`] whose slot indices
/// point into the evaluated tuple, so the same tree can reference multiple objects
/// at different positions and freely compose boolean predicates across slot sets.
///
/// Interior nodes combine sub-expressions with temporal sequencing (`Then`),
/// containment (`Contained`), proper overlap (`Overlap`), or duration filtering
/// (`Duration`).
#[derive(Clone, Debug)]
pub enum TrajExpr {
    /// Analytical predicate evaluated via polynomial root-finding — no per-frame sampling.
    /// Slot indices are embedded inside the [`AnalyticalPred`] leaves.
    AnalyticPred(AnalyticalPred),
    /// Temporal sequencing: intervals from `rhs` that follow `lhs` within the gap window.
    Then {
        lhs: Box<TrajExpr>,
        rhs: Box<TrajExpr>,
        min_gap: u32,
        max_gap: Option<u32>,
    },
    Contained {
        lhs: Box<TrajExpr>,
        rhs: Box<TrajExpr>,
    },
    /// Allen's *overlaps*: `rhs` starts strictly inside `lhs` and ends strictly after it.
    /// The result spans the union, `[lhs.start, rhs.stop]`.
    Overlap {
        lhs: Box<TrajExpr>,
        rhs: Box<TrajExpr>,
    },
    /// Filters to intervals lasting at least `min_frames`.
    Duration {
        inner: Box<TrajExpr>,
        min_frames: u32,
    },
}

impl TrajExpr {
    /// Wraps an [`AnalyticalPred`] as a `TrajExpr` leaf.
    pub fn analytic_pred(pred: AnalyticalPred) -> Self {
        Self::AnalyticPred(pred)
    }
}
