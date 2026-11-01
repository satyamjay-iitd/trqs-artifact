use std::sync::Arc;

use trql_bbox_compiler::collect_pred_slots;
use trql_types::{Slot, prop::Value};

use crate::expr::TrajExpr;

/// A property name–value equality constraint for filtering candidate objects.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PropConstraint {
    pub prop: String,
    pub val: Value,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PropConstraints {
    pub constraints: Vec<PropConstraint>,
}
impl PropConstraints {
    pub fn new(constraints: Vec<PropConstraint>) -> Self {
        Self { constraints }
    }
    pub fn empty_constraints() -> PropConstraints {
        Self::new(vec![])
    }

    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    pub fn get_constraint(&self, arg: usize) -> Option<&PropConstraint> {
        self.constraints.get(arg)
    }

    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    pub fn iter_constraints(&self) -> impl Iterator<Item = &PropConstraint> {
        self.constraints.iter()
    }
}

pub struct PropConstraintsBuilder {
    pub constraints: Vec<PropConstraint>,
}

impl PropConstraintsBuilder {
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
        }
    }

    pub fn add(&mut self, constraint: PropConstraint) {
        self.constraints.push(constraint);
    }

    pub fn build(self) -> PropConstraints {
        PropConstraints::new(self.constraints)
    }
}

/// One declared query variable: the name it is written with, the [`Slot`] it
/// was assigned, and the property constraints its object must satisfy.
#[derive(Clone, Debug)]
pub struct VariableWithConstraints {
    slot: Slot,
    name: Arc<str>,
    constraints: PropConstraints,
}

impl VariableWithConstraints {
    /// The tuple position this variable was assigned.
    pub fn slot(&self) -> Slot {
        self.slot
    }

    /// The name the variable is written with in the query.
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn constraints(&self) -> &PropConstraints {
        &self.constraints
    }
}

/// The query's declared slots — the only thing the caller needs to specify.
/// Group structure and temporal links are derived algorithmically from the `TrajExpr`.
///
/// Declaration order is tuple order: the variable declared at position `i` gets
/// [`Slot`] `i`, and `tuple[i]` is the ObjId bound to it.
/// [`VariablesWithConstraints::new`] is the one place that mapping is decided —
/// every variable of the query is known by then — and each
/// [`VariableWithConstraints`] records the result.
///
/// Construction enforces at least one variable — an empty SELECT is not a valid
/// query — so `num_slots() >= 1` holds for any value of this type, and planning
/// need not handle a slotless query.
#[derive(Clone, Debug)]
pub struct VariablesWithConstraints {
    /// One entry per slot in the query tuple.
    per_slot: Vec<VariableWithConstraints>,
}

impl VariablesWithConstraints {
    /// Declares `slots` in tuple order, assigning each variable its index.
    ///
    /// This is the *only* place a query slot's index is decided; from here on
    /// the rest of the system passes the resulting [`Slot`] handles around and
    /// never derives an index of its own.
    pub fn new<N: Into<Arc<str>>>(slots: impl IntoIterator<Item = (N, PropConstraints)>) -> Self {
        let per_slot: Vec<VariableWithConstraints> = slots
            .into_iter()
            .enumerate()
            .map(|(i, (name, constraints))| VariableWithConstraints {
                slot: i,
                name: name.into(),
                constraints,
            })
            .collect();
        assert!(
            !per_slot.is_empty(),
            "a query must declare at least one slot"
        );
        Self { per_slot }
    }

    /// Slots named `names`, none of them filtered by a property.
    pub fn unconstrained<N: Into<Arc<str>>>(names: impl IntoIterator<Item = N>) -> Self {
        Self::new(
            names
                .into_iter()
                .map(|n| (n, PropConstraints::empty_constraints())),
        )
    }

    /// Slots named `names`, all sharing the same property constraints.
    pub fn repeat_constraints<N: Into<Arc<str>>>(
        names: impl IntoIterator<Item = N>,
        constraints: PropConstraints,
    ) -> Self {
        Self::new(names.into_iter().map(|n| (n, constraints.clone())))
    }

    pub fn num_slots(&self) -> usize {
        self.per_slot.len()
    }

    pub fn get_slot_constraints(&self, s: usize) -> Option<&PropConstraints> {
        self.per_slot
            .get(s)
            .map(VariableWithConstraints::constraints)
    }

    /// The slot of the variable written `name` in the query, or `None` if the
    /// query never declared it. This is how the rest of the system gets at a
    /// slot: by the name the user wrote, never by inventing an index.
    pub fn slot_named(&self, name: &str) -> Option<Slot> {
        self.per_slot
            .iter()
            .find(|v| v.name() == name)
            .map(VariableWithConstraints::slot)
    }

    /// The name the variable at `slot` was declared with — the only way back
    /// from a position to something a reader recognises.
    pub fn name_of(&self, slot: Slot) -> Option<&str> {
        self.per_slot.get(slot).map(VariableWithConstraints::name)
    }

    /// Every declared slot, in tuple order.
    pub fn slots(&self) -> impl Iterator<Item = Slot> {
        self.per_slot.iter().map(VariableWithConstraints::slot)
    }
}

/// Collects all slot indices referenced by any `AnalyticPred` leaf in `expr`.
/// The result may contain duplicates; callers should dedup if needed.
pub fn collect_slots(expr: &TrajExpr) -> Vec<Slot> {
    let mut slots = Vec::new();
    collect_slots_into(expr, &mut slots);
    slots
}

fn collect_slots_into(expr: &TrajExpr, out: &mut Vec<Slot>) {
    match expr {
        TrajExpr::AnalyticPred(pred) => collect_pred_slots(pred, out),
        TrajExpr::Then { lhs, rhs, .. }
        | TrajExpr::Contained { lhs, rhs }
        | TrajExpr::Overlap { lhs, rhs } => {
            collect_slots_into(lhs, out);
            collect_slots_into(rhs, out);
        }
        TrajExpr::Duration { inner, .. } => collect_slots_into(inner, out),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trql_bbox_compiler::{AnalyticalPred, BBoxExpr};

    /// The slots these tests are written against: positions `0..N`, standing
    /// for the query variables `a`, `b`, `c`, `d`.
    fn query_slots<const N: usize>() -> [Slot; N] {
        std::array::from_fn(|i| i)
    }

    fn pred(slot: Slot) -> TrajExpr {
        TrajExpr::analytic_pred(AnalyticalPred::unary_gt(BBoxExpr::x1(), 0.0, slot))
    }

    fn binary_pred(s0: Slot, s1: Slot) -> TrajExpr {
        TrajExpr::analytic_pred(AnalyticalPred::coord_cmp(
            BBoxExpr::x1(),
            trql_bbox_compiler::CmpOp::Gt,
            BBoxExpr::x1(),
            [s0, s1],
        ))
    }

    #[test]
    fn collect_slots_walks_every_node_kind() {
        // Then / Contained / Overlap recurse into both sides, Duration into its inner expr.
        let [a, b, c, d, e] = query_slots();
        let expr = TrajExpr::Then {
            lhs: Box::new(TrajExpr::Duration {
                inner: Box::new(binary_pred(a, b)),
                min_frames: 3,
            }),
            rhs: Box::new(TrajExpr::Contained {
                lhs: Box::new(pred(c)),
                rhs: Box::new(TrajExpr::Overlap {
                    lhs: Box::new(pred(d)),
                    rhs: Box::new(pred(e)),
                }),
            }),
            min_gap: 0,
            max_gap: None,
        };

        let mut slots = collect_slots(&expr);
        slots.sort_unstable();
        assert_eq!(slots, vec![a, b, c, d, e]);
    }

    #[test]
    fn collect_slots_keeps_duplicates() {
        // Callers dedup themselves; `chain_groups` in eval.rs relies on this.
        let [b] = query_slots::<1>();
        let expr = TrajExpr::Then {
            lhs: Box::new(pred(b)),
            rhs: Box::new(pred(b)),
            min_gap: 0,
            max_gap: None,
        };
        assert_eq!(collect_slots(&expr), vec![b, b]);
    }
}
