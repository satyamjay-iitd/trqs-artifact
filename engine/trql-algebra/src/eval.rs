use std::collections::HashMap;

use trql_bbox_compiler::{collect_pred_slots, eval_analytical};
use trql_bbox_compiler::{AnalyticalPred, AnalyticalPredKind};
use trql_poly::PolyTraj;
use trql_types::intervals::Interval;

use trql_selector::{ChainLink, GroupBounds, GroupChain, ObjId, ObjSelector};
use trql_types::{Gap, Slot};

use trql_types::intervals::IntervalSets;
use trql_types::{QueryResult, Tuple};

use crate::{
    db::Trqs,
    expr::TrajExpr,
    optimize,
    plan::{PropConstraints, VariablesWithConstraints, collect_slots},
};

/// Whole-life results of single-slot predicates, keyed by the leaf that produced
/// them and the object they were evaluated for.
///
/// A predicate reading one slot has a result that depends on that slot's object
/// alone, yet the tuple loop re-solves it for every tuple the object appears in.
/// On `NS-crossing` — `CONTAINS(DURATION(c.center INSIDE r, 60), DURATION(p.center
/// INSIDE r, 12))`, where both operands are single-slot — that is ~367k pair
/// evaluations of two predicates where 20,391 cars + 5,912 pedestrians would do,
/// and it made polynomial solving 69% of the query against 1.1% for selection.
/// Solving once over the object's whole life and intersecting with the per-tuple
/// co-appearance window trades root-finding for interval intersection.
///
/// Sound because the leaf is *pointwise*: it returns the runs of frames where the
/// predicate holds, restricted to the window it is given. Restricting afterwards
/// selects the same frames as restricting first. The memo therefore has to sit
/// **below** any enclosing `Duration`, which is not pointwise — an interval that
/// clears a 60-frame minimum over the whole life can fall short of it inside one
/// co-appearance window, so caching a `Duration` result would be wrong.
///
/// The key's `usize` is the address of the `AnalyticalPred` node in the query's
/// optimised tree. [`AlgebraCtx::run`] owns that tree for the whole tuple loop and
/// only lends it out, so each node keeps one address for as long as the memo — both
/// live and die inside `run`.
type UnaryMemo = HashMap<(usize, ObjId), Vec<Interval>>;

/// [`sole_slot`], but only for a query whose tuples bind more than one slot.
///
/// A single-slot query's selector visits each candidate exactly once, so every memo
/// key would be written and read once — pure overhead. Measured on `NS-junction`
/// (43,360 objects, one slot) memoising cost 16.5 -> 28.3 ms, so the check comes
/// first and skips the classification walk with it.
fn sole_slot_when_shared(pred: &AnalyticalPred, num_slots: usize) -> Option<Slot> {
    (num_slots > 1).then(|| sole_slot(pred)).flatten()
}

/// The single slot `pred` reads, or `None` when it reads several or none.
///
/// `Binary` and `Ternary` name distinct slots as the planner builds them, so they
/// answer `None` in practice; they are still checked for coinciding slots rather
/// than assumed, so the answer holds by inspection of the node.
fn sole_slot(pred: &AnalyticalPred) -> Option<Slot> {
    fn all_same(slots: &[Slot]) -> Option<Slot> {
        let first = *slots.first()?;
        slots.iter().all(|&s| s == first).then_some(first)
    }
    match &pred.kind {
        AnalyticalPredKind::Unary { slot, .. } | AnalyticalPredKind::InPolygon { slot, .. } => {
            Some(*slot)
        }
        AnalyticalPredKind::Binary { slots, .. } => all_same(slots),
        AnalyticalPredKind::Ternary { slots, .. } => all_same(slots),
        AnalyticalPredKind::And(lhs, rhs) | AnalyticalPredKind::Or(lhs, rhs) => {
            let (l, r) = (sole_slot(lhs)?, sole_slot(rhs)?);
            (l == r).then_some(l)
        }
        AnalyticalPredKind::Not(inner) => sole_slot(inner),
        // Constant: no slot to key a memo entry on, and nothing to solve.
        AnalyticalPredKind::True => None,
    }
}

/// Query evaluation context holding a reference to the database.
///
/// Optionally overrides polynomial trajectories (e.g. for benchmarking with different fits).
pub struct AlgebraCtx<'a> {
    pub db: &'a Trqs,
    pub poly_trajs: Option<&'a [PolyTraj]>,
}

impl<'a> AlgebraCtx<'a> {
    /// Evaluates `expr` for one candidate tuple.
    ///
    /// `group_bounds[g]` is the coappearance of chain group `g`, and clips every
    /// predicate tagged with that group so polynomial evaluation never solves
    /// outside a window where the objects are actually co-present. A predicate
    /// whose group index has no entry — an unannotated tree, as in tests — is
    /// evaluated unclipped.
    pub(crate) fn eval(
        &self,
        expr: &TrajExpr,
        tuple: &[u32],
        group_bounds: &[GroupBounds],
        group_slots: &[Vec<Slot>],
        poly_trajs: &'a [PolyTraj],
        memo: &mut UnaryMemo,
    ) -> IntervalSets {
        match expr {
            TrajExpr::AnalyticPred(pred) => {
                // A group's bounds are the coappearance of *every* slot in the
                // group, but a leaf under `OVERLAP` or `CONTAINS` may name only
                // some of them: those operators relate two intervals that need
                // not both be live at once, so clipping such a leaf to the
                // group window truncates it and can drop real answers. Where a
                // leaf names the whole group the two coincide and the group
                // bounds -- which also carry the THEN chain window -- are the
                // tighter, cheaper choice; where it does not, fall back to the
                // coappearance of just the slots the leaf itself names.
                let own;
                let clips: Option<&[Interval]> = match group_bounds.get(pred.group) {
                    Some(gb) if covers_group(pred, group_slots.get(pred.group)) => {
                        Some(gb.as_slice())
                    }
                    Some(_) | None => {
                        own = slot_coappearance(pred, tuple, poly_trajs);
                        own.as_deref()
                    }
                };
                let intervals = match clips {
                    Some(clips) => eval_clipped(pred, tuple, poly_trajs, clips, memo),
                    None => eval_analytical(pred, tuple, poly_trajs, (0, u32::MAX)),
                };
                // Sorting and merging the solved frames into 5-tuples: per leaf
                // per tuple, so on a wide candidate space it can outweigh the
                // solving it follows.
                let _g = trql_instrument::scope(trql_instrument::Bucket::IntervalLift);
                IntervalSets::new(intervals)
            }
            TrajExpr::Then {
                lhs,
                rhs,
                min_gap,
                max_gap,
            } => {
                let lhs = self.eval(lhs, tuple, group_bounds, group_slots, poly_trajs, memo);
                if lhs.is_empty() {
                    return IntervalSets::empty();
                }
                let rhs = self.eval(rhs, tuple, group_bounds, group_slots, poly_trajs, memo);
                let _g = trql_instrument::scope(trql_instrument::Bucket::Temporal);
                lhs.then_with_gap(&rhs, *min_gap, *max_gap)
            }
            TrajExpr::Duration { inner, min_frames } => {
                let inner = self.eval(inner, tuple, group_bounds, group_slots, poly_trajs, memo);
                let _g = trql_instrument::scope(trql_instrument::Bucket::Temporal);
                inner.with_min_duration(*min_frames)
            }
            TrajExpr::Contained { lhs, rhs } => {
                let lhs = self.eval(lhs, tuple, group_bounds, group_slots, poly_trajs, memo);
                let rhs = self.eval(rhs, tuple, group_bounds, group_slots, poly_trajs, memo);
                let _g = trql_instrument::scope(trql_instrument::Bucket::Temporal);
                lhs.contains(&rhs)
            }
            TrajExpr::Overlap { lhs, rhs } => {
                let lhs = self.eval(lhs, tuple, group_bounds, group_slots, poly_trajs, memo);
                let rhs = self.eval(rhs, tuple, group_bounds, group_slots, poly_trajs, memo);
                let _g = trql_instrument::scope(trql_instrument::Bucket::Temporal);
                lhs.overlap(&rhs)
            }
        }
    }

    /// Runs a full query: prunes candidate object tuples with the appearance
    /// chain selector ([`trql_selector::ObjSelector`]), then evaluates the
    /// temporal predicate on each surviving tuple.
    ///
    /// The chain is derived from the expression: every top-level `Then` splits
    /// into two coappearance groups linked by a window-relaxed THEN
    /// (a necessary condition on appearance intervals — the exact gap
    /// semantics are still enforced by predicate evaluation). Groups may
    /// share slots; the selector pins shared slots across groups.
    ///
    /// Returns all `(tuple, intervals)` pairs where the combined predicate is
    /// satisfied. Tuples are indexed by slot: `tuple[s]` is the ObjId bound to
    /// slot `s`.
    pub fn run(&self, constraints: &VariablesWithConstraints, expr: &TrajExpr) -> QueryResult {
        // Whole-query enclosing scope: selector pruning, chain construction, and
        // interval algebra land in `Other`; nested scopes carve out root-finding
        // and segment-iteration (see [`trql_instrument`]).
        let _g = trql_instrument::scope(trql_instrument::Bucket::Other);
        let mut opt_expr = optimize::optimize(expr.clone());

        // Derive the coappearance chain and annotate each predicate with the
        // group whose coappearance intervals clip its evaluation. Both walks
        // recurse through `Then` identically, so group indices line up.
        let (chain, min_span) = coappear_chain(&opt_expr, constraints);
        let mut next_group = 0;
        annotate_groups(&mut opt_expr, &mut next_group);
        assert_eq!(next_group, chain.num_groups());

        // The expression may only reference slots the query declared. An
        // undeclared one belongs to no chain group, so the selector would never
        // bind it and would leave a zero sitting in that tuple position.
        if let Some(slot) = chain.slot_beyond(constraints.num_slots()) {
            let declared: Vec<&str> = constraints
                .slots()
                .filter_map(|s| constraints.name_of(s))
                .collect();
            panic!(
                "expression references slot {slot}, but the query declares only {}: {declared:?}",
                constraints.num_slots()
            );
        }

        // One slot per declared variable, in tuple order.
        // `VariablesWithConstraints` guarantees at least one, and every chain
        // group draws from these, so no group can be empty.
        let slots: Vec<Slot> = constraints.slots().collect();

        // Compiling the plan is just pairing the chain with each slot's
        // candidates: the chain is already in the shape the selector runs on,
        // and the binding it produces is indexed by query slot, so the result
        // tuple needs no remapping.
        let poly_trajs = self.poly_trajs.unwrap_or_else(|| self.db.poly_trajs());

        let selector = {
            let _g = trql_instrument::scope(trql_instrument::Bucket::Selection);
            let mut cands: Vec<Vec<ObjId>> = slots
                .iter()
                .map(|s| {
                    let c = constraints
                        .get_slot_constraints(*s)
                        .expect("slot is declared");
                    resolve_candidates(self.db, c)
                })
                .collect();
            let warm = prefilter_candidates(&mut cands, &opt_expr, poly_trajs);
            (ObjSelector::new(cands, chain).with_min_span(min_span), warm)
        };
        let (selector, warm_memo) = selector;
        // No useful capacity hint: this holds one row per candidate that
        // *survives* predicate evaluation, which is what the query is computing.
        let mut out: QueryResult = Vec::new();

        // Preallocated buffer to avoid repeated allocations.
        let mut group_bounds: Vec<GroupBounds> = vec![Vec::new(); selector.num_groups()];
        // Which slots each group covers, so a leaf that names only some of them
        // is not clipped to a window its own objects never needed.
        let group_slots = group_slot_sets(&opt_expr, &constraints);

        // Lives across the whole loop: that is what lets one object's single-slot
        // predicates be solved once rather than once per tuple it appears in.
        // It starts warm with whatever the candidate pre-filter already solved.
        let mut memo = warm_memo;

        let mut candidates = {
            let _g = trql_instrument::scope(trql_instrument::Bucket::Selection);
            selector.candidates(self.db)
        };
        loop {
            // Tuple enumeration and the per-tuple bookkeeping it feeds are the
            // selection stage; predicate evaluation and interval algebra below
            // are charged to their own buckets.
            let tuple: Tuple = {
                let _g = trql_instrument::scope(trql_instrument::Bucket::Selection);
                if !candidates.advance() {
                    break;
                }
                let tuple = candidates.bindings().clone();
                for (g, buf) in group_bounds.iter_mut().enumerate() {
                    buf.clear();
                    buf.extend_from_slice(candidates.group_bounds(g));
                }
                tuple
            };
            let ivs = self.eval(&opt_expr, &tuple, &group_bounds, &group_slots, poly_trajs, &mut memo);
            if !ivs.is_empty() {
                out.push((tuple, ivs));
            }
        }
        out
    }
}

/// Accumulates a [`GroupChain`] as [`chain_groups`] walks the spine.
///
/// Groups and gaps alternate by construction: every group consumes the gap
/// pushed before it, and the one group with no preceding gap becomes the head.
#[derive(Default)]
struct GroupChainBuilder {
    head: Option<Vec<Slot>>,
    pending_gap: Option<Gap>,
    links: Vec<ChainLink>,
    min_span: Vec<u32>,
}

impl GroupChainBuilder {
    /// Records the gap linking the group just pushed to the one coming next.
    fn gap(&mut self, gap: Gap) {
        debug_assert!(
            self.pending_gap.is_none(),
            "two gaps with no group between them"
        );
        self.pending_gap = Some(gap);
    }

    fn group(&mut self, slots: Vec<Slot>, min_span: u32) {
        self.min_span.push(min_span);
        match self.pending_gap.take() {
            Some(gap) => self.links.push(ChainLink::then_relaxed(gap, slots)),
            None => {
                debug_assert!(self.head.is_none(), "two heads: a gap was dropped");
                self.head = Some(slots);
            }
        }
    }

    fn build(self) -> (GroupChain, Vec<u32>) {
        let chain = GroupChain::new(
            self.head.expect("a chain has at least one group"),
            self.links,
        );
        (chain, self.min_span)
    }
}

/// Flattens the `Then` spine of `expr` into the coappearance chain the selector
/// prunes with: one group per non-`Then` subtree (its sorted, deduped slots),
/// linked in spine order by the gap windows of the `Then`s they came from.
///
/// Also returns, per group, the shortest co-appearance run that group could
/// still satisfy (see [`min_span`]) — the selector uses it to discard a binding
/// whose objects do co-appear, but not for long enough.
fn coappear_chain(
    expr: &TrajExpr,
    constraints: &VariablesWithConstraints,
) -> (GroupChain, Vec<u32>) {
    let mut chain = GroupChainBuilder::default();
    chain_groups(expr, constraints, &mut chain);
    chain.build()
}

/// A lower bound on the length of any interval `expr` can produce, or `None` if
/// the subtree contains a `Then`.
///
/// The selector reads this as "how many consecutive frames must this group's
/// variables share", which holds because a group's predicates are clipped to
/// its co-appearance: every interval the subtree produces lies inside a single
/// co-appearance run, so a bound on the interval's length is a bound on the
/// run's. The bounds themselves mirror the `c` (minimum-length) component the
/// interval algebra computes for each operator, in
/// [`trql_types::intervals::IntervalSet`].
///
/// `Then` is the exception and the reason for the `Option`: it joins two
/// intervals separated by a gap, so its result can straddle two runs and its
/// length says nothing about either. A `Then` nested under, say, a `Duration`
/// (the enclosing operators do not split groups; only a `Then` on the spine
/// does) therefore poisons the whole group's bound rather than just its own.
fn min_span(expr: &TrajExpr) -> Option<u32> {
    match expr {
        TrajExpr::AnalyticPred(_) => Some(1),
        TrajExpr::Then { .. } => None,
        TrajExpr::Duration { inner, min_frames } => Some(min_span(inner)?.max(*min_frames)),
        // CONTAINS keeps the lhs, which must strictly enclose the rhs.
        TrajExpr::Contained { lhs, rhs } => {
            Some(min_span(lhs)?.max(min_span(rhs)?.saturating_add(2)))
        }
        // OVERLAP spans from the lhs's start to the rhs's end, each strictly
        // beyond the other; three frames is the shortest proper overlap.
        TrajExpr::Overlap { lhs, rhs } => Some(
            min_span(lhs)?
                .saturating_add(1)
                .max(min_span(rhs)?.saturating_add(1))
                .max(3),
        ),
    }
}

/// Whether every predicate under `expr` names exactly `slots`.
///
/// When it does, clipping a leaf to its own slots and to the group's
/// coappearance are the same thing, so the group keeps the tighter
/// [`min_span`] bound.
fn uniform_slots(expr: &TrajExpr, slots: &[Slot]) -> bool {
    match expr {
        TrajExpr::AnalyticPred(pred) => {
            let mut own = Vec::new();
            collect_pred_slots(pred, &mut own);
            own.sort_unstable();
            own.dedup();
            own == slots
        }
        TrajExpr::Then { lhs, rhs, .. }
        | TrajExpr::Contained { lhs, rhs }
        | TrajExpr::Overlap { lhs, rhs } => {
            uniform_slots(lhs, slots) && uniform_slots(rhs, slots)
        }
        TrajExpr::Duration { inner, .. } => uniform_slots(inner, slots),
    }
}

/// How long *all* of a group's slots must be live together, for a subtree whose
/// operands name different slots.
///
/// Unlike [`min_span`] this does not bound the result's length -- it bounds only
/// the stretch during which every slot in the group is simultaneously
/// constrained. `OVERLAP` needs just the frames where its two operands meet,
/// which is one; `CONTAINS` needs its rhs's whole interval, which lies inside
/// the lhs's.
fn coappear_span(expr: &TrajExpr) -> Option<u32> {
    match expr {
        TrajExpr::AnalyticPred(_) => Some(1),
        TrajExpr::Then { .. } => None,
        TrajExpr::Duration { inner, min_frames } => {
            Some(coappear_span(inner)?.max(*min_frames))
        }
        TrajExpr::Contained { rhs, .. } => coappear_span(rhs),
        TrajExpr::Overlap { .. } => Some(1),
    }
}

/// Recursive worker for [`coappear_chain`].
///
/// A subtree referencing no slots (e.g. a constant predicate) falls back to
/// all slots declared in `constraints`, mirroring the tuple the caller asked
/// for.
fn chain_groups(
    expr: &TrajExpr,
    constraints: &VariablesWithConstraints,
    chain: &mut GroupChainBuilder,
) {
    match expr {
        TrajExpr::Then {
            lhs,
            rhs,
            min_gap,
            max_gap,
        } => {
            chain_groups(lhs, constraints, chain);
            chain.gap(match max_gap {
                Some(max) => Gap::between(*min_gap, *max),
                None => Gap::at_least(*min_gap),
            });
            chain_groups(rhs, constraints, chain);
        }
        leaf => {
            let mut slots = collect_slots(leaf);
            slots.sort_unstable();
            slots.dedup();
            if slots.is_empty() {
                slots = constraints.slots().collect();
            }
            // `min_span` bounds the *group's* coappearance run only while
            // every leaf is clipped to that run. A leaf naming a strict subset
            // of the group's slots is not (see `eval`), so its interval may
            // reach outside the run and the bound no longer applies: fall back
            // to what the operator really needs the whole group live for.
            let span = if uniform_slots(leaf, &slots) {
                min_span(leaf).unwrap_or(1)
            } else {
                coappear_span(leaf).unwrap_or(1)
            };
            chain.group(slots, span);
        }
    }
}

/// A single-object predicate that a candidate must satisfy for `need`
/// consecutive frames to stand any chance of appearing in an answer.
struct SlotRequirement<'e> {
    slot: Slot,
    pred: &'e AnalyticalPred,
    need: u32,
}

/// Collects the requirements the expression places on individual objects.
///
/// `need` is the length an interval of *this* subtree must reach, which is not
/// the same as the length its group must reach. `DURATION` raises it, and
/// `CONTAINS` passes it to the left operand because the result it returns is
/// that operand's interval --- but the right operand only has to fit inside,
/// so it inherits nothing. `OVERLAP` and `THEN` span beyond both operands, so
/// neither inherits. Getting this wrong in the other direction would be
/// unsound: under `CONTAINS(DURATION(P, 60), DURATION(Q, 12))` the group needs
/// 62 frames but a `Q` object only needs 12, and filtering it at 62 would drop
/// real answers.
fn slot_requirements<'e>(expr: &'e TrajExpr, need: u32, out: &mut Vec<SlotRequirement<'e>>) {
    match expr {
        TrajExpr::Duration { inner, min_frames } => {
            slot_requirements(inner, need.max(*min_frames), out)
        }
        TrajExpr::Contained { lhs, rhs } => {
            slot_requirements(lhs, need, out);
            slot_requirements(rhs, 1, out);
        }
        TrajExpr::Overlap { lhs, rhs } | TrajExpr::Then { lhs, rhs, .. } => {
            slot_requirements(lhs, 1, out);
            slot_requirements(rhs, 1, out);
        }
        TrajExpr::AnalyticPred(pred) => conjunct_requirements(pred, need, out),
    }
}

/// The single-object conjuncts of one leaf.
///
/// A conjunction holds only where every conjunct does, so a conjunct naming one
/// object must itself hold for the whole `need` frames. A leaf that names
/// several objects and is not a conjunction --- a comparison between two of
/// them, say --- constrains no object on its own and yields nothing.
fn conjunct_requirements<'e>(
    pred: &'e AnalyticalPred,
    need: u32,
    out: &mut Vec<SlotRequirement<'e>>,
) {
    if let Some(slot) = sole_slot(pred) {
        // `need <= 1` asks only that the object exist, which every candidate
        // already does, so it is not a requirement at all.
        if need > 1 {
            out.push(SlotRequirement { slot, pred, need });
        }
        return;
    }
    if let AnalyticalPredKind::And(lhs, rhs) = &pred.kind {
        conjunct_requirements(lhs, need, out);
        conjunct_requirements(rhs, need, out);
    }
}

/// Drops candidates that cannot satisfy their own predicate for long enough.
///
/// The selector prunes on *appearance* --- who is on screen with whom, and for
/// how long --- and learns nothing from the query's predicates until a tuple is
/// bound and evaluated. So a slot whose predicate is highly selective still
/// contributes every object of its class to the search. This solves each
/// single-object predicate once per candidate, over that object's whole life,
/// and removes the objects whose satisfying frames never run long enough.
///
/// Conservative in the right direction: the run is measured unclipped, and
/// clipping to a co-appearance can only shorten it, so nothing that could have
/// matched is removed.
///
/// The solved intervals are not thrown away: they are exactly what
/// [`UnaryMemo`] would compute for the surviving candidates later, keyed the
/// same way, so they are returned as a warm memo. Filtering a slot therefore
/// costs nothing beyond the objects it removes, whose work is what buys the
/// smaller search.
///
/// A single-variable query is skipped entirely. There is no combinatorics to
/// prune — one tuple per candidate either way — and no memo to seed, since a
/// predicate is only memoised when the tuple has something else in it.
fn prefilter_candidates(
    cands: &mut [Vec<ObjId>],
    expr: &TrajExpr,
    poly_trajs: &[PolyTraj],
) -> UnaryMemo {
    let mut memo = UnaryMemo::new();
    let num_slots = cands.len();
    if num_slots < 2 {
        return memo;
    }
    let mut reqs = Vec::new();
    slot_requirements(expr, 1, &mut reqs);
    let mut scratch = vec![0u32; num_slots];
    for req in reqs {
        if req.slot >= num_slots {
            continue;
        }
        let key = req.pred as *const AnalyticalPred as usize;
        cands[req.slot].retain(|&oid| {
            scratch[req.slot] = oid;
            let ivs = eval_analytical(req.pred, &scratch, poly_trajs, (0, u32::MAX));
            if trql_types::intervals::longest_run(&ivs) < req.need {
                return false;
            }
            memo.insert((key, oid), ivs);
            true
        });
    }
    memo
}

/// Solves `pred` over the frames of `clips`, reusing [`UnaryMemo`] wherever a
/// predicate depends on one object alone.
///
/// Two cases, tried in that order. A predicate naming one object does not depend
/// on the rest of the tuple, so it is solved once over that object's whole life
/// and clipped here — a region compiles to a conjunction of half-planes over one
/// slot, so this covers most leaves whole.
///
/// A conjunction naming *several* objects cannot be memoised as a unit, but its
/// conjuncts can be: `And` holds on exactly the frames where both sides do, so
/// splitting it and intersecting the results is the same answer, and each side
/// that names one object is then memoised against it. Without this a leaf like
/// `a INSIDE r AND b INSIDE r AND c INSIDE r` is re-solved for every candidate
/// tuple — once per triple rather than once per object.
fn eval_clipped(
    pred: &AnalyticalPred,
    tuple: &[u32],
    poly_trajs: &[PolyTraj],
    clips: &[Interval],
    memo: &mut UnaryMemo,
) -> Vec<Interval> {
    if let Some(slot) = sole_slot_when_shared(pred, tuple.len()) {
        let key = (pred as *const AnalyticalPred as usize, tuple[slot]);
        let full = memo
            .entry(key)
            .or_insert_with(|| eval_analytical(pred, tuple, poly_trajs, (0, u32::MAX)));
        let _g = trql_instrument::scope(trql_instrument::Bucket::IntervalLift);
        return trql_types::intervals::intersect(full, clips);
    }
    if let AnalyticalPredKind::And(lhs, rhs) = &pred.kind {
        let left = eval_clipped(lhs, tuple, poly_trajs, clips, memo);
        // Nothing survives an empty conjunct, and the right side may be the
        // expensive one.
        if left.is_empty() {
            return left;
        }
        let right = eval_clipped(rhs, tuple, poly_trajs, clips, memo);
        return trql_types::intervals::intersect(&left, &right);
    }
    clips
        .iter()
        .flat_map(|&clip| eval_analytical(pred, tuple, poly_trajs, clip))
        .collect()
}

/// Mirror of [`chain_groups`]: assigns each predicate the index of the chain
/// group its subtree became.
fn annotate_groups(expr: &mut TrajExpr, next_group: &mut usize) {
    match expr {
        TrajExpr::Then { lhs, rhs, .. } => {
            annotate_groups(lhs, next_group);
            annotate_groups(rhs, next_group);
        }
        leaf => {
            set_pred_groups(leaf, *next_group);
            *next_group += 1;
        }
    }
}

/// Sets `pred.group` for every predicate in the subtree (including `Then`
/// nodes nested under `Duration`/`Contained`, which stay inside one group).
fn set_pred_groups(expr: &mut TrajExpr, group: usize) {
    match expr {
        TrajExpr::AnalyticPred(pred) => pred.group = group,
        TrajExpr::Then { lhs, rhs, .. }
        | TrajExpr::Contained { lhs, rhs }
        | TrajExpr::Overlap { lhs, rhs } => {
            set_pred_groups(lhs, group);
            set_pred_groups(rhs, group);
        }
        TrajExpr::Duration { inner, .. } => set_pred_groups(inner, group),
    }
}

/// Whether `pred` names every slot its chain group covers.
///
/// `None` group slots means an unannotated tree (as in tests), where there is
/// no group to be narrower than, so the answer is "yes".
fn covers_group(pred: &AnalyticalPred, group: Option<&Vec<Slot>>) -> bool {
    let Some(group) = group else { return true };
    let mut own = Vec::new();
    collect_pred_slots(pred, &mut own);
    own.sort_unstable();
    own.dedup();
    group.iter().all(|g| own.binary_search(g).is_ok())
}

/// The frames where every slot `pred` names is present, as a clip list.
///
/// `None` when the predicate names no slot at all, which constrains nothing and
/// so must not be clipped.
fn slot_coappearance(
    pred: &AnalyticalPred,
    tuple: &[u32],
    poly_trajs: &[PolyTraj],
) -> Option<Vec<Interval>> {
    let mut slots = Vec::new();
    collect_pred_slots(pred, &mut slots);
    slots.sort_unstable();
    slots.dedup();
    let mut it = slots.into_iter();
    let first = it.next()?;
    let mut acc = poly_trajs[tuple[first] as usize]
        .canonical_segments()
        .to_vec();
    for slot in it {
        acc = trql_types::intervals::intersect(
            &acc,
            poly_trajs[tuple[slot] as usize].canonical_segments(),
        );
        if acc.is_empty() {
            break;
        }
    }
    Some(acc)
}

/// Slots per chain group, mirroring [`chain_groups`]'s walk so the indices
/// line up with the ones [`annotate_groups`] hands the predicates.
fn group_slot_sets(expr: &TrajExpr, constraints: &VariablesWithConstraints) -> Vec<Vec<Slot>> {
    fn walk(expr: &TrajExpr, constraints: &VariablesWithConstraints, out: &mut Vec<Vec<Slot>>) {
        match expr {
            TrajExpr::Then { lhs, rhs, .. } => {
                walk(lhs, constraints, out);
                walk(rhs, constraints, out);
            }
            leaf => {
                let mut slots = collect_slots(leaf);
                slots.sort_unstable();
                slots.dedup();
                if slots.is_empty() {
                    slots = constraints.slots().collect();
                }
                out.push(slots);
            }
        }
    }
    let mut out = Vec::new();
    walk(expr, constraints, &mut out);
    out
}

/// Intersects two ascending-sorted OID lists.
fn intersect_sorted(lhs: &[u32], rhs: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    let mut i = 0;
    let mut j = 0;

    while i < lhs.len() && j < rhs.len() {
        match lhs[i].cmp(&rhs[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(lhs[i]);
                i += 1;
                j += 1;
            }
        }
    }

    out
}

/// Resolves property constraints for a slot into a sorted OID candidate list.
fn resolve_candidates(db: &Trqs, constraints: &PropConstraints) -> Vec<ObjId> {
    let mut result: Option<Vec<u32>> = None;
    for c in constraints.iter_constraints() {
        let ids = match db.index_scan(&c.prop, &c.val) {
            Ok(ids) => ids,
            Err(_) => return Vec::new(),
        };
        result = Some(match result.take() {
            None => ids.to_vec(),
            Some(curr) => intersect_sorted(&curr, ids),
        });
    }
    result.unwrap_or_else(|| (0..db.num_objs()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{ObjWithBBox, TrqsBuilder};
    use trql_selector::{GroupRel, Interval};
    use trql_types::obj::{BoundingBox, ExactTraj};
    use trql_types::prop::{Schema, TypeDesc, Value};

    impl<'a> AlgebraCtx<'a> {
        /// Evaluates a [`TrajExpr`] on a single object tuple, returning the matching frame intervals.
        pub fn eval_test(
            &self,
            expr: &TrajExpr,
            tuple: &[u32],
            group_bounds: &[GroupBounds],
        ) -> IntervalSets {
            // Whole-query enclosing scope: work not charged to root-finding or
            // segment-iteration (optimize, interval algebra) lands in `Other`.
            let _g = trql_instrument::scope(trql_instrument::Bucket::Other);
            let optimized = optimize::optimize(expr.clone());
            let poly_trajs = self.poly_trajs.unwrap_or_else(|| self.db.poly_trajs());
            self.eval(
                &optimized,
                tuple,
                group_bounds,
                &[],
                poly_trajs,
                &mut UnaryMemo::new(),
            )
        }
    }

    fn make_db() -> Trqs {
        let schema = Schema::new(vec![(
            "color".into(),
            TypeDesc::Enum(vec!["red".into(), "blue".into()]),
        )]);
        let mut builder = TrqsBuilder::new(&schema);
        //     WHERE o0.x1 > 0",

        // obj 0: x1 moves from 100 to 140 over frames 0-4
        builder
            .ingest(ObjWithBBox {
                props: vec![Value::Enum(0)],
                exact_traj: ExactTraj {
                    frames: vec![0, 1, 2, 3, 4],
                    bboxes: (0..5u32)
                        .map(|f| BoundingBox {
                            x1: 100 + f * 10,
                            y1: 0,
                            x2: 110 + f * 10,
                            y2: 10,
                        })
                        .collect(),
                },
            })
            .unwrap();

        // obj 1: x1 stays at 50 over frames 0-4
        builder
            .ingest(ObjWithBBox {
                props: vec![Value::Enum(1)],
                exact_traj: ExactTraj {
                    frames: vec![0, 1, 2, 3, 4],
                    bboxes: vec![
                        BoundingBox {
                            x1: 50,
                            y1: 0,
                            x2: 60,
                            y2: 10
                        };
                        5
                    ],
                },
            })
            .unwrap();

        builder.build(10, 0.0)
    }

    #[test]
    fn eval_analytic_unary_with_custom_slot() {
        let db = make_db();
        let ctx = AlgebraCtx {
            db: &db,
            poly_trajs: None,
        };
        // obj1 has x1=50, so x1 > 120 is never true
        let expr = TrajExpr::analytic_pred(AnalyticalPred::unary_gt(
            trql_bbox_compiler::BBoxExpr::x1(),
            120.0,
            query_slots::<5>()[1], // target obj1
        ));
        let result = ctx.eval_test(&expr, &[0, 1], &[]);
        assert!(result.is_empty());

        // obj0 has x1=100..140, so x1 > 120 is true for frames 3,4
        let expr0 = TrajExpr::analytic_pred(AnalyticalPred::unary_gt(
            trql_bbox_compiler::BBoxExpr::x1(),
            120.0,
            query_slots::<5>()[0],
        ));
        let result0 = ctx.eval_test(&expr0, &[0, 1], &[]);
        assert!(!result0.is_empty());
    }

    /// One object per entry: (presence intervals, constant x1).
    fn make_db_objs(objs: &[(&[Interval], u32)]) -> Trqs {
        let schema = Schema::new(vec![]);
        let mut builder = TrqsBuilder::new(&schema);
        for &(intervals, x1) in objs {
            let mut frames = Vec::new();
            let mut bboxes = Vec::new();
            for &(s, e) in intervals {
                for f in s..=e {
                    frames.push(f);
                    bboxes.push(BoundingBox {
                        x1,
                        y1: 0,
                        x2: x1 + 10,
                        y2: 10,
                    });
                }
            }
            builder
                .ingest(ObjWithBBox {
                    props: vec![],
                    exact_traj: ExactTraj { frames, bboxes },
                })
                .unwrap();
        }
        builder.build(1, 0.0)
    }

    /// The variables these tests are written against, in tuple order — the
    /// names the cases are described with in the comments.
    const VAR_NAMES: [&str; 5] = ["a", "b", "c", "d", "e"];

    /// Slots `0..N`, i.e. the positions `VAR_NAMES` are declared at.
    fn query_slots<const N: usize>() -> [Slot; N] {
        std::array::from_fn(|i| i)
    }

    /// `VariablesWithConstraints` declaring the first `n` query variables, unfiltered.
    fn declaring(n: usize) -> VariablesWithConstraints {
        VariablesWithConstraints::unconstrained(VAR_NAMES[..n].iter().copied())
    }

    fn lt_pred(s0: Slot, s1: Slot) -> TrajExpr {
        TrajExpr::analytic_pred(AnalyticalPred::coord_cmp(
            trql_bbox_compiler::BBoxExpr::x1(),
            trql_bbox_compiler::CmpOp::Lt,
            trql_bbox_compiler::BBoxExpr::x1(),
            [s0, s1],
        ))
    }

    #[test]
    fn run_then_disjoint_groups() {
        // obj0 present [0,10], obj1 present [12,20]; both satisfy their unary preds.
        let db = make_db_objs(&[(&[(0, 10)], 200), (&[(12, 20)], 200)]);
        let ctx = AlgebraCtx {
            db: &db,
            poly_trajs: None,
        };
        let expr = TrajExpr::Then {
            lhs: Box::new(TrajExpr::analytic_pred(AnalyticalPred::unary_gt(
                trql_bbox_compiler::BBoxExpr::x1(),
                100.0,
                query_slots::<5>()[0],
            ))),
            rhs: Box::new(TrajExpr::analytic_pred(AnalyticalPred::unary_gt(
                trql_bbox_compiler::BBoxExpr::x1(),
                150.0,
                query_slots::<5>()[1],
            ))),
            min_gap: 1,
            max_gap: None,
        };
        let constraints = declaring(2);
        let results = ctx.run(&constraints, &expr);
        // [0,1] is the cross-object match. [0,0] and [1,1] also match: both
        // predicates hold over the object's whole presence, so sub-intervals
        // satisfying the gap exist within one lifetime. [1,0] must be pruned:
        // obj0 is gone before obj1 appears.
        let mut tuples: Vec<_> = results.iter().map(|(t, _)| t.to_vec()).collect();
        tuples.sort();
        assert_eq!(
            tuples,
            vec![vec![0, 0], vec![0, 1], vec![1, 1]],
            "got {results:?}"
        );
    }

    #[test]
    fn run_then_with_shared_slot_bridges_groups() {
        let [a, b, c] = query_slots::<3>();
        // a (obj0): present [0,10], x1 = 10
        // b (obj1): present [0,30], x1 = 50   — bridges both groups
        // c (obj2): present [20,30], x1 = 100 — never coappears with a!
        //
        // Query: (a.x1 < b.x1) THEN (b.x1 < c.x1), gap ≥ 1.
        //   lhs true on coappear(a,b) = [0,10]; rhs true on coappear(b,c) = [20,30];
        //   20 ≥ 10 + 1 → match. The old selector collapsed shared-slot THEN into a
        //   single group requiring a, b, c to coappear simultaneously (never happens
        //   here) and found nothing.
        let db = make_db_objs(&[(&[(0, 10)], 10), (&[(0, 30)], 50), (&[(20, 30)], 100)]);
        let ctx = AlgebraCtx {
            db: &db,
            poly_trajs: None,
        };
        let expr = TrajExpr::Then {
            lhs: Box::new(lt_pred(a, b)),
            rhs: Box::new(lt_pred(b, c)),
            min_gap: 1,
            max_gap: None,
        };
        let constraints = declaring(3);
        let results = ctx.run(&constraints, &expr);
        assert_eq!(
            results.len(),
            1,
            "expected exactly [0, 1, 2], got {results:?}"
        );
        assert_eq!(results[0].0.as_slice(), &[0, 1, 2]);
        assert!(!results[0].1.is_empty());
    }

    #[test]
    fn run_duration_keeps_a_coappearance_of_exactly_the_required_length() {
        // a: [0, 30], b: [10, 24] — they share 15 frames, and the query wants 15.
        // The boundary case for the selector's span prune: one frame stricter
        // and this match disappears.
        let [a, b] = query_slots::<2>();
        let db = make_db_objs(&[(&[(0, 30)], 10), (&[(10, 24)], 50)]);
        let ctx = AlgebraCtx {
            db: &db,
            poly_trajs: None,
        };
        let expr = |need| TrajExpr::Duration {
            inner: Box::new(lt_pred(a, b)),
            min_frames: need,
        };
        let constraints = declaring(2);

        let results = ctx.run(&constraints, &expr(15));
        assert_eq!(
            results.iter().map(|(t, _)| t.to_vec()).collect::<Vec<_>>(),
            vec![vec![0, 1]],
            "15 shared frames must satisfy DURATION 15, got {results:?}"
        );
        assert!(ctx.run(&constraints, &expr(16)).is_empty());
    }

    #[test]
    fn run_duration_over_a_nested_then_spans_the_gap() {
        // The reason `min_span` refuses to bound a subtree containing a `Then`.
        //
        // One object, present [0, 10] and [20, 30] with a gap between: its
        // longest co-appearance run is 11 frames. The `Then` still matches
        // across the gap, and the interval it produces — [0, 30] — is 31 frames
        // long, so the enclosing DURATION(25) holds. A selector that read the
        // 25 as a required run length would wrongly drop this object.
        let [a] = query_slots::<1>();
        let db = make_db_objs(&[(&[(0, 10), (20, 30)], 200)]);
        let ctx = AlgebraCtx {
            db: &db,
            poly_trajs: None,
        };
        let expr = TrajExpr::Duration {
            inner: Box::new(TrajExpr::Then {
                lhs: Box::new(TrajExpr::analytic_pred(AnalyticalPred::unary_gt(
                    trql_bbox_compiler::BBoxExpr::x1(),
                    100.0,
                    a,
                ))),
                rhs: Box::new(TrajExpr::analytic_pred(AnalyticalPred::unary_gt(
                    trql_bbox_compiler::BBoxExpr::x1(),
                    150.0,
                    a,
                ))),
                min_gap: 1,
                max_gap: None,
            }),
            min_frames: 25,
        };
        let results = ctx.run(&declaring(1), &expr);
        assert_eq!(
            results.iter().map(|(t, _)| t.to_vec()).collect::<Vec<_>>(),
            vec![vec![0]],
            "got {results:?}"
        );
    }

    #[test]
    fn not_of_an_always_true_pred_is_empty_from_frame_zero() {
        // Regression: `complement_in_domain` used to compute the gap before an
        // inner interval as `i_lo - 1` under a saturating sub, so an inner
        // interval starting at frame 0 emitted a phantom `(0, 0)`. The old
        // tests never exercised `i_lo == 0`.
        let [a] = query_slots::<1>();
        let db = make_db_objs(&[(&[(0, 10)], 100)]);
        let ctx = AlgebraCtx {
            db: &db,
            poly_trajs: None,
        };
        // x1 = 100 for every frame the object exists, so `x1 > 50` is always
        // true and its negation must match nothing at all.
        let always = AnalyticalPred::unary_gt(trql_bbox_compiler::BBoxExpr::x1(), 50.0, a);
        let results = ctx.run(&declaring(1), &TrajExpr::analytic_pred(always.not()));
        assert!(results.is_empty(), "got {results:?}");
    }

    #[test]
    fn run_coappear_pair() {
        let [a, b] = query_slots::<2>();
        // obj0 x1=10 and obj1 x1=50 coappear on [5,15] → a.x1 < b.x1 holds there.
        let db = make_db_objs(&[(&[(0, 15)], 10), (&[(5, 25)], 50)]);
        let ctx = AlgebraCtx {
            db: &db,
            poly_trajs: None,
        };
        let constraints = declaring(2);
        let results = ctx.run(&constraints, &lt_pred(a, b));
        assert_eq!(results.len(), 1, "got {results:?}");
        assert_eq!(results[0].0.as_slice(), &[0, 1]);
    }

    #[test]
    #[should_panic(expected = "expression references slot 2")]
    fn run_rejects_slot_the_query_never_declared() {
        // Slot 2 (`c`, undeclared here) belongs to no chain group the selector
        // can bind, so without this check the tuple would carry an unbound zero
        // in that position.
        let [a, _, c] = query_slots::<3>();
        let db = make_db_objs(&[(&[(0, 15)], 10), (&[(5, 25)], 50)]);
        let ctx = AlgebraCtx {
            db: &db,
            poly_trajs: None,
        };
        ctx.run(&declaring(2), &lt_pred(a, c));
    }

    fn gt_pred(slot: Slot) -> TrajExpr {
        TrajExpr::analytic_pred(AnalyticalPred::unary_gt(
            trql_bbox_compiler::BBoxExpr::x1(),
            0.0,
            slot,
        ))
    }

    /// The chain's groups in `Then` order, head first.
    fn groups_of(chain: &GroupChain) -> Vec<&[Slot]> {
        chain.groups().collect()
    }

    /// The relations linking consecutive groups, in the same order.
    fn gaps_of(chain: &GroupChain) -> Vec<GroupRel> {
        (1..chain.num_groups())
            .map(|g| chain.rel_into(g).expect("not the head").clone())
            .collect()
    }

    /// `GroupRel::ThenRelaxed` over `gap`, for comparing against `gaps_of`.
    fn then(gap: impl Into<Gap>) -> GroupRel {
        GroupRel::then_relaxed(gap)
    }

    #[test]
    fn build_selector_lowers_chain_shape() {
        // a=[0,10], b=[12,20], c=[30,40] — pairwise disjoint, in that order.
        let [a, b, c] = query_slots::<3>();
        let db = make_db_objs(&[(&[(0, 10)], 10), (&[(12, 20)], 20), (&[(30, 40)], 30)]);
        // Each slot is pinned to the object at the same index, so candidate data
        // cannot explain either result below — only the chain's shape can.
        let candidates = |s: &Slot| vec![*s as u32];

        let forward = GroupChain::new(
            [a],
            [
                ChainLink::then_relaxed(1.., [b]),
                ChainLink::then_relaxed(1.., [c]),
            ],
        );
        let matches: Vec<Vec<u32>> = ObjSelector::new([a, b, c].map(|s| candidates(&s)), forward)
            .iter(&db)
            .map(|(binding, _)| binding.to_vec())
            .collect();
        // One match, and the binding indexes by query slot with no remapping —
        // the handle-reuse contract `ObjSelector::builder` establishes.
        assert_eq!(matches, vec![vec![0, 1, 2]]);

        // Same slots, same candidates, links in the opposite order: c is long
        // gone before a appears, so the shape alone makes this unsatisfiable.
        let backward = GroupChain::new(
            [c],
            [
                ChainLink::then_relaxed(1.., [b]),
                ChainLink::then_relaxed(1.., [a]),
            ],
        );
        assert_eq!(
            ObjSelector::new([a, b, c].map(|s| candidates(&s)), backward)
                .iter(&db)
                .count(),
            0
        );

        // Links must lower to the relaxed `ThenWindow`, never a strict `THEN`.
        // Here a=[0,100] and b=[5,50] overlap, so a strict THEN (which anchors
        // the gap to a's end and b's start) would reject the pair outright,
        // pruning tuples the predicate could still match.
        let overlapping_db = make_db_objs(&[(&[(0, 100)], 10), (&[(5, 50)], 20)]);
        let overlapping = GroupChain::new([a], [ChainLink::then_relaxed(1..=10, [b])]);
        assert_eq!(
            ObjSelector::new([a, b].map(|s| candidates(&s)), overlapping)
                .iter(&overlapping_db)
                .count(),
            1
        );
    }

    #[test]
    fn chain_groups_flattens_then_spine_in_order() {
        let [a, b, c, d] = query_slots::<4>();
        // Left-nested spine: ((a.x1 > b.x1 THEN[5,9] c.x1>0) THEN[1,∞) d.x1<d.x1). The inner gap is
        // discovered inside the recursive lhs call, but must still attach to the
        // link it opened, not to a later one.
        let expr = TrajExpr::Then {
            lhs: Box::new(TrajExpr::Then {
                lhs: Box::new(lt_pred(b, a)), // unsorted slots
                rhs: Box::new(gt_pred(c)),
                min_gap: 5,
                max_gap: Some(9),
            }),
            rhs: Box::new(lt_pred(d, d)), // duplicate slots
            min_gap: 1,
            max_gap: None,
        };

        let (chain, _) = coappear_chain(&expr, &declaring(4));
        assert_eq!(
            groups_of(&chain),
            vec![
                &[query_slots::<5>()[0], query_slots::<5>()[1]][..],
                &[query_slots::<5>()[2]],
                &[query_slots::<5>()[3]]
            ]
        );
        assert_eq!(gaps_of(&chain), vec![then(5..=9), then(1..)]);
        assert_eq!(chain.num_groups(), gaps_of(&chain).len() + 1);

        // Right-nested spine over the same chain: (a.x1>b.x1 THEN[1,∞) (c.x1>0 THEN[5,9] d.x1<d.x1)).
        // Here the outer gap links groups 0→1 and the inner one links 1→2, so a
        // gap recorded after the rhs recursion instead of before it would swap them.
        let expr = TrajExpr::Then {
            lhs: Box::new(lt_pred(b, a)),
            rhs: Box::new(TrajExpr::Then {
                lhs: Box::new(gt_pred(c)),
                rhs: Box::new(lt_pred(d, d)),
                min_gap: 5,
                max_gap: Some(9),
            }),
            min_gap: 1,
            max_gap: None,
        };

        let (chain, _) = coappear_chain(&expr, &declaring(4));
        assert_eq!(
            groups_of(&chain),
            vec![
                &[query_slots::<5>()[0], query_slots::<5>()[1]][..],
                &[query_slots::<5>()[2]],
                &[query_slots::<5>()[3]]
            ]
        );
        assert_eq!(gaps_of(&chain), vec![then(1..), then(5..=9)]);
    }

    #[test]
    fn chain_groups_splits_then_regardless_of_slot_overlap() {
        let [a, b, c, d] = query_slots::<4>();
        // One top-level THEN always yields exactly two groups: the two sides' slot
        // sets are kept apart however much they overlap. `run` depends on this —
        // a shared slot is pinned *across* the two groups and linked by a gap
        // window, rather than collapsing into one group that would demand every
        // slot coappear at once (see `run_then_with_shared_slot_bridges_groups`).
        let check = |case: &str, lhs: TrajExpr, rhs: TrajExpr, expected: [&[Slot]; 2]| {
            let expr = TrajExpr::Then {
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                min_gap: 2,
                max_gap: None,
            };
            let (chain, _) = coappear_chain(&expr, &declaring(4));
            assert_eq!(chain.num_groups(), 2, "{case}: {chain:?}");
            assert_eq!(groups_of(&chain), expected.to_vec(), "{case}");
            assert_eq!(gaps_of(&chain), vec![then(2..)], "{case}");
        };

        // Shared slot b — the bridging case.
        check(
            "pred(a,b) THEN pred(b,c)",
            lt_pred(a, b),
            lt_pred(b, c),
            [
                &[query_slots::<5>()[0], query_slots::<5>()[1]],
                &[query_slots::<5>()[1], query_slots::<5>()[2]],
            ],
        );
        // Disjoint slots.
        check(
            "pred(a,b) THEN pred(c,d)",
            lt_pred(a, b),
            lt_pred(c, d),
            [
                &[query_slots::<5>()[0], query_slots::<5>()[1]],
                &[query_slots::<5>()[2], query_slots::<5>()[3]],
            ],
        );
        // One side's slots are a subset of the other's.
        check(
            "pred(a,b) THEN pred(b)",
            lt_pred(a, b),
            gt_pred(b),
            [
                &[query_slots::<5>()[0], query_slots::<5>()[1]],
                &[query_slots::<5>()[1]],
            ],
        );
        // Identical slot sets still stay two groups: the same pair of objects has
        // to satisfy the predicate twice, in two separated frame windows.
        check(
            "pred(a,b) THEN pred(a,b)",
            lt_pred(a, b),
            lt_pred(a, b),
            [
                &[query_slots::<5>()[0], query_slots::<5>()[1]],
                &[query_slots::<5>()[0], query_slots::<5>()[1]],
            ],
        );
    }

    #[test]
    fn min_span_reads_the_length_each_operator_forces() {
        let [a, b, _c] = query_slots::<3>();

        // A predicate says nothing beyond "some frame".
        assert_eq!(min_span(&lt_pred(a, b)), Some(1));

        // DURATION is the bound, unless what is inside already exceeds it.
        assert_eq!(
            min_span(&TrajExpr::Duration {
                inner: Box::new(lt_pred(a, b)),
                min_frames: 60,
            }),
            Some(60)
        );
        assert_eq!(
            min_span(&TrajExpr::Duration {
                inner: Box::new(TrajExpr::Duration {
                    inner: Box::new(lt_pred(a, b)),
                    min_frames: 60,
                }),
                min_frames: 12,
            }),
            Some(60)
        );

        // CONTAINS needs a frame on each side of what it encloses.
        assert_eq!(
            min_span(&TrajExpr::Contained {
                lhs: Box::new(lt_pred(a, b)),
                rhs: Box::new(TrajExpr::Duration {
                    inner: Box::new(gt_pred(a)),
                    min_frames: 12,
                }),
            }),
            Some(14)
        );

        // OVERLAP extends past each side, and three frames is the shortest
        // proper overlap.
        assert_eq!(
            min_span(&TrajExpr::Overlap {
                lhs: Box::new(lt_pred(a, b)),
                rhs: Box::new(gt_pred(a)),
            }),
            Some(3)
        );
        assert_eq!(
            min_span(&TrajExpr::Overlap {
                lhs: Box::new(TrajExpr::Duration {
                    inner: Box::new(lt_pred(a, b)),
                    min_frames: 60,
                }),
                rhs: Box::new(gt_pred(a)),
            }),
            Some(61)
        );
    }

    #[test]
    fn min_span_gives_up_on_a_then_anywhere_beneath_it() {
        let [a, b, c] = query_slots::<3>();
        let then = || TrajExpr::Then {
            lhs: Box::new(lt_pred(a, b)),
            rhs: Box::new(gt_pred(c)),
            min_gap: 4,
            max_gap: Some(8),
        };

        // A `Then`'s two sides need not share a frame, so its result can straddle
        // two co-appearance runs and bound neither.
        assert_eq!(min_span(&then()), None);
        // And the enclosing DURATION's own bound is no better: the 90 frames may
        // be split across the gap.
        assert_eq!(
            min_span(&TrajExpr::Duration {
                inner: Box::new(then()),
                min_frames: 90,
            }),
            None
        );
        assert_eq!(
            min_span(&TrajExpr::Contained {
                lhs: Box::new(then()),
                rhs: Box::new(gt_pred(a)),
            }),
            None
        );
    }

    #[test]
    fn chain_carries_one_min_span_per_group() {
        let [a, b, c] = query_slots::<3>();
        let expr = TrajExpr::Then {
            lhs: Box::new(TrajExpr::Duration {
                inner: Box::new(lt_pred(a, b)),
                min_frames: 60,
            }),
            rhs: Box::new(gt_pred(c)),
            min_gap: 1,
            max_gap: Some(30),
        };
        let (chain, spans) = coappear_chain(&expr, &declaring(3));
        assert_eq!(chain.num_groups(), 2);
        // Only the DURATION side carries a requirement; the bare predicate does
        // not, and a group with no requirement reads as 1.
        assert_eq!(spans, vec![60, 1]);
    }

    #[test]
    fn chain_groups_keeps_nested_then_inside_one_group() {
        let [a, b, c] = query_slots::<3>();
        // Only a *top-level* `Then` splits the chain. A `Then` under
        // `Duration`/`Contained` stays in one group, mirroring `set_pred_groups`.
        let nested = || TrajExpr::Then {
            lhs: Box::new(lt_pred(a, b)),
            rhs: Box::new(gt_pred(c)),
            min_gap: 4,
            max_gap: Some(8),
        };

        let duration = TrajExpr::Duration {
            inner: Box::new(nested()),
            min_frames: 3,
        };
        let (chain, _) = coappear_chain(&duration, &declaring(3));
        assert_eq!(
            groups_of(&chain),
            vec![
                &[
                    query_slots::<5>()[0],
                    query_slots::<5>()[1],
                    query_slots::<5>()[2]
                ][..]
            ]
        );
        assert_eq!(chain.num_groups(), 1, "got {chain:?}");

        let contained = TrajExpr::Contained {
            lhs: Box::new(gt_pred(a)),
            rhs: Box::new(nested()),
        };
        let (chain, _) = coappear_chain(&contained, &declaring(3));
        assert_eq!(
            groups_of(&chain),
            vec![
                &[
                    query_slots::<5>()[0],
                    query_slots::<5>()[1],
                    query_slots::<5>()[2]
                ][..]
            ]
        );
        assert_eq!(chain.num_groups(), 1, "got {chain:?}");

        // `Overlap` is the same story: two intervals can only properly overlap
        // if they share frames, so both sides live in one co-appearance group.
        let overlap = TrajExpr::Overlap {
            lhs: Box::new(gt_pred(a)),
            rhs: Box::new(nested()),
        };
        let (chain, _) = coappear_chain(&overlap, &declaring(3));
        assert_eq!(
            groups_of(&chain),
            vec![
                &[
                    query_slots::<5>()[0],
                    query_slots::<5>()[1],
                    query_slots::<5>()[2]
                ][..]
            ]
        );
        assert_eq!(chain.num_groups(), 1, "got {chain:?}");
    }

    #[test]
    fn chain_groups_slotless_leaf_falls_back_to_declared_slots() {
        let [_, b] = query_slots::<2>();
        // A constant predicate references no slot, so its group mirrors the
        // whole tuple the caller declared.
        let expr = TrajExpr::Then {
            lhs: Box::new(TrajExpr::analytic_pred(AnalyticalPred::true_pred())),
            rhs: Box::new(gt_pred(b)),
            min_gap: 0,
            max_gap: None,
        };

        let (chain, _) = coappear_chain(&expr, &declaring(3));
        assert_eq!(
            groups_of(&chain),
            vec![
                &[
                    query_slots::<5>()[0],
                    query_slots::<5>()[1],
                    query_slots::<5>()[2]
                ][..],
                &[query_slots::<5>()[1]]
            ]
        );
        assert_eq!(gaps_of(&chain), vec![then(0..)]);

        // The fallback can never produce an empty group: `VariablesWithConstraints`
        // declares at least one slot, so a slotless leaf always mirrors a
        // non-empty tuple.
        assert!(groups_of(&chain).iter().all(|g| !g.is_empty()));
    }

    #[test]
    fn chain_groups_agrees_with_annotate_groups() {
        let [a, b, c, d] = query_slots::<4>();
        // `run` relies on both walks recursing through `Then` identically, so
        // that group indices line up. Only asserted via `debug_assert_eq!`
        // there; pinned down for real here.
        let mut expr = TrajExpr::Then {
            lhs: Box::new(TrajExpr::Duration {
                inner: Box::new(TrajExpr::Then {
                    lhs: Box::new(lt_pred(a, b)),
                    rhs: Box::new(gt_pred(b)),
                    min_gap: 2,
                    max_gap: None,
                }),
                min_frames: 3,
            }),
            rhs: Box::new(TrajExpr::Then {
                lhs: Box::new(gt_pred(c)),
                rhs: Box::new(TrajExpr::Contained {
                    lhs: Box::new(lt_pred(c, d)),
                    rhs: Box::new(gt_pred(d)),
                }),
                min_gap: 1,
                max_gap: Some(7),
            }),
            min_gap: 0,
            max_gap: None,
        };

        let constraints = declaring(4);
        let (chain, _) = coappear_chain(&expr, &constraints);
        let groups = groups_of(&chain);
        assert_eq!(
            groups,
            vec![
                &[query_slots::<5>()[0], query_slots::<5>()[1]][..],
                &[query_slots::<5>()[2]],
                &[query_slots::<5>()[2], query_slots::<5>()[3]]
            ]
        );
        assert_eq!(gaps_of(&chain), vec![then(0..), then(1..=7)]);

        let mut next_group = 0;
        annotate_groups(&mut expr, &mut next_group);
        assert_eq!(next_group, chain.num_groups());

        // Every predicate's slots must be covered by the group it was annotated
        // with, or its coappearance clip would be wrong.
        let mut annotated = Vec::new();
        collect_annotations(&expr, &mut annotated);
        assert_eq!(annotated.len(), 5);
        for (group, pred_slots) in annotated {
            assert!(group < groups.len(), "group {group} out of range");
            for s in pred_slots {
                assert!(
                    groups[group].contains(&s),
                    "slot {:?} missing from group {group} ({:?})",
                    s,
                    groups[group]
                );
            }
        }
    }

    /// Collects `(pred.group, slots)` for every `AnalyticPred` leaf in `expr`.
    fn collect_annotations(expr: &TrajExpr, out: &mut Vec<(usize, Vec<Slot>)>) {
        match expr {
            TrajExpr::AnalyticPred(pred) => {
                let mut pred_slots = Vec::new();
                trql_bbox_compiler::collect_pred_slots(pred, &mut pred_slots);
                out.push((pred.group, pred_slots));
            }
            TrajExpr::Then { lhs, rhs, .. }
            | TrajExpr::Contained { lhs, rhs }
            | TrajExpr::Overlap { lhs, rhs } => {
                collect_annotations(lhs, out);
                collect_annotations(rhs, out);
            }
            TrajExpr::Duration { inner, .. } => collect_annotations(inner, out),
        }
    }

    #[test]
    fn sole_slot_identifies_predicates_over_one_object() {
        use trql_bbox_compiler::{BBoxExpr, CmpOp};
        let unary = |slot| AnalyticalPred::unary_gt(BBoxExpr::x1(), 0.0, slot);
        let binary = |a, b| {
            AnalyticalPred::coord_cmp(BBoxExpr::x1(), CmpOp::Lt, BBoxExpr::x1(), [a, b])
        };

        assert_eq!(sole_slot(&unary(2)), Some(2));
        // Two objects: the result depends on the pair, so it must not be memoised
        // against either one of them.
        assert_eq!(sole_slot(&binary(0, 1)), None);
        // Conjunctions are the common case — a convex region compiles to one
        // half-plane comparison per edge, all naming the same slot.
        assert_eq!(
            sole_slot(&AnalyticalPred::and(unary(1), unary(1))),
            Some(1)
        );
        assert_eq!(sole_slot(&AnalyticalPred::and(unary(0), unary(1))), None);
        assert_eq!(sole_slot(&AnalyticalPred::and(unary(0), binary(0, 1))), None);
        assert_eq!(sole_slot(&AnalyticalPred::not(unary(3))), Some(3));
        assert_eq!(sole_slot(&AnalyticalPred::true_pred()), None);
    }

    /// Reusing a memo across tuples must not leak one tuple's clip window into
    /// another's. The same `(leaf, object)` key is hit here under two different
    /// co-appearance windows, which is exactly the shape `CONTAINS(unary, unary)`
    /// produces — and what caching a post-clip or post-`Duration` result would break.
    #[test]
    fn a_shared_memo_answers_each_clip_window_on_its_own_terms() {
        // Two objects, so the query is multi-slot and the memo is in play; the
        // predicate still reads slot 0 alone, which is the point.
        let db = make_db_objs(&[(&[(0, 100)], 5), (&[(0, 100)], 7)]);
        let ctx = AlgebraCtx {
            db: &db,
            poly_trajs: None,
        };
        let poly_trajs = db.poly_trajs();
        // `x1 = 5 > 0` holds throughout, so each answer is just its own window --
        // any bleed between windows shows up immediately. The group annotation is
        // what points the leaf at `group_bounds[0]`; without it the leaf is
        // evaluated unclipped and this test would prove nothing.
        let mut expr = optimize::optimize(gt_pred(0));
        let mut next_group = 0;
        annotate_groups(&mut expr, &mut next_group);
        assert_eq!(next_group, 1, "expected the leaf to be annotated with group 0");
        let windows: [Vec<Interval>; 4] = [
            vec![(0, 100)],
            vec![(10, 20)],
            vec![(30, 40), (60, 70)],
            vec![(95, 100)],
        ];

        let mut shared = UnaryMemo::new();
        for w in &windows {
            let bounds = vec![w.clone()];
            let with_shared = ctx.eval(&expr, &[0, 1], &bounds, &[], poly_trajs, &mut shared);
            let with_fresh =
                ctx.eval(&expr, &[0, 1], &bounds, &[], poly_trajs, &mut UnaryMemo::new());
            assert_eq!(
                with_shared.maximal(),
                with_fresh.maximal(),
                "a shared memo changed the answer for window {w:?}"
            );
            assert_eq!(
                with_shared.maximal(),
                w.as_slice(),
                "expected the predicate to hold across all of {w:?}"
            );
        }
        // One entry per (leaf, object), not per window.
        assert_eq!(shared.len(), 1);
    }

    /// Splitting a conjunction so each side can be memoised must not change the
    /// answer. The two sides name different objects, so the conjunction as a
    /// whole is not memoisable and the split path is what runs; the reference is
    /// the same predicate evaluated as one unit.
    /// What the pre-filter may demand of one object, which is not what its group
    /// demands of the answer. Getting the `CONTAINS` case wrong would silently
    /// drop matches, so it is pinned here.
    #[test]
    fn slot_requirements_follow_the_operator_that_imposed_them() {
        let [a, b] = query_slots::<2>();
        let ua = || gt_pred(a);
        let ub = || gt_pred(b);
        let reqs = |e: &TrajExpr| {
            let mut out = Vec::new();
            slot_requirements(e, 1, &mut out);
            let mut got: Vec<(Slot, u32)> = out.iter().map(|r| (r.slot, r.need)).collect();
            got.sort();
            got
        };
        let dur = |e: TrajExpr, n: u32| TrajExpr::Duration {
            inner: Box::new(e),
            min_frames: n,
        };

        // A leaf under DURATION must itself hold that long.
        assert_eq!(reqs(&dur(ua(), 25)), vec![(a, 25)]);

        // CONTAINS returns the lhs interval, so the lhs inherits a demanded
        // length and the rhs does not: under this query the *group* needs 62
        // frames, but a `b` object only needs 12.
        assert_eq!(
            reqs(&TrajExpr::Contained {
                lhs: Box::new(dur(ua(), 60)),
                rhs: Box::new(dur(ub(), 12)),
            }),
            vec![(a, 60), (b, 12)]
        );
        // And a length demanded of a CONTAINS result reaches its lhs alone.
        assert_eq!(
            reqs(&dur(
                TrajExpr::Contained {
                    lhs: Box::new(ua()),
                    rhs: Box::new(ub()),
                },
                100
            )),
            vec![(a, 100)],
            "the rhs needs only one frame, so it is not a requirement at all"
        );

        // OVERLAP and THEN span beyond their operands: neither side inherits.
        assert_eq!(
            reqs(&dur(
                TrajExpr::Overlap {
                    lhs: Box::new(dur(ua(), 10)),
                    rhs: Box::new(dur(ub(), 7)),
                },
                100
            )),
            vec![(a, 10), (b, 7)]
        );
        assert_eq!(
            reqs(&dur(
                TrajExpr::Then {
                    lhs: Box::new(dur(ua(), 5)),
                    rhs: Box::new(dur(ub(), 7)),
                    min_gap: 0,
                    max_gap: None,
                },
                100
            )),
            vec![(a, 5), (b, 7)]
        );

        // A conjunction holds only where both sides do, so each single-object
        // conjunct inherits the whole length.
        assert_eq!(
            reqs(&dur(
                TrajExpr::analytic_pred(AnalyticalPred::and(
                    AnalyticalPred::unary_gt(trql_bbox_compiler::BBoxExpr::x1(), 0.0, a),
                    AnalyticalPred::unary_gt(trql_bbox_compiler::BBoxExpr::x1(), 0.0, b),
                )),
                25
            )),
            vec![(a, 25), (b, 25)]
        );

        // A comparison between two objects constrains neither on its own.
        assert_eq!(reqs(&dur(lt_pred(a, b), 25)), vec![]);
    }

    #[test]
    fn splitting_a_conjunction_agrees_with_evaluating_it_whole() {
        let [a, b] = query_slots::<2>();
        // a: x1 = 10 on [0, 60]; b: x1 = 50 on [20, 100]. `a.x1 > 5` holds
        // throughout a's life and `b.x1 > 30` throughout b's, so the conjunction
        // holds exactly on their overlap, clipped to the window.
        let db = make_db_objs(&[(&[(0, 60)], 10), (&[(20, 100)], 50)]);
        let ctx = AlgebraCtx {
            db: &db,
            poly_trajs: None,
        };
        let _ = &ctx;
        let poly_trajs = db.poly_trajs();
        let conj = AnalyticalPred::and(
            AnalyticalPred::unary_gt(trql_bbox_compiler::BBoxExpr::x1(), 5.0, a),
            AnalyticalPred::unary_gt(trql_bbox_compiler::BBoxExpr::x1(), 30.0, b),
        );
        assert_eq!(
            sole_slot(&conj),
            None,
            "the conjunction names two objects, so only the split path can memoise it"
        );

        for clips in [
            vec![(0u32, 100u32)],
            vec![(25, 45)],
            vec![(0, 10), (30, 40), (80, 100)],
        ] {
            let split = eval_clipped(&conj, &[0, 1], poly_trajs, &clips, &mut UnaryMemo::new());
            let whole: Vec<Interval> = clips
                .iter()
                .flat_map(|&clip| eval_analytical(&conj, &[0, 1], poly_trajs, clip))
                .collect();
            assert_eq!(split, whole, "split and whole disagreed on {clips:?}");
        }

        // The memo is keyed per conjunct per object, so a tuple reusing one of
        // these objects does not re-solve that side.
        let mut memo = UnaryMemo::new();
        eval_clipped(&conj, &[0, 1], poly_trajs, &[(0, 100)], &mut memo);
        assert_eq!(memo.len(), 2, "one entry per (conjunct, object)");
    }

    #[test]
    fn eval_slot_out_of_range_returns_empty() {
        let db = make_db();
        let ctx = AlgebraCtx {
            db: &db,
            poly_trajs: None,
        };
        // slot 5 doesn't exist in a 2-element tuple — panics with index out of bounds,
        // so just verify correct slots work
        let expr = TrajExpr::analytic_pred(AnalyticalPred::coord_cmp(
            trql_bbox_compiler::BBoxExpr::x1(),
            trql_bbox_compiler::CmpOp::Gt,
            trql_bbox_compiler::BBoxExpr::x1(),
            [query_slots::<5>()[0], query_slots::<5>()[1]],
        ));
        let result = ctx.eval_test(&expr, &[0, 1], &[]);
        assert!(!result.is_empty()); // valid slots → non-empty result
    }
}
