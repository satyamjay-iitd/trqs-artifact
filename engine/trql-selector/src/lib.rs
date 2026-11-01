mod lazy;
pub mod synth;
mod v2;
pub use lazy::CandidateCursor;
use trql_types::{Gap, Slot};
pub use v2::ObjSelectorIterV2;

use smallvec::SmallVec;

pub use trql_types::ObjId;
pub use trql_types::intervals::Interval;

/// Full object binding across all slots: `binding[slot_idx]` = the ObjId for that slot.
pub type Binding = SmallVec<[ObjId; 4]>;

/// Coappearance intervals of one chain group — the frames on which every object
/// of that group is simultaneously present.
///
/// Handed to the caller by `CandidateCursor::group_bounds`, which expects it to
/// become the clip domain for whatever predicate that group carries: evaluation
/// then never looks outside a window where the objects actually co-appear.
pub type GroupBounds = Vec<Interval>;

/// One `GroupBounds` per group in the chain.
pub type ChainBounds = SmallVec<[GroupBounds; 4]>;

// ---- GroupRel -----------------------------------------------------------------------

/// Temporal relationship between two consecutive coappearance groups.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GroupRel {
    /// Window-relaxed THEN, for pruning tuples before an exact temporal
    /// predicate is evaluated on the survivors: some interval of group B must
    /// overlap the window `[A.start + min_gap, A.end + max_gap]` for some
    /// interval A of group A.
    ///
    /// This is the appearance-level *necessary* condition for a
    /// predicate-level `lhs THEN rhs` match: the lhs predicate may become
    /// false anywhere inside A (so the gap window opens as early as
    /// `A.start + min_gap`) and the rhs predicate may become true anywhere
    /// inside B. An exact appearance-level THEN — anchoring the gap to A's end
    /// and B's start — would wrongly prune such tuples, so it is deliberately
    /// not offered.
    ThenRelaxed { min_gap: u32, max_gap: Option<u32> },
    /// Group B's intervals must be fully contained within group A's intervals.
    Contains,
}

impl GroupRel {
    /// [`GroupRel::ThenRelaxed`] over `gap`, which accepts ranges and integers
    /// — see [`Gap`].
    pub fn then_relaxed(gap: impl Into<Gap>) -> Self {
        let gap = gap.into();
        GroupRel::ThenRelaxed {
            min_gap: gap.min,
            max_gap: gap.max,
        }
    }
}

// ---- Plan ---------------------------------------------------------------------

/// One step of the chain: how this group relates to the one before it, and the
/// slots that must coappear in it.
///
/// The relation is owned by the group it links *to*, so it is never optional —
/// only the head group has none, and the head is stored separately.
#[derive(Clone, Debug, PartialEq)]
pub struct ChainLink {
    pub rel: GroupRel,
    pub slots: Vec<Slot>,
}

impl ChainLink {
    /// A window-relaxed THEN into `slots` ([`GroupRel::ThenRelaxed`]): this
    /// group must have presence overlapping `[prev.start + gap.min,
    /// prev.end + gap.max]`. `gap` accepts ranges and integers — see [`Gap`].
    ///
    /// The chain is a conservative prefilter, so this is the only THEN offered:
    /// an exact one would prune tuples whose predicate still matches somewhere
    /// inside the groups. See [`GroupRel::ThenRelaxed`].
    pub fn then_relaxed(gap: impl Into<Gap>, slots: impl IntoIterator<Item = Slot>) -> Self {
        Self {
            rel: GroupRel::then_relaxed(gap),
            slots: slots.into_iter().collect(),
        }
    }

    /// A CONTAINS step into `slots`: this group's intervals must be fully
    /// contained within the previous group's.
    pub fn containing(slots: impl IntoIterator<Item = Slot>) -> Self {
        Self {
            rel: GroupRel::Contains,
            slots: slots.into_iter().collect(),
        }
    }
}

/// The shape of a selection: a head coappearance group followed by one
/// [`ChainLink`] per subsequent group.
///
/// This is the query planner's output and carries no data — deriving it needs
/// only the expression and the declared variables, never a database. Pair it
/// with per-slot candidates via [`ObjSelector::new`] to get a runnable plan.
///
/// Storing the head separately from the links makes two invariants structural:
/// a chain always has at least one group, and each relation is owned by the
/// group it links to rather than living in a parallel `Vec` that has to be
/// zipped back into position.
#[derive(Clone, Debug, PartialEq)]
pub struct GroupChain {
    head: Vec<Slot>,
    links: Vec<ChainLink>,
}

impl GroupChain {
    /// Panics if any group is empty — a group with no slots can never bind.
    pub fn new(
        head: impl IntoIterator<Item = Slot>,
        links: impl IntoIterator<Item = ChainLink>,
    ) -> Self {
        let chain = GroupChain {
            head: head.into_iter().collect(),
            links: links.into_iter().collect(),
        };
        for slots in chain.groups() {
            assert!(!slots.is_empty(), "a group must contain at least one slot");
        }
        chain
    }

    /// Number of coappearance groups — one more than the number of links.
    pub fn num_groups(&self) -> usize {
        self.links.len() + 1
    }

    /// Groups in chain order, head first.
    pub fn groups(&self) -> impl Iterator<Item = &[Slot]> {
        std::iter::once(self.head.as_slice()).chain(self.links.iter().map(|l| l.slots.as_slice()))
    }

    /// The slots of group `g`, head first.
    pub fn group_slots(&self, g: usize) -> &[Slot] {
        match g.checked_sub(1) {
            None => &self.head,
            Some(i) => &self.links[i].slots,
        }
    }

    /// The relation linking group `g - 1` to group `g`; `None` for the head,
    /// which has no predecessor.
    pub fn rel_into(&self, g: usize) -> Option<&GroupRel> {
        g.checked_sub(1).map(|i| &self.links[i].rel)
    }

    /// The first slot the chain references that a `num_slots`-wide tuple cannot
    /// hold. Checked before binding candidates, so the caller can report it in
    /// its own terms.
    pub fn slot_beyond(&self, num_slots: usize) -> Option<Slot> {
        self.groups().flatten().copied().find(|&s| s >= num_slots)
    }
}

/// A runnable object-selection plan: a [`GroupChain`] plus the candidate ObjIds
/// each slot may bind.
///
/// Built with [`ObjSelector::new`]; consumed via [`ObjSelector::candidates`]
/// (zero-copy lending cursor over candidates) or [`ObjSelector::iter`]
/// (owned items).
pub struct ObjSelector {
    chain: GroupChain,
    num_slots: usize,
    /// `slot_2_objects[s]` = candidate ObjIds for slot `s` (already filtered by
    /// property constraints).
    slot_2_objects: Vec<Vec<ObjId>>,
    /// Per group, the shortest co-appearance run that could still satisfy it.
    ///
    /// A group whose subtree demands `DURATION(P, k)` cannot match unless its
    /// variables share `k` consecutive frames, because `P` can only hold where
    /// they are all present. Enumerating such a tuple is wasted work: the length
    /// test would otherwise be applied by the interval algebra, after the
    /// predicates it makes irrelevant have already been solved.
    ///
    /// Empty, or `1` for a group, means "no requirement beyond co-appearing at
    /// all" — which is what an unannotated chain gets, so a caller that does not
    /// supply these loses nothing but the pruning.
    min_span: Vec<u32>,
}

// ---- Construction --------------------------------------------------------------

impl ObjSelector {
    /// Pairs `chain` with `slot_2_objects` — one entry per query slot, in tuple
    /// order, holding that slot's candidate ObjIds (the property-scan result).
    ///
    /// The selector does not assign slots — whoever built the query did that
    /// once, when every variable was known. Taking them all at once makes that
    /// ordering structural: a slot *is* its position here, so a binding is
    /// indexed by query slot with no remapping. The chain arrives already
    /// shaped, so nothing is translated or re-derived here.
    ///
    /// Panics if a group names a slot with no candidates.
    ///
    /// ```
    /// use trql_selector::{ChainLink, GroupChain, ObjSelector};
    /// use trql_selector::synth::{SynthConfig, gen_db};
    /// use trql_types::Slot;
    ///
    /// let db = gen_db(&SynthConfig {
    ///     num_objs: 10, timeline: 1_000, segs_per_obj: (1, 2), seg_len: (10, 50), seed: 1,
    /// });
    ///
    /// // Slots come from the query: they are tuple positions, named there.
    /// let (cars, peds): (Slot, Slot) = (0, 1);
    ///
    /// // The chain is the plan's shape; it needs no database to build.
    /// let chain = GroupChain::new(
    ///     [cars, peds],                               // car and ped coappear,
    ///     [ChainLink::then_relaxed(1..=100, [peds])], // then ped reappears
    /// );
    /// let sel = ObjSelector::new([0..5, 5..10], chain); // + candidates per slot
    ///
    /// let mut m = sel.candidates(&db);
    /// while m.advance() {
    ///     let _car_id = m.get(cars);
    ///     let _coappear_intervals = m.group_bounds(0);
    /// }
    /// ```
    pub fn new<C>(slot_2_objects: impl IntoIterator<Item = C>, chain: GroupChain) -> Self
    where
        C: IntoIterator<Item = ObjId>,
    {
        let slot_2_objects: Vec<Vec<ObjId>> = slot_2_objects
            .into_iter()
            .map(|c| c.into_iter().collect())
            .collect();
        let num_slots = slot_2_objects.len();
        if let Some(s) = chain.slot_beyond(num_slots) {
            panic!(
                "slot #{s} has no candidates: only {num_slots} were given to `ObjSelector::new`"
            );
        }
        ObjSelector {
            chain,
            num_slots,
            slot_2_objects,
            min_span: Vec::new(),
        }
    }

    /// Declares the shortest co-appearance run each group could still satisfy.
    ///
    /// Optional, and omitting it only forgoes the pruning — see
    /// [`ObjSelector::min_span`]. A value of `0` or `1` for a group means the
    /// same thing: co-appearing at all is enough.
    ///
    /// Panics if `min_span` does not have one entry per group, since a silently
    /// mismatched vector would prune the wrong groups.
    pub fn with_min_span(mut self, min_span: impl IntoIterator<Item = u32>) -> Self {
        let min_span: Vec<u32> = min_span.into_iter().collect();
        assert_eq!(
            min_span.len(),
            self.num_groups(),
            "min_span must have one entry per group"
        );
        self.min_span = min_span;
        self
    }

    /// The shortest co-appearance run group `g` could still satisfy; `1` when
    /// nothing was declared.
    pub(crate) fn min_span(&self, g: usize) -> u32 {
        self.min_span.get(g).copied().unwrap_or(1).max(1)
    }

    /// The plan's shape.
    pub fn chain(&self) -> &GroupChain {
        &self.chain
    }

    /// Number of coappearance groups — one more than the number of links.
    pub fn num_groups(&self) -> usize {
        self.chain.num_groups()
    }
}

// ---- Interval compatibility -----------------------------------------------------

/// Returns `true` if any pair `(prev_iv, curr_iv)` satisfies the `GroupRel` constraint.
pub(crate) fn intervals_compatible(prev: &[Interval], curr: &[Interval], op: &GroupRel) -> bool {
    match op {
        GroupRel::ThenRelaxed { min_gap, max_gap } => {
            for &(ps, pe) in prev {
                for &(cs, ce) in curr {
                    if ce < ps.saturating_add(*min_gap) {
                        continue;
                    }
                    if let Some(mg) = max_gap {
                        if cs > pe.saturating_add(*mg) {
                            continue;
                        }
                    }
                    return true;
                }
            }
            false
        }
        GroupRel::Contains => {
            for &(ps, pe) in prev {
                for &(cs, ce) in curr {
                    if ps <= cs && ce <= pe {
                        return true;
                    }
                }
            }
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trql_storage::db::{ObjWithBBox, Trqs, TrqsBuilder};
    use trql_types::obj::{BoundingBox, ExactTraj};
    use trql_types::prop::Schema;

    fn dummy_bbox() -> BoundingBox {
        BoundingBox {
            x1: 0,
            y1: 0,
            x2: 10,
            y2: 10,
        }
    }

    /// Build an ExactTraj that is present during each of the given `(start, end)` intervals
    /// (consecutive frames within each interval, gap between intervals > 1).
    fn make_exact_traj(intervals: &[Interval]) -> ExactTraj {
        let mut frames = Vec::new();
        let mut bboxes = Vec::new();
        for &(start, end) in intervals {
            for f in start..=end {
                frames.push(f);
                bboxes.push(dummy_bbox());
            }
        }
        ExactTraj { frames, bboxes }
    }

    /// Build a Trqs with one object per entry in `obj_intervals`.
    /// gap_threshold=1 means a gap of 2+ frames between consecutive observed frames
    /// creates a new canonical segment.
    fn make_db(obj_intervals: &[&[Interval]]) -> Trqs {
        let schema = Schema::new(vec![]);
        let mut builder = TrqsBuilder::new(&schema);
        for &intervals in obj_intervals {
            builder
                .ingest(ObjWithBBox {
                    props: vec![],
                    exact_traj: make_exact_traj(intervals),
                })
                .unwrap();
        }
        builder.build(1, 0.0)
    }

    // ---- single group, single slot -----------------------------------------------

    #[test]
    fn single_slot_single_object() {
        let db = make_db(&[&[(0, 10)]]);
        let s0 = 0;
        let sel = ObjSelector::new([[0]], GroupChain::new([s0], []));
        let results: Vec<(SmallVec<[u32; 4]>, SmallVec<[Vec<Interval>; 4]>)> =
            sel.iter(&db).collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0[0], 0);
        assert_eq!(results[0].1[0], vec![(0, 10)]);
    }

    #[test]
    fn single_slot_two_objects_both_returned() {
        let db = make_db(&[&[(0, 10)], &[(5, 15)]]);
        let s0 = 0;
        let sel = ObjSelector::new([[0, 1]], GroupChain::new([s0], []));
        let mut results: Vec<_> = sel.iter(&db).collect();
        results.sort_by_key(|(b, _)| b[0]);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0[0], 0);
        assert_eq!(results[0].1[0], vec![(0, 10)]);
        assert_eq!(results[1].0[0], 1);
        assert_eq!(results[1].1[0], vec![(5, 15)]);
    }

    #[test]
    fn single_slot_empty_candidate_list_yields_nothing() {
        let db = make_db(&[&[(0, 10)]]);
        let s0 = 0;
        let sel = ObjSelector::new([std::iter::empty()], GroupChain::new([s0], []));
        assert_eq!(sel.iter(&db).count(), 0);
    }

    // ---- two slots, coappearance -------------------------------------------------

    #[test]
    fn two_slots_coappear_yields_both_orderings() {
        // A=[0,10], B=[5,15] → coappear at [5,10]
        // both assignments (slot0=A, slot1=B) and (slot0=B, slot1=A) are valid
        let db = make_db(&[&[(0, 10)], &[(5, 15)]]);
        let s0 = 0;
        let s1 = 1;
        let sel = ObjSelector::new([[0, 1], [0, 1]], GroupChain::new([s0, s1], []));
        let results: Vec<_> = sel.iter(&db).collect();
        assert_eq!(results.len(), 2);
        for (_, bounds) in &results {
            assert_eq!(
                bounds[0],
                vec![(5, 10)],
                "coappear interval should be [5,10]"
            );
        }
    }

    #[test]
    fn two_slots_no_temporal_overlap_yields_nothing() {
        // A=[0,5], B=[10,15] — gap of 5 frames, no simultaneous presence
        let db = make_db(&[&[(0, 5)], &[(10, 15)]]);
        let s0 = 0;
        let s1 = 1;
        let sel = ObjSelector::new([[0, 1], [0, 1]], GroupChain::new([s0, s1], []));
        assert_eq!(sel.iter(&db).count(), 0);
    }

    #[test]
    fn self_join_is_rejected() {
        // Only one object — it cannot fill two slots at once (enumerate rejects duplicate ObjIds)
        let db = make_db(&[&[(0, 10)]]);
        let s0 = 0;
        let s1 = 1;
        let sel = ObjSelector::new([[0], [0]], GroupChain::new([s0, s1], []));
        assert_eq!(sel.iter(&db).count(), 0);
    }

    // ---- Then --------------------------------------------------------------------

    #[test]
    fn then_finds_sequential_pair() {
        // A=[0,10], B=[12,20]; gap = 12 - 10 = 2 ≥ min_gap=1
        let db = make_db(&[&[(0, 10)], &[(12, 20)]]);
        let s0 = 0;
        let s1 = 1;
        let sel = ObjSelector::new(
            [[0], [1]],
            GroupChain::new([s0], [ChainLink::then_relaxed(1.., [s1])]),
        );
        let results: Vec<_> = sel.iter(&db).collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0[0], 0); // slot 0 = A
        assert_eq!(results[0].0[1], 1); // slot 1 = B
        assert_eq!(results[0].1[0], vec![(0, 10)]);
        assert_eq!(results[0].1[1], vec![(12, 20)]);
    }

    #[test]
    fn then_rejects_reversed_order() {
        // slot 0 candidates = [B=1], slot 1 candidates = [A=0]
        // A ends at 10, before B.start + min_gap = 13 → outside the window
        let db = make_db(&[&[(0, 10)], &[(12, 20)]]);
        let s0 = 0;
        let s1 = 1;
        let sel = ObjSelector::new(
            [[1], [0]],
            GroupChain::new([s0], [ChainLink::then_relaxed(1.., [s1])]),
        );
        assert_eq!(sel.iter(&db).count(), 0);
    }

    #[test]
    fn then_max_gap_filters_too_distant() {
        // A=[0,10], B=[15,25]; gap = 15 - 10 = 5 > max_gap=3
        let db = make_db(&[&[(0, 10)], &[(15, 25)]]);
        let s0 = 0;
        let s1 = 1;
        let sel = ObjSelector::new(
            [[0], [1]],
            GroupChain::new([s0], [ChainLink::then_relaxed(1..=3, [s1])]),
        );
        assert_eq!(sel.iter(&db).count(), 0);
    }

    #[test]
    fn then_max_gap_accepts_within_window() {
        // A=[0,10], B=[12,20]; gap = 2 ≤ max_gap=5
        let db = make_db(&[&[(0, 10)], &[(12, 20)]]);
        let s0 = 0;
        let s1 = 1;
        let sel = ObjSelector::new(
            [[0], [1]],
            GroupChain::new([s0], [ChainLink::then_relaxed(1..=5, [s1])]),
        );
        assert_eq!(sel.iter(&db).count(), 1);
    }

    // ---- Contains ----------------------------------------------------------------

    #[test]
    fn contains_accepts_inner_inside_outer() {
        // A=[0,20], B=[5,15]; B's interval is strictly inside A's
        let db = make_db(&[&[(0, 20)], &[(5, 15)]]);
        let s0 = 0;
        let s1 = 1;
        let sel = ObjSelector::new(
            [[0], [1]],
            GroupChain::new([s0], [ChainLink::containing([s1])]),
        );
        let results: Vec<_> = sel.iter(&db).collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0[0], 0); // outer = A
        assert_eq!(results[0].0[1], 1); // inner = B
        assert_eq!(results[0].1[0], vec![(0, 20)]);
        assert_eq!(results[0].1[1], vec![(5, 15)]);
    }

    #[test]
    fn contains_rejects_when_inner_exceeds_outer() {
        // A=[5,10] (outer), B=[0,20] (inner) — B extends beyond A on both sides
        let db = make_db(&[&[(5, 10)], &[(0, 20)]]);
        let s0 = 0;
        let s1 = 1;
        let sel = ObjSelector::new(
            [[0], [1]],
            GroupChain::new([s0], [ChainLink::containing([s1])]),
        );
        assert_eq!(sel.iter(&db).count(), 0);
    }

    // ---- Shared-slot pinning -----------------------------------------------------

    #[test]
    fn shared_slot_is_pinned_across_chain_steps() {
        // Objects: A=[0,10], B=[5,20], C=[15,30]
        // Group 0: slots [0,1]  — finds (A,B) coappearing at [5,10]
        //                         and (B,C) coappearing at [15,20]
        // Group 1: slots [1,2], linked by ThenRelaxed{min_gap=1} — slot 1 is
        // shared, so the object group 0 bound there is the one group 1 must
        // coappear with. The relaxed window opens at group 0's start, so the
        // link check is `group1.end >= group0.start + 1`.
        //
        // Slots in *different* groups may hold the same object — only a
        // duplicate within one group is a self-join — so [A,B,A] is a legitimate
        // match here, not an accident.
        let db = make_db(&[&[(0, 10)], &[(5, 20)], &[(15, 30)]]);
        let (s0, s1, s2) = (0, 1, 2);
        let sel = ObjSelector::new(
            [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
            GroupChain::new([s0, s1], [ChainLink::then_relaxed(1.., [s1, s2])]),
        );

        let mut results: Vec<(Vec<u32>, Vec<GroupBounds>)> = sel
            .iter(&db)
            .map(|(binding, bounds)| (binding.to_vec(), bounds.to_vec()))
            .collect();
        results.sort();

        let bindings: Vec<&Vec<u32>> = results.iter().map(|(b, _)| b).collect();
        assert_eq!(
            bindings,
            vec![
                &vec![0, 1, 0], // (A,B) then (B,A)
                &vec![0, 1, 2], // (A,B) then (B,C)
                &vec![1, 0, 1], // (B,A) then (A,B)
                &vec![1, 2, 1], // (B,C) then (C,B)
                &vec![2, 1, 2], // (C,B) then (B,C)
            ],
        );

        // The pinning itself: in [A,B,C], slot 1 = B bridges both groups, so
        // group 0's bounds are coappear(A,B) and group 1's are coappear(B,C).
        let (_, bounds) = &results[1];
        assert_eq!(bounds[0], vec![(5, 10)], "group 0 = coappear(A,B)");
        assert_eq!(bounds[1], vec![(15, 20)], "group 1 = coappear(B,C)");
    }

    // ---- Multi-segment trajectories ----------------------------------------------

    #[test]
    fn multi_segment_traj_coappear_uses_exact_overlap() {
        // A has two canonical segments: [0,5] and [20,30]  (gap of 14 frames)
        // B has one segment: [2,8]
        // They overlap only in A's first segment → coappear at [2,5]
        let db = make_db(&[&[(0, 5), (20, 30)], &[(2, 8)]]);
        let s0 = 0;
        let s1 = 1;
        let sel = ObjSelector::new([[0, 1], [0, 1]], GroupChain::new([s0, s1], []));
        let results: Vec<_> = sel.iter(&db).collect();
        let ab = results.iter().find(|(b, _)| b[0] == 0 && b[1] == 1);
        assert!(ab.is_some(), "expected (A,B) coappearance pair");
        assert_eq!(ab.unwrap().1[0], vec![(2, 5)]);
    }

    // ---- intervals_compatible unit tests -----------------------------------------

    #[test]
    fn then_compatible_with_sufficient_gap() {
        let prev = [(0u32, 10u32)];
        let curr = [(12u32, 20u32)];
        assert!(intervals_compatible(
            &prev,
            &curr,
            &GroupRel::ThenRelaxed {
                min_gap: 1,
                max_gap: None
            }
        ));
    }

    #[test]
    fn then_incompatible_exceeds_max_gap() {
        // curr starts at 15, past the window's close at prev.end + 3 = 13.
        let prev = [(0u32, 10u32)];
        let curr = [(15u32, 20u32)];
        assert!(!intervals_compatible(
            &prev,
            &curr,
            &GroupRel::ThenRelaxed {
                min_gap: 1,
                max_gap: Some(3)
            }
        ));
    }

    #[test]
    fn contains_exact_boundary_is_accepted() {
        // Inner = outer (boundary case: ps<=cs && ce<=pe with equality)
        let outer = [(5u32, 15u32)];
        let inner = [(5u32, 15u32)];
        assert!(intervals_compatible(&outer, &inner, &GroupRel::Contains));
    }

    #[test]
    fn contains_inner_starts_before_outer() {
        let outer = [(5u32, 15u32)];
        let inner = [(3u32, 12u32)]; // inner starts before outer
        assert!(!intervals_compatible(&outer, &inner, &GroupRel::Contains));
    }

    // ---- ThenRelaxed ----------------------------------------------------------------

    #[test]
    fn then_accepts_overlapping_presence() {
        // A = [0,100], B = [5,50], gap 1..=10: B overlaps the window
        // [A.start+1, A.end+10] = [1,110] even though it starts inside A.
        // An exact THEN would reject this, but a predicate-level match exists
        // (lhs true in [0,10], rhs in [12,20]), so pruning it would be wrong.
        let prev = [(0u32, 100u32)];
        let curr = [(5u32, 50u32)];
        assert!(intervals_compatible(
            &prev,
            &curr,
            &GroupRel::ThenRelaxed {
                min_gap: 1,
                max_gap: Some(10)
            }
        ));
    }

    #[test]
    fn then_window_rejects_curr_entirely_before_window() {
        // B ends before A.start + min_gap.
        let prev = [(50u32, 100u32)];
        let curr = [(0u32, 52u32)];
        assert!(!intervals_compatible(
            &prev,
            &curr,
            &GroupRel::ThenRelaxed {
                min_gap: 5,
                max_gap: None
            }
        ));
    }

    #[test]
    fn then_window_rejects_curr_entirely_after_window() {
        // B starts after A.end + max_gap.
        let prev = [(0u32, 10u32)];
        let curr = [(25u32, 40u32)];
        assert!(!intervals_compatible(
            &prev,
            &curr,
            &GroupRel::ThenRelaxed {
                min_gap: 1,
                max_gap: Some(10)
            }
        ));
    }

    #[test]
    fn then_relaxed_chain_keeps_overlapping_presence() {
        // A = [0,100], B = [5,50]: B starts inside A, so an exact THEN would
        // drop this pair. The relaxed window keeps it for the predicate to judge.
        let db = make_db(&[&[(0, 100)], &[(5, 50)]]);
        let (s0, s1) = (0, 1);
        let sel = ObjSelector::new(
            [[0], [1]],
            GroupChain::new([s0], [ChainLink::then_relaxed(1..=10, [s1])]),
        );
        assert_eq!(sel.iter(&db).count(), 1);
    }

    // ---- Builder ergonomics --------------------------------------------------------

    #[test]
    fn gap_conversions() {
        let g: Gap = (1..).into();
        assert_eq!((g.min, g.max), (1, None));
        let g: Gap = (0..=30).into();
        assert_eq!((g.min, g.max), (0, Some(30)));
        let g: Gap = (1..30).into();
        assert_eq!((g.min, g.max), (1, Some(29)));
        let g: Gap = (..).into();
        assert_eq!((g.min, g.max), (0, None));
        let g: Gap = 5.into();
        assert_eq!((g.min, g.max), (5, Some(5)));
    }

    #[test]
    #[should_panic(expected = "at least one slot")]
    fn empty_group_is_rejected_at_build_time() {
        let _ = ObjSelector::new([[0]], GroupChain::new([], []));
    }

    #[test]
    #[should_panic(expected = "has no candidates")]
    fn slot_without_candidates_is_rejected_at_build_time() {
        // Only one slot's candidates were given, so slot 1 names a tuple
        // position this selector has nothing for.
        let foreign = 1;
        let _ = ObjSelector::new([[0]], GroupChain::new([foreign], []));
    }

    #[test]
    fn slot_handles_resolve_bindings_via_cursor() {
        // A=[0,10], B=[12,20]; A then B
        let db = make_db(&[&[(0, 10)], &[(12, 20)]]);
        let first = 0;
        let second = 1;
        let sel = ObjSelector::new(
            [[0], [1]],
            GroupChain::new([first], [ChainLink::then_relaxed(1.., [second])]),
        );

        let mut m = sel.candidates(&db);
        assert!(m.advance());
        assert_eq!(m.get(first), 0);
        assert_eq!(m.get(second), 1);
        assert_eq!(m.group_bounds(0), &[(0, 10)]);
        assert_eq!(m.group_bounds(1), &[(12, 20)]);
        assert!(!m.advance());
    }
}
