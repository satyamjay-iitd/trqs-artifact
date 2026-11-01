//! Materialising selector execution strategy.
//!
//! Enumerates each chain group's coappearing tuples into a per-group candidate
//! vector and backtracks over those vectors. Key techniques:
//!
//! 1. **Segment-exact incremental intersection.** Group tuples are built one
//!    slot at a time while maintaining the running intersection of canonical
//!    segments. A partial tuple with an empty intersection prunes the whole
//!    subtree, instead of enumerating the full cross product of span-overlapping
//!    objects and running an exact k-way check per combination.
//! 2. **Selectivity ordering.** Within a group, slots with fewer candidates
//!    (pinned slots in particular) are bound first.
//! 3. **relation-region prefilter.** When materialising a group under a
//!    `ThenRelaxed` / `Contains` constraint, candidates are first filtered to objects with a
//!    segment overlapping the feasible region implied by the previous group's
//!    intervals (a necessary condition; the exact check still runs at the end).
//! 4. **Memoisation of unpinned groups.** A group that shares no slot with
//!    earlier groups enumerates the same raw tuple set for every parent binding;
//!    it is computed once and only the (cheap) GroupRel filter runs per parent.
//!
//! Result semantics are identical to the lazy strategy (verified by
//! differential tests); only tuple emission order may differ.

use crate::{
    Binding, ChainBounds, GroupBounds, GroupRel, Interval, ObjId, ObjSelector, intervals_compatible,
};
use trql_poly::PolyTraj;
use trql_storage::db::Trqs;
use trql_types::Slot;

// ---- Interval helpers -----------------------------------------------------------

/// Computes the feasible time region for the next group, given the previous
/// group's intervals and the GroupRel between them.
///
/// Any valid tuple of the next group has an interval whose start lies in this
/// region (`ThenRelaxed`) or that is contained in it (`Contains`); since a group
/// interval is contained in every member's segment, every member must have a
/// segment overlapping the region. This is a *necessary* condition only.
fn feasible_region(prev: &[Interval], op: &GroupRel) -> Vec<Interval> {
    let mut windows: Vec<Interval> = match op {
        GroupRel::ThenRelaxed { min_gap, max_gap } => prev
            .iter()
            .filter_map(|&(ps, pe)| {
                let lo = ps.saturating_add(*min_gap);
                let hi = max_gap.map_or(u32::MAX, |mg| pe.saturating_add(mg));
                (lo <= hi).then_some((lo, hi))
            })
            .collect(),
        GroupRel::Contains => prev.to_vec(),
    };
    windows.sort_unstable();
    let mut merged: Vec<Interval> = Vec::with_capacity(windows.len());
    for (lo, hi) in windows {
        match merged.last_mut() {
            Some(last) if lo <= last.1.saturating_add(1) => last.1 = last.1.max(hi),
            _ => merged.push((lo, hi)),
        }
    }
    merged
}

// ---- Group enumeration ----------------------------------------------------------

/// Enumerates all coappearing tuples for one group via backtracking with
/// incremental interval intersection.
///
/// `slot_lists[i]` are the candidates for the group's i-th slot (group-local
/// order). `region`, when present, prefilters candidates to those with a
/// segment overlapping it.
fn enumerate_group(
    trajs: &[PolyTraj],
    slot_lists: &[&[ObjId]],
    region: Option<&[Interval]>,
) -> Vec<(Vec<ObjId>, GroupBounds)> {
    let k = slot_lists.len();
    if k == 0 {
        return vec![(vec![], vec![])];
    }

    let mut lists: Vec<Vec<ObjId>> = Vec::with_capacity(k);
    for list in slot_lists {
        let filtered: Vec<ObjId> = list
            .iter()
            .copied()
            .filter(|&oid| {
                let segs = trajs[oid as usize].canonical_segments();
                !segs.is_empty()
                    && region.is_none_or(|r| trql_types::intervals::overlaps_any(segs, r))
            })
            .collect();
        if filtered.is_empty() {
            return vec![];
        }
        lists.push(filtered);
    }

    // Bind the most selective slots first (pinned singletons sort to the front).
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by_key(|&i| lists[i].len());

    let mut out = Vec::new();
    let mut chosen: Vec<ObjId> = Vec::with_capacity(k);
    descend(trajs, &lists, &order, &mut chosen, &[], &mut out);
    out
}

fn descend(
    trajs: &[PolyTraj],
    lists: &[Vec<ObjId>],
    order: &[usize],
    chosen: &mut Vec<ObjId>,
    cur: &[Interval],
    out: &mut Vec<(Vec<ObjId>, GroupBounds)>,
) {
    let pos = chosen.len();
    for &oid in &lists[order[pos]] {
        if chosen.contains(&oid) {
            continue;
        }
        let segs = trajs[oid as usize].canonical_segments();
        let next: Vec<Interval> = if pos == 0 {
            segs.to_vec()
        } else {
            trql_types::intervals::intersect(cur, segs)
        };
        if next.is_empty() {
            continue;
        }
        chosen.push(oid);
        if pos + 1 == lists.len() {
            // Restore the group's local slot order.
            let mut local = vec![0u32; lists.len()];
            for (p, &slot_pos) in order.iter().enumerate() {
                local[slot_pos] = chosen[p];
            }
            out.push((local, next));
        } else {
            descend(trajs, lists, order, chosen, &next, out);
        }
        chosen.pop();
    }
}

// ---- Iterator -------------------------------------------------------------------

struct GroupState {
    candidates: Vec<(Vec<ObjId>, GroupBounds)>,
    idx: usize,
}

/// Iterator over `(Binding, ChainBounds)` pairs; same semantics as
/// [`crate::ObjSelector::iter`], different execution strategy.
pub struct ObjSelectorIterV2<'a> {
    db: &'a Trqs,
    selector: &'a ObjSelector,
    stack: Vec<GroupState>,
    /// `memo[d]` caches the raw (unfiltered) tuples of group `d` when the
    /// group shares no slot with earlier groups.
    memo: Vec<Option<Vec<(Vec<ObjId>, GroupBounds)>>>,
    initialized: bool,
    done: bool,
}

impl ObjSelector {
    /// Iterate with the improved execution strategy. Yields the same result
    /// set as [`ObjSelector::iter`], possibly in a different order.
    pub fn iter_v2<'a>(&'a self, db: &'a Trqs) -> ObjSelectorIterV2<'a> {
        ObjSelectorIterV2 {
            db,
            selector: self,
            stack: Vec::with_capacity(self.num_groups()),
            memo: vec![None; self.num_groups()],
            initialized: false,
            done: false,
        }
    }
}

impl<'a> ObjSelectorIterV2<'a> {
    fn materialize(&mut self, depth: usize) -> Vec<(Vec<ObjId>, GroupBounds)> {
        let group_slots = self.selector.chain().group_slots(depth);
        let trajs = self.db.poly_trajs();

        // Collect slots already bound by a shallower group.
        let mut pinned: Vec<(Slot, ObjId)> = Vec::new();
        for prev in 0..depth {
            let prev_slots = self.selector.chain().group_slots(prev);
            let (prev_oids, _) = &self.stack[prev].candidates[self.stack[prev].idx];
            for (local_i, &slot) in prev_slots.iter().enumerate() {
                if group_slots.contains(&slot) {
                    pinned.push((slot, prev_oids[local_i]));
                }
            }
        }

        let op = self.selector.chain().rel_into(depth);
        let prev_ivs: Option<&GroupBounds> = if depth > 0 {
            let prev = &self.stack[depth - 1];
            Some(&prev.candidates[prev.idx].1)
        } else {
            None
        };

        // Unpinned group: its raw tuple set is parent-independent — compute it
        // once, then only run the GroupRel filter per parent binding.
        if pinned.is_empty() && depth > 0 {
            if self.memo[depth].is_none() {
                let slot_lists: Vec<&[ObjId]> = group_slots
                    .iter()
                    .map(|&slot| self.selector.slot_2_objects[slot].as_slice())
                    .collect();
                self.memo[depth] = Some(enumerate_group(trajs, &slot_lists, None));
            }
            let raw = self.memo[depth].as_ref().unwrap();
            return match (op, prev_ivs) {
                (Some(op), Some(p)) => raw
                    .iter()
                    .filter(|(_, ivs)| intervals_compatible(p, ivs, op))
                    .cloned()
                    .collect(),
                _ => raw.clone(),
            };
        }

        // Pinned group (or group 0): enumerate per parent, but prefilter
        // candidates by the feasible region implied by the GroupRel.
        let pinned_storage: Vec<[ObjId; 1]> = pinned.iter().map(|&(_, oid)| [oid]).collect();
        let slot_lists: Vec<&[ObjId]> = group_slots
            .iter()
            .map(|&slot| {
                if let Some(pos) = pinned.iter().position(|&(s, _)| s == slot) {
                    &pinned_storage[pos][..]
                } else {
                    self.selector.slot_2_objects[slot].as_slice()
                }
            })
            .collect();

        let region = match (op, prev_ivs) {
            (Some(op), Some(p)) => Some(feasible_region(p, op)),
            _ => None,
        };
        let mut res = enumerate_group(trajs, &slot_lists, region.as_deref());
        if let (Some(op), Some(p)) = (op, prev_ivs) {
            res.retain(|(_, ivs)| intervals_compatible(p, ivs, op));
        }
        res
    }

    fn build_result(&self) -> (Binding, ChainBounds) {
        let mut binding = vec![0u32; self.selector.num_slots];
        let mut bounds = ChainBounds::new();

        for (depth, state) in self.stack.iter().enumerate() {
            let (local_oids, ivs) = &state.candidates[state.idx];
            for (local_i, &slot) in self.selector.chain().group_slots(depth).iter().enumerate() {
                binding[slot] = local_oids[local_i];
            }
            bounds.push(ivs.clone());
        }

        (binding.into_iter().collect(), bounds)
    }
}

impl<'a> Iterator for ObjSelectorIterV2<'a> {
    type Item = (Binding, ChainBounds);

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let n = self.selector.num_groups();

        if !self.initialized {
            let c0 = self.materialize(0);
            self.stack.push(GroupState {
                candidates: c0,
                idx: 0,
            });
            self.initialized = true;
        }

        loop {
            let depth = self.stack.len();

            if depth == 0 {
                self.done = true;
                return None;
            }

            let (top_idx, top_len) = {
                let top = self.stack.last().unwrap();
                (top.idx, top.candidates.len())
            };

            if top_idx >= top_len {
                self.stack.pop();
                if let Some(parent) = self.stack.last_mut() {
                    parent.idx += 1;
                }
                continue;
            }

            if depth < n {
                let candidates = self.materialize(depth);
                self.stack.push(GroupState { candidates, idx: 0 });
                continue;
            }

            let result = self.build_result();
            self.stack.last_mut().unwrap().idx += 1;
            return Some(result);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intersect_basic() {
        assert_eq!(
            trql_types::intervals::intersect(&[(0, 10), (20, 30)], &[(5, 25)]),
            vec![(5, 10), (20, 25)]
        );
        assert_eq!(
            trql_types::intervals::intersect(&[(0, 4)], &[(5, 9)]),
            vec![]
        );
    }

    #[test]
    fn feasible_region_then_merges_windows() {
        // Windows open at prev.start + min_gap and close at prev.end + max_gap.
        let rel = GroupRel::ThenRelaxed {
            min_gap: 1,
            max_gap: Some(5),
        };
        // [1,15] and [41,55] are far apart and stay separate.
        assert_eq!(
            feasible_region(&[(0, 10), (40, 50)], &rel),
            vec![(1, 15), (41, 55)]
        );
        // [1,15] and [13,25] overlap, so they merge.
        assert_eq!(feasible_region(&[(0, 10), (12, 20)], &rel), vec![(1, 25)]);
    }

    #[test]
    fn feasible_region_then_unbounded() {
        let prev = [(0, 10)];
        let r = feasible_region(
            &prev,
            &GroupRel::ThenRelaxed {
                min_gap: 2,
                max_gap: None,
            },
        );
        assert_eq!(r, vec![(2, u32::MAX)]);
    }

    #[test]
    fn feasible_region_contains_is_prev() {
        let prev = [(5, 15), (20, 30)];
        let r = feasible_region(&prev, &GroupRel::Contains);
        assert_eq!(r, vec![(5, 15), (20, 30)]);
    }
}
