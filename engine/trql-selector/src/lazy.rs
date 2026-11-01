//! Fully lazy selector execution with steady-state zero allocation.
//!
//! The chain is compiled into a flat sequence of *positions* (one per slot
//! occurrence, group by group). Matching is a single backtracking search over
//! positions: binding a position intersects its object's canonical segments
//! into a per-position interval buffer; an empty intersection, a duplicate
//! object, a failed relation-region prefilter, or a failed exact relation check at a group
//! boundary all prune immediately.
//!
//! All buffers (cursors, binding, interval buffers, GroupRel regions) are allocated
//! once when the cursor is created and reused for the whole scan — advancing
//! to the next match performs no heap allocation. Because the `Iterator`
//! trait cannot yield items that borrow the iterator, the core API is a
//! lending-style [`CandidateCursor`] (`advance()` + accessors);
//! [`ObjSelector::iter`] wraps it for callers who want owned
//! `(Binding, ChainBounds)` items.
//!
//! Result semantics are identical to the v2 strategy (verified by
//! differential tests); only emission order may differ.

use crate::{Binding, ChainBounds, GroupRel, Interval, ObjId, ObjSelector, intervals_compatible};
use smallvec::SmallVec;
use trql_storage::db::Trqs;

/// Candidates per block of the `slot_blk_max` rollup. Small enough that a dead
/// stretch is found at useful granularity, large enough that the rollup costs a
/// fraction of the list it summarises and a live block is one cache line's worth
/// of scanning.
const SKIP_BLOCK: usize = 64;

/// `Some(bounds)` spanning a sorted interval list, `None` when it is empty.
/// The outer `Option` distinguishes "no window at all" at the call site.
fn hull_of(ivs: &[Interval]) -> Option<Option<(u32, u32)>> {
    Some(match (ivs.first(), ivs.last()) {
        (Some(&(s, _)), Some(&(_, e))) => Some((s, e)),
        _ => None,
    })
}
use trql_types::Slot;
use trql_types::intervals::{intersect_into, longest_run, overlaps_any};

/// Computes the feasible time region for the group following `prev` under
/// `op`, writing the merged windows into `out` (cleared first).
///
/// Every member of a valid next-group tuple must have a segment overlapping
/// this region (necessary condition only; the exact `intervals_compatible`
/// check still runs at group completion). `prev` is sorted and disjoint, so
/// the windows are produced in order and merged on the fly.
fn feasible_region_into(prev: &[Interval], op: &GroupRel, out: &mut Vec<Interval>) {
    out.clear();
    match op {
        GroupRel::ThenRelaxed { min_gap, max_gap } => {
            // The window opens at the previous interval's start and closes at
            // its end plus the max gap.
            for &(ps, pe) in prev {
                let lo = ps.saturating_add(*min_gap);
                let hi = max_gap.map_or(u32::MAX, |mg| pe.saturating_add(mg));
                if lo > hi {
                    continue;
                }
                match out.last_mut() {
                    Some(last) if lo <= last.1.saturating_add(1) => last.1 = last.1.max(hi),
                    _ => out.push((lo, hi)),
                }
            }
        }
        GroupRel::Contains => out.extend_from_slice(prev),
    }
}

// ---- Compiled positions -----------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct Position {
    group: usize,
    slot: Slot,
    /// Slot already bound by an earlier group: candidate list is the singleton
    /// `binding[slot]`.
    pinned: bool,
    first_in_group: bool,
    last_in_group: bool,
}

// ---- Cursor -----------------------------------------------------------------------

impl ObjSelector {
    /// Creates a lazy candidate cursor. All per-query tables and working
    /// buffers are allocated here; advancing the cursor does not allocate.
    ///
    /// Each slot's candidates are ordered by the frame they first appear, so a
    /// position bounded by an already-bound sibling walks only the stretch of
    /// that list which could still overlap, instead of all of it.
    pub fn candidates<'a>(&'a self, db: &'a Trqs) -> CandidateCursor<'a> {
        let num_groups = self.num_groups();
        let mut positions: Vec<Position> = Vec::new();
        let mut group_first_pos = Vec::with_capacity(num_groups);
        let mut group_last_pos = Vec::with_capacity(num_groups);
        let mut seen_slots: Vec<Slot> = Vec::new();

        // `GroupChain::new` rejects an empty group, so every group contributes
        // at least one position and `positions` is never empty.
        for g in 0..num_groups {
            let group_slots = self.chain().group_slots(g);
            let mut grp: Vec<Position> = group_slots
                .iter()
                .map(|&slot| Position {
                    group: g,
                    slot,
                    pinned: seen_slots.contains(&slot),
                    first_in_group: false,
                    last_in_group: false,
                })
                .collect();
            // Bind pinned slots first, then free slots by ascending selectivity.
            grp.sort_by_key(|p| (!p.pinned, self.slot_2_objects[p.slot].len()));
            if let Some(first) = grp.first_mut() {
                first.first_in_group = true;
            }
            if let Some(last) = grp.last_mut() {
                last.last_in_group = true;
            }
            group_first_pos.push(positions.len());
            positions.extend_from_slice(&grp);
            group_last_pos.push(positions.len() - 1);
            seen_slots.extend_from_slice(group_slots);
        }

        let npos = positions.len();

        // Order every slot's candidates by the frame they first appear, and roll
        // up the frame each stops covering. Together these turn "who could be
        // present during this window?" into a binary search plus a forward scan
        // that skips dead blocks and stops, instead of a walk over the whole list.
        let starts = |oid: ObjId| {
            db.poly_trajs()[oid as usize]
                .canonical_segments()
                .first()
                .map_or((u32::MAX, 0), |&(s, _)| (s, s))
        };
        let slot_by_start: Vec<Vec<(u32, ObjId)>> = self
            .slot_2_objects
            .iter()
            .map(|cands| {
                let mut v: Vec<(u32, ObjId)> = cands.iter().map(|&o| (starts(o).0, o)).collect();
                // Object ids are already roughly appearance-ordered in every
                // dataset measured, so this is near-linear on real input.
                v.sort_unstable();
                v
            })
            .collect();
        // Last frame each candidate covers, in `slot_by_start` order, rolled up
        // two ways so a window can skip the candidates that already ended.
        let last_frame = |oid: ObjId| {
            db.poly_trajs()[oid as usize]
                .canonical_segments()
                .last()
                .map_or(0, |&(_, e)| e)
        };
        // Running maximum, hence non-decreasing, hence binary-searchable: the
        // first entry reaching the window start bounds the prefix that cannot.
        let slot_pmax_end: Vec<Vec<u32>> = slot_by_start
            .iter()
            .map(|cands| {
                let mut running = 0;
                cands
                    .iter()
                    .map(|&(_, o)| {
                        running = running.max(last_frame(o));
                        running
                    })
                    .collect()
            })
            .collect();
        // Per fixed-size block, the same maximum without the running carry. One
        // long-lived candidate pins `slot_pmax_end` for everything behind it;
        // this recovers the dead stretches past it, a block at a time.
        let slot_blk_max: Vec<Vec<u32>> = slot_by_start
            .iter()
            .map(|cands| {
                cands
                    .chunks(SKIP_BLOCK)
                    .map(|blk| blk.iter().map(|&(_, o)| last_frame(o)).max().unwrap_or(0))
                    .collect()
            })
            .collect();

        CandidateCursor {
            db,
            slot_by_start,
            slot_pmax_end,
            slot_blk_max,
            scan_lo: vec![0; npos],
            scan_hi: vec![u32::MAX; npos],
            scan_ws: vec![0; npos],
            selector: self,
            positions,
            group_first_pos,
            group_last_pos,
            cursors: vec![0; npos],
            bindings: vec![0; self.num_slots].into(),
            ivbufs: vec![Vec::new(); npos],
            regions: vec![Vec::new(); num_groups],
            started: false,
            done: false,
        }
    }

    /// Convenience adapter over [`ObjSelector::candidates`] yielding owned items.
    /// Each yielded result clones the binding and bounds — that is the price of
    /// being a real `Iterator`; drive the cursor directly to avoid it.
    pub fn iter<'a>(&'a self, db: &'a Trqs) -> impl Iterator<Item = (Binding, ChainBounds)> + 'a {
        let mut cursor = self.candidates(db);
        std::iter::from_fn(move || {
            if !cursor.advance() {
                return None;
            }
            let binding: Binding = cursor.bindings().iter().copied().collect();
            let bounds: ChainBounds = (0..cursor.num_groups())
                .map(|g| cursor.group_bounds(g).to_vec())
                .collect();
            Some((binding, bounds))
        })
    }
}
/// Lending-style cursor over the tuples that satisfy the selector's
/// [`GroupChain`]: call [`CandidateCursor::advance`], then read the current one
/// through [`CandidateCursor::bindings`] / [`CandidateCursor::group_bounds`].
///
/// These are *candidates*, not query results. The chain constrains appearance
/// only — which objects coappear, and how those groups relate in time — which
/// is a necessary but not sufficient condition for the query's predicate to
/// hold (see [`GroupRel::ThenRelaxed`]). The caller is expected to evaluate the
/// exact predicate on each candidate and discard the ones that fail.
///
/// # Running example
///
/// Every field below is described against this one query. Three objects, each
/// present over a single span:
///
/// ```text
/// A = [0,10]      B = [5,20]      C = [15,30]
/// ```
///
/// The query is `pred(a, b) THEN pred(b, c)` — slots `a = 0`, `b = 1`,
/// `c = 2` — which the planner turns into a two-group chain, `b` shared
/// across the link:
///
/// ```text
/// coappear(a, b)  THEN_relaxed(1..)  coappear(b, c)
/// ```
///
/// with candidate lists `a → {A, B}`, `b → {B}`, `c → {C}`.
/// This will have **3 slots**, **2 groups**, **4 positions**.
/// Driving the cursor to exhaustion yields exactly one
/// candidate, `[A, B, C]`.
///
/// [`GroupChain`]: crate::GroupChain
/// [`GroupRel::ThenRelaxed`]: crate::GroupRel::ThenRelaxed
pub struct CandidateCursor<'a> {
    db: &'a Trqs,
    selector: &'a ObjSelector,

    /// Per slot, its candidates as `(first_frame, oid)` ordered by first frame.
    /// Built once per cursor; the enumeration walks a sub-range of this.
    slot_by_start: Vec<Vec<(u32, ObjId)>>,

    /// Per slot, the running maximum last-frame over `slot_by_start` — so
    /// `slot_pmax_end[slot][i]` is the last frame any of the first `i + 1`
    /// candidates covers. Non-decreasing by construction, so the prefix of
    /// candidates that all ended before a window opened is one binary search
    /// away (see [`Self::set_scan_range`]).
    ///
    /// This replaced a single per-slot maximum *span*, which one long-lived
    /// candidate could inflate for the whole slot: on i24 `pickup`, a median
    /// span of 230 frames against a maximum of 16,892 made every window scan
    /// ~6,500 candidates where ~200 could overlap.
    slot_pmax_end: Vec<Vec<u32>>,

    /// Per slot, the maximum last-frame within each [`SKIP_BLOCK`]-sized block
    /// of `slot_by_start` — the same rollup as `slot_pmax_end` but *without*
    /// the running carry, so a single long-lived candidate no longer masks the
    /// dead stretches behind it. `slot_pmax_end` can only skip a contiguous
    /// prefix; this skips blocks past it during the forward scan.
    slot_blk_max: Vec<Vec<u32>>,

    /// Per position, where its scan of `slot_by_start` begins — the binary
    /// search result for the current window. `cursors` counts from here.
    scan_lo: Vec<usize>,

    /// Per position, the last first-frame worth visiting: an object starting
    /// after the window's end cannot overlap it, and neither can any later one,
    /// so the scan stops rather than running to the end of the list.
    scan_hi: Vec<u32>,

    /// Per position, the start of the window that bounds it — what
    /// [`Self::skip_dead_blocks`] tests `slot_blk_max` against. `0` means
    /// nothing bounds this position, which admits every block.
    scan_ws: Vec<u32>,

    /// One entry per *slot occurrence*, laid out group by group — so its length
    /// is the total size of all groups.
    ///
    /// In the running example, length 4 — `b` occurs in both groups. Within a
    /// group, pinned slots sort first, then the rest by ascending candidate
    /// count, so the most selective slot is bound first and failures prune
    /// high. `b` has one candidate to `a`'s two, so it leads group 0:
    ///
    /// ```text
    /// index:  0        1        2                 3
    ///         (g0, b)  (g0, a)  (g1, b) pinned    (g1, c)
    /// ```
    positions: Vec<Position>,

    /// Where each group starts and ends in `positions`; length is the number of
    /// groups. In the running example `[0, 2]` and `[1, 3]`.
    ///
    /// `group_first_pos` bounds the backward scan that rejects binding the same
    /// object twice *within* one group; `group_last_pos[g]` is where group `g`'s
    /// completed coappearance lives in `ivbufs`.
    group_first_pos: Vec<usize>,
    group_last_pos: Vec<usize>,

    /// Per position, the *index into that slot's candidate list*.
    /// Length matches `positions`, so 4 in the running example. This is the
    /// depth-first search stack: it is the whole of the cursor's saved place,
    /// which is why resuming is just `cursors[npos - 1] += 1`.
    ///
    /// It starts `[0, 0, 0, 0]`, is still `[0, 0, 0, 0]` when `[A, B, C]` is
    /// returned, and ends `[1, 2, 1, 1]` — every level one past its list — once
    /// exhausted.
    cursors: Vec<usize>,

    /// The ObjId currently bound to each **slot**, so its length is the number
    /// of slots — 3 in the running example, not 4. That mismatch with
    /// `positions` is deliberate: positions 0 and 2 name the same slot `b`, so
    /// the pinned position 2 reads back the cell position 0 wrote instead of
    /// enumerating. It reaches `[A, B, C]` = `[0, 1, 2]` for the one candidate.
    bindings: SmallVec<[ObjId; 4]>,

    /// Per position, the running intersection of the group's objects' segments
    /// *as far as this depth*; length matches `positions`. A `first_in_group`
    /// position seeds with its object's raw segments, every later one
    /// intersects the buffer to its left, so `ivbufs[group_last_pos[g]]` is
    /// group `g`'s full coappearance — what `group_bounds(g)` returns.
    ///
    /// At the single candidate of the running example:
    ///
    /// ```text
    /// ivbufs[0] = [(5,20)]   B alone, seeding group 0
    /// ivbufs[1] = [(5,10)]   coappear(A, B) — group 0 complete
    /// ivbufs[2] = [(5,20)]   B alone again, seeding group 1 from scratch
    /// ivbufs[3] = [(15,20)]  coappear(B, C) — group 1 complete
    /// ```
    ///
    /// Groups measure their own coappearance independently: position 2 reseeds
    /// from B's raw span rather than chaining onto group 0's `[(5,10)]`.
    ivbufs: Vec<Vec<Interval>>,

    /// Per group, the window group `g` must show presence in, given group
    /// `g - 1`'s completed intervals and the relation between them; length is
    /// the number of groups. Refreshed when a group's last position binds, and
    /// used as a cheap overlap prefilter *before* the intersection work.
    ///
    /// `regions[0]` is never written — the head group has no predecessor. In
    /// the running example `regions[1] = [(6, u32::MAX)]`: group 0 coappears
    /// from frame 5, and `THEN_relaxed(1..)` opens the window one frame later.
    regions: Vec<Vec<Interval>>,

    /// Whether `advance` has run: distinguishes "start at depth 0" from
    /// "resume past the candidate just yielded".
    started: bool,

    /// Set once the search backtracks past depth 0; every later `advance`
    /// returns `false` immediately.
    done: bool,
}

impl<'a> CandidateCursor<'a> {
    /// Advances to the next candidate. Returns `false` when exhausted.
    pub fn advance(&mut self) -> bool {
        if self.done {
            return false;
        }

        let npos = self.positions.len();
        let mut pos_idx;
        if self.started {
            // Resume: move past the match we yielded last time.
            pos_idx = npos - 1;
            self.cursors[pos_idx] += 1;
        } else {
            self.started = true;
            self.cursors[0] = 0;
            pos_idx = 0;
            self.set_scan_range(0);
        }

        loop {
            let pos = self.positions[pos_idx];

            let mut found = false;
            loop {
                let oid = if pos.pinned {
                    if self.cursors[pos_idx] > 0 {
                        break;
                    }
                    self.bindings[pos.slot]
                } else {
                    let i = self.scan_lo[pos_idx] + self.cursors[pos_idx];
                    // Jumping the dead blocks here rather than in `set_scan_range`
                    // keeps `i == scan_lo + cursors`, so resuming still works.
                    let i = self.skip_dead_blocks(pos.slot, i, self.scan_ws[pos_idx]);
                    self.cursors[pos_idx] = i - self.scan_lo[pos_idx];
                    let Some(&(first_frame, oid)) = self.slot_by_start[pos.slot].get(i) else {
                        break;
                    };
                    // Sorted by first frame, so once one starts after the window
                    // ends, so does every candidate behind it.
                    if first_frame > self.scan_hi[pos_idx] {
                        break;
                    }
                    oid
                };
                if self.try_bind(pos_idx, oid) {
                    found = true;
                    break;
                }
                self.cursors[pos_idx] += 1;
            }

            if !found {
                if pos_idx == 0 {
                    self.done = true;
                    return false;
                }
                pos_idx -= 1;
                self.cursors[pos_idx] += 1;
                continue;
            }

            if pos_idx + 1 == npos {
                return true;
            }
            pos_idx += 1;
            self.cursors[pos_idx] = 0;
            self.set_scan_range(pos_idx);
        }
    }

    /// Fixes the sub-range of `slot_by_start` this position will walk, from the
    /// window that bounds it: the group's coappearance so far, or — at a group's
    /// first slot — the window the previous group published. The head group's
    /// first slot has neither and scans everything.
    ///
    /// Called only where the cursor resets, since that is where the bound prefix
    /// changes. Backtracking keeps the range: the prefix above it is unchanged.
    fn set_scan_range(&mut self, pos_idx: usize) {
        let pos = self.positions[pos_idx];
        if pos.pinned {
            return;
        }
        // Hull of the bounding interval list — conservative, and all the sort
        // order can exploit. `None` means nothing bounds this position.
        let hull = if pos.first_in_group && pos.group == 0 {
            None
        } else if pos.first_in_group {
            hull_of(&self.regions[pos.group])
        } else {
            hull_of(&self.ivbufs[pos_idx - 1])
        };

        let list = &self.slot_by_start[pos.slot];
        match hull {
            None => {
                self.scan_lo[pos_idx] = 0;
                self.scan_hi[pos_idx] = u32::MAX;
                self.scan_ws[pos_idx] = 0;
            }
            // An empty bounding list admits nothing: park past the end.
            Some(None) => {
                self.scan_lo[pos_idx] = list.len();
                self.scan_hi[pos_idx] = 0;
                self.scan_ws[pos_idx] = 0;
            }
            Some(Some((ws, we))) => {
                // Every candidate in the prefix below this ended before the
                // window opened, so none of them can overlap it. `slot_pmax_end`
                // is non-decreasing, so this is the longest such prefix.
                self.scan_lo[pos_idx] = self.slot_pmax_end[pos.slot].partition_point(|&m| m < ws);
                self.scan_hi[pos_idx] = we;
                self.scan_ws[pos_idx] = ws;
            }
        }
    }

    /// Advances `i` past whole blocks of `slot_by_start` in which every
    /// candidate ended before `ws`, and so cannot reach the window.
    ///
    /// This is what `scan_lo` alone cannot do: `slot_pmax_end` carries a running
    /// maximum, so one long-lived candidate stops the prefix skip dead, even
    /// when everything behind it is long gone. Only ever moves forward, so the
    /// caller can fold the result back into its resume counter.
    fn skip_dead_blocks(&self, slot: Slot, mut i: usize, ws: u32) -> usize {
        let len = self.slot_by_start[slot].len();
        let blk = &self.slot_blk_max[slot];
        while i < len && blk[i / SKIP_BLOCK] < ws {
            i = (i / SKIP_BLOCK + 1) * SKIP_BLOCK;
        }
        i
    }

    /// Attempts to bind `oid` at `pos_idx`; updates the position's interval
    /// buffer and, at group boundaries, runs the exact GroupRel check and refreshes
    /// the next group's feasible region.
    fn try_bind(&mut self, pos_idx: usize, oid: ObjId) -> bool {
        let pos = self.positions[pos_idx];

        // Not already taken by an earlier slot of this group
        if self.already_bound_in_group(pos, pos_idx, oid) {
            return false;
        }
        // Present at all, and able to reach the window the previous group opened.
        let Some(segs) = self.presence_reaching_region(pos, oid) else {
            return false;
        };
        // Still coappearing with this group's earlier slots.
        if !self.extend_coappearance(pos, pos_idx, segs) {
            return false;
        }
        // At a group's last slot: exact relation must hold,
        // then hand the next group its window.
        if pos.last_in_group && !self.close_group(pos, pos_idx) {
            return false;
        }

        // bind the object to this slot
        self.bindings[pos.slot] = oid;
        true
    }

    /// Whether an earlier slot of this same group already took `oid`. Slots in
    /// *different* groups may share an object; a repeat within one group would
    /// be a self-join.
    fn already_bound_in_group(&self, pos: Position, pos_idx: usize, oid: ObjId) -> bool {
        self.positions[self.group_first_pos[pos.group]..pos_idx]
            .iter()
            .any(|p| self.bindings[p.slot] == oid)
    }

    /// The object's canonical segments, or `None` if it is absent entirely or
    /// cannot reach the window the previous group opened for this one
    /// (`regions[group]`, published when that group completed). The head group
    /// has no predecessor, so nothing constrains it.
    ///
    /// A necessary condition per object, checked before any intersection work;
    /// the exact test on the group's full coappearance is [`Self::close_group`].
    fn presence_reaching_region(&self, pos: Position, oid: ObjId) -> Option<&'a [Interval]> {
        let segs = self.db.poly_trajs()[oid as usize].canonical_segments();
        if segs.is_empty() {
            return None;
        }
        if pos.group != 0 && !overlaps_any(segs, &self.regions[pos.group]) {
            return None;
        }
        Some(segs)
    }

    /// Folds `segs` into this group's running coappearance, returning whether
    /// anything survives. The group's first slot seeds the buffer; every later
    /// one intersects the buffer to its left, so an empty result means this
    /// object shares no frame with the slots already bound in the group.
    ///
    /// "Survives" also means long enough. A group whose subtree demands
    /// `DURATION(P, k)` needs its variables present together for `k` consecutive
    /// frames, since `P` can only hold where they all are; a co-appearance whose
    /// longest run is shorter than that cannot match however the remaining slots
    /// bind. Checking it here rather than when the group closes is sound because
    /// intersection only ever shrinks the buffer — a prefix already too short
    /// can never recover — and it prunes higher in the search.
    fn extend_coappearance(&mut self, pos: Position, pos_idx: usize, segs: &[Interval]) -> bool {
        if pos.first_in_group {
            let buf = &mut self.ivbufs[pos_idx];
            buf.clear();
            buf.extend_from_slice(segs);
        } else {
            let (prev, rest) = self.ivbufs.split_at_mut(pos_idx);
            intersect_into(&prev[pos_idx - 1], segs, &mut rest[0]);
        }
        let need = self.selector.min_span(pos.group);
        let buf = &self.ivbufs[pos_idx];
        if need <= 1 {
            !buf.is_empty()
        } else {
            longest_run(buf) >= need
        }
    }

    /// Runs at a group's last slot, where its coappearance is complete: checks
    /// the exact relation against the previous group, then publishes the window
    /// the next group must fall in. Returns whether the group holds.
    ///
    /// Only the relation check can fail, so the head group never returns
    /// `false` — it has no predecessor to be incompatible with, and publishing
    /// a window cannot fail.
    fn close_group(&mut self, pos: Position, pos_idx: usize) -> bool {
        if pos.group != 0 {
            let op = self
                .selector
                .chain()
                .rel_into(pos.group)
                .expect("not the head");
            let prev_ivs = &self.ivbufs[self.group_last_pos[pos.group - 1]];
            // `intervals_compatible` is existential over pairs, so this rejects
            // only when *every* (prev interval, this interval) pair fails:
            //   ThenRelaxed — none where `curr.end >= prev.start + min_gap`, and
            //                 within `prev.end + max_gap` when bounded;
            //   Contains    — no interval of this group nested inside one of the
            //                 previous group's.
            //
            // `presence_reaching_region` cannot have caught this: it tests each
            // object's *raw* segments against the window individually, whereas
            // this tests the group's n-way intersection. Every member can reach
            // the window while their intersection does not — e.g. window
            // `[50, ∞)` with `b = [0,20] ∪ [50,60]` and `c = [10,15] ∪ [70,80]`,
            // where both pass individually but `coappear(b, c) = [10,15]`.
            if !intervals_compatible(prev_ivs, &self.ivbufs[pos_idx], op) {
                return false;
            }
        }
        if pos.group + 1 < self.selector.num_groups() {
            let op = self
                .selector
                .chain()
                .rel_into(pos.group + 1)
                .expect("group + 1 is not the head");
            feasible_region_into(&self.ivbufs[pos_idx], op, &mut self.regions[pos.group + 1]);
        }
        true
    }

    /// `binding()[slot]` = ObjId currently bound to that slot. Valid after
    /// `advance()` returned `true`, until the next `advance()` call.
    pub fn bindings(&self) -> &SmallVec<[ObjId; 4]> {
        &self.bindings
    }

    /// ObjId currently bound to `slot`. Valid after `advance()` returned
    /// `true`, until the next `advance()` call.
    pub fn get(&self, slot: crate::Slot) -> ObjId {
        self.bindings[slot]
    }

    pub fn num_groups(&self) -> usize {
        self.selector.num_groups()
    }

    /// Coappearance intervals of group `g` for the current candidate — the
    /// witness for why it was admitted, and what the caller clips predicate
    /// evaluation to.
    pub fn group_bounds(&self, g: usize) -> &[Interval] {
        &self.ivbufs[self.group_last_pos[g]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feasible_region_into_then_merges() {
        // Windows open at prev.start + min_gap and close at prev.end + max_gap.
        let mut out = Vec::new();

        // [1,15] and [41,55] are far apart and stay separate.
        feasible_region_into(
            &[(0, 10), (40, 50)],
            &GroupRel::ThenRelaxed {
                min_gap: 1,
                max_gap: Some(5),
            },
            &mut out,
        );
        assert_eq!(out, vec![(1, 15), (41, 55)]);

        // [1,15] and [13,25] overlap, so they merge on the fly.
        feasible_region_into(
            &[(0, 10), (12, 20)],
            &GroupRel::ThenRelaxed {
                min_gap: 1,
                max_gap: Some(5),
            },
            &mut out,
        );
        assert_eq!(out, vec![(1, 25)]);

        feasible_region_into(
            &[(0, 10)],
            &GroupRel::ThenRelaxed {
                min_gap: 2,
                max_gap: None,
            },
            &mut out,
        );
        assert_eq!(out, vec![(2, u32::MAX)]);
    }

    #[test]
    fn feasible_region_into_contains_copies_prev() {
        let mut out = vec![(99, 100)];
        feasible_region_into(&[(5, 15), (20, 30)], &GroupRel::Contains, &mut out);
        assert_eq!(out, vec![(5, 15), (20, 30)]);
    }

    use crate::synth::{SynthConfig, gen_db_heavy_tail};
    use crate::{GroupChain, ObjSelector};

    /// Last frame `oid` covers, the quantity both rollups summarise.
    fn end_of(db: &Trqs, oid: ObjId) -> u32 {
        db.poly_trajs()[oid as usize]
            .canonical_segments()
            .last()
            .map_or(0, |&(_, e)| e)
    }

    /// Runs `f` against a cursor whose slot 1 enumerates `oids`, so the scan
    /// bounds built for that slot can be inspected directly. Takes a callback
    /// because the cursor borrows the selector it came from.
    fn with_cursor<R>(db: &Trqs, oids: Vec<ObjId>, f: impl FnOnce(&CandidateCursor<'_>) -> R) -> R {
        let sel = ObjSelector::new([vec![0], oids], GroupChain::new([0, 1], []));
        f(&sel.candidates(db))
    }

    /// The prefix skip must be both sound (everything below it really did end
    /// before the window) and maximal (the entry it lands on did not) — which is
    /// what makes the binary search over a *running* maximum the right rollup.
    #[test]
    fn the_prefix_skip_is_sound_and_maximal() {
        let db = gen_db_heavy_tail(
            &SynthConfig {
                num_objs: 400,
                timeline: 20_000,
                segs_per_obj: (1, 1),
                seg_len: (5, 20),
                seed: 11,
            },
            0,
            0,
        );
        with_cursor(&db, (0..db.num_objs()).collect(), |cursor| {
            let pmax = &cursor.slot_pmax_end[1];
            let list = &cursor.slot_by_start[1];

            assert!(
                pmax.windows(2).all(|w| w[0] <= w[1]),
                "slot_pmax_end must be non-decreasing for the binary search to be valid"
            );

            let mut fired = false;
            for ws in (0..20_000).step_by(500) {
                let lo = pmax.partition_point(|&m| m < ws);
                for &(_, oid) in &list[..lo] {
                    let e = end_of(&db, oid);
                    assert!(e < ws, "skipped oid {oid} ends at {e}, which reaches {ws}");
                }
                if let Some(&(_, oid)) = list.get(lo) {
                    // Maximal: had the prefix gone one further it would have dropped
                    // a candidate that can still overlap.
                    assert!(
                        end_of(&db, oid) >= ws || lo == 0,
                        "the prefix stopped short of a candidate that ended before {ws}"
                    );
                }
                fired |= lo > 0;
            }
            assert!(
                fired,
                "no window skipped any prefix; the test proves nothing"
            );
        });
    }

    /// The block rollup recovers what the prefix skip cannot: with one long-lived
    /// candidate sorting to the front, `slot_pmax_end` is pinned at 0 for every
    /// window inside that object's life, yet the short candidates behind it are
    /// long dead and must still be skipped a block at a time.
    #[test]
    fn dead_blocks_are_skipped_past_a_long_lived_candidate() {
        let db = gen_db_heavy_tail(
            &SynthConfig {
                num_objs: 400,
                timeline: 20_000,
                segs_per_obj: (1, 1),
                seg_len: (5, 20),
                seed: 5,
            },
            1,
            19_900,
        );
        with_cursor(&db, (0..db.num_objs()).collect(), |cursor| {
            let list = &cursor.slot_by_start[1];
            let ws = 19_500;

            // The long-lived object starts in [0, 100] and reaches past `ws`, so it
            // sits in block 0 and pins the running maximum from the very first entry.
            assert_eq!(
                cursor.slot_pmax_end[1].partition_point(|&m| m < ws),
                0,
                "expected the prefix skip to be pinned by the long-lived candidate"
            );

            // Agreement with a straightforward per-block scan, from every block edge.
            for start in (0..list.len()).step_by(SKIP_BLOCK) {
                let want = {
                    let mut j = start;
                    while j < list.len() {
                        let b = j / SKIP_BLOCK;
                        let block = &list[b * SKIP_BLOCK..((b + 1) * SKIP_BLOCK).min(list.len())];
                        if block.iter().any(|&(_, o)| end_of(&db, o) >= ws) {
                            break;
                        }
                        j = (b + 1) * SKIP_BLOCK;
                    }
                    j
                };
                assert_eq!(
                    cursor.skip_dead_blocks(1, start, ws),
                    want,
                    "block skip disagreed starting from {start}"
                );
            }

            // And it genuinely fires past block 0, which is the whole point.
            assert!(
                cursor.skip_dead_blocks(1, SKIP_BLOCK, ws) > SKIP_BLOCK,
                "expected dead blocks behind the long-lived candidate"
            );
            // With nothing bounding the position, no block may be skipped.
            assert_eq!(cursor.skip_dead_blocks(1, 0, 0), 0);
        });
    }
}
