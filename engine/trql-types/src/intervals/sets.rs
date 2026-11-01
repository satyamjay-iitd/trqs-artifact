//! The 5-tuple interval-set algebra.
//!
//! An [`IntervalSet`] `(S, E, c, L, R)` represents the set of valid
//! `(start, end)` pairs:
//!
//! ```text
//! T(S, E, c, L, R) = { (s, e) | S ≤ s ≤ L, R ≤ e ≤ E, e - s + 1 ≥ c }
//! ```
//!
//! An [`IntervalSets`] is a list of those; its semantic content is the union of
//! their pair sets. Raw `(start, end)` lists — what a predicate evaluator
//! produces — are lifted in by [`IntervalSets::new`] and read back out by
//! [`IntervalSets::maximal`].

use super::Interval;

/// 5-tuple interval-set representation.
///
/// Encodes all the interval set contained within the bound whose size ≥ c,
/// and there are further restrictions on their start and end.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IntervalSet {
    /// `S`: lower bound on start position.
    pub s: u32,
    /// `E`: upper bound on end position.
    pub e: u32,
    /// `c`: minimum required length (`e − s + 1 ≥ c`).
    pub c: u32,
    /// `L`: upper bound on start position (`s ≤ L`).
    pub l: u32,
    /// `R`: lower bound on end position (`e ≥ R`).
    pub r: u32,
}

impl IntervalSet {
    /// Creates a Interval from a DURATION interval `[S, E]` with minimum length `c`.
    ///
    /// Sets `L = E` and `R = S` (vacuous bounds). Returns `None` if the interval
    /// is too short to satisfy the length constraint.
    pub fn with_duration(s: u32, e: u32, c: u32) -> Option<Self> {
        // Check s > e before computing e - s to avoid underflow in debug mode.
        if c < 1 || s > e || (e - s).saturating_add(1) < c {
            return None;
        }
        Some(Self {
            s,
            e,
            c,
            l: e,
            r: s,
        })
    }

    /// Creates a base Interval from a maximal interval `[S, E]` with `c = 1`.
    ///
    /// This is the encoding for any leaf predicate's output interval.
    pub fn all_between(s: u32, e: u32) -> Option<Self> {
        Self::with_duration(s, e, 1)
    }

    /// Creates a Interval from raw parameters. Returns `None` if infeasible.
    pub fn new(s: u32, e: u32, c: u32, l: u32, r: u32) -> Option<Self> {
        let ft = Self { s, e, c, l, r };
        if ft.is_feasible() { Some(ft) } else { None }
    }

    /// Returns `true` if this 5-tuple represents a non-empty set of `(s, e)` pairs.
    pub fn is_feasible(&self) -> bool {
        // s <= e must be checked before e - s to avoid underflow.
        self.c >= 1
            && self.s <= self.e
            && self.s <= self.l
            && self.r <= self.e
            && (self.e - self.s).saturating_add(1) >= self.c
    }

    /// Returns `true` if any `(s, e)` pair actually exists.
    ///
    /// Stricter than `is_feasible`: also checks that the tightest start and end
    /// can form a pair of sufficient length.
    pub fn is_nonempty(&self) -> bool {
        if !self.is_feasible() {
            return false;
        }
        // Best case: s = self.s (smallest start), e = self.e (largest end)
        // If even that doesn't satisfy length, empty.
        // But we also need s ≤ L and e ≥ R.
        // The most constrained pair is s = L, e = R.
        // If R - L + 1 < c AND E - S + 1 >= c, there may still be valid pairs
        // where s < L or e > R. The length constraint is diagonal.
        // A non-empty set exists iff ∃ s ∈ [S, L], e ∈ [R, E] with e - s + 1 ≥ c.
        // Best: s = S, e = E → length = E - S + 1 ≥ c (checked in is_feasible).
        // But we also need S ≤ L and R ≤ E (checked in is_feasible).
        // So is_feasible implies nonempty when E - S + 1 ≥ c, S ≤ L, R ≤ E.
        true
    }

    /// AND of two 5-tuples. Returns `None` if the result is infeasible.
    ///
    /// # Formula
    /// ```text
    /// S_out = max(S₁, S₂)
    /// E_out = min(E₁, E₂)
    /// c_out = max(c₁, c₂)
    /// L_out = min(L₁, L₂)
    /// R_out = max(R₁, R₂)
    /// ```
    pub fn and_pair(&self, other: &Self) -> Option<Self> {
        Self::new(
            self.s.max(other.s),
            self.e.min(other.e),
            self.c.max(other.c),
            self.l.min(other.l),
            self.r.max(other.r),
        )
    }

    /// THEN of two 5-tuples (adjacent chaining: s₂ = e₁ + 1). Returns `None` if infeasible.
    /// This is an alias for the 0-gap sequence.
    ///
    /// # Formula (Theorem 4.1)
    /// ```text
    /// S_out = S₁
    /// E_out = E₂
    /// c_out = c₁ + c₂
    /// L_out = min(L₁, E₁ − c₁ + 1, L₂ − c₁)
    /// R_out = max(R₂, R₁ + c₂, S₂ + c₂ − 1)
    /// ```
    pub fn then_pair(&self, other: &Self) -> Option<Self> {
        self.then_pair_with_gap(other, 0, Some(0))
    }

    /// Generalized THEN with gap constraints. Returns `None` if infeasible.
    ///
    /// # Formula (Theorem 4.1)
    /// ```text
    /// S_out = S₁
    /// E_out = E₂
    /// c_out = c₁ + c₂ + g_min
    /// L_out = min(L₁, E₁ − c₁ + 1, L₂ − c₁ - g_min)
    /// R_out = max(R₂, R₁ + c₂ + g_min, S₂ + c₂ − 1)
    /// ```
    ///
    /// # Feasibility guards
    /// The formula is only valid behind these conditions:
    /// ```text
    /// R₁ ≤ E₁, S₂ ≤ L₂, 1 ≤ c₁, 1 ≤ c₂   structural (is_feasible)
    /// S₂ ≤ E₁ + g_max + 1                 the second interval can start soon enough
    /// R₁ + g_min + 1 ≤ L₂                 the earliest second start is within L₂
    /// c₁ + g_min ≤ L₂                     keeps the L₂ − c₁ − g_min term well-defined
    /// ```
    /// Mechanized as `SplitIntervalSet.thenGapOp` / `thenGapOp_spec` in
    /// `lean-proof/Intervalset/ThenWithGap.lean`.
    pub fn then_pair_with_gap(&self, other: &Self, g_min: u32, g_max: Option<u32>) -> Option<Self> {
        if !self.is_feasible() || !other.is_feasible() {
            return None;
        }
        if let Some(g_max) = g_max {
            if other.s > self.e.saturating_add(g_max).saturating_add(1) {
                return None;
            }
        }
        // `checked_sub` encodes `R₁ + g_min + 1 ≤ L₂` and `c₁ + g_min ≤ L₂` exactly: an
        // underflow means the guard is violated, which is the same `None`. Writing them as
        // additions would overflow at the u32::MAX boundary instead.
        let l2_less_gap = other.l.checked_sub(g_min)?;
        if self.r > l2_less_gap.checked_sub(1)? {
            return None;
        }
        let l2_term = l2_less_gap.checked_sub(self.c)?; // L₂ − c₁ − g_min

        let s_out = self.s;
        // E₁ + 1 − c₁, written so it cannot overflow when E₁ == u32::MAX (1 ≤ c₁ ≤ E₁ + 1).
        let l_out = self.l.min(self.e - (self.c - 1)).min(l2_term);
        // Saturating past u32::MAX only tightens the tuple; `Self::new` then rejects it.
        let r_out = other
            .r
            .max(self.r.saturating_add(other.c).saturating_add(g_min))
            .max(other.s.saturating_add(other.c).saturating_sub(1));
        let e_out = other.e;
        let c_out = self.c.saturating_add(other.c).saturating_add(g_min);

        Self::new(s_out, e_out, c_out, l_out, r_out)
    }

    /// OVERLAP of two 5-tuples (Allen's *overlaps*). Returns `None` if infeasible.
    ///
    /// Merges two properly overlapping intervals: the second starts strictly after the
    /// first starts, meets it, and ends strictly after it —
    /// `[a₁, b₁] OVERLAP [a₂, b₂] = [a₁, b₂]` when `a₁ < a₂ ≤ b₁ < b₂`.
    ///
    /// # Formula
    /// ```text
    /// S_out = S₁
    /// E_out = E₂
    /// c_out = max(c₁ + 1, c₂ + 1, 3)
    /// L_out = min(L₁, E₁ − c₁ + 1, E₁ − 1, L₂ − 1)
    /// R_out = max(R₂, R₁ + 1, S₂ + c₂ − 1, S₂ + 1)
    /// ```
    /// The floor of `3` on `c_out` is the smallest proper overlap: `a < a+1 ≤ a+1 < a+2`
    /// already spans three frames.
    ///
    /// # Feasibility guards
    /// ```text
    /// R₁ ≤ E₁, S₂ ≤ L₂, 1 ≤ c₁, 1 ≤ c₂   structural (is_feasible)
    /// S₂ ≤ E₁                             the second interval starts inside the first
    /// S₁ + 1 ≤ E₁                         keeps the E₁ − 1 term well-defined
    /// S₁ + 1 ≤ L₂                         keeps the L₂ − 1 term well-defined
    /// ```
    /// Mechanized as `SplitIntervalSet.overlapOp` / `overlapOp_spec` in
    /// `lean-proof/Intervalset/Overlap.lean`.
    pub fn overlap(&self, other: &Self) -> Option<Self> {
        if !self.is_feasible() || !other.is_feasible() {
            return None;
        }
        if other.s > self.e || self.s >= self.e || self.s >= other.l {
            return None;
        }

        let s_out = self.s;
        // The guards above make `E₁ − 1` and `L₂ − 1` non-negative; `E₁ + 1 − c₁` is
        // written as `E₁ − (c₁ − 1)` so it cannot overflow when E₁ == u32::MAX.
        let l_out = self
            .l
            .min(self.e - (self.c - 1))
            .min(self.e - 1)
            .min(other.l - 1);
        // Saturating past u32::MAX only tightens the tuple; `Self::new` then rejects it.
        let r_out = other
            .r
            .max(self.r.saturating_add(1))
            .max(other.s.saturating_add(other.c).saturating_sub(1))
            .max(other.s.saturating_add(1));
        let e_out = other.e;
        let c_out = self
            .c
            .saturating_add(1)
            .max(other.c.saturating_add(1))
            .max(3);

        Self::new(s_out, e_out, c_out, l_out, r_out)
    }

    /// CONTAINS of two 5-tuples (Allen's *contains*, `di`). Returns `None` if infeasible.
    ///
    /// Keeps the *first* interval when the second lies strictly inside it —
    /// `[a₁, b₁] CONTAINS [a₂, b₂] = [a₁, b₁]` when `a₁ < a₂ ∧ b₂ < b₁`. Unlike AND/THEN
    /// the result is shaped like the lhs: `S_out = S₁` and `E_out = E₁`.
    ///
    /// # Formula
    /// ```text
    /// S_out = S₁
    /// E_out = E₁
    /// c_out = max(c₁, c₂ + 2)
    /// L_out = min(L₁, L₂ − 1, E₂ − c₂)
    /// R_out = max(R₁, R₂ + 1, S₂ + c₂)
    /// ```
    /// The `c₂ + 2` size floor reflects that the lhs must *strictly* enclose an interval
    /// of size `≥ c₂`, so it needs one extra frame on each side.
    ///
    /// # Feasibility guards
    /// ```text
    /// S₂ ≤ L₂, R₂ ≤ E₂, 1 ≤ c₂            structural (is_feasible)
    /// S₁ + 1 ≤ L₂                         keeps the L₂ − 1 term well-defined
    /// c₂ ≤ E₂                             keeps the E₂ − c₂ term well-defined
    /// ```
    /// Mechanized as `SplitIntervalSet.containsOp` / `containsOp_spec` in
    /// `lean-proof/Intervalset/Contains.lean`.
    pub fn contains(&self, other: &Self) -> Option<Self> {
        if !self.is_feasible() || !other.is_feasible() {
            return None;
        }
        if self.s >= other.l || other.c > other.e {
            return None;
        }

        let s_out = self.s;
        // The guards above make `L₂ − 1` and `E₂ − c₂` non-negative.
        let l_out = self.l.min(other.l - 1).min(other.e - other.c);
        // Saturating past u32::MAX only tightens the tuple; `Self::new` then rejects it.
        let r_out = self
            .r
            .max(other.r.saturating_add(1))
            .max(other.s.saturating_add(other.c));
        let e_out = self.e;
        let c_out = self.c.max(other.c.saturating_add(2));

        Self::new(s_out, e_out, c_out, l_out, r_out)
    }
}

/// A intervals is a set of non-redundant 5-tuples.
///
/// The semantic content is the union: `T(Intervals) = ∪ T(I)` for `I` in the intervals.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IntervalSets(Vec<IntervalSet>);

impl IntervalSets {
    /// Creates a intervals from raw `(start, end)` intervals, each with `c = 1`.
    ///
    /// This is the standard way to lift leaf predicate output into the algebra.
    /// Input is sorted by `(s, e)` to establish the invariant required for loop pruning.
    pub fn new(mut intervals: Vec<Interval>) -> Self {
        intervals.sort_unstable_by_key(|&(s, e)| (s, e));

        let intervals: Vec<Interval> = intervals.into_iter().fold(Vec::new(), |mut acc, (s, e)| {
            if let Some(last) = acc.last_mut() {
                if s <= last.1 + 1 {
                    last.1 = last.1.max(e);
                    return acc;
                }
            }
            acc.push((s, e));
            acc
        });

        let tuples: Vec<IntervalSet> = intervals
            .iter()
            .filter_map(|&(s, e)| IntervalSet::all_between(s, e))
            .collect();
        Self(tuples)
    }

    /// Creates a intervals from a single Interval. Returns empty if infeasible.
    pub fn single(ft: IntervalSet) -> Self {
        if ft.is_feasible() {
            Self(vec![ft])
        } else {
            Self::empty()
        }
    }

    /// Creates an empty intervals.
    pub fn empty() -> Self {
        Self(Vec::new())
    }

    /// AND of two intervals: cross-product of individual ANDs.
    ///
    /// Complexity: O(|A| × |B|), output size ≤ |A| × |B|.
    pub fn intersect(&self, other: &Self) -> Self {
        let mut result = Vec::new();
        for ft_a in &self.0 {
            for ft_b in &other.0 {
                // other is sorted by s; once ft_b.s > ft_a.e, s_out > e_out → infeasible.
                if ft_b.s > ft_a.e {
                    break;
                }
                if let Some(ft) = ft_a.and_pair(ft_b) {
                    result.push(ft);
                }
            }
        }
        let mut out = Self(result);
        out.canonicalize();
        out
    }

    /// OR of two intervals: set-theoretic union.
    ///
    /// Complexity: O(|A| + |B|), output size = |A| + |B|.
    pub fn union(&self, other: &Self) -> Self {
        let mut result = self.0.clone();
        result.extend_from_slice(&other.0);
        let mut out = Self(result);
        out.canonicalize();
        out
    }

    /// Generalized THEN composition with a strictly bounded or unbounded gap.
    ///
    /// Complexity: O(|A| × |B|), output size ≤ |A| × |B|.
    pub fn then_with_gap(&self, other: &Self, min_gap: u32, max_gap: Option<u32>) -> Self {
        let mut result = Vec::new();
        for ft_a in &self.0 {
            for ft_b in &other.0 {
                if let Some(mg) = max_gap {
                    // Because `other` is sorted by `s` (canonicalize invariant),
                    // if ft_b.s exceeds the max allowed start position, all subsequent
                    // ft_b will also exceed it, so we can safely break inner loop.
                    if ft_b.s > ft_a.e.saturating_add(mg).saturating_add(1) {
                        break;
                    }
                }
                if let Some(ft) = ft_a.then_pair_with_gap(ft_b, min_gap, max_gap) {
                    result.push(ft);
                }
            }
        }
        let mut out = Self(result);
        out.canonicalize();
        out
    }

    /// OVERLAP (Allen's *overlaps*) of two intervals: cross-product of individual OVERLAPs.
    ///
    /// Complexity: O(|A| × |B|), output size ≤ |A| × |B|.
    pub fn overlap(&self, other: &Self) -> Self {
        let mut result = Vec::new();
        for ft_a in &self.0 {
            for ft_b in &other.0 {
                // `other` is sorted by `s` (canonicalize invariant), so once ft_b.s passes
                // ft_a.e the `S₂ ≤ E₁` guard fails for every later ft_b too.
                if ft_b.s > ft_a.e {
                    break;
                }
                if let Some(ft) = ft_a.overlap(ft_b) {
                    result.push(ft);
                }
            }
        }
        let mut out = Self(result);
        out.canonicalize();
        out
    }

    /// CONTAINS (Allen's *contains*) of two intervals: cross-product of individual CONTAINS.
    ///
    /// Complexity: O(|A| × |B|), output size ≤ |A| × |B|.
    pub fn contains(&self, other: &Self) -> Self {
        let mut result = Vec::new();
        for ft_a in &self.0 {
            for ft_b in &other.0 {
                // Feasibility needs `R_out ≥ S₂ + c₂` to stay within `E_out = E₁`, and
                // c₂ ≥ 1, so `ft_b.s < ft_a.e` is required. `other` is sorted by `s`, so
                // every later ft_b fails too.
                if ft_b.s >= ft_a.e {
                    break;
                }
                if let Some(ft) = ft_a.contains(ft_b) {
                    result.push(ft);
                }
            }
        }
        let mut out = Self(result);
        out.canonicalize();
        out
    }

    /// DURATION: applies a minimum length constraint to every tuple in a intervals.
    ///
    /// Complexity: O(|S|), output size ≤ |S|.
    pub fn with_min_duration(&self, min_len: u32) -> Self {
        let result: Vec<IntervalSet> = self
            .0
            .iter()
            .filter_map(|ft| {
                let new_c = ft.c.max(min_len);
                IntervalSet::new(ft.s, ft.e, new_c, ft.l, ft.r)
            })
            .collect();
        Self(result)
    }

    /// Returns `true` if the intervals contains no 5-tuples.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns the frame window `[lo, hi]` that an rhs candidate must overlap for a
    /// `self Then rhs` with the given gap to be satisfiable.
    ///
    /// For each lhs interval end `e`, valid rhs starts are `[e + min_gap, e + max_gap]`.
    /// The returned window is the union over all lhs ends, i.e.
    /// `lo = min(e) + min_gap`, `hi = max(e) + max_gap` (or `u32::MAX` if unbounded).
    /// Returns `None` if `self` is empty.
    pub fn then_rhs_window(&self, min_gap: u32, max_gap: Option<u32>) -> Option<Interval> {
        if self.0.is_empty() {
            return None;
        }
        let min_e = self.0.first().unwrap().e;
        let max_e = self.0.last().unwrap().e;
        let lo = min_e.saturating_add(min_gap);
        let hi = max_gap
            .map(|mg| max_e.saturating_add(mg))
            .unwrap_or(u32::MAX);
        Some((lo, hi))
    }

    /// Returns an iterator over all frame IDs covered by this `Intervals`, sorted and deduplicated.
    ///
    /// Each tuple contributes the range `[ft.s, ft.e]`. Overlapping windows are merged
    /// in a single O(n) pass (valid because `canonicalize` keeps the inner vec sorted by `ft.s`).
    pub fn frames_iter(&self) -> impl Iterator<Item = u32> {
        let mut merged: Vec<Interval> = Vec::new();
        for ft in &self.0 {
            match merged.last_mut() {
                Some(last) if ft.s <= last.1 => last.1 = last.1.max(ft.e),
                _ => merged.push((ft.s, ft.e)),
            }
        }
        merged.into_iter().flat_map(|(s, e)| s..=e)
    }

    pub fn maximal(&self) -> Vec<Interval> {
        let mut merged: Vec<Interval> = Vec::new();
        for ft in &self.0 {
            match merged.last_mut() {
                Some(last) if ft.s <= last.1 => last.1 = last.1.max(ft.e),
                _ => merged.push((ft.s, ft.e)),
            }
        }
        merged
    }

    //^                     API ends

    /// Returns the number of 5-tuples in this intervals.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Constructs Intervals directly from a vector of 5-tuples.
    pub fn from_intervals(tuples: Vec<IntervalSet>) -> Self {
        Self(tuples)
    }

    /// Removes infeasible tuples and exact duplicates.
    pub fn canonicalize(&mut self) {
        self.0.retain(|ft| ft.is_feasible());
        self.0.sort_by_key(|ft| (ft.s, ft.e, ft.c, ft.l, ft.r));
        self.0.dedup();
    }

    /// THEN of two intervals: cross-product of individual THENs.
    ///
    /// Complexity: O(|A| × |B|), output size ≤ |A| × |B|.
    pub fn then_intervals(&self, other: &Self) -> Self {
        self.then_with_gap(other, 0, Some(0))
    }

    /// N-fold THEN chain: `intervals[0]` THEN `intervals[1]` THEN … THEN `intervals[n-1]`.
    ///
    /// Left-fold computation. Returns empty if any intermediate result is empty.
    pub fn chain(intervals: &[Self]) -> Self {
        if intervals.is_empty() {
            return Self::empty();
        }
        let gaps = vec![(0, Some(0)); intervals.len().saturating_sub(1)];
        Self::chain_with_gap(intervals, &gaps)
    }

    /// Generalized N-fold sequence chain with custom gap steps.
    pub fn chain_with_gap(intervals: &[Self], gaps: &[(u32, Option<u32>)]) -> Self {
        if intervals.is_empty() {
            return Self::empty();
        }
        assert_eq!(
            intervals.len(),
            gaps.len() + 1,
            "Mismatched intervals and gaps lengths"
        );

        let mut current = intervals[0].clone();
        for i in 1..intervals.len() {
            let (g_min, g_max) = gaps[i - 1];
            current = current.then_with_gap(&intervals[i], g_min, g_max);
            if current.is_empty() {
                return Self::empty();
            }
        }
        current
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    impl IntervalSets {
        /// Total number of `(s, e)` pairs across all 5-tuples.
        ///
        /// Sums `Interval::pair_count()` for each tuple in O(n). **Exact** when the
        /// pair sets of the individual tuples are disjoint (always true for leaf inputs
        /// from `from_intervals`). May overcount after AND/THEN if pair sets overlap.
        pub fn total_frames(&self) -> usize {
            self.0.iter().map(|ft| ft.pair_count()).sum()
        }

        // Testing
        /// Enumerates all `(s, e)` pairs across all 5-tuples. For testing only.
        pub fn pair_set(&self) -> BTreeSet<Interval> {
            let mut pairs = BTreeSet::new();
            for ft in &self.0 {
                pairs.extend(ft.pair_set());
            }
            pairs
        }

        /// Brute-force TRUE set for OVERLAP: every `(s₁, e₂)` from a properly
        /// overlapping pair `s₁ < s₂ ≤ e₁ < e₂`. For testing only.
        pub fn brute_force_overlap(a: &Self, b: &Self) -> BTreeSet<Interval> {
            let mut out = BTreeSet::new();
            let b_pairs = b.pair_set();
            for &(s1, e1) in &a.pair_set() {
                for &(s2, e2) in &b_pairs {
                    if s1 < s2 && s2 <= e1 && e1 < e2 {
                        out.insert((s1, e2));
                    }
                }
            }
            out
        }

        /// Brute-force TRUE set for CONTAINS: every lhs pair `(s₁, e₁)` that strictly
        /// encloses some rhs pair, `s₁ < s₂ ≤ e₂ < e₁`. For testing only.
        pub fn brute_force_contains(a: &Self, b: &Self) -> BTreeSet<Interval> {
            let mut out = BTreeSet::new();
            let b_pairs = b.pair_set();
            for &(s1, e1) in &a.pair_set() {
                if b_pairs.iter().any(|&(s2, e2)| s1 < s2 && e2 < e1) {
                    out.insert((s1, e1));
                }
            }
            out
        }

        /// Generalized gap brute-force TRUE set.
        pub fn brute_force_chain_with_gap(
            intervals: &[Self],
            gaps: &[(u32, Option<u32>)],
        ) -> BTreeSet<Interval> {
            if intervals.is_empty() {
                return BTreeSet::new();
            }
            let n = intervals.len();
            let stage_pairs: Vec<BTreeSet<Interval>> =
                intervals.iter().map(|s| s.pair_set()).collect();

            if n == 1 {
                return stage_pairs[0].clone();
            }
            let mut current = stage_pairs[0].clone();
            for k in 1..n {
                let mut next = BTreeSet::new();
                let (g_min, g_max) = gaps[k - 1];
                for &(s, e_prev) in &current {
                    for &(s_next, e_next) in &stage_pairs[k] {
                        if s_next > e_prev {
                            let dist = s_next - e_prev - 1;
                            if dist >= g_min && g_max.map_or(true, |m| dist <= m) {
                                next.insert((s, e_next));
                            }
                        }
                    }
                }
                current = next;
                if current.is_empty() {
                    return BTreeSet::new();
                }
            }
            current
        }
        /// Brute-force TRUE set for an n-fold THEN chain.
        ///
        /// Enumerates all split points to find valid (s, e) pairs. For testing only.
        pub fn brute_force_then_chain(intervals: &[Self]) -> BTreeSet<Interval> {
            let gaps = vec![(0, Some(0)); intervals.len().saturating_sub(1)];
            Self::brute_force_chain_with_gap(intervals, &gaps)
        }
    }

    impl IntervalSet {
        // ─── Interval construction ─────────────────────────────────
        /// Enumerates all `(s, e)` pairs in this 5-tuple's set. For testing only.
        pub fn pair_set(&self) -> BTreeSet<Interval> {
            let mut pairs = BTreeSet::new();
            if !self.is_feasible() {
                return pairs;
            }
            for s in self.s..=self.l {
                // s + c - 1: since c >= 1, use s.saturating_add(c - 1) to avoid overflow.
                let e_min = self.r.max(s.saturating_add(self.c - 1));
                for e in e_min..=self.e {
                    pairs.insert((s, e));
                }
            }
            pairs
        }
        /// Number of valid `(s, e)` pairs. O(1) closed-form formula.
        ///
        /// Splits the sum over `s ∈ [S, L]` at the crossover point `s_split = R − c + 1`
        /// where `e_min(s)` transitions from the constant `R` to the diagonal `s + c − 1`.
        ///
        /// - **Case 1** (`s < s_split`): each row contributes `E − R + 1` pairs → rectangular block.
        /// - **Case 2** (`s ≥ s_split`): each row contributes `E − s − c + 2` pairs → triangular sum.
        pub fn pair_count(&self) -> usize {
            if !self.is_feasible() {
                return 0;
            }
            // Use i64 for signed intermediate arithmetic; values are bounded by frame counts.
            let s_lo = self.s as i64;
            let s_hi = self.l as i64;
            let e_hi = self.e as i64;
            let c = self.c as i64;
            let r = self.r as i64;

            // Crossover: smallest s where s + c − 1 ≥ R.
            let s_split = r - c + 1;

            // Case 1: s ∈ [s_lo, min(s_hi, s_split − 1)], e_min = R (constant).
            let c1_end = s_hi.min(s_split - 1);
            let n1 = (c1_end - s_lo + 1).max(0);
            let per1 = (e_hi - r + 1).max(0);
            let total1 = n1 * per1;

            // Case 2: s ∈ [max(s_lo, s_split), min(s_hi, E−c+1)], e_min = s + c − 1.
            // Upper bound on s: need E − (s+c−1) + 1 > 0  →  s ≤ E − c + 1.
            let c2_start = s_lo.max(s_split);
            let c2_end = s_hi.min(e_hi - c + 1);
            let total2 = if c2_start > c2_end {
                0
            } else {
                let n2 = c2_end - c2_start + 1;
                let a = e_hi - c2_start - c + 2; // count at s = c2_start
                let b = e_hi - c2_end - c + 2; // count at s = c2_end
                n2 * (a + b) / 2 // always integer: see proof in doc
            };

            (total1 + total2) as usize
        }
    }

    // // ─── Interval construction ─────────────────────────────────
    // /// Enumerates all `(s, e)` pairs in this 5-tuple's set. For testing only.
    // fn pair_set(interval_set: &IntervalSet) -> BTreeSet<Interval> {
    //     let mut pairs = BTreeSet::new();
    //     if !interval_set.is_feasible() {
    //         return pairs;
    //     }
    //     for s in interval_set.s..=interval_set.l {
    //         // s + c - 1: since c >= 1, use s.saturating_add(c - 1) to avoid overflow.
    //         let e_min = interval_set.r.max(s.saturating_add(interval_set.c - 1));
    //         for e in e_min..=interval_set.e {
    //             pairs.insert((s, e));
    //         }
    //     }
    //     pairs
    // }

    #[test]
    fn base_interval() {
        let ft = IntervalSet::all_between(1, 5).unwrap();
        assert_eq!(
            ft,
            IntervalSet {
                s: 1,
                e: 5,
                c: 1,
                l: 5,
                r: 1
            }
        );
        assert!(ft.is_feasible());
    }

    #[test]
    fn duration_interval() {
        let ft = IntervalSet::with_duration(1, 10, 3).unwrap();
        assert_eq!(ft.c, 3);
        assert_eq!(ft.l, 10); // vacuous
        assert_eq!(ft.r, 1); // vacuous
    }

    #[test]
    fn infeasible_too_short() {
        assert!(IntervalSet::with_duration(1, 3, 5).is_none());
    }

    #[test]
    fn infeasible_inverted() {
        assert!(IntervalSet::with_duration(5, 3, 1).is_none());
    }

    #[test]
    fn pair_set_base() {
        let ft = IntervalSet::all_between(1, 3).unwrap();
        let ps = ft.pair_set();
        // All sub-intervals of [1,3] with length ≥ 1:
        // (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)
        assert_eq!(ps.len(), 6);
        assert!(ps.contains(&(1, 1)));
        assert!(ps.contains(&(1, 3)));
        assert!(ps.contains(&(3, 3)));
    }

    #[test]
    fn pair_set_duration() {
        let ft = IntervalSet::with_duration(1, 5, 3).unwrap();
        let ps = ft.pair_set();
        // All (s, e) with 1 ≤ s ≤ 5, 1 ≤ e ≤ 5, e-s+1 ≥ 3
        for &(s, e) in &ps {
            assert!(s >= 1 && e <= 5 && e - s + 1 >= 3);
        }
        // (1,3), (1,4), (1,5), (2,4), (2,5), (3,5)
        assert_eq!(ps.len(), 6);
    }

    #[test]
    fn pair_set_with_tight_bounds() {
        let ft = IntervalSet::new(1, 10, 4, 3, 6).unwrap();
        let ps = ft.pair_set();
        for &(s, e) in &ps {
            assert!(s >= 1 && s <= 3);
            assert!(e >= 6 && e <= 10);
            assert!(e - s + 1 >= 4);
        }
    }

    // ─── pair_count closed-form ────────────────────────────────

    #[test]
    fn pair_count_matches_pair_set_base() {
        let ft = IntervalSet::all_between(1, 3).unwrap();
        assert_eq!(ft.pair_count(), ft.pair_set().len());
    }

    #[test]
    fn pair_count_matches_pair_set_duration() {
        let ft = IntervalSet::with_duration(1, 5, 3).unwrap();
        assert_eq!(ft.pair_count(), ft.pair_set().len());
    }

    #[test]
    fn pair_count_matches_pair_set_tight_bounds() {
        let ft = IntervalSet::new(1, 10, 4, 3, 6).unwrap();
        assert_eq!(ft.pair_count(), ft.pair_set().len());
    }

    #[test]
    fn pair_count_infeasible_is_zero() {
        // c > E - S + 1
        assert!(IntervalSet::with_duration(1, 3, 5).is_none());
        // Direct construction of an infeasible tuple
        let ft = IntervalSet {
            s: 5,
            e: 3,
            c: 1,
            l: 5,
            r: 3,
        };
        assert_eq!(ft.pair_count(), 0);
    }

    #[test]
    fn pair_count_various() {
        // Exhaustive check over a range of small 5-tuples
        for s in 0u32..6 {
            for e in s..8 {
                for c in 1..=(e - s + 1) {
                    for l in s..=e {
                        for r in s..=e {
                            if let Some(ft) = IntervalSet::new(s, e, c, l, r) {
                                assert_eq!(
                                    ft.pair_count(),
                                    ft.pair_set().len(),
                                    "mismatch for {:?}",
                                    ft
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    // ─── AND of two Intervals ──────────────────────────────────

    #[test]
    fn and_pair_c1() {
        let a = IntervalSet::all_between(1, 5).unwrap();
        let b = IntervalSet::all_between(3, 8).unwrap();
        let result = a.and_pair(&b).unwrap();
        assert_eq!(result.s, 3);
        assert_eq!(result.e, 5);
        assert_eq!(result.c, 1);

        let expected: BTreeSet<_> = a.pair_set().intersection(&b.pair_set()).copied().collect();
        assert_eq!(result.pair_set(), expected);
    }

    #[test]
    fn and_pair_different_c() {
        let a = IntervalSet::with_duration(1, 10, 3).unwrap();
        let b = IntervalSet::with_duration(5, 15, 5).unwrap();
        let result = a.and_pair(&b).unwrap();
        assert_eq!(result.s, 5);
        assert_eq!(result.e, 10);
        assert_eq!(result.c, 5); // max(3, 5)

        let expected: BTreeSet<_> = a.pair_set().intersection(&b.pair_set()).copied().collect();
        assert_eq!(result.pair_set(), expected);
    }

    #[test]
    fn and_pair_disjoint() {
        let a = IntervalSet::all_between(1, 3).unwrap();
        let b = IntervalSet::all_between(5, 8).unwrap();
        assert!(a.and_pair(&b).is_none());
    }

    #[test]
    fn and_pair_with_tight_bounds() {
        let a = IntervalSet::new(1, 10, 2, 5, 3).unwrap();
        let b = IntervalSet::new(3, 8, 3, 7, 4).unwrap();
        let result = a.and_pair(&b).unwrap();

        let expected: BTreeSet<_> = a.pair_set().intersection(&b.pair_set()).copied().collect();
        assert_eq!(result.pair_set(), expected);
    }

    #[test]
    fn and_pair_c_too_large_for_overlap() {
        let a = IntervalSet::with_duration(1, 5, 4).unwrap();
        let b = IntervalSet::with_duration(4, 6, 3).unwrap();
        // Overlap is [4, 5], length 2. max(c) = 4 > 2 → empty
        let result = a.and_pair(&b);
        if let Some(ft) = result {
            assert!(ft.pair_set().is_empty());
        }
    }

    // ─── THEN of two Intervals ─────────────────────────────────

    #[test]
    fn then_pair_basic_c1() {
        let a = IntervalSet::all_between(1, 4).unwrap();
        let b = IntervalSet::all_between(3, 8).unwrap();
        let result = a.then_pair(&b).unwrap();

        let mut expected = BTreeSet::new();
        for &(s, e1) in &a.pair_set() {
            for &(s2, e) in &b.pair_set() {
                if s2 == e1 + 1 {
                    expected.insert((s, e));
                }
            }
        }
        assert_eq!(result.pair_set(), expected);
    }

    #[test]
    fn then_pair_basic_c2() {
        let a = IntervalSet::all_between(0, 10).unwrap();

        let b = IntervalSet::all_between(0, 10).unwrap();

        let mut expected = BTreeSet::new();
        for &(s, e1) in &a.pair_set() {
            for &(s2, e) in &b.pair_set() {
                if s2 >= e1 + 1 {
                    let gap = s2 - e1 - 1;
                    if gap >= 2 && gap <= 4 {
                        expected.insert((s, e));
                    }
                }
            }
        }
        let result = a.then_pair_with_gap(&b, 2, Some(4)).unwrap();
        assert_eq!(result.pair_set(), expected);
    }

    #[test]
    fn then_pair_with_duration() {
        let a = IntervalSet::with_duration(1, 4, 2).unwrap();
        let b = IntervalSet::with_duration(3, 8, 2).unwrap();
        let result = a.then_pair(&b).unwrap();

        let mut expected = BTreeSet::new();
        for &(s, e1) in &a.pair_set() {
            for &(s2, e) in &b.pair_set() {
                if s2 == e1 + 1 {
                    expected.insert((s, e));
                }
            }
        }
        assert_eq!(result.pair_set(), expected);
    }

    #[test]
    fn then_pair_disjoint() {
        let a = IntervalSet::all_between(1, 3).unwrap();
        let b = IntervalSet::all_between(20, 25).unwrap();
        assert!(a.then_pair(&b).is_none());
    }

    #[test]
    fn then_pair_tight_overlap() {
        let a = IntervalSet::with_duration(1, 3, 2).unwrap();
        let b = IntervalSet::with_duration(3, 10, 2).unwrap();
        let result = a.then_pair(&b).unwrap();

        let mut expected = BTreeSet::new();
        for &(s, e1) in &a.pair_set() {
            for &(s2, e) in &b.pair_set() {
                if s2 == e1 + 1 {
                    expected.insert((s, e));
                }
            }
        }
        assert_eq!(result.pair_set(), expected);
    }

    // ─── OVERLAP (Allen's overlaps) ────────────────────────────

    /// Brute-force OVERLAP over two single 5-tuples, via their pair sets.
    fn brute_overlap(a: &IntervalSet, b: &IntervalSet) -> BTreeSet<Interval> {
        IntervalSets::brute_force_overlap(&IntervalSets::single(*a), &IntervalSets::single(*b))
    }

    #[test]
    fn overlap_pair_basic() {
        let a = IntervalSet::all_between(1, 3).unwrap();
        let b = IntervalSet::all_between(2, 5).unwrap();
        let result = a.overlap(&b).unwrap();

        // c_out is floored at 3, so the shortest proper overlap spans three frames.
        assert_eq!(result.c, 3);
        assert_eq!(
            result.pair_set(),
            BTreeSet::from([(1, 3), (1, 4), (1, 5), (2, 4), (2, 5)])
        );
        assert_eq!(result.pair_set(), brute_overlap(&a, &b));
    }

    #[test]
    fn overlap_pair_containment_still_overlaps() {
        // b is nested inside a: proper overlaps still exist between their sub-intervals.
        let a = IntervalSet::all_between(1, 10).unwrap();
        let b = IntervalSet::all_between(2, 5).unwrap();
        let result = a.overlap(&b).unwrap();
        assert_eq!(result.pair_set(), brute_overlap(&a, &b));
    }

    #[test]
    fn overlap_pair_disjoint_and_adjacent_are_none() {
        let a = IntervalSet::all_between(1, 3).unwrap();

        // Disjoint: S₂ > E₁.
        let far = IntervalSet::all_between(7, 9).unwrap();
        assert!(a.overlap(&far).is_none());
        assert!(brute_overlap(&a, &far).is_empty());

        // Adjacent (THEN, not overlaps): S₂ == E₁ + 1 still fails S₂ ≤ E₁.
        let next = IntervalSet::all_between(4, 7).unwrap();
        assert!(a.overlap(&next).is_none());
        assert!(brute_overlap(&a, &next).is_empty());
    }

    #[test]
    fn overlap_pair_singleton_lhs_is_none() {
        // A one-frame lhs cannot satisfy `a₁ < a₂ ≤ b₁`: the `S₁ + 1 ≤ E₁` guard rejects it.
        let a = IntervalSet::all_between(4, 4).unwrap();
        let b = IntervalSet::all_between(4, 9).unwrap();
        assert!(a.overlap(&b).is_none());
        assert!(brute_overlap(&a, &b).is_empty());
    }

    #[test]
    fn overlap_pair_rhs_start_bound_too_early() {
        // `S₁ + 1 ≤ L₂` fails: every rhs start is at or before the lhs start.
        let a = IntervalSet::all_between(5, 12).unwrap();
        let b = IntervalSet::new(0, 20, 1, 5, 0).unwrap();
        assert!(a.overlap(&b).is_none());
        assert!(brute_overlap(&a, &b).is_empty());
    }

    #[test]
    fn overlap_pair_with_durations() {
        let a = IntervalSet::with_duration(1, 8, 3).unwrap();
        let b = IntervalSet::with_duration(4, 14, 4).unwrap();
        let result = a.overlap(&b).unwrap();

        // c_out = max(c₁ + 1, c₂ + 1, 3) = 5.
        assert_eq!(result.c, 5);
        assert_eq!(result.pair_set(), brute_overlap(&a, &b));
    }

    #[test]
    fn overlap_pair_with_tight_bounds() {
        let a = IntervalSet::new(2, 12, 3, 6, 7).unwrap();
        let b = IntervalSet::new(4, 18, 2, 9, 11).unwrap();
        let result = a.overlap(&b).unwrap();
        assert_eq!(result.pair_set(), brute_overlap(&a, &b));
    }

    #[test]
    fn overlap_pair_handles_u32_max_boundaries() {
        let max = u32::MAX;
        let a = IntervalSet::all_between(max - 4, max - 1).unwrap();
        let b = IntervalSet::all_between(max - 3, max).unwrap();
        let result = a.overlap(&b).unwrap();
        assert_eq!(result.pair_set(), brute_overlap(&a, &b));
    }

    #[test]
    fn overlap_is_not_commutative() {
        // Allen's `overlaps` is asymmetric. Both directions are non-empty here because an
        // interval set is closed under sub-intervals, but they are different sets: B
        // overlaps A only via the single sub-interval pair [4,5] / [5,6].
        let a = IntervalSets::new(vec![(1, 6)]);
        let b = IntervalSets::new(vec![(4, 10)]);

        let ab = a.overlap(&b);
        let ba = b.overlap(&a);
        assert_eq!(ab.pair_set(), IntervalSets::brute_force_overlap(&a, &b));
        assert_eq!(ba.pair_set(), IntervalSets::brute_force_overlap(&b, &a));

        assert_eq!(ba.pair_set(), BTreeSet::from([(4, 6)]));
        assert_ne!(ab.pair_set(), ba.pair_set());
    }

    #[test]
    fn overlap_multi_tuple() {
        let a = IntervalSets::new(vec![(1, 6), (20, 28)]);
        let b = IntervalSets::new(vec![(3, 9), (24, 33)]);

        let result = a.overlap(&b);
        assert!(!result.is_empty());
        assert_eq!(result.pair_set(), IntervalSets::brute_force_overlap(&a, &b));
    }

    #[test]
    fn overlap_with_empty_is_empty() {
        let a = IntervalSets::new(vec![(1, 6)]);
        assert!(a.overlap(&IntervalSets::empty()).is_empty());
        assert!(IntervalSets::empty().overlap(&a).is_empty());
    }

    #[test]
    fn property_style_randomized_overlap_matches_bruteforce() {
        let mut seed: u64 = 0xA11E_0BE1_4A9C_D00D;
        let mut next_u32 = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 32) as u32
        };

        let mut gen_stage = || {
            let count = (next_u32() % 4 + 1) as usize;
            let mut raw = Vec::with_capacity(count);
            for _ in 0..count {
                let s = next_u32() % 20;
                let len = next_u32() % 10;
                raw.push((s, s.saturating_add(len)));
            }
            let stage = IntervalSets::new(raw).with_min_duration(next_u32() % 6);
            if next_u32() % 2 == 0 {
                return stage;
            }
            // Tighten L/R away from the vacuous L = E, R = S so the L₂ and R₁ terms bite.
            let tightened: Vec<IntervalSet> = stage
                .0
                .iter()
                .filter_map(|ft| {
                    let span = ft.e - ft.s;
                    let l = ft.s + next_u32() % (span + 1);
                    let r = ft.e - next_u32() % (span + 1);
                    IntervalSet::new(ft.s, ft.e, ft.c, l, r)
                })
                .collect();
            let mut out = IntervalSets::from_intervals(tightened);
            out.canonicalize();
            out
        };

        for _ in 0..300 {
            let a = gen_stage();
            let b = gen_stage();
            assert_eq!(
                a.overlap(&b).pair_set(),
                IntervalSets::brute_force_overlap(&a, &b),
                "OVERLAP mismatch for a={:?}, b={:?}",
                a,
                b
            );
        }
    }

    // ─── CONTAINS (Allen's contains) ───────────────────────────

    /// Brute-force CONTAINS over two single 5-tuples, via their pair sets.
    fn brute_contains(a: &IntervalSet, b: &IntervalSet) -> BTreeSet<Interval> {
        IntervalSets::brute_force_contains(&IntervalSets::single(*a), &IntervalSets::single(*b))
    }

    #[test]
    fn contains_pair_basic() {
        let a = IntervalSet::all_between(1, 10).unwrap();
        let b = IntervalSet::all_between(3, 6).unwrap();
        let result = a.contains(&b).unwrap();

        // The result is shaped like the lhs, with a size floor of c₂ + 2 = 3.
        assert_eq!((result.s, result.e), (a.s, a.e));
        assert_eq!(result.c, 3);
        assert_eq!(result.pair_set(), brute_contains(&a, &b));
    }

    #[test]
    fn contains_pair_result_is_a_subset_of_lhs() {
        let a = IntervalSet::all_between(1, 10).unwrap();
        let b = IntervalSet::all_between(3, 6).unwrap();
        let result = a.contains(&b).unwrap();

        // CONTAINS filters the lhs; it never invents pairs outside it.
        assert!(result.pair_set().is_subset(&a.pair_set()));
    }

    #[test]
    fn contains_pair_rejects_when_rhs_starts_too_early() {
        // `S₁ + 1 ≤ L₂` fails: every rhs start is at or before the lhs start.
        let a = IntervalSet::all_between(5, 20).unwrap();
        let b = IntervalSet::new(0, 20, 1, 5, 0).unwrap();
        assert!(a.contains(&b).is_none());
        assert!(brute_contains(&a, &b).is_empty());
    }

    #[test]
    fn contains_pair_rejects_when_rhs_is_too_long_to_be_enclosed() {
        // `c₂ ≤ E₂` fails: the only rhs interval is [0, 2], which nothing can start before.
        let a = IntervalSet::all_between(0, 20).unwrap();
        let b = IntervalSet::with_duration(0, 2, 3).unwrap();
        assert!(a.contains(&b).is_none());
        assert!(brute_contains(&a, &b).is_empty());
    }

    #[test]
    fn contains_pair_needs_two_extra_frames() {
        // c_out = max(c₁, c₂ + 2): a 4-frame rhs forces a 6-frame lhs, so a 5-frame lhs
        // window has nothing left to report.
        let a = IntervalSet::all_between(0, 4).unwrap();
        let b = IntervalSet::with_duration(1, 9, 4).unwrap();
        assert!(a.contains(&b).is_none());
        assert!(brute_contains(&a, &b).is_empty());

        let wider = IntervalSet::all_between(0, 5).unwrap();
        let result = wider.contains(&b).unwrap();
        assert_eq!(result.c, 6);
        assert_eq!(result.pair_set(), brute_contains(&wider, &b));
    }

    #[test]
    fn contains_pair_with_durations_and_tight_bounds() {
        let a = IntervalSet::new(2, 18, 4, 9, 11).unwrap();
        let b = IntervalSet::new(4, 15, 2, 10, 7).unwrap();
        let result = a.contains(&b).unwrap();
        assert_eq!(result.pair_set(), brute_contains(&a, &b));
    }

    #[test]
    fn contains_pair_handles_u32_max_boundaries() {
        let max = u32::MAX;
        let a = IntervalSet::all_between(max - 6, max).unwrap();
        let b = IntervalSet::all_between(max - 4, max - 2).unwrap();
        let result = a.contains(&b).unwrap();
        assert_eq!(result.pair_set(), brute_contains(&a, &b));
    }

    #[test]
    fn contains_is_not_commutative() {
        // Both directions are non-empty (interval sets are closed under sub-intervals),
        // but each keeps its own lhs, so the results differ.
        let a = IntervalSets::new(vec![(1, 10)]);
        let b = IntervalSets::new(vec![(3, 6)]);

        let ab = a.contains(&b);
        let ba = b.contains(&a);
        assert_eq!(ab.pair_set(), IntervalSets::brute_force_contains(&a, &b));
        assert_eq!(ba.pair_set(), IntervalSets::brute_force_contains(&b, &a));

        assert_eq!(ba.pair_set(), BTreeSet::from([(3, 5), (3, 6), (4, 6)]));
        assert_ne!(ab.pair_set(), ba.pair_set());
    }

    #[test]
    fn contains_multi_tuple() {
        let a = IntervalSets::new(vec![(1, 12), (20, 32)]);
        let b = IntervalSets::new(vec![(3, 8), (23, 28)]);

        let result = a.contains(&b);
        assert!(!result.is_empty());
        assert_eq!(
            result.pair_set(),
            IntervalSets::brute_force_contains(&a, &b)
        );
    }

    #[test]
    fn contains_with_empty_is_empty() {
        let a = IntervalSets::new(vec![(1, 10)]);
        assert!(a.contains(&IntervalSets::empty()).is_empty());
        assert!(IntervalSets::empty().contains(&a).is_empty());
    }

    #[test]
    fn property_style_randomized_contains_matches_bruteforce() {
        let mut seed: u64 = 0xD1C0_47A1_9BEE_F00D;
        let mut next_u32 = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 32) as u32
        };

        let mut gen_stage = || {
            let count = (next_u32() % 4 + 1) as usize;
            let mut raw = Vec::with_capacity(count);
            for _ in 0..count {
                let s = next_u32() % 20;
                let len = next_u32() % 10;
                raw.push((s, s.saturating_add(len)));
            }
            let stage = IntervalSets::new(raw).with_min_duration(next_u32() % 6);
            if next_u32() % 2 == 0 {
                return stage;
            }
            // Tighten L/R away from the vacuous L = E, R = S so the L₂ and R₂ terms bite.
            let tightened: Vec<IntervalSet> = stage
                .0
                .iter()
                .filter_map(|ft| {
                    let span = ft.e - ft.s;
                    let l = ft.s + next_u32() % (span + 1);
                    let r = ft.e - next_u32() % (span + 1);
                    IntervalSet::new(ft.s, ft.e, ft.c, l, r)
                })
                .collect();
            let mut out = IntervalSets::from_intervals(tightened);
            out.canonicalize();
            out
        };

        for _ in 0..300 {
            let a = gen_stage();
            let b = gen_stage();
            assert_eq!(
                a.contains(&b).pair_set(),
                IntervalSets::brute_force_contains(&a, &b),
                "CONTAINS mismatch for a={:?}, b={:?}",
                a,
                b
            );
        }
    }

    // ─── Triple THEN chain ──────────────────────────────────────

    #[test]
    fn triple_then_chain() {
        let a = IntervalSets::single(IntervalSet::with_duration(1, 4, 2).unwrap());
        let b = IntervalSets::single(IntervalSet::with_duration(3, 6, 2).unwrap());
        let c = IntervalSets::single(IntervalSet::with_duration(5, 10, 2).unwrap());

        let result = IntervalSets::chain(&[a.clone(), b.clone(), c.clone()]);
        let brute = IntervalSets::brute_force_then_chain(&[a, b, c]);

        assert_eq!(result.pair_set(), brute);
    }

    #[test]
    fn four_stage_chain() {
        let intervals: Vec<IntervalSets> = vec![
            IntervalSets::single(IntervalSet::with_duration(1, 5, 2).unwrap()),
            IntervalSets::single(IntervalSet::with_duration(4, 8, 2).unwrap()),
            IntervalSets::single(IntervalSet::with_duration(7, 12, 3).unwrap()),
            IntervalSets::single(IntervalSet::with_duration(11, 16, 2).unwrap()),
        ];
        let result = IntervalSets::chain(&intervals);
        let brute = IntervalSets::brute_force_then_chain(&intervals);
        assert_eq!(result.pair_set(), brute);
    }

    #[test]
    fn five_stage_chain() {
        let intervals: Vec<IntervalSets> = vec![
            IntervalSets::single(IntervalSet::with_duration(1, 5, 2).unwrap()),
            IntervalSets::single(IntervalSet::with_duration(4, 8, 2).unwrap()),
            IntervalSets::single(IntervalSet::with_duration(7, 11, 2).unwrap()),
            IntervalSets::single(IntervalSet::with_duration(10, 14, 2).unwrap()),
            IntervalSets::single(IntervalSet::with_duration(13, 18, 2).unwrap()),
        ];
        let result = IntervalSets::chain(&intervals);
        let brute = IntervalSets::brute_force_then_chain(&intervals);
        assert_eq!(result.pair_set(), brute);
    }

    // ─── DURATION ───────────────────────────────────────────────

    #[test]
    fn duration_filters_short() {
        let intervals = IntervalSets::new(vec![(1, 2), (10, 15), (20, 20)]);
        let filtered = intervals.with_min_duration(3);
        // Only [10, 15] has length ≥ 3
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered.0[0].s, 10);
        assert_eq!(filtered.0[0].e, 15);
        assert_eq!(filtered.0[0].c, 3);
    }

    #[test]
    fn duration_preserves_existing_c() {
        let ft = IntervalSet::with_duration(1, 10, 5).unwrap();
        let intervals = IntervalSets::single(ft);
        let filtered = intervals.with_min_duration(3);
        // c stays 5 (max of 5, 3)
        assert_eq!(filtered.0[0].c, 5);
    }

    #[test]
    fn duration_raises_c() {
        let ft = IntervalSet::with_duration(1, 10, 2).unwrap();
        let intervals = IntervalSets::single(ft);
        let filtered = intervals.with_min_duration(7);
        assert_eq!(filtered.0[0].c, 7);
    }

    // ─── Intervals-level AND ────────────────────────────────────────

    #[test]
    fn stage_and_c1() {
        let a = IntervalSets::new(vec![(1, 5), (10, 15)]);
        let b = IntervalSets::new(vec![(3, 8), (12, 18)]);
        let result = a.intersect(&b);

        let expected: BTreeSet<_> = a.pair_set().intersection(&b.pair_set()).copied().collect();
        assert_eq!(result.pair_set(), expected);
    }

    #[test]
    fn stage_and_with_duration() {
        let a = IntervalSets::new(vec![(1, 10)]).with_min_duration(3);
        let b = IntervalSets::new(vec![(5, 15)]).with_min_duration(4);
        let result = a.intersect(&b);

        let expected: BTreeSet<_> = a.pair_set().intersection(&b.pair_set()).copied().collect();
        assert_eq!(result.pair_set(), expected);
    }

    // ─── Intervals-level OR ─────────────────────────────────────────

    #[test]
    fn stage_or() {
        let a = IntervalSets::new(vec![(1, 3)]);
        let b = IntervalSets::new(vec![(10, 12)]);
        let result = a.union(&b);

        let expected: BTreeSet<_> = a.pair_set().union(&b.pair_set()).copied().collect();
        assert_eq!(result.pair_set(), expected);
    }

    #[test]
    fn stage_or_overlapping() {
        let a = IntervalSets::new(vec![(1, 5)]);
        let b = IntervalSets::new(vec![(3, 8)]);
        let result = a.union(&b);

        let expected: BTreeSet<_> = a.pair_set().union(&b.pair_set()).copied().collect();
        assert_eq!(result.pair_set(), expected);
    }

    // ─── Multi-tuple intervals ─────────────────────────────────────

    #[test]
    fn multi_tuple_then() {
        let a = IntervalSets::new(vec![(1, 5), (10, 14)]);
        let b = IntervalSets::new(vec![(4, 9), (13, 18)]);

        let result = a.then_intervals(&b);
        let brute = IntervalSets::brute_force_then_chain(&[a, b]);
        assert_eq!(result.pair_set(), brute);
    }

    #[test]
    fn multi_tuple_three_stage_chain() {
        let intervals = vec![
            IntervalSets::new(vec![(1, 5), (10, 14)]),
            IntervalSets::new(vec![(4, 9), (13, 18)]),
            IntervalSets::new(vec![(8, 15), (17, 22)]),
        ];

        let intervals_dur: Vec<IntervalSets> = intervals
            .iter()
            .enumerate()
            .map(|(i, s)| s.with_min_duration([2, 2, 3][i]))
            .collect();

        let result = IntervalSets::chain(&intervals_dur);
        let brute = IntervalSets::brute_force_then_chain(&intervals_dur);
        assert_eq!(result.pair_set(), brute);
    }

    // ─── Disjoint intervals ────────────────────────────────────────

    #[test]
    fn disjoint_then_empty() {
        let a = IntervalSets::new(vec![(1, 5)]);
        let b = IntervalSets::new(vec![(20, 25)]);
        let result = a.then_intervals(&b);
        assert!(result.is_empty());
    }

    #[test]
    fn empty_stage_propagates() {
        let a = IntervalSets::new(vec![(1, 5)]);
        let b = IntervalSets::empty();
        let c = IntervalSets::new(vec![(10, 15)]);
        let result = IntervalSets::chain(&[a, b, c]);
        assert!(result.is_empty());
    }

    // ─── Associativity ──────────────────────────────────────────

    #[test]
    fn then_is_associative() {
        let a = IntervalSets::single(IntervalSet::with_duration(1, 5, 2).unwrap());
        let b = IntervalSets::single(IntervalSet::with_duration(4, 8, 2).unwrap());
        let c = IntervalSets::single(IntervalSet::with_duration(7, 12, 3).unwrap());

        let left = a.then_intervals(&b).then_intervals(&c);
        let right = a.then_intervals(&b.then_intervals(&c));

        assert_eq!(left.pair_set(), right.pair_set());
    }

    // ─── THEN after AND composition ─────────────────────────────

    #[test]
    fn then_after_and() {
        let a = IntervalSets::new(vec![(1, 8)]).with_min_duration(3);
        let b = IntervalSets::new(vec![(3, 10)]).with_min_duration(2);
        let ab = a.intersect(&b);

        let c = IntervalSets::new(vec![(7, 15)]).with_min_duration(2);
        let result = ab.then_intervals(&c);

        let mut expected = BTreeSet::new();
        for &(s, e1) in &ab.pair_set() {
            for &(s2, e) in &c.pair_set() {
                if s2 == e1 + 1 {
                    expected.insert((s, e));
                }
            }
        }
        assert_eq!(result.pair_set(), expected);
    }

    // ─── Large c values ─────────────────────────────────────────

    #[test]
    fn large_c_relative_to_interval() {
        let a = IntervalSets::new(vec![(1, 6)]).with_min_duration(4);
        let b = IntervalSets::new(vec![(5, 10)]).with_min_duration(3);
        let result = a.then_intervals(&b);

        let brute = IntervalSets::brute_force_then_chain(&[a, b]);
        assert_eq!(result.pair_set(), brute);
    }

    // ─── AND commutativity ──────────────────────────────────────

    #[test]
    fn and_is_commutative() {
        let a = IntervalSets::new(vec![(1, 10)]).with_min_duration(3);
        let b = IntervalSets::new(vec![(5, 12)]).with_min_duration(5);

        let ab = a.intersect(&b);
        let ba = b.intersect(&a);
        assert_eq!(ab.pair_set(), ba.pair_set());
    }

    // ─── OR identity ────────────────────────────────────────────

    #[test]
    fn or_with_empty_is_identity() {
        let a = IntervalSets::new(vec![(1, 5), (10, 15)]);
        let result = a.union(&IntervalSets::empty());
        assert_eq!(result.pair_set(), a.pair_set());
    }

    // ─── AND with self ──────────────────────────────────────────

    #[test]
    fn and_with_self() {
        let a = IntervalSets::new(vec![(1, 10)]).with_min_duration(3);
        let result = a.intersect(&a);
        assert_eq!(result.pair_set(), a.pair_set());
    }

    // ─── Canonicalization + counting contracts ────────────────

    #[test]
    fn canonicalize_drops_infeasible_sorts_and_dedups() {
        let a = IntervalSet::with_duration(1, 5, 2).unwrap();
        let b = IntervalSet::with_duration(2, 6, 1).unwrap();
        let infeasible = IntervalSet {
            s: 9,
            e: 3,
            c: 1,
            l: 9,
            r: 3,
        };

        let mut intervals = IntervalSets(vec![b, a, infeasible, a]);
        intervals.canonicalize();

        assert_eq!(intervals.len(), 2);
        assert_eq!(intervals.0[0], a);
        assert_eq!(intervals.0[1], b);
    }

    #[test]
    fn intervals_pair_count_can_overcount_when_overlap_exists() {
        let a = IntervalSets::new(vec![(1, 5)]);
        let b = IntervalSets::new(vec![(3, 8)]);
        let u = a.union(&b);

        let exact = u.pair_set().len();
        let sum = u.total_frames();
        assert!(sum >= exact);
        assert!(sum > exact);
    }

    // ─── Edge/fencepost behavior ───────────────────────────────

    #[test]
    fn then_pair_fencepost_adjacency_equal_passes_and_next_fails() {
        let a = IntervalSet::all_between(1, 3).unwrap();
        let b_pass = IntervalSet::all_between(4, 7).unwrap();
        let b_fail = IntervalSet::all_between(5, 7).unwrap();

        assert!(a.then_pair(&b_pass).is_some());
        assert!(a.then_pair(&b_fail).is_none());
    }

    #[test]
    fn then_pair_fencepost_structural_equal_passes_and_next_fails() {
        let a_pass = IntervalSet::new(1, 10, 2, 8, 4).unwrap();
        let a_fail = IntervalSet::new(1, 10, 2, 8, 5).unwrap();
        let b = IntervalSet::new(3, 12, 2, 5, 6).unwrap();

        assert!(a_pass.then_pair(&b).is_some()); // self.r == other.l - 1
        assert!(a_fail.then_pair(&b).is_none()); // self.r > other.l - 1
    }

    #[test]
    fn then_pair_rejects_when_min_length_exceeds_rhs_start_bound() {
        // The `c₁ + g_min ≤ L₂` guard: every first stop is at least `s + c₁ - 1`, so the
        // earliest second start is `c₁` frames past the lhs start — beyond `L₂ = 3` here.
        let a = IntervalSet::with_duration(0, 20, 6).unwrap();
        let b = IntervalSet::new(0, 20, 1, 3, 0).unwrap();

        // The sibling guard `R₁ + g_min + 1 ≤ L₂` is satisfied (0 + 1 ≤ 3), so this case is
        // only rejected by the length guard.
        assert!(a.r < b.l);
        assert!(a.then_pair(&b).is_none());

        let brute = IntervalSets::brute_force_then_chain(&[
            IntervalSets::single(a),
            IntervalSets::single(b),
        ]);
        assert!(brute.is_empty());
    }

    #[test]
    fn duration_zero_is_noop_and_too_large_filters_out() {
        let a = IntervalSets::new(vec![(10, 15)]).with_min_duration(3);
        let same = a.with_min_duration(0);
        assert_eq!(same.pair_set(), a.pair_set());

        let dropped = a.with_min_duration(100);
        assert!(dropped.is_empty());
    }

    #[test]
    fn chain_empty_and_singleton() {
        assert!(IntervalSets::chain(&[]).is_empty());

        let one = IntervalSets::new(vec![(2, 6)]).with_min_duration(2);
        let out = IntervalSets::chain(std::slice::from_ref(&one));
        assert_eq!(out.pair_set(), one.pair_set());
    }

    // ─── Algebraic laws on pair semantics ──────────────────────

    #[test]
    fn and_with_empty_is_empty() {
        let a = IntervalSets::new(vec![(1, 5), (10, 15)]).with_min_duration(2);
        assert!(a.intersect(&IntervalSets::empty()).is_empty());
        assert!(IntervalSets::empty().intersect(&a).is_empty());
    }

    #[test]
    fn then_with_empty_is_empty() {
        let a = IntervalSets::new(vec![(1, 5)]).with_min_duration(2);
        assert!(a.then_intervals(&IntervalSets::empty()).is_empty());
        assert!(IntervalSets::empty().then_intervals(&a).is_empty());
    }

    #[test]
    fn or_is_idempotent_commutative_associative() {
        let a = IntervalSets::new(vec![(1, 5)]);
        let b = IntervalSets::new(vec![(3, 7)]);
        let c = IntervalSets::new(vec![(10, 12)]);

        assert_eq!(a.union(&a).pair_set(), a.pair_set());

        let ab = a.union(&b);
        let ba = b.union(&a);
        assert_eq!(ab.pair_set(), ba.pair_set());

        let left = a.union(&b).union(&c);
        let right = a.union(&b.union(&c));
        assert_eq!(left.pair_set(), right.pair_set());
    }

    #[test]
    fn and_is_associative_on_pair_set() {
        let a = IntervalSets::new(vec![(1, 10)]).with_min_duration(2);
        let b = IntervalSets::new(vec![(3, 12)]).with_min_duration(3);
        let c = IntervalSets::new(vec![(5, 14)]).with_min_duration(2);

        let left = a.intersect(&b).intersect(&c);
        let right = a.intersect(&b.intersect(&c));
        assert_eq!(left.pair_set(), right.pair_set());
    }

    // ─── Saturation boundary coverage ──────────────────────────

    #[test]
    fn then_pair_handles_u32_max_boundaries() {
        let max = u32::MAX;

        let a = IntervalSet::all_between(max - 2, max).unwrap();
        let b = IntervalSet::all_between(max, max).unwrap();
        assert!(a.then_pair(&b).is_some());

        let c = IntervalSet::all_between(max - 4, max - 2).unwrap();
        assert!(c.then_pair(&b).is_none());
    }

    #[test]
    fn then_pair_saturating_r_out_stays_feasible() {
        let max = u32::MAX;

        let a = IntervalSet::new(max - 10, max, 1, max, max - 1).unwrap();
        let b = IntervalSet::with_duration(max - 5, max, 2).unwrap();
        let out = a.then_pair(&b).unwrap();

        assert_eq!(out.r, max);
        assert!(out.is_feasible());
    }

    #[test]
    fn property_style_randomized_and_then_match_pair_semantics() {
        // Deterministic LCG to keep the test reproducible without external deps.
        let mut seed: u64 = 0xC0FFEE_F00D_BAAD;
        let mut next_u32 = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 32) as u32
        };

        let mut gen_stage = || {
            let count = (next_u32() % 5 + 1) as usize; // 1..=5 tuples
            let mut raw = Vec::with_capacity(count);
            for _ in 0..count {
                let s = next_u32() % 25;
                let len = next_u32() % 12; // 0..=11, allows singletons
                let e = s.saturating_add(len);
                raw.push((s, e));
            }
            let min_len = next_u32() % 8; // includes 0 edge
            let stage = IntervalSets::new(raw).with_min_duration(min_len);

            // `with_min_duration` leaves L = E and R = S, so half the stages get tightened
            // bounds instead — those are what exercise the L₂ and R₁ guards in THEN.
            if next_u32() % 2 == 0 {
                return stage;
            }
            let tightened: Vec<IntervalSet> = stage
                .0
                .iter()
                .filter_map(|ft| {
                    let span = ft.e - ft.s;
                    let l = ft.s + next_u32() % (span + 1);
                    let r = ft.e - next_u32() % (span + 1);
                    IntervalSet::new(ft.s, ft.e, ft.c, l, r)
                })
                .collect();
            let mut out = IntervalSets::from_intervals(tightened);
            out.canonicalize();
            out
        };

        for _ in 0..300 {
            let a = gen_stage();
            let b = gen_stage();

            // AND semantics: exact pair-set intersection.
            let and_res = a.intersect(&b);
            let and_expected: BTreeSet<_> =
                a.pair_set().intersection(&b.pair_set()).copied().collect();
            assert_eq!(
                and_res.pair_set(),
                and_expected,
                "AND mismatch for a={:?}, b={:?}",
                a,
                b
            );

            // THEN semantics: exact brute-force chaining over two stages.
            let then_res = a.then_intervals(&b);
            let then_expected = IntervalSets::brute_force_then_chain(&[a.clone(), b.clone()]);
            assert_eq!(
                then_res.pair_set(),
                then_expected,
                "THEN mismatch for a={:?}, b={:?}",
                a,
                b
            );
        }
    }

    #[test]
    fn property_style_randomized_chain_3_to_5_matches_bruteforce() {
        // Deterministic LCG for reproducibility.
        let mut seed: u64 = 0x1234_5678_9ABC_DEF0;
        fn next_u32(seed: &mut u64) -> u32 {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*seed >> 32) as u32
        }

        let gen_stage = |seed: &mut u64| {
            let count = (next_u32(seed) % 4 + 1) as usize; // 1..=4 tuples
            let mut raw = Vec::with_capacity(count);
            for _ in 0..count {
                let s = next_u32(seed) % 20;
                let len = next_u32(seed) % 10; // 0..=9
                let e = s.saturating_add(len);
                raw.push((s, e));
            }
            let min_len = next_u32(seed) % 7; // includes 0 edge
            IntervalSets::new(raw).with_min_duration(min_len)
        };

        for _ in 0..200 {
            let n = (next_u32(&mut seed) % 3 + 3) as usize; // 3..=5 stages
            let stages: Vec<IntervalSets> = (0..n).map(|_| gen_stage(&mut seed)).collect();

            let chain_res = IntervalSets::chain(&stages);
            let brute = IntervalSets::brute_force_then_chain(&stages);
            assert_eq!(
                chain_res.pair_set(),
                brute,
                "CHAIN mismatch for stages={:?}",
                stages
            );
        }
    }

    #[test]
    fn property_style_duration_is_monotone_under_min_len() {
        // If k2 >= k1 then with_min_duration(k2) must be a subset of with_min_duration(k1).
        let mut seed: u64 = 0xDEAD_BEEF_CAFE_BABE;
        let mut next_u32 = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 32) as u32
        };

        for _ in 0..400 {
            let count = (next_u32() % 8 + 1) as usize; // 1..=8 tuples
            let mut raw = Vec::with_capacity(count);
            for _ in 0..count {
                let s = next_u32() % 30;
                let len = next_u32() % 12;
                let e = s.saturating_add(len);
                raw.push((s, e));
            }

            let base = IntervalSets::new(raw);
            let k1 = next_u32() % 10;
            let k2 = k1.saturating_add(next_u32() % 10); // ensure k2 >= k1

            let d1 = base.with_min_duration(k1).pair_set();
            let d2 = base.with_min_duration(k2).pair_set();
            assert!(
                d2.is_subset(&d1),
                "with_min_duration monotonicity violated for k1={}, k2={}, base={:?}",
                k1,
                k2,
                base
            );
        }
    }

    // ─── Stress test: brute force on random-ish inputs ──────────

    #[test]
    fn stress_chain_various_configs() {
        let configs: Vec<Vec<(u32, u32, u32)>> = vec![
            vec![(1, 6, 2), (4, 9, 3), (7, 14, 2)],
            vec![(1, 3, 1), (2, 5, 1), (4, 8, 2), (7, 10, 1)],
            vec![(1, 10, 5), (8, 15, 3), (13, 20, 4)],
            vec![(1, 4, 2), (3, 7, 2), (6, 9, 2), (8, 12, 2)],
        ];

        for config in &configs {
            let intervals: Vec<IntervalSets> = config
                .iter()
                .map(|&(s, e, c)| IntervalSets::new(vec![(s, e)]).with_min_duration(c))
                .collect();

            let result = IntervalSets::chain(&intervals);
            let brute = IntervalSets::brute_force_then_chain(&intervals);
            assert_eq!(result.pair_set(), brute, "Mismatch for config {:?}", config);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Gap Algebra Tests (THEN_WITH_GAP)
    // ═══════════════════════════════════════════════════════════════════════════
    mod gap_tests {
        use super::*;

        #[test]
        fn then_with_gap_zero_gap_equivalence() {
            // Proves `then_intervals` is mathematically strictly equivalent to `then_with_gap(0, Some(0))`
            let a = IntervalSets::new(vec![(1, 5), (10, 15)]);
            let b = IntervalSets::new(vec![(4, 8), (14, 20)]);

            let standard_then = a.then_intervals(&b);
            let gap_then = a.then_with_gap(&b, 0, Some(0));

            assert_eq!(standard_then.pair_set(), gap_then.pair_set());
            assert_eq!(standard_then, gap_then); // The exact structure should map exactly identically
        }

        #[test]
        fn then_with_gap_exact_distance() {
            let a = IntervalSets::new(vec![(1, 4)]);
            let b = IntervalSets::new(vec![(8, 10)]);
            // e1_max = 4. s2_min = 8. Min distance = 8 - 4 - 1 = 3.

            // Exactly 3 gap is perfectly valid
            let res_exact = a.then_with_gap(&b, 3, Some(3));
            assert!(!res_exact.is_empty());

            // Gap 2 is mathematically impossible (too small)
            let res_too_small = a.then_with_gap(&b, 2, Some(2));
            assert!(res_too_small.is_empty());
        }

        #[test]
        fn then_with_gap_impossible_too_large() {
            let a = IntervalSets::new(vec![(1, 5)]);
            let b = IntervalSets::new(vec![(8, 10)]);
            // The largest distance possible is e1_min to s2_max:
            // e1 can be as small as 1. s2 can be up to 10.
            // However, with max constraints in real intervals, the largest split point gap
            // is 10 - 1 - 1 = 8.

            // Requiring a minimum gap of 15 is impossible.
            let res_impossible = a.then_with_gap(&b, 15, None);
            assert!(res_impossible.is_empty());
        }

        #[test]
        fn then_with_gap_range_filters() {
            let a = IntervalSets::new(vec![(1, 5)]);
            let b = IntervalSets::new(vec![(8, 12)]);

            // e1 can be up to 5. s2 can be down to 8.
            // Min gap = 8 - 5 - 1 = 2
            // Max gap = 12 - 1 - 1 = 10

            let res_valid = a.then_with_gap(&b, 2, Some(10));
            assert!(!res_valid.is_empty());

            // Check against exact pair existential brute force
            let brute =
                IntervalSets::brute_force_chain_with_gap(&[a.clone(), b.clone()], &[(2, Some(10))]);
            assert_eq!(res_valid.pair_set(), brute);
        }

        #[test]
        fn then_with_gap_open_ended() {
            let a = IntervalSets::new(vec![(1, 5)]);
            let b = IntervalSets::new(vec![(20, 25)]);

            // Unbounded gap (None) should seamlessly connect them regardless of how far
            let res = a.then_with_gap(&b, 5, None);
            let brute =
                IntervalSets::brute_force_chain_with_gap(&[a.clone(), b.clone()], &[(5, None)]);

            assert!(!res.is_empty());
            assert_eq!(res.pair_set(), brute);
        }

        #[test]
        fn then_with_gap_associativity() {
            // Proves `(A THEN_{g1} B) THEN_{g2} C == A THEN_{g1} (B THEN_{g2} C)`
            let a = IntervalSets::new(vec![(1, 5), (10, 15)]);
            let b = IntervalSets::new(vec![(4, 10), (20, 25)]);
            let c = IntervalSets::new(vec![(8, 15), (30, 35)]);

            let g1 = (1, Some(3));
            let g2 = (0, None);

            // Left association
            let left_ab = a.then_with_gap(&b, g1.0, g1.1);
            let left_result = left_ab.then_with_gap(&c, g2.0, g2.1);

            // Right association
            let right_bc = b.then_with_gap(&c, g2.0, g2.1);
            let right_result = a.then_with_gap(&right_bc, g1.0, g1.1);

            assert_eq!(
                left_result.pair_set(),
                right_result.pair_set(),
                "THEN_WITH_GAP is not associative!"
            );
        }

        #[test]
        fn then_with_gap_monotonicity() {
            // If we relax the gap constraints, the resulting set should strictly be a superset
            let a = IntervalSets::new(vec![(1, 10)]);
            let b = IntervalSets::new(vec![(12, 20)]);

            let strict_res = a.then_with_gap(&b, 2, Some(4));
            let relaxed_res = a.then_with_gap(&b, 1, Some(6));

            let strict_pairs = strict_res.pair_set();
            let relaxed_pairs = relaxed_res.pair_set();

            assert!(relaxed_pairs.is_superset(&strict_pairs));
            assert!(relaxed_pairs.len() > strict_pairs.len());
        }

        #[test]
        fn chain_with_gap_empty_propagation() {
            let a = IntervalSets::new(vec![(1, 5)]);
            let empty = IntervalSets::empty();
            let b = IntervalSets::new(vec![(10, 15)]);

            let gaps = vec![(0, None), (1, Some(2))];

            let result = IntervalSets::chain_with_gap(&[a, empty, b], &gaps);
            assert!(result.is_empty());
        }

        #[test]
        #[should_panic(expected = "Mismatched intervals and gaps lengths")]
        fn chain_with_gap_mismatched_lengths_panics() {
            let a = IntervalSets::new(vec![(1, 5)]);
            let b = IntervalSets::new(vec![(10, 15)]);
            // We provide 2 intervals, but 2 gaps (should be 1 gap)
            let gaps = vec![(0, None), (1, Some(2))];

            IntervalSets::chain_with_gap(&[a, b], &gaps);
        }

        #[test]
        fn then_with_gap_min_length_plus_gap_exceeds_rhs_start_bound() {
            // `c₁ = 3` alone fits inside `L₂ = 4`, but `c₁ + g_min = 5` does not.
            let a = IntervalSets::single(IntervalSet::with_duration(0, 20, 3).unwrap());
            let b = IntervalSets::single(IntervalSet::new(0, 20, 1, 4, 0).unwrap());

            let no_gap = a.then_with_gap(&b, 0, None);
            assert!(!no_gap.is_empty());
            assert_eq!(
                no_gap.pair_set(),
                IntervalSets::brute_force_chain_with_gap(&[a.clone(), b.clone()], &[(0, None)])
            );

            let with_gap = a.then_with_gap(&b, 2, None);
            assert!(with_gap.is_empty());
            assert!(IntervalSets::brute_force_chain_with_gap(&[a, b], &[(2, None)]).is_empty());
        }

        #[test]
        fn then_with_gap_saturating_bounds() {
            let max = u32::MAX;

            // Use the gap logic right at the absolute boundary of u32 math
            let a = IntervalSet::all_between(max - 10, max - 5).unwrap();
            let b = IntervalSet::all_between(max - 2, max).unwrap();

            let a_iv = IntervalSets::single(a);
            let b_iv = IntervalSets::single(b);

            // A gap of 2 separates (max - 5) and (max - 2).
            let res = a_iv.then_with_gap(&b_iv, 2, Some(5));
            assert!(!res.is_empty());

            // This huge required gap should be safely filtered out without panicking
            let impossible = a_iv.then_with_gap(&b_iv, max / 2, None);
            assert!(impossible.is_empty());
        }

        #[test]
        fn property_style_randomized_chain_with_gap_matches_bruteforce() {
            let mut seed: u64 = 0x5555_4444_3333_2222;
            fn next_u32(seed: &mut u64) -> u32 {
                *seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (*seed >> 32) as u32
            }

            let gen_stage = |seed: &mut u64| {
                let count = (next_u32(seed) % 4 + 1) as usize;
                let mut raw = Vec::with_capacity(count);
                for _ in 0..count {
                    let s = next_u32(seed) % 40;
                    let len = next_u32(seed) % 15;
                    let e = s.saturating_add(len);
                    raw.push((s, e));
                }
                let min_len = next_u32(seed) % 5;
                IntervalSets::new(raw).with_min_duration(min_len)
            };

            for _ in 0..200 {
                let n = (next_u32(&mut seed) % 3 + 3) as usize; // 3..=5 stages
                let stages: Vec<IntervalSets> = (0..n).map(|_| gen_stage(&mut seed)).collect();

                let mut gaps = Vec::new();
                for _ in 0..(n - 1) {
                    let g_min = next_u32(&mut seed) % 5;
                    let is_unbounded = next_u32(&mut seed) % 4 == 0;
                    let g_max = if is_unbounded {
                        None
                    } else {
                        Some(g_min + next_u32(&mut seed) % 5)
                    };
                    gaps.push((g_min, g_max));
                }

                let chain_res = IntervalSets::chain_with_gap(&stages, &gaps);
                let brute = IntervalSets::brute_force_chain_with_gap(&stages, &gaps);

                assert_eq!(
                    chain_res.pair_set(),
                    brute,
                    "CHAIN_WITH_GAP mismatch for stages={:?} gaps={:?}",
                    stages,
                    gaps
                );
            }
        }

        #[test]
        fn stress_chain_with_gap_various_configs() {
            // Hardcoded intensive setups
            let configs: Vec<(Vec<(u32, u32, u32)>, Vec<(u32, Option<u32>)>)> = vec![
                (
                    vec![(1, 6, 2), (5, 12, 3), (10, 20, 2)],
                    vec![(0, Some(2)), (1, None)],
                ),
                (
                    vec![(1, 4, 1), (3, 8, 1), (6, 11, 2), (10, 15, 1)],
                    vec![(1, Some(1)), (0, Some(3)), (2, None)],
                ),
                (
                    vec![(1, 20, 5), (10, 30, 3), (25, 40, 4)],
                    vec![(5, Some(10)), (0, None)],
                ),
            ];

            for (stage_cfg, gaps) in &configs {
                let intervals: Vec<IntervalSets> = stage_cfg
                    .iter()
                    .map(|&(s, e, c)| IntervalSets::new(vec![(s, e)]).with_min_duration(c))
                    .collect();

                let result = IntervalSets::chain_with_gap(&intervals, gaps);
                let brute = IntervalSets::brute_force_chain_with_gap(&intervals, gaps);

                assert_eq!(
                    result.pair_set(),
                    brute,
                    "Mismatch for configuration with gaps {:?}",
                    gaps
                );
            }
        }
    }

    // ─── frames_iter ────────────────────────────────────────────

    #[test]
    fn frames_iter_single_tuple() {
        let iv = IntervalSets::new(vec![(3, 6)]);
        let frames: Vec<u32> = iv.frames_iter().collect();
        assert_eq!(frames, vec![3, 4, 5, 6]);
    }

    #[test]
    fn frames_iter_empty() {
        let frames: Vec<u32> = IntervalSets::empty().frames_iter().collect();
        assert!(frames.is_empty());
    }

    #[test]
    fn frames_iter_non_overlapping_tuples() {
        // Two disjoint base tuples: [1,3] and [7,9]
        let iv = IntervalSets::new(vec![(1, 3), (7, 9)]);
        let frames: Vec<u32> = iv.frames_iter().collect();
        assert_eq!(frames, vec![1, 2, 3, 7, 8, 9]);
    }

    #[test]
    fn frames_iter_overlapping_tuples_dedup() {
        // After union, two overlapping tuples: [1,5] and [3,8] → merged to [1,8]
        let a = IntervalSets::new(vec![(1, 5)]);
        let b = IntervalSets::new(vec![(3, 8)]);
        let u = a.union(&b);

        let frames: Vec<u32> = u.frames_iter().collect();
        // Each fid in [1,8] should appear exactly once
        assert_eq!(frames, (1u32..=8).collect::<Vec<_>>());
    }
}
