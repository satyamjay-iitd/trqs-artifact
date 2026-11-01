//! The naive, fully materialized interval-set representation — an ablation
//! baseline for the 5-tuple algebra in [`super::sets`].
//!
//! An [`IntervalSets`] here is the literal set of every `(start, end)` pair the
//! query admits, sorted by `(start, end)` and deduplicated. The operators are the
//! set-theoretic definitions applied directly, the same thing the brute-force
//! oracles in the `sets` tests compute. A run of `k` frames lifts to
//! `k (k + 1) / 2` pairs, which is the cost the 5-tuple exists to avoid.
//!
//! Selected in place of the 5-tuple by the `naive-intervals` feature; the public
//! API mirrors `sets::IntervalSets` so the evaluator compiles against either.
//!
//! Sizes are capped by `TRQL_NAIVE_MAX_PAIRS` (default 2^28 pairs, ~2 GiB) so a
//! long run fails loudly instead of swapping the machine.

// Unreferenced outside its tests unless the ablation feature selects it.
#![cfg_attr(not(feature = "naive-intervals"), allow(dead_code))]

use std::sync::OnceLock;

use super::{Interval, IntervalSet};

const DEFAULT_MAX_PAIRS: usize = 1 << 28;

/// The pair-count ceiling, read once from `TRQL_NAIVE_MAX_PAIRS`.
fn max_pairs() -> usize {
    static MAX: OnceLock<usize> = OnceLock::new();
    *MAX.get_or_init(|| {
        std::env::var("TRQL_NAIVE_MAX_PAIRS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_MAX_PAIRS)
    })
}

/// Panics if a set about to hold `n` pairs exceeds the ceiling.
fn check_size(n: usize) {
    let max = max_pairs();
    if n > max {
        panic!("naive interval set exceeded {max} pairs (needs {n}); raise TRQL_NAIVE_MAX_PAIRS");
    }
}

/// Fully materialized interval-set representation: every valid `(s, e)` pair.
///
/// Invariant: sorted by `(s, e)`, deduplicated, `s ≤ e` for every pair.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IntervalSets(Vec<Interval>);

impl IntervalSets {
    /// Creates a set from raw `(start, end)` runs: every sub-interval of every run.
    ///
    /// Runs are merged exactly as `sets::IntervalSets::new` merges them, so both
    /// representations lift the same leaf output to the same pair set.
    pub fn new(mut intervals: Vec<Interval>) -> Self {
        intervals.sort_unstable_by_key(|&(s, e)| (s, e));

        let runs: Vec<Interval> = intervals.into_iter().fold(Vec::new(), |mut acc, (s, e)| {
            if let Some(last) = acc.last_mut() {
                if s <= last.1.saturating_add(1) {
                    last.1 = last.1.max(e);
                    return acc;
                }
            }
            acc.push((s, e));
            acc
        });

        let total: usize = runs
            .iter()
            .filter(|&&(s, e)| s <= e)
            .map(|&(s, e)| {
                let k = (e - s) as usize + 1;
                k * (k + 1) / 2
            })
            .sum();
        check_size(total);

        // Runs are disjoint and sorted, so emitting each in order keeps the pairs sorted.
        let mut pairs = Vec::with_capacity(total);
        for (s, e) in runs {
            if s > e {
                continue;
            }
            for a in s..=e {
                pairs.extend((a..=e).map(|b| (a, b)));
            }
        }
        Self(pairs)
    }

    /// Creates a set holding every pair of a single 5-tuple. Empty if infeasible.
    pub fn single(ft: IntervalSet) -> Self {
        Self::from_intervals(vec![ft])
    }

    /// Creates an empty set.
    pub fn empty() -> Self {
        Self(Vec::new())
    }

    /// AND: pairs present in both sets.
    pub fn intersect(&self, other: &Self) -> Self {
        let (a, b) = (&self.0, &other.0);
        let mut result = Vec::new();
        let (mut i, mut j) = (0, 0);
        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    result.push(a[i]);
                    i += 1;
                    j += 1;
                }
            }
        }
        Self(result)
    }

    /// OR: pairs present in either set.
    pub fn union(&self, other: &Self) -> Self {
        check_size(self.0.len() + other.0.len());
        let (a, b) = (&self.0, &other.0);
        let mut result = Vec::with_capacity(a.len() + b.len());
        let (mut i, mut j) = (0, 0);
        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Less => {
                    result.push(a[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    result.push(b[j]);
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    result.push(a[i]);
                    i += 1;
                    j += 1;
                }
            }
        }
        result.extend_from_slice(&a[i..]);
        result.extend_from_slice(&b[j..]);
        Self(result)
    }

    /// THEN with a gap window: `(s₁, e₂)` for every `(s₁, e₁)`, `(s₂, e₂)` with
    /// `g_min ≤ s₂ − e₁ − 1 ≤ g_max`.
    ///
    /// `other` is sorted by start, so each lhs pair binary-searches its window of
    /// rhs starts rather than scanning all of `other`.
    pub fn then_with_gap(&self, other: &Self, min_gap: u32, max_gap: Option<u32>) -> Self {
        let b = &other.0;
        let mut result = Vec::new();
        for &(s1, e1) in &self.0 {
            // Valid rhs starts are [e₁ + 1 + g_min, e₁ + 1 + g_max], in u64 so the
            // window past u32::MAX is empty rather than wrapped.
            let lo = e1 as u64 + 1 + min_gap as u64;
            if lo > u32::MAX as u64 {
                continue;
            }
            let hi = max_gap.map_or(u32::MAX as u64, |g| e1 as u64 + 1 + g as u64);
            let i = b.partition_point(|p| (p.0 as u64) < lo);
            let j = b.partition_point(|p| (p.0 as u64) <= hi);
            result.extend(b[i..j].iter().map(|&(_, e2)| (s1, e2)));
            check_size(result.len());
        }
        let mut out = Self(result);
        out.canonicalize();
        out
    }

    /// OVERLAP (Allen's *overlaps*): `(s₁, e₂)` for every pair with `s₁ < s₂ ≤ e₁ < e₂`.
    pub fn overlap(&self, other: &Self) -> Self {
        let b = &other.0;
        let mut result = Vec::new();
        for &(s1, e1) in &self.0 {
            let i = b.partition_point(|p| p.0 <= s1);
            let j = b.partition_point(|p| p.0 <= e1);
            result.extend(
                b[i..j]
                    .iter()
                    .filter(|&&(_, e2)| e2 > e1)
                    .map(|&(_, e2)| (s1, e2)),
            );
            check_size(result.len());
        }
        let mut out = Self(result);
        out.canonicalize();
        out
    }

    /// CONTAINS (Allen's *contains*): every lhs pair `(s₁, e₁)` strictly enclosing
    /// some rhs pair, `s₁ < s₂ ≤ e₂ < e₁`.
    ///
    /// A suffix minimum of rhs ends (rhs is sorted by start) answers "is there an
    /// rhs pair starting after `s₁` that ends before `e₁`" with one binary search.
    pub fn contains(&self, other: &Self) -> Self {
        let b = &other.0;
        let mut suffix_min_end = vec![u32::MAX; b.len() + 1];
        for k in (0..b.len()).rev() {
            suffix_min_end[k] = suffix_min_end[k + 1].min(b[k].1);
        }
        let result: Vec<Interval> = self
            .0
            .iter()
            .copied()
            .filter(|&(s1, e1)| {
                let i = b.partition_point(|p| p.0 <= s1);
                i < b.len() && suffix_min_end[i] < e1
            })
            .collect();
        // A filter of a sorted, deduplicated list stays so.
        Self(result)
    }

    /// DURATION: keeps pairs at least `min_len` frames long.
    pub fn with_min_duration(&self, min_len: u32) -> Self {
        Self(
            self.0
                .iter()
                .copied()
                .filter(|&(s, e)| (e - s) as u64 + 1 >= min_len as u64)
                .collect(),
        )
    }

    /// Returns `true` if the set holds no pairs.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns the frame window `[lo, hi]` that an rhs candidate must overlap for a
    /// `self Then rhs` with the given gap to be satisfiable. `None` if `self` is empty.
    pub fn then_rhs_window(&self, min_gap: u32, max_gap: Option<u32>) -> Option<Interval> {
        // Sorted by start, so the extreme ends need a scan.
        let min_e = self.0.iter().map(|p| p.1).min()?;
        let max_e = self.0.iter().map(|p| p.1).max()?;
        let lo = min_e.saturating_add(min_gap);
        let hi = max_gap
            .map(|mg| max_e.saturating_add(mg))
            .unwrap_or(u32::MAX);
        Some((lo, hi))
    }

    /// Returns an iterator over all frame IDs covered by this set, sorted and deduplicated.
    pub fn frames_iter(&self) -> impl Iterator<Item = u32> {
        self.maximal().into_iter().flat_map(|(s, e)| s..=e)
    }

    pub fn maximal(&self) -> Vec<Interval> {
        let mut merged: Vec<Interval> = Vec::new();
        for &(s, e) in &self.0 {
            match merged.last_mut() {
                Some(last) if s <= last.1 => last.1 = last.1.max(e),
                _ => merged.push((s, e)),
            }
        }
        merged
    }

    //^                     API ends

    /// Returns the number of materialized pairs.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Constructs a set holding every pair of every given 5-tuple.
    pub fn from_intervals(tuples: Vec<IntervalSet>) -> Self {
        let mut pairs = Vec::new();
        for ft in tuples.into_iter().filter(IntervalSet::is_feasible) {
            for s in ft.s..=ft.l {
                // s + c − 1: c ≥ 1 on a feasible tuple, so this cannot underflow.
                let e_min = ft.r.max(s.saturating_add(ft.c - 1));
                if e_min <= ft.e {
                    pairs.extend((e_min..=ft.e).map(|e| (s, e)));
                }
            }
            check_size(pairs.len());
        }
        let mut out = Self(pairs);
        out.canonicalize();
        out
    }

    /// Restores the sorted, deduplicated invariant.
    pub fn canonicalize(&mut self) {
        self.0.sort_unstable();
        self.0.dedup();
    }

    /// THEN with no gap.
    pub fn then_intervals(&self, other: &Self) -> Self {
        self.then_with_gap(other, 0, Some(0))
    }

    /// N-fold THEN chain: `intervals[0]` THEN `intervals[1]` THEN … THEN `intervals[n-1]`.
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
// Tests — differential against the 5-tuple algebra
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::super::sets;
    use super::*;

    /// One operand in both representations.
    #[derive(Clone)]
    struct Both {
        five: sets::IntervalSets,
        naive: IntervalSets,
    }

    impl Both {
        fn assert_agree(&self, what: &str) {
            let expected = self.five.pair_set();
            let got: BTreeSet<Interval> = self.naive.0.iter().copied().collect();
            assert_eq!(got.len(), self.naive.len(), "{what}: naive set has duplicates");
            assert!(
                self.naive.0.windows(2).all(|w| w[0] < w[1]),
                "{what}: naive set not sorted"
            );
            assert_eq!(got, expected, "{what}: pair sets differ");
            assert_eq!(self.naive.maximal(), self.five.maximal(), "{what}: maximal differs");
            assert_eq!(self.naive.is_empty(), self.five.is_empty(), "{what}: emptiness differs");
        }

        fn then(&self, o: &Self, g_min: u32, g_max: Option<u32>) -> Self {
            Self {
                five: self.five.then_with_gap(&o.five, g_min, g_max),
                naive: self.naive.then_with_gap(&o.naive, g_min, g_max),
            }
        }

        fn overlap(&self, o: &Self) -> Self {
            Self {
                five: self.five.overlap(&o.five),
                naive: self.naive.overlap(&o.naive),
            }
        }

        fn contains(&self, o: &Self) -> Self {
            Self {
                five: self.five.contains(&o.five),
                naive: self.naive.contains(&o.naive),
            }
        }

        fn intersect(&self, o: &Self) -> Self {
            Self {
                five: self.five.intersect(&o.five),
                naive: self.naive.intersect(&o.naive),
            }
        }

        fn union(&self, o: &Self) -> Self {
            Self {
                five: self.five.union(&o.five),
                naive: self.naive.union(&o.naive),
            }
        }

        fn duration(&self, c: u32) -> Self {
            Self {
                five: self.five.with_min_duration(c),
                naive: self.naive.with_min_duration(c),
            }
        }
    }

    /// Deterministic LCG, as in the `sets` tests, to stay reproducible without deps.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 32) as u32
        }

        /// A leaf: raw runs lifted by both `new`s, optionally with a duration.
        fn leaf(&mut self) -> Both {
            let count = (self.next() % 4 + 1) as usize;
            let raw: Vec<Interval> = (0..count)
                .map(|_| {
                    let s = self.next() % 25;
                    (s, s + self.next() % 10)
                })
                .collect();
            let both = Both {
                five: sets::IntervalSets::new(raw.clone()),
                naive: IntervalSets::new(raw),
            };
            both.assert_agree("leaf");
            match self.next() % 3 {
                0 => both,
                _ => both.duration(self.next() % 6),
            }
        }

        /// Random 5-tuples with tightened `L`/`R`, which leaves never produce.
        fn tuples(&mut self) -> Both {
            let count = (self.next() % 3 + 1) as usize;
            let mut tuples = Vec::new();
            for _ in 0..count {
                let s = self.next() % 25;
                let e = s + self.next() % 10;
                let span = e - s;
                let c = 1 + self.next() % (span + 1);
                let l = s + self.next() % (span + 1);
                let r = e - self.next() % (span + 1);
                tuples.extend(IntervalSet::new(s, e, c, l, r));
            }
            let mut five = sets::IntervalSets::from_intervals(tuples.clone());
            five.canonicalize();
            Both {
                five,
                naive: IntervalSets::from_intervals(tuples),
            }
        }

        fn operand(&mut self) -> Both {
            if self.next() % 2 == 0 { self.leaf() } else { self.tuples() }
        }

        fn gap(&mut self) -> (u32, Option<u32>) {
            let g_min = self.next() % 4;
            let g_max = match self.next() % 3 {
                0 => None,
                _ => Some(g_min + self.next() % 6),
            };
            (g_min, g_max)
        }
    }

    #[test]
    fn new_materializes_every_subinterval() {
        let set = IntervalSets::new(vec![(5, 7), (8, 8), (20, 21)]);
        // [5, 7] and [8, 8] are adjacent, so they lift as one run [5, 8].
        assert_eq!(set.len(), 4 * 5 / 2 + 3);
        assert_eq!(set.maximal(), vec![(5, 8), (20, 21)]);
    }

    #[test]
    fn single_operators_match_five_tuple() {
        let mut rng = Lcg(0xC0FFEE_F00D_BAAD);
        for i in 0..400 {
            let (a, b) = (rng.operand(), rng.operand());
            a.assert_agree(&format!("lhs #{i}"));
            b.assert_agree(&format!("rhs #{i}"));
            let (g_min, g_max) = rng.gap();
            a.then(&b, g_min, g_max)
                .assert_agree(&format!("then({g_min}, {g_max:?}) #{i}"));
            a.then(&b, 0, Some(0)).assert_agree(&format!("then adjacent #{i}"));
            a.overlap(&b).assert_agree(&format!("overlap #{i}"));
            a.contains(&b).assert_agree(&format!("contains #{i}"));
            a.intersect(&b).assert_agree(&format!("intersect #{i}"));
            a.union(&b).assert_agree(&format!("union #{i}"));
            a.duration(rng.next() % 8).assert_agree(&format!("duration #{i}"));
        }
    }

    #[test]
    fn nested_compositions_match_five_tuple() {
        let mut rng = Lcg(0x5EED_1234_ABCD);
        for i in 0..300 {
            let (a, b, c) = (rng.operand(), rng.operand(), rng.operand());
            let (g_min, g_max) = rng.gap();
            let d = rng.next() % 6;
            a.then(&b, g_min, g_max)
                .then(&c, 0, None)
                .assert_agree(&format!("then∘then #{i}"));
            a.then(&b, g_min, g_max)
                .duration(d)
                .assert_agree(&format!("duration∘then #{i}"));
            a.duration(d)
                .contains(&b)
                .then(&c, g_min, g_max)
                .assert_agree(&format!("then∘contains #{i}"));
            a.overlap(&b)
                .contains(&c.duration(d))
                .assert_agree(&format!("contains∘overlap #{i}"));
            a.contains(&b.overlap(&c))
                .assert_agree(&format!("contains(overlap) #{i}"));
        }
    }

    #[test]
    fn chain_matches_five_tuple() {
        let mut rng = Lcg(0xBAD_CAFE);
        for i in 0..200 {
            let stages: Vec<Both> = (0..3).map(|_| rng.leaf()).collect();
            let gaps = vec![rng.gap(), rng.gap()];
            let five: Vec<_> = stages.iter().map(|s| s.five.clone()).collect();
            let naive: Vec<_> = stages.iter().map(|s| s.naive.clone()).collect();
            Both {
                five: sets::IntervalSets::chain_with_gap(&five, &gaps),
                naive: IntervalSets::chain_with_gap(&naive, &gaps),
            }
            .assert_agree(&format!("chain #{i}"));
        }
    }
}
