//! Set operations on sorted, disjoint frame-interval lists.
//!
//! The primitives every layer above needs: leaf predicate evaluation
//! ([`trql_bbox_compiler`]) composes root-finding output with them, and candidate
//! enumeration ([`trql_selector`]) uses them to intersect coappearance windows.
//! Both used to carry their own copies.
//!
//! Every function takes and returns lists that are **sorted by start and
//! pairwise disjoint**, and preserves that invariant. Adjacent intervals are
//! *not* coalesced except by [`union`], which merges anything touching.
//!
//! This crate deliberately holds no notion of *bounds* on an interval — minimum
//! length, latest start, earliest end. Those belong to the temporal algebra
//! (`trql_algebra::intervals`), which sits above the crates using this one.

/// A closed frame range `[start, end]`, both inclusive.
pub type Interval = (u32, u32);

/// Intersects two sorted, disjoint lists.
pub fn intersect(a: &[Interval], b: &[Interval]) -> Vec<Interval> {
    let mut out = Vec::new();
    intersect_into(a, b, &mut out);
    out
}

/// [`intersect`], writing into `out` (cleared first) so a caller looping over
/// many pairs can reuse one allocation.
pub fn intersect_into(a: &[Interval], b: &[Interval], out: &mut Vec<Interval>) {
    out.clear();
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        let lo = a[i].0.max(b[j].0);
        let hi = a[i].1.min(b[j].1);
        if lo <= hi {
            out.push((lo, hi));
        }
        // Advance the list that ends first; the other may still meet the next one.
        if a[i].1 < b[j].1 {
            i += 1;
        } else {
            j += 1;
        }
    }
}

/// Length, in frames, of the longest consecutive run `ivs` covers.
///
/// Adjacent intervals count as one run — `[3, 5]` then `[6, 9]` is seven
/// consecutive frames, not four — since this crate does not coalesce them
/// (see the module docs) but the frames they name are contiguous all the same.
/// Returns `0` for an empty list.
pub fn longest_run(ivs: &[Interval]) -> u32 {
    let mut best = 0;
    let mut run: Option<Interval> = None;
    for &(s, e) in ivs {
        run = match run {
            Some((rs, re)) if s <= re.saturating_add(1) => Some((rs, e.max(re))),
            _ => {
                best = best.max(run.map_or(0, |(rs, re)| re - rs + 1));
                Some((s, e))
            }
        };
    }
    best.max(run.map_or(0, |(rs, re)| re - rs + 1))
}

/// Unions two lists, merging overlapping *and* adjacent intervals — `[0,4]` and
/// `[5,9]` become `[0,9]`, since frames are discrete.
pub fn union(mut a: Vec<Interval>, mut b: Vec<Interval>) -> Vec<Interval> {
    a.append(&mut b);
    a.sort_unstable();
    let mut merged: Vec<Interval> = Vec::with_capacity(a.len());
    for iv in a {
        if let Some(last) = merged.last_mut() {
            if iv.0 <= last.1.saturating_add(1) {
                last.1 = last.1.max(iv.1);
                continue;
            }
        }
        merged.push(iv);
    }
    merged
}

/// The parts of `domain` not covered by `inner` — the complement, clipped to a
/// finite domain rather than to all of `u32`.
pub fn complement_in_domain(inner: &[Interval], domain: &[Interval]) -> Vec<Interval> {
    let mut result = Vec::new();
    for &(d_lo, d_hi) in domain {
        let mut cursor = d_lo;
        for &(i_lo, i_hi) in inner {
            if i_hi < cursor {
                continue;
            }
            if i_lo > d_hi {
                break;
            }
            // Only a strictly-later start leaves a gap. Computing `i_lo - 1`
            // first would underflow at frame 0 and emit a phantom `(0, 0)`.
            if i_lo > cursor {
                result.push((cursor, i_lo - 1));
            }
            cursor = i_hi.saturating_add(1);
        }
        if cursor <= d_hi {
            result.push((cursor, d_hi));
        }
    }
    result
}

/// Whether any interval of `a` overlaps any interval of `b`. Cheaper than
/// [`intersect`] when only the existence of an overlap matters, since it stops
/// at the first one and allocates nothing.
pub fn overlaps_any(a: &[Interval], b: &[Interval]) -> bool {
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        if a[i].0.max(b[j].0) <= a[i].1.min(b[j].1) {
            return true;
        }
        if a[i].1 < b[j].1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intersect_overlapping_and_disjoint() {
        assert_eq!(
            intersect(&[(0, 10), (20, 30)], &[(5, 25)]),
            vec![(5, 10), (20, 25)]
        );
        assert_eq!(intersect(&[(0, 4)], &[(5, 9)]), vec![]);
    }

    #[test]
    fn intersect_into_clears_first() {
        let mut out = vec![(99, 99)];
        intersect_into(&[(0, 10)], &[(5, 20)], &mut out);
        assert_eq!(out, vec![(5, 10)]);
        intersect_into(&[(0, 1)], &[(8, 9)], &mut out);
        assert!(out.is_empty(), "stale contents must not survive");
    }

    #[test]
    fn union_merges_adjacent() {
        // Frames are discrete, so [0,4] and [5,9] touch and become one range.
        assert_eq!(union(vec![(0, 4)], vec![(5, 9)]), vec![(0, 9)]);
        assert_eq!(union(vec![(0, 4)], vec![(6, 9)]), vec![(0, 4), (6, 9)]);
        assert_eq!(union(vec![(5, 9)], vec![(0, 20)]), vec![(0, 20)]);
    }

    #[test]
    fn complement_carves_gaps_out_of_the_domain() {
        assert_eq!(complement_in_domain(&[], &[(0, 10)]), vec![(0, 10)]);
        assert_eq!(complement_in_domain(&[(0, 10)], &[(0, 10)]), vec![]);
        assert_eq!(
            complement_in_domain(&[(3, 5)], &[(0, 10)]),
            vec![(0, 2), (6, 10)]
        );
        // Multiple domain pieces are handled independently.
        assert_eq!(
            complement_in_domain(&[(3, 5)], &[(0, 4), (8, 9)]),
            vec![(0, 2), (8, 9)]
        );
    }

    #[test]
    fn overlaps_any_stops_at_the_first_touch() {
        assert!(overlaps_any(&[(0, 10)], &[(10, 20)]), "sharing one frame");
        assert!(
            !overlaps_any(&[(0, 9)], &[(10, 20)]),
            "adjacent is not overlap"
        );
        assert!(overlaps_any(&[(0, 2), (8, 9)], &[(3, 8)]));
        assert!(!overlaps_any(&[], &[(0, 9)]));
    }
}

#[cfg(test)]
mod longest_run_tests {
    use super::longest_run;

    #[test]
    fn empty_list_has_no_run() {
        assert_eq!(longest_run(&[]), 0);
    }

    #[test]
    fn single_interval_is_its_own_length() {
        assert_eq!(longest_run(&[(10, 10)]), 1);
        assert_eq!(longest_run(&[(10, 19)]), 10);
    }

    #[test]
    fn abutting_intervals_form_one_run() {
        assert_eq!(longest_run(&[(3, 5), (6, 9)]), 7);
        assert_eq!(longest_run(&[(3, 5), (6, 9), (11, 12)]), 7);
    }

    #[test]
    fn a_gap_of_one_frame_breaks_the_run() {
        assert_eq!(longest_run(&[(3, 5), (7, 9)]), 3);
    }

    #[test]
    fn the_longest_run_need_not_be_first_or_last() {
        assert_eq!(longest_run(&[(0, 1), (5, 8), (9, 12), (20, 21)]), 8);
    }
}
