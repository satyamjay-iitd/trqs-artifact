//! Trajectory-level predicates that produce frame intervals.

use std::sync::Arc;

use trql_poly::PolyTraj;
use trql_types::{
    ObjId,
    intervals::Interval,
    obj::{BoundingBox, frames_to_intervals},
};

/// Composable trajectory predicate tree.
///
/// Evaluation returns a list of `(start, end)` frame intervals where
/// the predicate holds. Boolean combinators merge intervals accordingly.
#[derive(Clone)]
pub enum TrajBody {
    Atomic(Arc<dyn Fn(&PolyTraj) -> Vec<Interval> + Send + Sync>),
    And(Box<TrajBody>, Box<TrajBody>),
    Or(Box<TrajBody>, Box<TrajBody>),
    Not(Box<TrajBody>),
    True,
}

impl std::fmt::Debug for TrajBody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Atomic(_) => f.debug_tuple("Atomic").finish(),
            Self::And(arg0, arg1) => f.debug_tuple("And").field(arg0).field(arg1).finish(),
            Self::Or(arg0, arg1) => f.debug_tuple("Or").field(arg0).field(arg1).finish(),
            Self::Not(arg0) => f.debug_tuple("Not").field(arg0).finish(),
            Self::True => write!(f, "True"),
        }
    }
}

impl TrajBody {
    /// Evaluates this predicate on an exact trajectory, returning canonical frame intervals.
    pub fn eval(&self, traj: &PolyTraj) -> Vec<Interval> {
        match self {
            TrajBody::Atomic(f) => canonicalize(f(traj)),
            TrajBody::And(a, b) => intersect(&a.eval(traj), &b.eval(traj)),
            TrajBody::Or(a, b) => union(&a.eval(traj), &b.eval(traj)),
            TrajBody::Not(a) => complement_within(&a.eval(traj), &presence_intervals(traj)),
            TrajBody::True => presence_intervals(traj),
        }
    }

    /// Returns `true` if the predicate holds for any frames in the trajectory.
    pub fn eval_bool(&self, traj: &PolyTraj) -> bool {
        !self.eval(traj).is_empty()
    }

    /// Returns `true` if this is the trivially-true predicate.
    pub fn is_true(&self) -> bool {
        matches!(self, TrajBody::True)
    }

    /// Creates a predicate that detects center-x direction reversals of at least `min_streak` consecutive frames.
    pub fn reverses_direction(min_streak: usize) -> Self {
        TrajBody::Atomic(Arc::new(move |traj| {
            fn commit(
                dir: i8,
                start: u32,
                end: u32,
                len: usize,
                min_streak: usize,
                best_left: &mut Option<Interval>,
                out: &mut Vec<Interval>,
            ) {
                if len < min_streak {
                    return;
                }
                if dir == -1 {
                    *best_left = Some(match *best_left {
                        Some((best_start, best_end)) if best_end <= end => (best_start, best_end),
                        _ => (start, end),
                    });
                } else if let Some((left_start, left_end)) = *best_left
                    && start > left_end
                {
                    out.push((left_start, end));
                }
            }

            let mut prev: Option<(ObjId, BoundingBox)> = None;
            let mut s_dir: i8 = 0;
            let mut s_start: u32 = 0;
            let mut s_end: u32 = 0;
            let mut s_len: usize = 0;
            let mut s_last_fid: u32 = 0;
            let mut best_left: Option<Interval> = None;
            let mut out = Vec::new();

            for (fid, bbox) in traj.into_iter() {
                if let Some((prev_fid, prev_bbox)) = prev
                    && fid == prev_fid + 1
                {
                    let cx0 = prev_bbox.center_x() as i64;
                    let cx1 = bbox.center_x() as i64;
                    let d = cx0 - cx1;
                    if d != 0 {
                        let dir = d.signum() as i8;
                        let move_fid = prev_fid;

                        if s_len > 0 && dir == s_dir && move_fid == s_last_fid + 1 {
                            s_end = move_fid;
                            s_len += 1;
                            s_last_fid = move_fid;
                        } else {
                            commit(
                                s_dir,
                                s_start,
                                s_end,
                                s_len,
                                min_streak,
                                &mut best_left,
                                &mut out,
                            );
                            s_dir = dir;
                            s_start = move_fid;
                            s_end = move_fid;
                            s_len = 1;
                            s_last_fid = move_fid;
                        }
                    }
                }
                prev = Some((fid, bbox));
            }

            commit(
                s_dir,
                s_start,
                s_end,
                s_len,
                min_streak,
                &mut best_left,
                &mut out,
            );
            out
        }))
    }

    /// Creates a predicate matching frames where the per-frame center displacement exceeds `threshold` pixels.
    pub fn avg_displacement_gt(threshold: f64) -> Self {
        TrajBody::Atomic(Arc::new(move |traj| {
            let mut prev: Option<(ObjId, BoundingBox)> = None;
            let mut matching_frames = Vec::new();

            for (fid, bbox) in traj.into_iter() {
                if let Some((prev_fid, prev_bbox)) = prev
                    && fid == prev_fid + 1
                {
                    let dx = bbox.center_x() as f64 - prev_bbox.center_x() as f64;
                    let dy = bbox.center_y() as f64 - prev_bbox.center_y() as f64;
                    if (dx * dx + dy * dy).sqrt() > threshold {
                        matching_frames.push(prev_fid);
                        matching_frames.push(fid);
                    }
                }
                prev = Some((fid, bbox));
            }

            matching_frames.sort_unstable();
            matching_frames.dedup();
            canonicalize(frames_to_intervals(&matching_frames))
        }))
    }
}

fn presence_intervals(traj: &PolyTraj) -> Vec<Interval> {
    frames_to_intervals(&traj.into_iter().map(|(fid, _)| fid).collect::<Vec<_>>())
}

fn canonicalize(mut raw: Vec<Interval>) -> Vec<Interval> {
    raw.retain(|(start, end)| start <= end);
    raw.sort_unstable_by_key(|&(start, end)| (start, end));

    let mut out: Vec<Interval> = Vec::with_capacity(raw.len());
    for (start, end) in raw {
        if let Some((_, prev_end)) = out.last_mut()
            && start <= prev_end.saturating_add(1)
        {
            *prev_end = (*prev_end).max(end);
            continue;
        }
        out.push((start, end));
    }
    out
}

fn intersect(lhs: &[Interval], rhs: &[Interval]) -> Vec<Interval> {
    let mut out = Vec::new();
    let mut i = 0;
    let mut j = 0;

    while i < lhs.len() && j < rhs.len() {
        let (a_start, a_end) = lhs[i];
        let (b_start, b_end) = rhs[j];
        let start = a_start.max(b_start);
        let end = a_end.min(b_end);
        if start <= end {
            out.push((start, end));
        }

        if a_end < b_end {
            i += 1;
        } else {
            j += 1;
        }
    }

    out
}

fn union(lhs: &[Interval], rhs: &[Interval]) -> Vec<Interval> {
    let mut raw = lhs.to_vec();
    raw.extend_from_slice(rhs);
    canonicalize(raw)
}

fn complement_within(inner: &[Interval], domain: &[Interval]) -> Vec<Interval> {
    let mut out = Vec::new();
    let mut inner_idx = 0;

    for &(domain_start, domain_end) in domain {
        while inner_idx < inner.len() && inner[inner_idx].1 < domain_start {
            inner_idx += 1;
        }

        let mut cursor = domain_start;
        let mut probe = inner_idx;
        while probe < inner.len() {
            let (inner_start, inner_end) = inner[probe];
            if inner_start > domain_end {
                break;
            }

            if cursor < inner_start {
                out.push((cursor, inner_start - 1));
            }

            cursor = cursor.max(inner_end.saturating_add(1));
            if cursor > domain_end {
                break;
            }
            probe += 1;
        }

        if cursor <= domain_end {
            out.push((cursor, domain_end));
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use trql_types::obj::ExactTraj;

    use super::*;

    fn bb(cx: u32) -> BoundingBox {
        BoundingBox {
            x1: cx - 5,
            y1: 0,
            x2: cx + 5,
            y2: 10,
        }
    }

    fn make_traj(frames: Vec<u32>, centers: Vec<u32>) -> PolyTraj {
        let traj = ExactTraj {
            frames,
            bboxes: centers.into_iter().map(bb).collect(),
        };
        PolyTraj::from_traj_with_bboxes(&traj, 0, 0.0)
    }

    // --- TrajBody::True ---

    #[test]
    fn true_returns_presence() {
        let traj = make_traj(vec![1, 2, 3, 10, 11], vec![50, 50, 50, 50, 50]);
        let result = TrajBody::True.eval(&traj);
        assert_eq!(result, vec![(1, 3), (10, 11)]);
    }

    #[test]
    fn true_eval_bool() {
        let traj = make_traj(vec![1, 2], vec![50, 50]);
        assert!(TrajBody::True.eval_bool(&traj));
    }

    #[test]
    fn is_true() {
        assert!(TrajBody::True.is_true());
        let atomic = TrajBody::Atomic(Arc::new(|_| vec![]));
        assert!(!atomic.is_true());
    }

    // --- TrajBody::Atomic ---

    #[test]
    fn atomic_returns_matching_intervals() {
        let pred = TrajBody::Atomic(Arc::new(|_traj| vec![(1, 3)]));
        let traj = make_traj(vec![1, 2, 3, 4, 5], vec![50; 5]);
        assert_eq!(pred.eval(&traj), vec![(1, 3)]);
    }

    #[test]
    fn atomic_empty_result() {
        let pred = TrajBody::Atomic(Arc::new(|_| vec![]));
        let traj = make_traj(vec![1, 2, 3], vec![50, 50, 50]);
        assert!(!pred.eval_bool(&traj));
    }

    // --- TrajBody::And ---

    #[test]
    fn and_intersects_intervals() {
        let a = TrajBody::Atomic(Arc::new(|_| vec![(1, 5)]));
        let b = TrajBody::Atomic(Arc::new(|_| vec![(3, 7)]));
        let pred = TrajBody::And(Box::new(a), Box::new(b));
        let traj = make_traj(vec![1, 2, 3, 4, 5, 6, 7], vec![50; 7]);
        assert_eq!(pred.eval(&traj), vec![(3, 5)]);
    }

    #[test]
    fn and_disjoint() {
        let a = TrajBody::Atomic(Arc::new(|_| vec![(1, 3)]));
        let b = TrajBody::Atomic(Arc::new(|_| vec![(5, 7)]));
        let pred = TrajBody::And(Box::new(a), Box::new(b));
        let traj = make_traj(vec![1, 2, 3, 5, 6, 7], vec![50; 6]);
        assert!(pred.eval(&traj).is_empty());
    }

    // --- TrajBody::Or ---

    #[test]
    fn or_unions_intervals() {
        let a = TrajBody::Atomic(Arc::new(|_| vec![(1, 3)]));
        let b = TrajBody::Atomic(Arc::new(|_| vec![(5, 7)]));
        let pred = TrajBody::Or(Box::new(a), Box::new(b));
        let traj = make_traj(vec![1, 2, 3, 5, 6, 7], vec![50; 6]);
        assert_eq!(pred.eval(&traj), vec![(1, 3), (5, 7)]);
    }

    #[test]
    fn or_merges_overlapping() {
        let a = TrajBody::Atomic(Arc::new(|_| vec![(1, 5)]));
        let b = TrajBody::Atomic(Arc::new(|_| vec![(3, 7)]));
        let pred = TrajBody::Or(Box::new(a), Box::new(b));
        let traj = make_traj(vec![1, 2, 3, 4, 5, 6, 7], vec![50; 7]);
        assert_eq!(pred.eval(&traj), vec![(1, 7)]);
    }

    // --- TrajBody::Not ---

    #[test]
    fn not_complements_within_presence() {
        let inner = TrajBody::Atomic(Arc::new(|_| vec![(3, 5)]));
        let pred = TrajBody::Not(Box::new(inner));
        let traj = make_traj(vec![1, 2, 3, 4, 5, 6, 7], vec![50; 7]);
        assert_eq!(pred.eval(&traj), vec![(1, 2), (6, 7)]);
    }

    // --- avg_displacement_gt ---

    #[test]
    fn avg_displacement_gt_detects_movement() {
        // Objects moving 20px per frame horizontally
        let traj = make_traj(vec![1, 2, 3, 4], vec![100, 120, 140, 160]);
        let pred = TrajBody::avg_displacement_gt(10.0);
        let result = pred.eval(&traj);
        assert!(!result.is_empty());
    }

    #[test]
    fn avg_displacement_gt_rejects_stationary() {
        let traj = make_traj(vec![1, 2, 3, 4], vec![100, 100, 100, 100]);
        let pred = TrajBody::avg_displacement_gt(10.0);
        assert!(pred.eval(&traj).is_empty());
    }
}
