//! Object and trajectory types for tracked entities.

use bincode::{Decode, Encode};

use crate::intervals::Interval;

/// Index into a shared bounding-box table.
pub type BBID = u32;

/// Axis-aligned bounding box with pixel coordinates.
#[derive(Debug, Clone, Encode, Decode, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BoundingBox {
    pub x1: u32,
    pub y1: u32,
    pub x2: u32,
    pub y2: u32,
}

impl BoundingBox {
    /// Returns `true` if `other` is fully contained within this box.
    pub fn contains(&self, other: &BoundingBox) -> bool {
        self.x1 <= other.x1 && self.y1 <= other.y1 && self.x2 >= other.x2 && self.y2 >= other.y2
    }

    /// X coordinate of the center.
    pub fn center_x(&self) -> u32 {
        (self.x1 + self.x2) / 2
    }

    /// Y coordinate of the center.
    pub fn center_y(&self) -> u32 {
        (self.y1 + self.y2) / 2
    }

    /// A zero-area bounding box at the origin.
    pub fn empty() -> Self {
        Self {
            x1: 0,
            y1: 0,
            x2: 0,
            y2: 0,
        }
    }
}

/// Trajectory with inline bounding boxes (one per frame).
#[derive(Debug, Clone, Encode, Decode)]
pub struct ExactTraj {
    pub frames: Vec<u32>,
    pub bboxes: Vec<BoundingBox>,
}

/// Alias for [`ExactTraj`].
pub type TrajWithBBox = ExactTraj;

/// Trajectory with indexed bounding-box references into a shared table.
#[derive(Debug, Clone, Encode, Decode)]
pub struct Trajectory {
    pub frames: Vec<u32>,
    pub bboxes: Vec<BBID>,
}

impl Trajectory {
    /// Groups consecutive frames into clips of at least `duration` frames.
    pub fn group_by_duration(self, duration: u32) -> Option<Clips> {
        find_consecutive_runs(self.frames, duration)
    }

    /// Returns `true` if the trajectory has no frames.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Returns `(first_frame, last_frame)`.
    pub fn span(&self) -> Interval {
        span(&self.frames)
    }
}

impl ExactTraj {
    /// Groups consecutive frames into clips of at least `duration` frames.
    pub fn group_by_duration(self, duration: u32) -> Option<Clips> {
        find_consecutive_runs(self.frames, duration)
    }

    /// Returns `true` if the trajectory has no frames.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Returns `(first_frame, last_frame)`.
    pub fn span(&self) -> Interval {
        span(&self.frames)
    }
}

/// A set of `(start, end)` frame intervals (clips).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Clips {
    pub clips: Vec<Interval>,
}

fn find_consecutive_runs(frames: Vec<u32>, duration: u32) -> Option<Clips> {
    let clips = frames_to_intervals(&frames)
        .into_iter()
        .filter(|(start, end)| end - start + 1 >= duration)
        .collect::<Vec<_>>();
    (!clips.is_empty()).then_some(Clips { clips })
}

/// Converts a sorted list of frame IDs into merged `(start, end)` intervals
/// by grouping consecutive frames.
pub fn frames_to_intervals(frames: &[u32]) -> Vec<Interval> {
    if frames.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::new();
    let mut start = frames[0];
    let mut prev = frames[0];

    for &fid in &frames[1..] {
        if fid == prev + 1 {
            prev = fid;
            continue;
        }
        out.push((start, prev));
        start = fid;
        prev = fid;
    }
    out.push((start, prev));
    out
}

fn span(frames: &[u32]) -> Interval {
    (
        *frames.first().expect("non-empty trajectory"),
        *frames.last().expect("non-empty trajectory"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- frames_to_intervals ---

    #[test]
    fn frames_to_intervals_empty() {
        assert_eq!(frames_to_intervals(&[]), Vec::<Interval>::new());
    }

    #[test]
    fn frames_to_intervals_single() {
        assert_eq!(frames_to_intervals(&[5]), vec![(5, 5)]);
    }

    #[test]
    fn frames_to_intervals_contiguous() {
        assert_eq!(frames_to_intervals(&[1, 2, 3, 4, 5]), vec![(1, 5)]);
    }

    #[test]
    fn frames_to_intervals_multiple_runs() {
        assert_eq!(
            frames_to_intervals(&[1, 2, 3, 10, 11, 20]),
            vec![(1, 3), (10, 11), (20, 20)]
        );
    }

    #[test]
    fn frames_to_intervals_all_gaps() {
        assert_eq!(
            frames_to_intervals(&[1, 5, 9]),
            vec![(1, 1), (5, 5), (9, 9)]
        );
    }

    // --- BoundingBox ---

    #[test]
    fn bbox_contains_self() {
        let b = BoundingBox {
            x1: 10,
            y1: 10,
            x2: 20,
            y2: 20,
        };
        assert!(b.contains(&b));
    }

    #[test]
    fn bbox_contains_inner() {
        let outer = BoundingBox {
            x1: 0,
            y1: 0,
            x2: 100,
            y2: 100,
        };
        let inner = BoundingBox {
            x1: 10,
            y1: 10,
            x2: 50,
            y2: 50,
        };
        assert!(outer.contains(&inner));
        assert!(!inner.contains(&outer));
    }

    #[test]
    fn bbox_contains_partial_overlap() {
        let a = BoundingBox {
            x1: 0,
            y1: 0,
            x2: 50,
            y2: 50,
        };
        let b = BoundingBox {
            x1: 25,
            y1: 25,
            x2: 75,
            y2: 75,
        };
        assert!(!a.contains(&b));
        assert!(!b.contains(&a));
    }

    #[test]
    fn bbox_center() {
        let b = BoundingBox {
            x1: 10,
            y1: 20,
            x2: 30,
            y2: 40,
        };
        assert_eq!(b.center_x(), 20);
        assert_eq!(b.center_y(), 30);
    }

    #[test]
    fn bbox_empty() {
        let b = BoundingBox::empty();
        assert_eq!(b.x1, 0);
        assert_eq!(b.y2, 0);
    }

    // --- Trajectory ---

    #[test]
    fn trajectory_is_empty() {
        let t = Trajectory {
            frames: vec![],
            bboxes: vec![],
        };
        assert!(t.is_empty());
    }

    #[test]
    fn trajectory_span() {
        let t = Trajectory {
            frames: vec![5, 6, 7, 10],
            bboxes: vec![0, 1, 2, 3],
        };
        assert_eq!(t.span(), (5, 10));
    }

    #[test]
    fn trajectory_group_by_duration_none() {
        let t = Trajectory {
            frames: vec![1, 2, 5, 6],
            bboxes: vec![0, 1, 2, 3],
        };
        assert_eq!(t.group_by_duration(3), None);
    }

    #[test]
    fn trajectory_group_by_duration_some() {
        let t = Trajectory {
            frames: vec![1, 2, 3, 10, 11],
            bboxes: vec![0, 1, 2, 3, 4],
        };
        let clips = t.group_by_duration(3).unwrap();
        assert_eq!(clips.clips, vec![(1, 3)]);
    }

    // --- ExactTraj ---

    #[test]
    fn exact_traj_group_by_duration() {
        let bb = || BoundingBox::empty();
        let t = ExactTraj {
            frames: vec![1, 2, 3, 4, 10, 11],
            bboxes: vec![bb(), bb(), bb(), bb(), bb(), bb()],
        };
        let clips = t.group_by_duration(2).unwrap();
        assert_eq!(clips.clips, vec![(1, 4), (10, 11)]);
    }

    #[test]
    fn exact_traj_span() {
        let bb = || BoundingBox::empty();
        let t = ExactTraj {
            frames: vec![3, 4, 5],
            bboxes: vec![bb(), bb(), bb()],
        };
        assert_eq!(t.span(), (3, 5));
    }
}
