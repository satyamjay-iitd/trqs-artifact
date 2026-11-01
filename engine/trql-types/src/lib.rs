//! Shared data types for the TrQL query engine.
//!
//! Provides core types for bounding boxes, trajectories, property schemas,
//! and typed values used across all TrQL crates.

use smallvec::SmallVec;

use crate::intervals::IntervalSets;

pub mod intervals;
pub mod obj;
pub mod prop;

// ---- Objects ---------------------------------------------------------------------

/// Identifier of one object in the database — an index into its trajectories.
pub type ObjId = u32;

/// One [`ObjId`] per query slot, indexed by [`Slot`]: `tuple[slot]` is the object
/// bound to that variable.
///
/// Inlined for arities ≤ 4 (the common binary/ternary/quaternary cases) with
/// automatic heap fallback for wider tuples.
pub type Tuple = SmallVec<[ObjId; 4]>;

/// What a query returns: one row per object tuple that satisfies it, paired with
/// the frame intervals over which it does.
pub type QueryResult = Vec<(Tuple, IntervalSets)>;

// ---- Slot & Gap -----------------------------------------------------------------

/// Position of one "object variable" of a query in the match binding.
///
/// A slot is just an index: `tuple[slot]` is the ObjId bound to it, and
/// `binding[slot]` reads it back from a match. The variable's name is not part
/// of it — that lives once in the query's declaration table (see
/// `trql_algebra::plan::VariablesWithConstraints`), which is what assigns these
/// indices in the first place and what to ask when a name is needed for
/// diagnostics.
pub type Slot = usize;

/// Frame gap allowed between two consecutive groups in a
/// `ChainLink::then_relaxed` step: the next group must have presence in the
/// window running from the previous group's start plus `min` to its end plus
/// `max`.
///
/// Converts from range syntax and integers:
///
/// | expression | meaning |
/// |---|---|
/// | `1..`      | at least 1 frame |
/// | `0..=30`   | between 0 and 30 frames (inclusive) |
/// | `1..30`    | between 1 and 29 frames |
/// | `..`       | any gap (≥ 0) |
/// | `5`        | exactly 5 frames |
///
/// A gap of 0 allows a shared boundary frame; a gap of 1 means adjacent with
/// no shared frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gap {
    pub min: u32,
    pub max: Option<u32>,
}

impl Gap {
    pub const fn at_least(min: u32) -> Self {
        Gap { min, max: None }
    }

    pub const fn between(min: u32, max: u32) -> Self {
        Gap {
            min,
            max: Some(max),
        }
    }

    pub const fn exactly(frames: u32) -> Self {
        Gap {
            min: frames,
            max: Some(frames),
        }
    }
}

impl From<u32> for Gap {
    fn from(frames: u32) -> Self {
        Gap::exactly(frames)
    }
}

impl From<std::ops::RangeFrom<u32>> for Gap {
    fn from(r: std::ops::RangeFrom<u32>) -> Self {
        Gap::at_least(r.start)
    }
}

impl From<std::ops::RangeInclusive<u32>> for Gap {
    fn from(r: std::ops::RangeInclusive<u32>) -> Self {
        Gap::between(*r.start(), *r.end())
    }
}

impl From<std::ops::Range<u32>> for Gap {
    fn from(r: std::ops::Range<u32>) -> Self {
        Gap::between(r.start, r.end.saturating_sub(1))
    }
}

impl From<std::ops::RangeFull> for Gap {
    fn from(_: std::ops::RangeFull) -> Self {
        Gap::at_least(0)
    }
}
