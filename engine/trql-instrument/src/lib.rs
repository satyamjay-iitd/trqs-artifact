//! Feature-gated query-time breakdown instrumentation.
//!
//! A query's wall time is attributed to five mutually-exclusive buckets, which
//! group into the three stages a query actually spends its time in:
//!
//! | Stage | Bucket | What it measures |
//! |---|---|---|
//! | object selection | [`Bucket::Selection`] | resolving each slot's candidates, and enumerating co-appearing tuples |
//! | polynomial solving | [`Bucket::RootFinding`] | analytic root-finding (`positive_integer_intervals` & friends) |
//! | | [`Bucket::SegmentIter`] | sweeping/clipping trajectory segments, excluding the solving it invokes |
//! | temporal relations | [`Bucket::Temporal`] | `THEN` / `CONTAINS` / `OVERLAP` / `DURATION` on interval sets |
//! | — | [`Bucket::Other`] | residual: planning, optimisation, tuple materialisation |
//!
//! `Selection` and `Temporal` were split out of `Other` because the query-time
//! breakdown wants them separately — the two are the cost drivers on opposite
//! ends (a query is selection-bound when its candidate set is large, and
//! temporal-bound when its interval sets are). A large `Other` now means real
//! unattributed work, which is the point of keeping it.
//!
//! Accounting is **exclusive self-time**: entering a nested scope pauses the
//! enclosing bucket's timer and resumes it on drop, so time spent root-finding
//! *inside* a segment sweep is charged only to `RootFinding`, never
//! double-counted. Wrapping the whole query in an `Other` scope then makes the
//! three buckets sum to (approximately) the query's total wall time.
//!
//! With the `enabled` feature off (the default) every entry point is an
//! `#[inline]` no-op and [`Guard`] is a zero-sized type, so production builds
//! pay nothing. Enable it only for benchmarks via the `instrument` feature that
//! each trql crate forwards here.

/// The bucket a [`scope`] charges its self-time to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bucket {
    /// Analytic polynomial root-finding (`positive_integer_intervals` & friends).
    RootFinding = 0,
    /// Segment sweep/clip iteration, excluding the root-finding it invokes.
    SegmentIter = 1,
    /// Residual: planning, expression optimisation, tuple materialisation.
    Other = 2,
    /// Resolving each slot's candidate objects, and enumerating the co-appearing
    /// tuples the predicates then run on.
    Selection = 3,
    /// Interval-set algebra: `THEN`, `CONTAINS`, `OVERLAP`, `DURATION`.
    Temporal = 4,
    /// Turning a leaf's solved frames into an interval-set, and clipping a
    /// memoised result to the tuple's co-appearance. Per leaf per tuple, so it
    /// scales with candidates rather than with segments — which is what
    /// separates it from [`Bucket::SegmentIter`].
    IntervalLift = 5,
}

/// Number of distinct buckets; indices match `Bucket as usize`.
pub const NUM_BUCKETS: usize = 6;

/// Whether instrumentation is compiled in (the `enabled` feature is active).
pub const ENABLED: bool = cfg!(feature = "enabled");

/// Human-readable bucket labels, indexed by `Bucket as usize`.
pub const LABELS: [&str; NUM_BUCKETS] = [
    "root-finding",
    "segment-iteration",
    "other",
    "object-selection",
    "temporal-relations",
    "interval-lifting",
];

/// The three stages the buckets roll up into, for reporting: each entry is a
/// label and the buckets that sum into it.
pub const STAGES: [(&str, &[Bucket]); 4] = [
    ("object selection", &[Bucket::Selection]),
    (
        "polynomial solving",
        &[Bucket::RootFinding, Bucket::SegmentIter],
    ),
    ("interval lifting", &[Bucket::IntervalLift]),
    ("temporal relations", &[Bucket::Temporal]),
];

#[cfg(feature = "enabled")]
mod imp {
    use super::{Bucket, NUM_BUCKETS};
    use std::cell::RefCell;
    use std::time::{Duration, Instant};

    struct Frame {
        bucket: usize,
        /// When this frame last (re)started accumulating into its bucket.
        start: Instant,
    }

    #[derive(Default)]
    struct State {
        buckets: [Duration; NUM_BUCKETS],
        stack: Vec<Frame>,
    }

    thread_local! {
        static STATE: RefCell<State> = RefCell::new(State::default());
    }

    /// RAII scope: charges elapsed self-time to its bucket on drop.
    pub struct Guard(());

    /// Enters a timing scope for `bucket`. The enclosing scope (if any) is
    /// paused for the duration, so nested scopes never double-count.
    #[inline]
    pub fn scope(bucket: Bucket) -> Guard {
        let now = Instant::now();
        STATE.with(|s| {
            let mut s = s.borrow_mut();
            if let Some(top) = s.stack.last() {
                let (bucket, start) = (top.bucket, top.start);
                s.buckets[bucket] += now.saturating_duration_since(start);
            }
            s.stack.push(Frame {
                bucket: bucket as usize,
                start: now,
            });
        });
        Guard(())
    }

    impl Drop for Guard {
        #[inline]
        fn drop(&mut self) {
            let now = Instant::now();
            STATE.with(|s| {
                let mut s = s.borrow_mut();
                if let Some(frame) = s.stack.pop() {
                    s.buckets[frame.bucket] += now.saturating_duration_since(frame.start);
                }
                // Resume the parent's timer from this instant.
                if let Some(top) = s.stack.last_mut() {
                    top.start = now;
                }
            });
        }
    }

    /// Clears accumulated bucket times and any dangling scope stack.
    #[inline]
    pub fn reset() {
        STATE.with(|s| {
            let mut s = s.borrow_mut();
            s.buckets = [Duration::ZERO; NUM_BUCKETS];
            s.stack.clear();
        });
    }

    /// Reads the accumulated self-time per bucket, indexed by `Bucket as usize`.
    #[inline]
    pub fn snapshot() -> [Duration; NUM_BUCKETS] {
        STATE.with(|s| s.borrow().buckets)
    }
}

#[cfg(not(feature = "enabled"))]
mod imp {
    use super::{Bucket, NUM_BUCKETS};
    use std::time::Duration;

    /// Zero-sized no-op guard.
    pub struct Guard;

    #[inline(always)]
    pub fn scope(_bucket: Bucket) -> Guard {
        Guard
    }

    #[inline(always)]
    pub fn reset() {}

    #[inline(always)]
    pub fn snapshot() -> [Duration; NUM_BUCKETS] {
        [Duration::ZERO; NUM_BUCKETS]
    }
}

pub use imp::{Guard, reset, scope, snapshot};
