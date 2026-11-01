//! Frame intervals and the algebra over them.
//!
//! Two layers, both self-contained — they name nothing else in this crate:
//!
//! * **list** — set operations on sorted, disjoint `(start, end)` lists. What
//!   leaf predicate evaluation and candidate enumeration need: intersect, union,
//!   complement, overlap test. No notion of bounds.
//! * **sets** — the 5-tuple algebra. An [`IntervalSet`] `(S, E, c, L, R)` denotes
//!   every `(start, end)` pair within `[S, E]` that is at least `c` long and
//!   respects the start/end bounds `L`/`R`; [`IntervalSets`] is a union of those,
//!   closed under the temporal operators (`then`, `contains`, `overlap`,
//!   minimum duration).
//!
//! The split matters: the list layer is what a predicate evaluator produces, and
//! the set layer is what the query algebra composes. `IntervalSets::new` lifts
//! the former into the latter.
//!
//! * **naive** — an ablation baseline for the set layer: the same API over the
//!   fully materialized set of `(start, end)` pairs. The `naive-intervals`
//!   feature swaps it in as [`IntervalSets`] so the engine can be measured with
//!   and without the 5-tuple.

mod list;
mod naive;
// Only `IntervalSet` is used from here when the naive ablation is selected.
#[cfg_attr(feature = "naive-intervals", allow(dead_code))]
mod sets;

pub use list::{
    Interval, complement_in_domain, intersect, intersect_into, longest_run, overlaps_any, union,
};
#[cfg(feature = "naive-intervals")]
pub use naive::IntervalSets;
pub use sets::IntervalSet;
#[cfg(not(feature = "naive-intervals"))]
pub use sets::IntervalSets;
