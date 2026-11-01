//! Predicate intermediate representation for TrQL.
//!
//! Defines per-frame spatial predicates ([`spatial::SpatialBody`]) and
//! trajectory-level predicates ([`traj_pred::TrajBody`]) that can be
//! composed with boolean operators.

pub mod spatial;
pub mod traj_pred;
