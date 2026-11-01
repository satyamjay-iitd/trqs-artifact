//! TrQL — a trajectory algebra engine for composable spatio-temporal queries.
//!
//! This crate is the public facade that re-exports the workspace crates:
//!
//! - [`algebra`] — expression AST, optimizer, evaluator, and candidate selector
//! - [`storage`] — columnar database with polynomial trajectories and property indexes
//! - [`poly`] — piecewise quadratic polynomial fitting and analytic interval solving
//! - [`predicates`] — spatial and trajectory predicate IR
//! - [`types`] — shared data types (bounding boxes, schemas, values)
//!
//! # Quick start
//!
//! ```ignore
//! use trql::{db::Trqs, dsl::parse_query, eval::AlgebraCtx};
//!
//! let db = Trqs::from_bin(&path)?;
//! let (constraints, expr) = parse_query(query, db.schema())?;
//! let ctx = AlgebraCtx { db: &db, poly_trajs: None };
//! for (tuple, intervals) in ctx.run(&constraints, &expr) {
//!     // process results
//! }
//! ```

pub use trql_algebra::{eval, expr, optimize, plan};
pub use trql_storage::db::FramePathIndex;
pub use trql_storage::{db, index, prop};
pub use trql_types::intervals;

pub use trql_algebra as algebra;
pub use trql_algebra::instrument;
pub use trql_dsl as dsl;
pub use trql_poly as poly;
pub use trql_predicates as predicates;
pub use trql_storage as storage;
pub use trql_types as types;
