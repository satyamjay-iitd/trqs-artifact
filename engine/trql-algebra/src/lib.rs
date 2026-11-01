//! TrQL trajectory algebra: expression AST, optimizer, evaluator, and candidate selector.
//!
//! The query pipeline is:
//! ```text
//! ObjSelector + TrajExpr → optimize() → AlgebraCtx::run() → (tuple, Intervals)
//! ```

pub use smallvec::SmallVec;
pub use trql_storage::db::FramePathIndex;
pub use trql_storage::{db, index, prop};

/// Query-time breakdown instrumentation (paper P4). See [`trql_instrument`].
///
/// A no-op unless the crate is built with the `instrument` feature. Callers
/// wrap a batch of queries with [`instrument::reset`] / [`instrument::snapshot`]
/// to read the root-finding / segment-iteration / other split.
pub use trql_instrument as instrument;

pub mod eval;
pub mod expr;
pub mod optimize;
pub mod plan;
