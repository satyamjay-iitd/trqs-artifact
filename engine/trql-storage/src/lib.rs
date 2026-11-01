//! Columnar storage engine for TrQL.
//!
//! Provides the [`db::Trqs`] database, property column storage with indexes,
//! and binary persistence via bincode.
//!
//! With the `duckdb` feature, it also provides the ingestion path that turns a
//! dataset's source `duckdb.db` into a `.trql`: [`duckdb_meta`] infers the
//! schema and an internal `convert` module writes the database, reached through
//! [`datasets::SubDataset::load_database`] and
//! [`datasets::SubDataset::rebuild_trql`]. [`merge`] folds several sub-datasets
//! into one — a storage operation, so it lives here rather than in a front end,
//! and both `trql-tui` and `trql-cli` drive the same implementation.

pub mod datasets;
pub mod db;
pub mod index;
pub mod prop;
pub mod stats;

#[cfg(feature = "duckdb")]
pub(crate) mod convert;
#[cfg(feature = "duckdb")]
pub mod duckdb_meta;
#[cfg(feature = "duckdb")]
pub mod merge;

// Detection and its scoring. Here rather than in a front end because both the
// TUI and the CLI drive the same pipeline, the same way `merge` is shared: a
// detection made from either is the same detection.
#[cfg(feature = "duckdb")]
pub mod align;
#[cfg(feature = "duckdb")]
pub mod detect;
#[cfg(feature = "duckdb")]
pub mod eval;
