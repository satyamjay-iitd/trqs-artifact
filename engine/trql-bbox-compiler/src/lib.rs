//! Analytical compiler for structured bounding-box predicates.
//!
//! The polynomial *degree* this compiler reasons about is selected at compile time via a
//! Cargo feature, matching the active [`trql_poly`] degree. Exactly one of the
//! following must be active:
//!
//! | feature     | module          | poly degree |
//! |-------------|-----------------|-------------|
//! | `linear`    | [`linear`]      | 1           |
//! | `quadratic` | [`quadratic`]   | 2 (default) |
//! | `cubic`     | [`cubic`]       | 3           |
//!
//! Each module exposes the same public surface (`BBoxExpr`, `AnalyticalPred`,
//! `eval_analytical`, …); `lib.rs` re-exports whichever one the active feature selects.

#[cfg(not(any(feature = "linear", feature = "quadratic", feature = "cubic")))]
compile_error!(
    "trql-bbox-compiler: enable exactly one of the `linear`, `quadratic`, or `cubic` features"
);

#[cfg(any(
    all(feature = "linear", feature = "quadratic"),
    all(feature = "linear", feature = "cubic"),
    all(feature = "quadratic", feature = "cubic"),
))]
compile_error!(
    "trql-bbox-compiler: the `linear`, `quadratic`, and `cubic` features are mutually exclusive"
);

// Degree-independent: a region is a region whatever the trajectory polynomials look like,
// so this is one module rather than a copy inside each degree module.
mod polygon;
pub use polygon::{Polygon, PolygonError, corner_extreme_weights};

#[cfg(feature = "linear")]
mod linear;
#[cfg(feature = "linear")]
pub use linear::*;

#[cfg(feature = "quadratic")]
mod quadratic;
#[cfg(feature = "quadratic")]
pub use quadratic::*;

#[cfg(feature = "cubic")]
mod cubic;
#[cfg(feature = "cubic")]
pub use cubic::*;
