//! Polynomial trajectory fitting and analytic interval solving.
//!
//! The polynomial *degree* used to approximate trajectories is selected at compile
//! time via a Cargo feature. Exactly one of the following must be active:
//!
//! | feature     | module          | fit degree |
//! |-------------|-----------------|------------|
//! | `linear`    | [`linear`]      | 1          |
//! | `quadratic` | [`quadratic`]   | 2 (default)|
//! | `cubic`     | [`cubic`]       | 3          |
//!
//! Each module exposes the same public surface (`PolyTraj`, `PolySegment`,
//! `ScalarSegmentData2`, `fit_polynomial`, `positive_integer_intervals`, …); `lib.rs`
//! simply re-exports whichever one the active feature selects.

#[cfg(not(any(feature = "linear", feature = "quadratic", feature = "cubic")))]
compile_error!("trql-poly: enable exactly one of the `linear`, `quadratic`, or `cubic` features");

#[cfg(any(
    all(feature = "linear", feature = "quadratic"),
    all(feature = "linear", feature = "cubic"),
    all(feature = "quadratic", feature = "cubic"),
))]
compile_error!("trql-poly: the `linear`, `quadratic`, and `cubic` features are mutually exclusive");

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
