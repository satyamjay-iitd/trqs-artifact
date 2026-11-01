//! SQL-like text DSL for TrQL queries.
//!
//! Parses a `[REGIONS …] SELECT … WHERE …` string into the
//! [`VariablesWithConstraints`](trql_algebra::plan::VariablesWithConstraints) /
//! [`TrajExpr`](trql_algebra::expr::TrajExpr) AST.
//!
//! # Grammar (informal)
//! ```text
//! query    := regions? SELECT selector WHERE expr
//! regions  := "REGIONS" "(" region ("," region)* ")"
//! region   := ident "=" shape
//! shape    := rect | polygon
//! rect     := "[" num "," num "," num "," num "]"
//! polygon  := "[" point ("," point)* "]"        -- at least 3 vertices
//! point    := "[" num "," num "]"
//! selector := "(" binding ("," binding)* ")"
//! binding  := ident ":" (prop "=" value)*
//! expr     := primary (THEN gap? primary)*
//! gap      := "[" int "]" | "[" int ".." "]" | "[" int ".." int "]"
//! primary  := DURATION "(" expr "," int ")"
//!           | CONTAINS "(" expr "," expr ")"
//!           | OVERLAP "(" expr "," expr ")"
//!           | pred
//!           | "(" expr ")"
//! pred     := pred_and ("OR" pred_and)*
//! pred_and := pred_not ("AND" pred_not)*
//! pred_not := "NOT" pred_not | pred_atom
//! pred_atom := "TRUE"
//!           | bbox_expr rel_op bbox_expr              -- analytic comparison
//!           | slot ("." "center")? INSIDE (shape | ident)
//!           | "(" pred ")"
//! bbox_expr := term (("+" | "-") term)*
//! term      := factor ("*" factor)*
//! factor    := number | slot "." coord | "(" bbox_expr ")" | ("+"|"-") factor
//! ```
//!
//! A rectangle and a polygon are told apart by the second bracket — `[0, 0, 10, 10]` is a
//! rectangle, `[[0,0],[10,0],[10,10],[0,10]]` a polygon — so no keyword is needed and no
//! query written before polygons existed changes meaning. A polygon's ring is implicitly
//! closed and its winding does not matter.
//!
//! `INSIDE` is mostly sugar: the parser expands a rectangle into the four affine comparisons
//! it stands for (`x1 ≥ … ∧ y1 ≥ … ∧ x2 ≤ … ∧ y2 ≤ …`, or the box centre for the `.center`
//! form), and a *convex* polygon into one half-plane comparison per edge — per corner too,
//! for the whole-box form. `REGIONS` names are resolved at parse time. None of it survives
//! into the AST, so nothing downstream of this crate knows any of it exists.
//!
//! A **concave** polygon is the one case with no such conjunctive form. `v.center INSIDE p`
//! then compiles to an `InPolygon` node, which is still solved analytically — the trajectory
//! polynomials are substituted into each edge's line, and the resulting sign changes bound
//! the frames where membership can flip.
//!
//! `v INSIDE p` for a concave `p` needs one condition more than its corners, since all four
//! can sit inside such a region while the box crosses a notch: no edge of the region may
//! pass through the box's interior. That condition is a separating-axis test per edge, and
//! it stays affine because the box corner extremal along an axis is fixed by the axis's
//! sign — so the whole thing is still comparisons, four `InPolygon` corner tests aside.
//!
//! # Example
//! ```no_run
//! use trql_dsl::parse_query;
//! use trql_types::prop::{Schema, TypeDesc};
//!
//! let schema = Schema::new(vec![
//!     ("car_type".into(), TypeDesc::Enum(vec!["SUV".into(), "Sedan".into()])),
//! ]);
//! let q = parse_query(
//!     "SELECT (a: car_type=SUV, b: car_type=Sedan) WHERE duration(a.x1 < b.x1, 30)",
//!     &schema,
//! ).unwrap();
//! ```

mod parser;
pub use parser::{DslError, QueryRegion, parse_query, parse_query_regions};
