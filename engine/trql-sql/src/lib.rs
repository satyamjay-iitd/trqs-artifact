//! Compiles a parsed TrQL query into an equivalent DuckDB SQL query.
//!
//! The generated SQL runs against the standard dataset schema
//! (`datasets/processed/<dataset>/<clip>/duckdb.db`):
//!
//! ```text
//! objects(oid PRIMARY KEY, class)
//! bbox(fid, oid -> objects.oid, x1, y1, x2, y2, aid)
//! ```
//!
//! and returns one row per `(object tuple, interval)` — a column per declared query
//! variable holding its bound `oid`, plus the interval bounds `s` and `e`.
//!
//! # How the translation works
//!
//! The query splits cleanly into a pointwise part and a temporal part:
//!
//! - **Pointwise.** Comparisons, `INSIDE`, `AND`, `OR`, `NOT` are all per-frame, so an
//!   entire predicate tree becomes a single SQL boolean (see [`pred`]).
//! - **Temporal.** Only `DURATION`, `THEN`, `CONTAINS`, and `OVERLAP` reason about
//!   intervals. Each node is a CTE over `(oids…, s, e, c, l, r)` rows.
//!
//! The five interval columns are [`trql_types::intervals::IntervalSet`]: the 5-tuple
//! denoting the set of concrete intervals `{(s', e') : s ≤ s' ≤ l, r ≤ e' ≤ e,
//! e' − s' + 1 ≥ c}`. Carrying all five is what makes the SQL *equivalent* to the engine
//! rather than merely similar — `DURATION(p, 30) THEN DURATION(q, 30)` is satisfiable by
//! splitting one long run of `p`/`q`, which a naive "maximal interval followed by maximal
//! interval" join would miss. The interval arithmetic here is a transcription of the
//! (Lean-proved) formulas in `trql-types/src/intervals/sets.rs`. The final projection
//! drops `c`/`l`/`r` and reports `[s, e]`, the maximal span, exactly as the engine's
//! `QueryResult` does.
//!
//! # Testing
//!
//! The Rust tests here only assert on the *shape* of the generated SQL. Correctness is
//! checked separately, by differential testing against a brute-force reference that
//! enumerates concrete intervals directly and never uses the 5-tuple encoding:
//!
//! Run from the repository root (requires Python `duckdb` and Rust/Cargo):
//!
//! ```text
//! .trqs-pyenv/bin/python engine/trql-sql/tests/differential.py
//! ```
//!
//! Add `--real-db path/to/soccernet/clip/duckdb.db` for the optional real-clip cases.
//!
//! It compares denotations — the full set of `(s, e)` pairs — over a synthetic fixture and
//! a slice of a real clip. The fixture is deliberately built to reach the awkward corners
//! (detection gaps, thresholds exactly equal to a run length, right-nested `THEN` so the
//! right operand has `l < e`); each was added because a deliberately introduced bug in the
//! corresponding formula would otherwise have gone unnoticed.
//!
//! # Where this diverges from the engine
//!
//! The engine evaluates predicates against *fitted polynomial trajectories* and solves for
//! roots continuously. This compiler evaluates against the *stored per-frame samples*. On
//! the frames that exist the two agree; they differ where the polynomial fit smooths over
//! detection gaps. Islands here break at any missing frame, so an object absent for a
//! frame splits the interval.

mod pred;

use std::fmt::Write as _;

use trql_algebra::{
    expr::TrajExpr,
    plan::{VariablesWithConstraints, collect_slots},
};
use trql_types::{
    Slot,
    prop::{Schema, TypeDesc, Value},
};

// ---------------------------------------------------------------------------
// Public error type
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum SqlError {
    #[error(
        "variable '{0}' is declared in SELECT but never used in WHERE — \
         it has no appearance group to bind it to"
    )]
    UnreferencedVar(String),
    #[error("unknown property '{0}'")]
    UnknownProp(String),
    #[error("property '{prop}' has enum value {index}, which is out of range for the schema")]
    BadEnumIndex { prop: String, index: usize },
    #[error(
        "the DuckDB schema stores object properties in the single `class` column, \
         so property '{0}' cannot be compiled"
    )]
    UnsupportedProp(String),
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Compiles `expr` over `vars` into a standalone DuckDB SQL query.
///
/// `schema` is needed to turn the enum-encoded property values back into the `class`
/// strings stored in the database — the DSL resolves `class=ball` to an index at parse
/// time, and only the schema knows what that index meant.
pub fn compile(
    vars: &VariablesWithConstraints,
    expr: &TrajExpr,
    schema: &Schema,
) -> Result<String, SqlError> {
    compile_with(vars, expr, schema, Output::Spans)
}

/// What the generated query projects for each result tuple.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Output {
    /// `s, e` — the maximal span, matching what the engine reports.
    Spans,
    /// `s, e, c, l, r` — the whole [`trql_types::intervals::IntervalSet`] 5-tuple, which
    /// denotes the set `{(s', e') : s ≤ s' ≤ l, r ≤ e' ≤ e, e' − s' + 1 ≥ c}`. Useful for
    /// checking the compiler against a reference implementation, where comparing spans
    /// alone would miss a wrong length bound.
    Tuples,
}

/// [`compile`], choosing what the final `SELECT` projects.
pub fn compile_with(
    vars: &VariablesWithConstraints,
    expr: &TrajExpr,
    schema: &Schema,
    output: Output,
) -> Result<String, SqlError> {
    let mut c = Compiler {
        vars,
        schema,
        ctes: Vec::new(),
        next_group: 0,
        next_node: 0,
    };

    let root = c.compile_top(expr)?;

    // Every declared variable must end up in the result tuple. A variable the WHERE clause
    // never mentions belongs to no appearance group, so there is nothing to bind it to —
    // the engine would leave a hole in the tuple, and here there is simply no column.
    for slot in vars.slots() {
        if !root.slots.contains(&slot) {
            let name = vars.name_of(slot).unwrap_or("?").to_owned();
            return Err(SqlError::UnreferencedVar(name));
        }
    }

    Ok(c.finish(&root, output))
}

// ---------------------------------------------------------------------------
// Column naming
// ---------------------------------------------------------------------------

/// Column holding the `oid` bound to `slot`.
fn oid_col(slot: Slot) -> String {
    format!("o{slot}")
}

/// Column holding one bounding-box coordinate of `slot`.
pub(crate) fn coord_col(slot: Slot, coord: &str) -> String {
    format!("s{slot}_{coord}")
}

/// Formats an `f64` as a SQL numeric literal.
///
/// `{:?}` always emits a decimal point or an exponent, so the literal can never be
/// re-read as an integer and silently change the arithmetic to integer division.
pub(crate) fn num(v: f64) -> String {
    format!("{v:?}")
}

/// Escapes a SQL single-quoted string literal.
fn quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

// ---------------------------------------------------------------------------
// Compiler
// ---------------------------------------------------------------------------

/// A compiled sub-expression: the CTE holding its interval rows, and which slots those
/// rows carry `oid` columns for.
struct Node {
    cte: String,
    slots: Vec<Slot>,
}

struct Compiler<'a> {
    vars: &'a VariablesWithConstraints,
    schema: &'a Schema,
    ctes: Vec<(String, String)>,
    next_group: usize,
    next_node: usize,
}

impl Compiler<'_> {
    fn add_cte(&mut self, prefix: &str, body: String) -> String {
        let name = format!("{prefix}{}", self.next_node);
        self.next_node += 1;
        self.ctes.push((name.clone(), body));
        name
    }

    // -- Top level: the THEN spine -----------------------------------------------------

    /// Walks the `Then` spine, mirroring `trql_algebra::eval::chain_groups`.
    ///
    /// Every non-`Then` subtree is one *appearance group*, evaluated over the frames where
    /// exactly that group's objects co-appear. Groups are deliberately independent: in
    /// `DURATION(a…) THEN DURATION(b…)`, `a` and `b` never need to be on screen together,
    /// which is why each group gets its own frame relation instead of one global join.
    fn compile_top(&mut self, expr: &TrajExpr) -> Result<Node, SqlError> {
        match expr {
            TrajExpr::Then {
                lhs,
                rhs,
                min_gap,
                max_gap,
            } => {
                let l = self.compile_top(lhs)?;
                let r = self.compile_top(rhs)?;
                Ok(self.emit_then(&l, &r, *min_gap, *max_gap))
            }
            leaf => {
                let mut slots = collect_slots(leaf);
                slots.sort_unstable();
                slots.dedup();
                // A subtree referencing no slot (a bare `TRUE`) still has to produce a
                // tuple, so it falls back to every declared variable — as the engine does.
                if slots.is_empty() {
                    slots = self.vars.slots().collect();
                }

                let frames = self.emit_frames(&slots)?;
                self.compile_in_group(leaf, &frames, &slots)
            }
        }
    }

    // -- Within one appearance group ---------------------------------------------------

    /// Compiles a subtree that lives entirely inside one appearance group.
    ///
    /// A nested `Then` reaches here when it sits under `DURATION`/`CONTAINS`; the algebra
    /// keeps such a `Then` inside its enclosing group, so both sides share `frames`.
    fn compile_in_group(
        &mut self,
        expr: &TrajExpr,
        frames: &str,
        slots: &[Slot],
    ) -> Result<Node, SqlError> {
        match expr {
            TrajExpr::AnalyticPred(p) => Ok(self.emit_leaf(&pred::render(p), frames, slots)),

            TrajExpr::Duration { inner, min_frames } => {
                let inner = self.compile_in_group(inner, frames, slots)?;
                Ok(self.emit_duration(&inner, *min_frames))
            }

            TrajExpr::Then {
                lhs,
                rhs,
                min_gap,
                max_gap,
            } => {
                let l = self.compile_in_group(lhs, frames, slots)?;
                let r = self.compile_in_group(rhs, frames, slots)?;
                Ok(self.emit_then(&l, &r, *min_gap, *max_gap))
            }

            TrajExpr::Contained { lhs, rhs } => {
                let l = self.compile_in_group(lhs, frames, slots)?;
                let r = self.compile_in_group(rhs, frames, slots)?;
                Ok(self.emit_contains(&l, &r))
            }

            TrajExpr::Overlap { lhs, rhs } => {
                let l = self.compile_in_group(lhs, frames, slots)?;
                let r = self.compile_in_group(rhs, frames, slots)?;
                Ok(self.emit_overlap(&l, &r))
            }
        }
    }

    // -- Frame relation for a group ----------------------------------------------------

    /// Emits the per-frame relation for one appearance group: one `bbox` row per slot, all
    /// at the same `fid`, so a row exists exactly on the frames where the group's objects
    /// co-appear. This is the SQL counterpart of the engine clipping evaluation to a
    /// group's co-appearance bounds.
    fn emit_frames(&mut self, slots: &[Slot]) -> Result<String, SqlError> {
        let lead = slots[0];
        let mut select: Vec<String> = vec![format!("b{lead}.fid AS fid")];
        let mut from = format!("bbox b{lead}");
        let mut wheres: Vec<String> = Vec::new();

        for (i, &slot) in slots.iter().enumerate() {
            if i > 0 {
                write!(from, "\n    JOIN bbox b{slot} ON b{slot}.fid = b{lead}.fid").unwrap();
            }
            select.push(format!("b{slot}.oid AS {}", oid_col(slot)));
            for coord in ["x1", "y1", "x2", "y2"] {
                // Cast to DOUBLE at the projection, not at each use.
                //
                // Predicates are affine combinations with real weights and real
                // thresholds (`0.5 * x1 + 0.5 * x2 >= 2110.0`), so DOUBLE is the type the
                // arithmetic already assumes -- and it is what the engine computes in,
                // which is what makes the two evaluators comparable.
                //
                // Without the cast a source whose boxes are an *unsigned* integer type
                // fails outright rather than silently: i24_motion stores UINTEGER, and
                // `a.x2 - b.x1 < 0` raises "Overflow in subtraction of UINT32" in DuckDB
                // precisely when the predicate is true, because the difference it is
                // asking about is the negative one.
                select.push(format!(
                    "CAST(b{slot}.{coord} AS DOUBLE) AS {}",
                    coord_col(slot, coord)
                ));
            }

            for lit in self.class_literals(slot)? {
                write!(
                    from,
                    "\n    JOIN objects j{slot} ON j{slot}.oid = b{slot}.oid AND j{slot}.class = {lit}"
                )
                .unwrap();
            }
        }

        // Two slots of one group binding the same object would be a self-join, which the
        // selector rejects. Across groups it is allowed, so this only applies within a group.
        for i in 0..slots.len() {
            for j in (i + 1)..slots.len() {
                wheres.push(format!("b{}.oid <> b{}.oid", slots[i], slots[j]));
            }
        }

        let mut body = format!(
            "SELECT\n        {}\n    FROM {from}",
            select.join(",\n        ")
        );
        if !wheres.is_empty() {
            write!(body, "\n    WHERE {}", wheres.join("\n      AND ")).unwrap();
        }

        let name = format!("g{}_frames", self.next_group);
        self.next_group += 1;
        self.ctes.push((name.clone(), body));
        Ok(name)
    }

    /// The `class` literals a slot's property constraints reduce to.
    ///
    /// The DuckDB schema keeps a single `class` column, so every constraint has to target
    /// it; a query constraining anything else cannot be expressed against this schema.
    fn class_literals(&self, slot: Slot) -> Result<Vec<String>, SqlError> {
        let constraints = match self.vars.get_slot_constraints(slot) {
            Some(c) => c,
            None => return Ok(Vec::new()),
        };

        let mut out = Vec::new();
        for c in constraints.iter_constraints() {
            if c.prop != "class" {
                return Err(SqlError::UnsupportedProp(c.prop.clone()));
            }
            let ty = self
                .schema
                .get_type(&c.prop)
                .ok_or_else(|| SqlError::UnknownProp(c.prop.clone()))?;

            let text = match (&c.val, ty) {
                (Value::Enum(i), TypeDesc::Enum(variants)) => variants
                    .get(*i)
                    .ok_or_else(|| SqlError::BadEnumIndex {
                        prop: c.prop.clone(),
                        index: *i,
                    })?
                    .clone(),
                (Value::String(s), _) => s.clone(),
                (Value::UInt(v), _) => v.to_string(),
                (Value::Int(v), _) => v.to_string(),
                (Value::Bool(v), _) => v.to_string(),
                (Value::Enum(i), _) => {
                    return Err(SqlError::BadEnumIndex {
                        prop: c.prop.clone(),
                        index: *i,
                    });
                }
            };
            out.push(quote(&text));
        }
        Ok(out)
    }

    // -- Interval nodes ----------------------------------------------------------------

    /// A leaf predicate's intervals: maximal runs of consecutive frames where it holds.
    ///
    /// The `fid − ROW_NUMBER()` trick is the standard gaps-and-islands identity — within a
    /// run of consecutive frames the difference is constant, and it changes at every break.
    /// Each run is lifted with `c = 1, l = e, r = s`, matching `IntervalSet::all_between`.
    fn emit_leaf(&mut self, predicate: &str, frames: &str, slots: &[Slot]) -> Node {
        let oids: Vec<String> = slots.iter().map(|&s| oid_col(s)).collect();
        let keys = oids.join(", ");

        let body = format!(
            "SELECT {keys},\n           MIN(fid) AS s,\n           MAX(fid) AS e,\n           \
             1 AS c,\n           MAX(fid) AS l,\n           MIN(fid) AS r\n    FROM (\n        \
             SELECT {keys}, fid,\n               fid - ROW_NUMBER() OVER \
             (PARTITION BY {keys} ORDER BY fid) AS island\n        FROM {frames}\n        \
             WHERE {predicate}\n    ) t\n    GROUP BY {keys}, island"
        );

        Node {
            cte: self.add_cte("n", body),
            slots: slots.to_vec(),
        }
    }

    /// `DURATION(inner, k)` — raises the minimum length `c` to `k`.
    ///
    /// Note it does not shrink `[s, e]`: the interval still spans the same frames, it just
    /// now denotes only the sub-intervals of at least `k` frames. Rows whose span can no
    /// longer host such a sub-interval drop out.
    fn emit_duration(&mut self, inner: &Node, min_frames: u32) -> Node {
        let oids: Vec<String> = inner.slots.iter().map(|&s| oid_col(s)).collect();
        let body = format!(
            "SELECT {},\n           s, e,\n           GREATEST(c, {min_frames}) AS c,\n           \
             l, r\n    FROM {}\n    WHERE e - s + 1 >= GREATEST(c, {min_frames})",
            oids.join(", "),
            inner.cte
        );

        Node {
            cte: self.add_cte("n", body),
            slots: inner.slots.clone(),
        }
    }

    /// `lhs THEN rhs` with a gap window — `IntervalSet::then_pair_with_gap`.
    ///
    /// The two sides join on whatever slots they share, which pins a variable appearing on
    /// both sides to one object while leaving the rest free.
    fn emit_then(&mut self, lhs: &Node, rhs: &Node, min_gap: u32, max_gap: Option<u32>) -> Node {
        let mut guards = vec![
            // R₁ + g_min + 1 ≤ L₂ and c₁ + g_min ≤ L₂, as in the `checked_sub` guards.
            format!("b.l - {min_gap} >= 1"),
            format!("b.l - {min_gap} - a.c >= 0"),
            format!("a.r <= b.l - {min_gap} - 1"),
        ];
        if let Some(g) = max_gap {
            // The second interval has to be able to start soon enough after the first ends.
            guards.push(format!("b.s <= a.e + {g} + 1"));
        }

        let cols = format!(
            "a.s AS s,\n           b.e AS e,\n           a.c + b.c + {min_gap} AS c,\n           \
             LEAST(a.l, a.e - a.c + 1, b.l - {min_gap} - a.c) AS l,\n           \
             GREATEST(b.r, a.r + b.c + {min_gap}, b.s + b.c - 1) AS r"
        );

        self.emit_join(lhs, rhs, &cols, &guards)
    }

    /// `lhs CONTAINS rhs` — `IntervalSet::contains`. The result keeps the *lhs* span.
    fn emit_contains(&mut self, lhs: &Node, rhs: &Node) -> Node {
        let guards = vec!["a.s < b.l".to_owned(), "b.c <= b.e".to_owned()];
        let cols =
            "a.s AS s,\n           a.e AS e,\n           GREATEST(a.c, b.c + 2) AS c,\n           \
             LEAST(a.l, b.l - 1, b.e - b.c) AS l,\n           \
             GREATEST(a.r, b.r + 1, b.s + b.c) AS r"
                .to_owned();

        self.emit_join(lhs, rhs, &cols, &guards)
    }

    /// `lhs OVERLAP rhs` (Allen's *overlaps*) — `IntervalSet::overlap`. The result spans the
    /// union, `[lhs.s, rhs.e]`, because a proper overlap merges the two intervals.
    ///
    /// Transcribed from `SplitIntervalSet.overlapOp` in `lean-proof/Intervalset/Overlap.lean`.
    /// Lean states the guards over `ℕ`, where `a.e - 1` and `b.l - 1` would truncate; here
    /// they are the same inequalities moved to the other side, which is why `a.s < a.e` and
    /// `a.s < b.l` appear rather than `a.s + 1 <= a.e` and `a.s + 1 <= b.l`. The structural
    /// `a.r <= a.e` and `b.s <= b.l` hold of every input row by construction.
    ///
    /// `a.s < a.e` is the one term the differential test cannot pin down: over SQL's signed
    /// arithmetic it is implied by the `a.e - 1` in `l_out`, which already drives `l` below
    /// `s` for a single-frame lhs. It is kept because dropping *both* would be wrong, and
    /// because staying a term-for-term transcription of the Lean is worth more than the
    /// one comparison it costs.
    fn emit_overlap(&mut self, lhs: &Node, rhs: &Node) -> Node {
        let guards = vec![
            // S₂ ≤ E₁ — the second interval starts inside the first.
            "b.s <= a.e".to_owned(),
            // S₁ + 1 ≤ E₁ and S₁ + 1 ≤ L₂ — keep `a.e - 1` and `b.l - 1` well-defined.
            "a.s < a.e".to_owned(),
            "a.s < b.l".to_owned(),
        ];
        let cols = "a.s AS s,\n           b.e AS e,\n           \
             GREATEST(a.c + 1, b.c + 1, 3) AS c,\n           \
             LEAST(a.l, a.e - a.c + 1, a.e - 1, b.l - 1) AS l,\n           \
             GREATEST(b.r, a.r + 1, b.s + b.c - 1, b.s + 1) AS r"
            .to_owned();

        self.emit_join(lhs, rhs, &cols, &guards)
    }

    /// Shared shape of the binary interval operators: join the sides on their common
    /// slots, apply the operator's guards, compute the new 5-tuple, then keep only feasible
    /// rows (`IntervalSet::new` returning `Some`).
    fn emit_join(&mut self, lhs: &Node, rhs: &Node, cols: &str, guards: &[String]) -> Node {
        let mut slots = lhs.slots.clone();
        for &s in &rhs.slots {
            if !slots.contains(&s) {
                slots.push(s);
            }
        }
        slots.sort_unstable();

        // A slot on both sides is the same variable, so it must be the same object.
        let shared: Vec<String> = lhs
            .slots
            .iter()
            .filter(|s| rhs.slots.contains(s))
            .map(|&s| {
                let c = oid_col(s);
                format!("a.{c} = b.{c}")
            })
            .collect();

        let projected: Vec<String> = slots
            .iter()
            .map(|&s| {
                let c = oid_col(s);
                let side = if lhs.slots.contains(&s) { "a" } else { "b" };
                format!("{side}.{c} AS {c}")
            })
            .collect();

        let join = if shared.is_empty() {
            format!("{} a\n        CROSS JOIN {} b", lhs.cte, rhs.cte)
        } else {
            format!(
                "{} a\n        JOIN {} b ON {}",
                lhs.cte,
                rhs.cte,
                shared.join(" AND ")
            )
        };

        let mut inner = format!(
            "SELECT {},\n           {cols}\n        FROM {join}",
            projected.join(",\n           ")
        );
        if !guards.is_empty() {
            write!(inner, "\n        WHERE {}", guards.join("\n          AND ")).unwrap();
        }

        let oids: Vec<String> = slots.iter().map(|&s| oid_col(s)).collect();
        let body = format!(
            "SELECT DISTINCT {}, s, e, c, l, r\n    FROM (\n        {inner}\n    ) t\n    \
             WHERE c >= 1 AND s <= e AND s <= l AND r <= e AND e - s + 1 >= c",
            oids.join(", ")
        );

        Node {
            cte: self.add_cte("n", body),
            slots,
        }
    }

    // -- Assembly ----------------------------------------------------------------------

    /// Stitches the CTEs together and projects the result tuple, naming each `oid` column
    /// after the variable it binds.
    fn finish(&self, root: &Node, output: Output) -> String {
        let mut sql = String::from("WITH ");
        for (i, (name, body)) in self.ctes.iter().enumerate() {
            if i > 0 {
                sql.push_str(",\n");
            }
            write!(sql, "{name} AS (\n    {body}\n)").unwrap();
        }

        let projected: Vec<String> = root
            .slots
            .iter()
            .map(|&s| {
                let col = oid_col(s);
                match self.vars.name_of(s) {
                    Some(n) => format!("{col} AS \"{}\"", n.replace('"', "\"\"")),
                    None => col,
                }
            })
            .collect();

        let order: Vec<String> = root.slots.iter().map(|&s| oid_col(s)).collect();

        let interval_cols = match output {
            Output::Spans => "s, e",
            Output::Tuples => "s, e, c, l, r",
        };

        write!(
            sql,
            "\nSELECT {}, {interval_cols}\nFROM {}\nORDER BY {}, s, e",
            projected.join(", "),
            root.cte,
            order.join(", ")
        )
        .unwrap();

        sql
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sql(query: &str) -> String {
        let schema = Schema::new(vec![("class".into(), TypeDesc::String)]);
        let (vars, expr) = trql_dsl::parse_query(query, &schema).expect("parses");
        compile(&vars, &expr, &schema).expect("compiles")
    }

    fn err(query: &str) -> SqlError {
        let schema = Schema::new(vec![("class".into(), TypeDesc::String)]);
        let (vars, expr) = trql_dsl::parse_query(query, &schema).expect("parses");
        compile(&vars, &expr, &schema).expect_err("should not compile")
    }

    /// Occurrences of `needle`, for asserting on structure rather than exact text.
    fn count(haystack: &str, needle: &str) -> usize {
        haystack.matches(needle).count()
    }

    #[test]
    fn leaf_builds_islands_of_consecutive_frames() {
        let out = sql("SELECT (a:) WHERE a.x1 < 100");
        assert!(out.contains("ROW_NUMBER() OVER (PARTITION BY o0 ORDER BY fid)"));
        assert!(out.contains("GROUP BY o0, island"));
        // A leaf lifts to c = 1, l = e, r = s — `IntervalSet::all_between`.
        assert!(out.contains("1 AS c"));
        assert!(out.contains("MAX(fid) AS l"));
        assert!(out.contains("MIN(fid) AS r"));
    }

    /// A convex polygon must never reach the SQL renderer as a polygon: it is expanded into
    /// half-plane comparisons upstream, so the generated SQL is ordinary arithmetic.
    #[test]
    fn convex_polygon_renders_as_plain_comparisons() {
        let out = sql("SELECT (a:) WHERE a.center INSIDE [[0,0],[10,0],[10,10],[0,10]]");
        assert!(!out.contains("CASE WHEN"));
        assert!(!out.contains("% 2"));
        // One comparison per edge, all over the same slot's coordinate columns.
        assert_eq!(count(&out, ">= 0.0"), 4);
    }

    /// A concave polygon has no conjunctive form, so the renderer states ray-crossing
    /// parity directly: one `CASE` per non-horizontal edge, summed and taken mod 2.
    #[test]
    fn concave_polygon_renders_ray_crossing_parity() {
        let out =
            sql("SELECT (a:) WHERE a.center INSIDE [[0,0],[10,0],[10,5],[5,5],[5,10],[0,10]]");
        // Six edges, three of them horizontal (y = 0, the y = 5 step, and y = 10).
        assert_eq!(count(&out, "CASE WHEN"), 3);
        assert!(out.contains(") % 2) = 1)"));
        // The horizontal edges are dropped rather than dividing by zero.
        assert!(!out.contains("inf"));
        assert!(!out.contains("NaN"));
    }

    #[test]
    fn class_constraint_joins_objects() {
        let out = sql("SELECT (a: class=ball) WHERE a.x1 < 100");
        assert!(out.contains("JOIN objects j0 ON j0.oid = b0.oid AND j0.class = 'ball'"));
    }

    #[test]
    fn class_literal_is_escaped() {
        let out = sql("SELECT (a: class=\"o'brien\") WHERE a.x1 < 100");
        assert!(out.contains("j0.class = 'o''brien'"));
    }

    #[test]
    fn slots_in_one_group_must_be_distinct_objects() {
        // Two slots of the same group binding one object would be a self-join, which the
        // selector rejects.
        let out = sql("SELECT (a:, b:) WHERE a.x1 < b.x1");
        assert!(out.contains("b0.oid <> b1.oid"));
        // Co-presence: both slots are read at the same frame.
        assert!(out.contains("JOIN bbox b1 ON b1.fid = b0.fid"));
    }

    #[test]
    fn duration_raises_c_without_shrinking_the_span() {
        let out = sql("SELECT (a:) WHERE DURATION(a.x1 < 100, 30)");
        assert!(out.contains("GREATEST(c, 30) AS c"));
        assert!(out.contains("WHERE e - s + 1 >= GREATEST(c, 30)"));
        // The span itself is carried through untouched.
        assert!(out.contains("s, e,"));
    }

    #[test]
    fn then_transcribes_the_gap_formula() {
        let out = sql("SELECT (a:, b:) WHERE a.x1 < b.x1 THEN[5..20] a.x1 > b.x1");
        assert!(out.contains("a.c + b.c + 5 AS c"));
        assert!(out.contains("LEAST(a.l, a.e - a.c + 1, b.l - 5 - a.c) AS l"));
        assert!(out.contains("GREATEST(b.r, a.r + b.c + 5, b.s + b.c - 1) AS r"));
        // max_gap bounds how late the second interval may start.
        assert!(out.contains("b.s <= a.e + 20 + 1"));
    }

    #[test]
    fn then_without_max_gap_omits_the_upper_bound() {
        let out = sql("SELECT (a:, b:) WHERE a.x1 < b.x1 THEN a.x1 > b.x1");
        assert!(!out.contains("b.s <= a.e"));
    }

    #[test]
    fn contains_keeps_the_lhs_span() {
        let out = sql("SELECT (a:) WHERE CONTAINS(a.x1 < 100, a.x1 < 50)");
        assert!(out.contains("a.s AS s"));
        assert!(out.contains("a.e AS e"));
        assert!(out.contains("GREATEST(a.c, b.c + 2) AS c"));
        assert!(out.contains("LEAST(a.l, b.l - 1, b.e - b.c) AS l"));
    }

    #[test]
    fn overlap_spans_the_union() {
        let out = sql("SELECT (a:) WHERE OVERLAP(a.x1 < 100, a.x1 > 50)");
        // Unlike CONTAINS, the stop comes from the *rhs* — a proper overlap merges the two.
        assert!(out.contains("a.s AS s"));
        assert!(out.contains("b.e AS e"));
        assert!(out.contains("GREATEST(a.c + 1, b.c + 1, 3) AS c"));
        assert!(out.contains("LEAST(a.l, a.e - a.c + 1, a.e - 1, b.l - 1) AS l"));
        assert!(out.contains("GREATEST(b.r, a.r + 1, b.s + b.c - 1, b.s + 1) AS r"));
    }

    #[test]
    fn overlap_stays_in_one_appearance_group() {
        // Two intervals can only properly overlap if they share frames, so both sides are
        // evaluated over the same co-appearance window — one frame relation, as CONTAINS.
        let out = sql("SELECT (a:, b:) WHERE OVERLAP(a.x1 < 100, b.x1 < 100)");
        assert_eq!(count(&out, "_frames AS ("), 1);
    }

    #[test]
    fn every_interval_node_filters_for_feasibility() {
        let out = sql("SELECT (a:, b:) WHERE a.x1 < b.x1 THEN a.x1 > b.x1");
        assert!(out.contains("WHERE c >= 1 AND s <= e AND s <= l AND r <= e AND e - s + 1 >= c"));
    }

    // -- Appearance groups -------------------------------------------------------------

    #[test]
    fn then_over_disjoint_slots_makes_two_groups() {
        // `a` and `b` never need to be on screen together, so each side gets its own frame
        // relation and the sides are combined by a cross join.
        let out = sql("SELECT (a:, b:) WHERE a.x1 < 100 THEN b.x1 < 100");
        assert_eq!(count(&out, "_frames AS ("), 2);
        assert!(out.contains("CROSS JOIN"));
    }

    #[test]
    fn then_sharing_a_slot_pins_it_to_one_object() {
        let out = sql("SELECT (a:, b:) WHERE a.x1 < b.x1 THEN a.x1 > 100");
        // Slot 0 appears on both sides, so it must be the same object.
        assert!(out.contains("ON a.o0 = b.o0"));
        assert!(!out.contains("CROSS JOIN"));
    }

    #[test]
    fn one_group_per_non_then_subtree() {
        let out = sql("SELECT (a:) WHERE a.x1 < 1 THEN a.x1 < 2 THEN a.x1 < 3");
        assert_eq!(count(&out, "_frames AS ("), 3);
    }

    #[test]
    fn nested_then_under_duration_stays_in_one_group() {
        // A `Then` beneath `DURATION` does not split the appearance chain, so both sides
        // read the same frame relation.
        let out = sql("SELECT (a:, b:) WHERE DURATION(a.x1 < b.x1 THEN a.x1 > b.x1, 10)");
        assert_eq!(count(&out, "_frames AS ("), 1);
    }

    // -- Projection and errors ---------------------------------------------------------

    #[test]
    fn result_columns_are_named_after_query_variables() {
        let out = sql("SELECT (ball: class=ball, plr: class=player_left) WHERE ball.x1 < plr.x1");
        assert!(out.contains("SELECT o0 AS \"ball\", o1 AS \"plr\", s, e"));
    }

    #[test]
    fn declared_but_unused_variable_is_rejected() {
        // It belongs to no appearance group, so there is no column to bind it to.
        let e = err("SELECT (a:, unused:) WHERE a.x1 < 100");
        assert!(matches!(e, SqlError::UnreferencedVar(n) if n == "unused"));
    }

    #[test]
    fn non_class_property_is_rejected() {
        let schema = Schema::new(vec![
            ("class".into(), TypeDesc::String),
            ("color".into(), TypeDesc::String),
        ]);
        let (vars, expr) =
            trql_dsl::parse_query("SELECT (a: color=red) WHERE a.x1 < 100", &schema).unwrap();
        let e = compile(&vars, &expr, &schema).expect_err("no such column in the DuckDB schema");
        assert!(matches!(e, SqlError::UnsupportedProp(p) if p == "color"));
    }

    #[test]
    fn enum_values_resolve_back_to_class_strings() {
        // The DSL resolves `class=ball` to an index at parse time; only the schema knows
        // which string that index meant.
        let schema = Schema::new(vec![(
            "class".into(),
            TypeDesc::Enum(vec!["ball".into(), "player".into()]),
        )]);
        let (vars, expr) =
            trql_dsl::parse_query("SELECT (a: class=player) WHERE a.x1 < 100", &schema).unwrap();
        let out = compile(&vars, &expr, &schema).unwrap();
        assert!(out.contains("j0.class = 'player'"));
    }

    // -- Predicate rendering -----------------------------------------------------------

    #[test]
    fn boolean_predicate_stays_pointwise() {
        // No interval machinery for AND/OR/NOT — one boolean, one island pass.
        let out = sql("SELECT (a:) WHERE (a.x1 > 10 OR a.x1 < 5) AND NOT a.y1 > 20");
        assert_eq!(count(&out, "ROW_NUMBER()"), 1);
        assert!(out.contains(" OR "));
        assert!(out.contains("NOT "));
    }

    #[test]
    fn coordinates_are_cast_to_double_at_the_projection() {
        // A source may store boxes as an unsigned integer type -- i24_motion uses
        // UINTEGER so the engine's `u32` read is lossless -- and every predicate here is
        // a *signed* affine combination. Projecting the raw column makes
        // `a.x2 - b.x1 < 0` raise "Overflow in subtraction of UINT32" in DuckDB exactly
        // when the predicate holds, so the query fails rather than answering.
        let out = sql("SELECT (a:, b:) WHERE a.x2 < b.x1");
        assert!(out.contains("CAST(b0.x2 AS DOUBLE) AS s0_x2"), "{out}");
        assert!(out.contains("CAST(b1.x1 AS DOUBLE) AS s1_x1"), "{out}");
        // The arithmetic itself is unchanged; only the column feeding it is retyped.
        assert!(out.contains("(s0_x2) - (s1_x1) < 0.0"), "{out}");
    }

    #[test]
    fn binary_comparison_renders_as_a_difference() {
        // `lhs(A) − rhs(B) op threshold`, matching what the analytical evaluator solves.
        let out = sql("SELECT (a:, b:) WHERE a.x1 < b.x1");
        assert!(out.contains("(s0_x1) - (s1_x1) < 0.0"));
    }

    #[test]
    fn derived_coords_expand_to_affine_combinations() {
        let out = sql("SELECT (a:) WHERE a.cx > 100");
        assert!(out.contains("0.5 * s0_x1 + 0.5 * s0_x2"));
    }

    #[test]
    fn inside_expands_to_four_bounds() {
        let out = sql("SELECT (a:) WHERE a INSIDE [0, 10, 100, 200]");
        assert!(out.contains("(s0_x1) >= 0.0"));
        assert!(out.contains("(s0_y1) >= 10.0"));
        assert!(out.contains("(s0_x2) <= 100.0"));
        assert!(out.contains("(s0_y2) <= 200.0"));
    }

    /// Named regions are resolved by the parser, so by the time SQL is generated there is
    /// nothing left of them — the two spellings must compile to identical text.
    #[test]
    fn named_regions_leave_no_trace_in_the_sql() {
        let named = sql("REGIONS (kerb = [0, 10, 100, 200]) SELECT (a:) WHERE a INSIDE kerb");
        let inlined = sql("SELECT (a:) WHERE a INSIDE [0, 10, 100, 200]");
        assert_eq!(named, inlined);
    }

    /// The centre form is still four affine comparisons, so it needs no backend support —
    /// `cx` renders as the half-sum the analytical evaluator uses.
    #[test]
    fn center_inside_renders_as_affine_midpoints() {
        let out = sql("SELECT (a:) WHERE a.center INSIDE [0, 10, 100, 200]");
        assert!(out.contains("0.5 * s0_x1 + 0.5 * s0_x2"));
        assert!(out.contains("0.5 * s0_y1 + 0.5 * s0_y2"));
        assert!(!out.contains("(s0_x1) >= 0.0"));
    }

    #[test]
    fn numeric_literals_never_render_as_bare_integers() {
        // A bare `50` beside integer columns would silently switch DuckDB to integer
        // arithmetic; the trailing `.0` keeps the comparison in floating point.
        let out = sql("SELECT (a:) WHERE a.x1 < 50");
        assert!(out.contains("50.0"));
    }
}
