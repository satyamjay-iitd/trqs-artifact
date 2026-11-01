use nom::{
    IResult,
    bytes::complete::{tag, tag_no_case, take_while1},
    character::complete::{char, u32 as nom_u32},
    number::complete::double,
};
use std::cell::RefCell;
use std::sync::Arc;
use trql_algebra::{
    expr::TrajExpr,
    plan::{PropConstraint, PropConstraints, PropConstraintsBuilder, VariablesWithConstraints},
};

use trql_bbox_compiler::{
    AnalyticalPred, AnalyticalPredKind, BBoxExpr, BinaryBBoxExpr, CmpOp, Polygon, PolygonError,
};
use trql_types::{Slot, prop::Schema};

// ---------------------------------------------------------------------------
// Public error type
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum DslError {
    #[error("parse error: expected {expected} at '{at}'")]
    Parse { expected: &'static str, at: String },
    #[error("unknown variable '{0}' — declare it in SELECT")]
    UnknownVar(String),
    #[error("unknown property '{0}'")]
    UnknownProp(String),
    #[error("unknown enum value '{value}' for property '{prop}'")]
    UnknownEnumValue { prop: String, value: String },
    #[error("unsupported comparison: {0}")]
    UnsupportedComparison(&'static str),
    #[error("unknown region '{0}' — declare it in REGIONS")]
    UnknownRegion(String),
    #[error("region '{0}' declared twice")]
    DuplicateRegion(String),
    #[error("empty region [{x1}, {y1}, {x2}, {y2}] — needs x1 ≤ x2 and y1 ≤ y2")]
    EmptyRegion { x1: f64, y1: f64, x2: f64, y2: f64 },
    #[error("bad polygon: {0}")]
    BadPolygon(#[from] PolygonError),
}

// ---------------------------------------------------------------------------
// Regions
// ---------------------------------------------------------------------------

/// An axis-aligned region, `[x1, y1, x2, y2]`, normalised so `x1 ≤ x2` and `y1 ≤ y2`.
type Rect = [f64; 4];

/// What a region name or an inline `INSIDE` literal can stand for.
///
/// The rectangle is not a degenerate polygon even though [`Polygon::from_rect`] would
/// produce one: keeping it separate lets `INSIDE` emit the four coordinate comparisons it
/// always has, which every existing test, plan, and generated query already expects.
#[derive(Clone, Debug, PartialEq)]
enum Shape {
    Rect(Rect),
    Poly(Arc<Polygon>),
}

/// A region the query tests an object against, in the coordinates the query was written in.
///
/// Reported so a caller can *show* the region — draw it over the video the query ran
/// against. The predicate tree cannot answer that: `INSIDE` is expanded at parse time into
/// comparisons that no longer name a shape.
///
/// Every shape is reported as a closed ring, rectangles included, so a consumer needs one
/// drawing path rather than two.
#[derive(Clone, Debug, PartialEq)]
pub struct QueryRegion {
    /// The name it was declared under in `REGIONS`, or `None` for an inline literal.
    pub name: Option<String>,
    /// The variable whose box is tested against it.
    pub var: String,
    /// Whether the whole box had to be inside, or only its centre.
    pub whole_box: bool,
    /// The ring, implicitly closed. Polygons come back CCW-normalised, as
    /// [`Polygon::new`] leaves them; a rectangle is given as its four corners.
    pub vertices: Vec<[f64; 2]>,
}

/// The regions declared by a query's optional `REGIONS` clause, in declaration order.
///
/// A `Vec` rather than a map: declaration order makes duplicate detection report the
/// *first* collision, and a query never declares enough regions for lookup cost to matter.
type Regions = Vec<(String, Shape)>;

// ---------------------------------------------------------------------------
// Internal parse context
// ---------------------------------------------------------------------------

/// The declared variables and regions of the query being parsed.
///
/// Slot indices are decided once, when the SELECT clause is turned into
/// [`VariablesWithConstraints`] — every variable is known at that point. The WHERE
/// clause then resolves names against that table, so nothing downstream ever
/// invents an index.
///
/// Regions are resolved here too, and only here: by the time the WHERE clause has been
/// parsed every region name has been replaced by the rectangle it stands for, so nothing
/// downstream ever learns that regions exist.
struct Ctx<'a> {
    slots: &'a VariablesWithConstraints,
    regions: &'a Regions,
    /// Every region an `INSIDE` in this query tested against, in the order met.
    ///
    /// Collected during the parse rather than recovered from the result, because by then
    /// the shapes are gone: a rectangle and a convex polygon have both become plain
    /// comparisons, and only a concave ring is still recognisable. Interior mutability
    /// keeps this off the signature of every recursive-descent function that merely
    /// passes the context through.
    used: RefCell<Vec<QueryRegion>>,
}

impl Ctx<'_> {
    /// Resolves a variable name written in the query to its slot handle, which
    /// carries that name through planning and into diagnostics.
    fn slot(&self, name: &str) -> Result<Slot, DslError> {
        self.slots
            .slot_named(name)
            .ok_or_else(|| DslError::UnknownVar(name.to_owned()))
    }

    /// Notes that `var` was tested against `shape`, for [`parse_query_regions`].
    ///
    /// Repeats are kept: `a INSIDE r THEN b INSIDE r` genuinely tests two different
    /// objects, and a caller listing what a query looks at wants both.
    fn record(&self, name: Option<&str>, var: &str, shape: &Shape, form: InsideForm) {
        self.used.borrow_mut().push(QueryRegion {
            name: name.map(str::to_owned),
            var: var.to_owned(),
            whole_box: form == InsideForm::Whole,
            vertices: match shape {
                Shape::Rect([x1, y1, x2, y2]) => {
                    vec![[*x1, *y1], [*x2, *y1], [*x2, *y2], [*x1, *y2]]
                }
                Shape::Poly(poly) => poly.verts().to_vec(),
            },
        });
    }

    /// Resolves a region name written in the query to the shape it was declared with.
    fn region(&self, name: &str) -> Result<Shape, DslError> {
        self.regions
            .iter()
            .find(|(declared, _)| declared == name)
            .map(|(_, shape)| shape.clone())
            .ok_or_else(|| DslError::UnknownRegion(name.to_owned()))
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Parse a full `[REGIONS …] SELECT … WHERE …` query string against the given schema.
///
/// Enum property values are resolved to numeric indices via the schema.
pub fn parse_query(
    input: &str,
    schema: &Schema,
) -> Result<(VariablesWithConstraints, TrajExpr), DslError> {
    let (slots, expr, _) = parse_query_regions(input, schema)?;
    Ok((slots, expr))
}

/// [`parse_query`], additionally reporting the regions the query tests against.
///
/// The extra return is what a caller needs to *draw* the query: the shapes are gone from
/// the expression by the time it is built, since `INSIDE` expands into comparisons. They
/// come back in the coordinates the query was written in, so a caller that translates
/// coordinates must translate these the same way.
pub fn parse_query_regions(
    input: &str,
    schema: &Schema,
) -> Result<(VariablesWithConstraints, TrajExpr, Vec<QueryRegion>), DslError> {
    let input = input.trim();

    // REGIONS — optional, and pure sugar: the returned expression is identical to the
    // one an equivalent query with inlined rectangles would produce.
    let (input, regions) = parse_regions(input)?;

    // SELECT
    let input = ws_str(input);
    let (input, _) =
        tag_no_case::<&str, &str, nom::error::Error<&str>>("SELECT")(input).map_err(|_| {
            DslError::Parse {
                expected: "SELECT",
                at: snip(input),
            }
        })?;

    // selector  →  variable names + bindings
    let (input, (vars_order, bindings)) = parse_selector(input, schema)?;

    // Every variable is known here, so this is where slot indices are decided:
    // declaration order is tuple order, and `VariablesWithConstraints` is the sole
    // authority on it from now on.
    let slots = VariablesWithConstraints::new(vars_order.iter().map(String::as_str).zip(bindings));
    let ctx = Ctx {
        slots: &slots,
        regions: &regions,
        used: RefCell::new(Vec::new()),
    };

    // WHERE
    let input = ws_str(input);
    let (input, _) =
        tag_no_case::<&str, &str, nom::error::Error<&str>>("WHERE")(input).map_err(|_| {
            DslError::Parse {
                expected: "WHERE",
                at: snip(input),
            }
        })?;

    // expr
    let (rest, expr) = parse_expr(input, &ctx)?;

    if !rest.trim().is_empty() {
        return Err(DslError::Parse {
            expected: "end of input",
            at: snip(rest),
        });
    }

    let used = ctx.used.into_inner();
    Ok((slots, expr, used))
}

// ---------------------------------------------------------------------------
// Regions:  "REGIONS" "(" ident "=" rect (, ident "=" rect)* ")"
// rect   :  "[" num "," num "," num "," num "]"
// ---------------------------------------------------------------------------

/// Parses the optional leading `REGIONS` clause.
///
/// Absent — the common case, and every query written before regions existed — yields an
/// empty table and leaves the input untouched for `SELECT` to consume.
fn parse_regions(input: &str) -> Result<(&str, Regions), DslError> {
    let trimmed = ws_str(input);
    let Ok((after_kw, _)) = tag_no_case::<&str, &str, nom::error::Error<&str>>("REGIONS")(trimmed)
    else {
        return Ok((input, Regions::new()));
    };
    if after_kw.starts_with(|c: char| c.is_alphanumeric() || c == '_') {
        return Ok((input, Regions::new()));
    }

    let after_kw = ws_str(after_kw);
    let (after_open, _) =
        char('(')(after_kw).map_err(|_: nom::Err<nom::error::Error<&str>>| DslError::Parse {
            expected: "( after REGIONS",
            at: snip(after_kw),
        })?;

    let mut regions = Regions::new();
    let mut remaining = after_open;

    loop {
        remaining = ws_str(remaining);
        let (after_name, name) =
            ident::<nom::error::Error<&str>>(remaining).map_err(|_| DslError::Parse {
                expected: "region name",
                at: snip(remaining),
            })?;
        let after_name = ws_str(after_name);
        let (after_eq, _) =
            char('=')(after_name).map_err(|_: nom::Err<nom::error::Error<&str>>| {
                DslError::Parse {
                    expected: "= after region name",
                    at: snip(after_name),
                }
            })?;

        let (after_shape, shape) = parse_shape(after_eq)?;
        if regions.iter().any(|(declared, _)| declared == name) {
            return Err(DslError::DuplicateRegion(name.to_owned()));
        }
        regions.push((name.to_owned(), shape));

        remaining = ws_str(after_shape);
        if let Ok((rest, _)) = char::<&str, nom::error::Error<&str>>(')')(remaining) {
            return Ok((rest, regions));
        }
        let (rest, _) =
            char::<&str, nom::error::Error<&str>>(',')(remaining).map_err(|_| DslError::Parse {
                expected: ", or ) in REGIONS",
                at: snip(remaining),
            })?;
        remaining = rest;
    }
}

/// Parses a region literal: either a rectangle or a polygon.
///
/// The two are told apart by their second character — a polygon's first element is itself
/// bracketed — so no keyword is needed and no existing query changes meaning:
///
/// ```text
/// [0, 0, 100, 200]                          rectangle
/// [[0, 0], [100, 0], [100, 200], [0, 200]]  polygon
/// ```
///
/// Shared by `REGIONS` declarations and inline `INSIDE` literals so the two can never drift
/// apart.
fn parse_shape(input: &str) -> Result<(&str, Shape), DslError> {
    let trimmed = ws_str(input);
    // `[` then `[`: the first element is a point, so this is a vertex list.
    if ws_str(trimmed.strip_prefix('[').unwrap_or("")).starts_with('[') {
        let (rest, poly) = parse_polygon(trimmed)?;
        return Ok((rest, Shape::Poly(Arc::new(poly))));
    }
    let (rest, rect) = parse_rect(trimmed)?;
    Ok((rest, Shape::Rect(rect)))
}

/// Parses `[ [num, num] , [num, num] , … ]` — a vertex ring, at least three points.
///
/// The ring is implicitly closed and winding does not matter; [`Polygon::new`] normalises
/// both. A ring that is degenerate (collinear, or too few distinct vertices) is rejected
/// here rather than silently matching nothing, for the same reason an inverted rectangle is.
fn parse_polygon(input: &str) -> Result<(&str, Polygon), DslError> {
    let input = ws_str(input);
    let (mut remaining, _) =
        char('[')(input).map_err(|_: nom::Err<nom::error::Error<&str>>| DslError::Parse {
            expected: "[ to open a polygon",
            at: snip(input),
        })?;

    let mut verts: Vec<[f64; 2]> = Vec::new();
    loop {
        remaining = ws_str(remaining);
        let (after_open, _) =
            char('[')(remaining).map_err(|_: nom::Err<nom::error::Error<&str>>| {
                DslError::Parse {
                    expected: "[ to open a polygon vertex",
                    at: snip(remaining),
                }
            })?;
        let (after_x, x) = parse_num(after_open, "x in polygon vertex")?;
        let (after_comma, _) = rect_comma(after_x)?;
        let (after_y, y) = parse_num(after_comma, "y in polygon vertex")?;
        let after_y = ws_str(after_y);
        let (after_close, _) =
            char(']')(after_y).map_err(|_: nom::Err<nom::error::Error<&str>>| DslError::Parse {
                expected: "] to close a polygon vertex",
                at: snip(after_y),
            })?;
        verts.push([x, y]);

        remaining = ws_str(after_close);
        if let Ok((rest, _)) = char::<&str, nom::error::Error<&str>>(']')(remaining) {
            return Ok((rest, Polygon::new(verts)?));
        }
        let (rest, _) =
            char::<&str, nom::error::Error<&str>>(',')(remaining).map_err(|_| DslError::Parse {
                expected: ", or ] in polygon",
                at: snip(remaining),
            })?;
        remaining = rest;
    }
}

/// Parses `[ num , num , num , num ]` — the literal region form.
///
/// Shared by `REGIONS` declarations and inline `INSIDE` rectangles so the two can never
/// drift apart. Coordinates are `f64`, so fractional and negative bounds are accepted.
fn parse_rect(input: &str) -> Result<(&str, Rect), DslError> {
    let input = ws_str(input);
    let (input, _) =
        char('[')(input).map_err(|_: nom::Err<nom::error::Error<&str>>| DslError::Parse {
            expected: "[ to open a region",
            at: snip(input),
        })?;

    let (input, x1) = parse_num(input, "x1 in region")?;
    let (input, _) = rect_comma(input)?;
    let (input, y1) = parse_num(input, "y1 in region")?;
    let (input, _) = rect_comma(input)?;
    let (input, x2) = parse_num(input, "x2 in region")?;
    let (input, _) = rect_comma(input)?;
    let (input, y2) = parse_num(input, "y2 in region")?;

    let input = ws_str(input);
    let (input, _) =
        char(']')(input).map_err(|_: nom::Err<nom::error::Error<&str>>| DslError::Parse {
            expected: "] to close a region",
            at: snip(input),
        })?;

    // An inverted rectangle parses fine and then matches nothing, which is a miserable
    // thing to debug in a literal that is just four numbers in a row. Reject it here.
    if x1 > x2 || y1 > y2 {
        return Err(DslError::EmptyRegion { x1, y1, x2, y2 });
    }

    Ok((input, [x1, y1, x2, y2]))
}

fn rect_comma(input: &str) -> Result<(&str, ()), DslError> {
    let input = ws_str(input);
    let (rest, _) =
        char(',')(input).map_err(|_: nom::Err<nom::error::Error<&str>>| DslError::Parse {
            expected: ", in region",
            at: snip(input),
        })?;
    Ok((rest, ()))
}

fn parse_num<'a>(input: &'a str, expected: &'static str) -> Result<(&'a str, f64), DslError> {
    let input = ws_str(input);
    double::<&str, nom::error::Error<&str>>(input).map_err(|_| DslError::Parse {
        expected,
        at: snip(input),
    })
}

// ---------------------------------------------------------------------------
// Selector:  "(" binding (, binding)* ")"
// binding  :  ident ":" (prop "=" value)*
// ---------------------------------------------------------------------------

fn parse_selector<'a>(
    input: &'a str,
    schema: &Schema,
) -> Result<(&'a str, (Vec<String>, Vec<PropConstraints>)), DslError> {
    let input = ws_str(input);

    // opening paren
    let (input, _) =
        char('(')(input).map_err(|_: nom::Err<nom::error::Error<&str>>| DslError::Parse {
            expected: "( after SELECT",
            at: snip(input),
        })?;

    let mut names = Vec::new();
    let mut bindings: Vec<PropConstraints> = Vec::new();
    let mut remaining = ws_str(input);

    loop {
        // binding: ident ":"
        let (after_colon, var_name) = parse_ident_colon(remaining)?;
        names.push(var_name);

        // property constraints: zero or more "prop=value" separated by whitespace
        let (after_constraints, constraints) = parse_constraints(after_colon, schema)?;
        bindings.push(constraints);
        remaining = ws_str(after_constraints);

        // ")" or ","
        if let Ok((rest, _)) = char::<&str, nom::error::Error<&str>>(')')(remaining) {
            return Ok((rest, (names, bindings)));
        }
        let (rest, _) =
            char::<&str, nom::error::Error<&str>>(',')(remaining).map_err(|_| DslError::Parse {
                expected: ", or ) in selector",
                at: snip(remaining),
            })?;
        remaining = ws_str(rest);
    }
}

/// Parses `ident ":"` and returns `(rest, var_name)`.
fn parse_ident_colon<'a>(input: &'a str) -> Result<(&'a str, String), DslError> {
    let (rest, name) =
        ident(input).map_err(|_: nom::Err<nom::error::Error<&str>>| DslError::Parse {
            expected: "variable name",
            at: snip(input),
        })?;
    let rest = ws_str(rest);
    let (rest, _) =
        char(':')(rest).map_err(|_: nom::Err<nom::error::Error<&str>>| DslError::Parse {
            expected: ": after variable name",
            at: snip(rest),
        })?;
    Ok((rest, name.to_owned()))
}

/// Parses zero or more `prop=value` constraints (whitespace-separated).
fn parse_constraints<'a>(
    input: &'a str,
    schema: &Schema,
) -> Result<(&'a str, PropConstraints), DslError> {
    let mut constraints = PropConstraintsBuilder::new();
    let mut remaining = ws_str(input);

    loop {
        // peek: is the next token an ident followed by '='?
        let Ok((after_name, prop_name)) = ident::<nom::error::Error<&str>>(remaining) else {
            break;
        };
        let after_name = ws_str(after_name);
        let Ok((after_eq, _)) = char::<&str, nom::error::Error<&str>>('=')(after_name) else {
            break;
        };
        let after_eq = ws_str(after_eq);

        // value token (ident or integer or quoted string)
        let (after_val, raw_val) = parse_value_token(after_eq).map_err(|_| DslError::Parse {
            expected: "property value",
            at: snip(after_eq),
        })?;

        let type_desc = schema
            .get_type(prop_name)
            .ok_or_else(|| DslError::UnknownProp(prop_name.to_owned()))?;

        let val = type_desc
            .parse_value(raw_val)
            .map_err(|_| DslError::UnknownEnumValue {
                prop: prop_name.to_owned(),
                value: raw_val.to_owned(),
            })?;

        constraints.add(PropConstraint {
            prop: prop_name.to_owned(),
            val,
        });
        remaining = ws_str(after_val);
    }

    Ok((remaining, constraints.build()))
}

// ---------------------------------------------------------------------------
// Expression grammar (recursive descent, explicit ctx threading)
// ---------------------------------------------------------------------------

/// expr := primary (THEN gap? primary)*
fn parse_expr<'a>(input: &'a str, ctx: &Ctx) -> Result<(&'a str, TrajExpr), DslError> {
    let (mut remaining, mut lhs) = parse_primary(input, ctx)?;

    loop {
        let trimmed = ws_str(remaining);
        let Ok((after_then, _)) =
            tag_no_case::<&str, &str, nom::error::Error<&str>>("THEN")(trimmed)
        else {
            break;
        };
        if after_then.starts_with(|c: char| c.is_alphanumeric() || c == '_') {
            break;
        }

        // optional gap
        let (after_gap, (min_gap, max_gap)) = parse_gap(after_then)?;
        let (after_rhs, rhs) = parse_primary(after_gap, ctx)?;
        lhs = TrajExpr::Then {
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            min_gap,
            max_gap,
        };
        remaining = after_rhs;
    }

    Ok((remaining, lhs))
}

/// primary := DURATION(...) | CONTAINS(expr, expr) | OVERLAP(expr, expr)
///          | slot.coord op slot.coord | slot.coord op int
///          | slot INSIDE [...] | "(" expr ")"
fn parse_primary<'a>(input: &'a str, ctx: &Ctx) -> Result<(&'a str, TrajExpr), DslError> {
    let input = ws_str(input);

    // DURATION ( expr , int )
    if let Ok((rest, _)) = tag_no_case::<&str, &str, nom::error::Error<&str>>("DURATION")(input) {
        if !rest.trim_start().starts_with('(') {
            // fall through — could be a variable named "duration" (unlikely but safe)
        } else {
            return parse_duration(rest, ctx);
        }
    }

    // CONTAINS ( expr , expr )
    if let Ok((rest, _)) = tag_no_case::<&str, &str, nom::error::Error<&str>>("CONTAINS")(input) {
        if !rest.starts_with(|c: char| c.is_alphanumeric() || c == '_') {
            return parse_contains(rest, ctx);
        }
    }

    // OVERLAP ( expr , expr )
    if let Ok((rest, _)) = tag_no_case::<&str, &str, nom::error::Error<&str>>("OVERLAP")(input) {
        if !rest.starts_with(|c: char| c.is_alphanumeric() || c == '_') {
            return parse_overlap(rest, ctx);
        }
    }

    // "(" … ")" — ambiguous: either a parenthesised predicate, whose booleans may
    // continue past the closing paren (`(A OR B) AND (C OR D)`), or a parenthesised
    // trajectory expression (`(X THEN Y)`). Booleans live only on the predicate side,
    // so try that first; it fails cleanly on THEN/DURATION/CONTAINS inside the parens,
    // which is precisely when this is a trajectory group.
    if input.starts_with('(') {
        if let Ok((rest, pred)) = parse_pred_expr(input, ctx) {
            return Ok((rest, TrajExpr::analytic_pred(pred)));
        }
        return parse_paren(input, ctx);
    }

    // predicate expression: comparison, INSIDE, NOT, AND, OR (and parens around them)
    parse_pred_expr(input, ctx).map(|(rest, pred)| (rest, TrajExpr::analytic_pred(pred)))
}

// ---------------------------------------------------------------------------
// Primary sub-parsers
// ---------------------------------------------------------------------------

/// `( expr , int )`  — already consumed "DURATION"
fn parse_duration<'a>(input: &'a str, ctx: &Ctx) -> Result<(&'a str, TrajExpr), DslError> {
    let input = ws_str(input);
    let (input, _) =
        char('(')(input).map_err(|_: nom::Err<nom::error::Error<&str>>| DslError::Parse {
            expected: "( after DURATION",
            at: snip(input),
        })?;
    let (input, inner) = parse_expr(input, ctx)?;
    let input = ws_str(input);
    let (input, _) =
        char(',')(input).map_err(|_: nom::Err<nom::error::Error<&str>>| DslError::Parse {
            expected: ", in DURATION",
            at: snip(input),
        })?;
    let input = ws_str(input);
    let (input, n) =
        nom_u32::<&str, nom::error::Error<&str>>(input).map_err(|_| DslError::Parse {
            expected: "integer in DURATION",
            at: snip(input),
        })?;
    let input = ws_str(input);
    let (input, _) =
        char(')')(input).map_err(|_: nom::Err<nom::error::Error<&str>>| DslError::Parse {
            expected: ") after DURATION",
            at: snip(input),
        })?;
    Ok((
        input,
        TrajExpr::Duration {
            inner: Box::new(inner),
            min_frames: n,
        },
    ))
}

/// The three error labels a `KEYWORD(expr, expr)` parser reports with, in the order
/// its delimiters are consumed.
struct BinaryLabels {
    open: &'static str,
    comma: &'static str,
    close: &'static str,
}

/// `( expr , expr )` — the shared body of the two-argument temporal keywords, called
/// with the keyword itself already consumed. `build` shapes the pair into its node.
fn parse_binary_temporal<'a>(
    input: &'a str,
    ctx: &Ctx,
    labels: &BinaryLabels,
    build: fn(Box<TrajExpr>, Box<TrajExpr>) -> TrajExpr,
) -> Result<(&'a str, TrajExpr), DslError> {
    let input = ws_str(input);
    let (input, _) =
        char('(')(input).map_err(|_: nom::Err<nom::error::Error<&str>>| DslError::Parse {
            expected: labels.open,
            at: snip(input),
        })?;
    let (input, lhs) = parse_expr(input, ctx)?;
    let input = ws_str(input);
    let (input, _) =
        char(',')(input).map_err(|_: nom::Err<nom::error::Error<&str>>| DslError::Parse {
            expected: labels.comma,
            at: snip(input),
        })?;
    let (input, rhs) = parse_expr(input, ctx)?;
    let input = ws_str(input);
    let (input, _) =
        char(')')(input).map_err(|_: nom::Err<nom::error::Error<&str>>| DslError::Parse {
            expected: labels.close,
            at: snip(input),
        })?;
    Ok((input, build(Box::new(lhs), Box::new(rhs))))
}

/// `( expr , expr )`  — already consumed "CONTAINS"
fn parse_contains<'a>(input: &'a str, ctx: &Ctx) -> Result<(&'a str, TrajExpr), DslError> {
    parse_binary_temporal(
        input,
        ctx,
        &BinaryLabels {
            open: "( after CONTAINS",
            comma: ", in CONTAINS",
            close: ") after CONTAINS",
        },
        |lhs, rhs| TrajExpr::Contained { lhs, rhs },
    )
}

/// `( expr , expr )`  — already consumed "OVERLAP"
fn parse_overlap<'a>(input: &'a str, ctx: &Ctx) -> Result<(&'a str, TrajExpr), DslError> {
    parse_binary_temporal(
        input,
        ctx,
        &BinaryLabels {
            open: "( after OVERLAP",
            comma: ", in OVERLAP",
            close: ") after OVERLAP",
        },
        |lhs, rhs| TrajExpr::Overlap { lhs, rhs },
    )
}

/// `( expr )`
fn parse_paren<'a>(input: &'a str, ctx: &Ctx) -> Result<(&'a str, TrajExpr), DslError> {
    let (input, _) =
        char('(')(input).map_err(|_: nom::Err<nom::error::Error<&str>>| DslError::Parse {
            expected: "(",
            at: snip(input),
        })?;
    let (input, expr) = parse_expr(input, ctx)?;
    let input = ws_str(input);
    let (input, _) =
        char(')')(input).map_err(|_: nom::Err<nom::error::Error<&str>>| DslError::Parse {
            expected: ")",
            at: snip(input),
        })?;
    Ok((input, expr))
}

// ---------------------------------------------------------------------------
// Predicate expression grammar (AND / OR / NOT over analytic comparisons)
// ---------------------------------------------------------------------------

/// Parsed predicate — wraps [`AnalyticalPred`] which already carries slot indices in its
/// leaves. Boolean combinators (`And`, `Or`, `Not`) freely mix predicates over any slots.
type ParsedPred = AnalyticalPred;

/// pred_expr := pred_and ("OR" pred_and)*
fn parse_pred_expr<'a>(input: &'a str, ctx: &Ctx) -> Result<(&'a str, ParsedPred), DslError> {
    let (mut remaining, mut lhs) = parse_pred_and(input, ctx)?;
    loop {
        let trimmed = ws_str(remaining);
        let Ok((after_or, _)) = tag_no_case::<&str, &str, nom::error::Error<&str>>("OR")(trimmed)
        else {
            break;
        };
        if after_or.starts_with(|c: char| c.is_alphanumeric() || c == '_') {
            break;
        }
        let (after_rhs, rhs) = parse_pred_and(after_or, ctx)?;
        lhs = lhs.or(rhs);
        remaining = after_rhs;
    }
    Ok((remaining, lhs))
}

/// pred_and := pred_not ("AND" pred_not)*
fn parse_pred_and<'a>(input: &'a str, ctx: &Ctx) -> Result<(&'a str, ParsedPred), DslError> {
    let (mut remaining, mut lhs) = parse_pred_not(input, ctx)?;
    loop {
        let trimmed = ws_str(remaining);
        let Ok((after_and, _)) = tag_no_case::<&str, &str, nom::error::Error<&str>>("AND")(trimmed)
        else {
            break;
        };
        if after_and.starts_with(|c: char| c.is_alphanumeric() || c == '_') {
            break;
        }
        let (after_rhs, rhs) = parse_pred_not(after_and, ctx)?;
        lhs = lhs.and(rhs);
        remaining = after_rhs;
    }
    Ok((remaining, lhs))
}

/// pred_not := "NOT" pred_not | pred_atom
fn parse_pred_not<'a>(input: &'a str, ctx: &Ctx) -> Result<(&'a str, ParsedPred), DslError> {
    let input = ws_str(input);
    if let Ok((rest, _)) = tag_no_case::<&str, &str, nom::error::Error<&str>>("NOT")(input) {
        if !rest.starts_with(|c: char| c.is_alphanumeric() || c == '_') {
            let (rest, inner) = parse_pred_not(rest, ctx)?;
            return Ok((rest, inner.not()));
        }
    }
    parse_pred_atom(input, ctx)
}

/// pred_atom := "(" pred_expr ")" | slot INSIDE [...] | analytic_cmp
fn parse_pred_atom<'a>(input: &'a str, ctx: &Ctx) -> Result<(&'a str, ParsedPred), DslError> {
    let input = ws_str(input);

    // TRUE keyword
    if let Ok((rest, _)) = tag_no_case::<&str, &str, nom::error::Error<&str>>("TRUE")(input) {
        if !rest.starts_with(|c: char| c.is_alphanumeric() || c == '_') {
            return Ok((rest, AnalyticalPred::true_pred()));
        }
    }

    // "(" pred_expr ")"
    if input.starts_with('(') {
        let (input, _) =
            char('(')(input).map_err(|_: nom::Err<nom::error::Error<&str>>| DslError::Parse {
                expected: "(",
                at: snip(input),
            })?;
        let (input, pred) = parse_pred_expr(input, ctx)?;
        let input = ws_str(input);
        let (input, _) =
            char(')')(input).map_err(|_: nom::Err<nom::error::Error<&str>>| DslError::Parse {
                expected: ") after predicate",
                at: snip(input),
            })?;
        return Ok((input, pred));
    }

    // slot INSIDE region | analytic_cmp — both start with an ident
    if let Ok((after_ident, var_name)) = ident::<nom::error::Error<&str>>(input) {
        let after_ident = ws_str(after_ident);

        // A `.center` suffix selects the centre form; any other dotted suffix is a
        // coordinate and belongs to the comparison path below.
        let (after_target, form) = match strip_center_suffix(after_ident) {
            Some(rest) => (ws_str(rest), InsideForm::Center),
            None => (after_ident, InsideForm::Whole),
        };

        // slot INSIDE [x1, y1, x2, y2] | slot INSIDE region_name
        if let Ok((after_inside, _)) =
            tag_no_case::<&str, &str, nom::error::Error<&str>>("INSIDE")(after_target)
        {
            if !after_inside.starts_with(|c: char| c.is_alphanumeric() || c == '_') {
                let slot = ctx.slot(var_name)?;
                let (rest, pred) = parse_inside_pred(after_inside, slot, var_name, form, ctx)?;
                return Ok((rest, pred));
            }
        }

        if after_ident.starts_with('.')
            || after_ident.starts_with('+')
            || after_ident.starts_with('-')
            || after_ident.starts_with('*')
            || after_ident.starts_with('/')
            || after_ident.starts_with('<')
            || after_ident.starts_with('>')
            || after_ident.starts_with(|c: char| c.is_whitespace())
        {
            return parse_analytic_cmp_pred(input, ctx);
        }
    }

    Err(DslError::Parse {
        expected: "predicate (comparison, INSIDE, NOT, or parenthesized predicate)",
        at: snip(input),
    })
}

/// Which part of an object's bounding box `INSIDE` tests against the region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InsideForm {
    /// `v INSIDE R` — the whole box lies within the region.
    Whole,
    /// `v.center INSIDE R` — only the box centre lies within the region.
    Center,
}

/// Expands `INSIDE` into the predicate its shape stands for.
///
/// A rectangle keeps the four affine comparisons it always had. A **convex** polygon expands
/// the same way, one half-plane comparison per edge — still `And` of `Unary`, so the
/// optimizer, the evaluator, and the SQL backend need to know nothing about polygons.
///
/// A concave polygon is the only case that reaches for the `InPolygon` node. Both forms are
/// supported and both are exact; the whole-box form additionally checks that no edge of the
/// region cuts through the box, since in a concave region the four corners alone do not
/// settle containment.
fn inside_shape(slot: Slot, shape: &Shape, form: InsideForm) -> ParsedPred {
    match shape {
        Shape::Rect(rect) => inside_rect(slot, *rect, form),
        Shape::Poly(poly) => match form {
            InsideForm::Center => AnalyticalPred::point_in_polygon(
                BBoxExpr::center_x(),
                BBoxExpr::center_y(),
                poly.clone(),
                slot,
            ),
            InsideForm::Whole => AnalyticalPred::box_in_polygon(poly.clone(), slot),
        },
    }
}

/// Expands a rectangular `INSIDE` into the four affine comparisons it stands for.
///
/// This is the entirety of a rectangle's meaning. Everything downstream — the optimizer, the
/// analytical evaluator, the SQL backend — only ever sees `And` of `Unary` predicates, so
/// the operator costs them nothing and the centre form needs no new machinery.
///
/// The left-nested `And` order is load-bearing: `set_pred_groups` and the SQL renderer walk
/// this tree, so it reproduces exactly what the old `AnalyticalPred::inside` built.
fn inside_rect(slot: Slot, [lx1, ly1, lx2, ly2]: Rect, form: InsideForm) -> ParsedPred {
    let (low_x, low_y, high_x, high_y) = match form {
        InsideForm::Whole => (
            BBoxExpr::x1(),
            BBoxExpr::y1(),
            BBoxExpr::x2(),
            BBoxExpr::y2(),
        ),
        // Both bounds test the same point, so the centre must clear the low edge and
        // stay under the high one.
        InsideForm::Center => (
            BBoxExpr::center_x(),
            BBoxExpr::center_y(),
            BBoxExpr::center_x(),
            BBoxExpr::center_y(),
        ),
    };

    AnalyticalPred::unary_gte(low_x, lx1, slot)
        .and(AnalyticalPred::unary_gte(low_y, ly1, slot))
        .and(AnalyticalPred::unary_lte(high_x, lx2, slot))
        .and(AnalyticalPred::unary_lte(high_y, ly2, slot))
}

/// A region literal (rectangle or polygon) or a region name — already consumed the target
/// and "INSIDE".
fn parse_inside_pred<'a>(
    input: &'a str,
    slot: Slot,
    var: &str,
    form: InsideForm,
    ctx: &Ctx,
) -> Result<(&'a str, ParsedPred), DslError> {
    let trimmed = ws_str(input);

    if trimmed.starts_with('[') {
        let (rest, shape) = parse_shape(trimmed)?;
        ctx.record(None, var, &shape, form);
        return Ok((rest, inside_shape(slot, &shape, form)));
    }

    let (rest, name) = ident::<nom::error::Error<&str>>(trimmed).map_err(|_| DslError::Parse {
        expected: "region literal or region name after INSIDE",
        at: snip(trimmed),
    })?;
    let shape = ctx.region(name)?;
    ctx.record(Some(name), var, &shape, form);
    Ok((rest, inside_shape(slot, &shape, form)))
}

fn parse_analytic_cmp_pred<'a>(
    input: &'a str,
    ctx: &Ctx,
) -> Result<(&'a str, ParsedPred), DslError> {
    let (input, lhs) = parse_bbox_expr(input, ctx)?;
    let input = ws_str(input);
    let (input, op) = parse_rel_op(input)?;
    let input = ws_str(input);
    let (rest, rhs) = parse_bbox_expr(input, ctx)?;
    build_analytic_cmp_pred(lhs, op, rhs).map(|pred| (rest, pred))
}

// ---------------------------------------------------------------------------
// Gap: "[N]" | "[N..]" | "[N..M]"
// Returns (min_gap, max_gap). Defaults to (0, None) if absent.
// ---------------------------------------------------------------------------

fn parse_gap(input: &str) -> Result<(&str, (u32, Option<u32>)), DslError> {
    let trimmed = ws_str(input);
    if !trimmed.starts_with('[') {
        return Ok((input, (0, None)));
    }
    let (input, _) =
        char('[')(trimmed).map_err(|_: nom::Err<nom::error::Error<&str>>| DslError::Parse {
            expected: "[ in gap",
            at: snip(trimmed),
        })?;
    let input = ws_str(input);
    let (input, min) =
        nom_u32::<&str, nom::error::Error<&str>>(input).map_err(|_| DslError::Parse {
            expected: "min in gap",
            at: snip(input),
        })?;
    let input = ws_str(input);

    // optional ".." and max
    if let Ok((after_dots, _)) = tag::<&str, &str, nom::error::Error<&str>>("..")(input) {
        let after_dots = ws_str(after_dots);
        // optional max
        let (after_max, max_opt) =
            if let Ok((r, n)) = nom_u32::<&str, nom::error::Error<&str>>(after_dots) {
                (r, Some(n))
            } else {
                (after_dots, None)
            };
        let after_max = ws_str(after_max);
        let (rest, _) = char(']')(after_max).map_err(|_: nom::Err<nom::error::Error<&str>>| {
            DslError::Parse {
                expected: "] in gap",
                at: snip(after_max),
            }
        })?;
        Ok((rest, (min, max_opt)))
    } else {
        // "[N]" — treat as min_gap = N, no max
        let input = ws_str(input);
        let (rest, _) =
            char(']')(input).map_err(|_: nom::Err<nom::error::Error<&str>>| DslError::Parse {
                expected: "] in gap",
                at: snip(input),
            })?;
        Ok((rest, (min, None)))
    }
}

// ---------------------------------------------------------------------------
// Coord and operator helpers
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct ParsedBBoxExpr {
    slot: Option<Slot>,
    expr: BBoxExpr,
}

fn parse_bbox_expr<'a>(input: &'a str, ctx: &Ctx) -> Result<(&'a str, ParsedBBoxExpr), DslError> {
    let (mut input, mut acc) = parse_bbox_term(input, ctx)?;

    loop {
        let trimmed = ws_str(input);
        if let Ok((rest, _)) = char::<&str, nom::error::Error<&str>>('+')(trimmed) {
            let (next, rhs) = parse_bbox_term(rest, ctx)?;
            acc = combine_bbox_exprs(acc, rhs, false)?;
            input = next;
            continue;
        }
        if let Ok((rest, _)) = char::<&str, nom::error::Error<&str>>('-')(trimmed) {
            let (next, rhs) = parse_bbox_term(rest, ctx)?;
            acc = combine_bbox_exprs(acc, rhs, true)?;
            input = next;
            continue;
        }
        break;
    }

    Ok((input, acc))
}

fn parse_bbox_term<'a>(input: &'a str, ctx: &Ctx) -> Result<(&'a str, ParsedBBoxExpr), DslError> {
    let (mut input, mut acc) = parse_bbox_unary(input, ctx)?;

    loop {
        let trimmed = ws_str(input);
        let Ok((rest, _)) = char::<&str, nom::error::Error<&str>>('*')(trimmed) else {
            break;
        };
        let (next, rhs) = parse_bbox_unary(rest, ctx)?;
        acc = scale_bbox_expr(acc, rhs)?;
        input = next;
    }

    Ok((input, acc))
}

fn parse_bbox_unary<'a>(input: &'a str, ctx: &Ctx) -> Result<(&'a str, ParsedBBoxExpr), DslError> {
    let input = ws_str(input);
    if let Ok((rest, _)) = char::<&str, nom::error::Error<&str>>('+')(input) {
        return parse_bbox_unary(rest, ctx);
    }
    if let Ok((rest, _)) = char::<&str, nom::error::Error<&str>>('-')(input) {
        let (rest, expr) = parse_bbox_unary(rest, ctx)?;
        return Ok((
            rest,
            ParsedBBoxExpr {
                slot: expr.slot,
                expr: -expr.expr,
            },
        ));
    }
    parse_bbox_atom(input, ctx)
}

fn parse_bbox_atom<'a>(input: &'a str, ctx: &Ctx) -> Result<(&'a str, ParsedBBoxExpr), DslError> {
    let input = ws_str(input);

    if let Ok((rest, value)) = double::<&str, nom::error::Error<&str>>(input) {
        return Ok((
            rest,
            ParsedBBoxExpr {
                slot: None,
                expr: BBoxExpr::constant(value),
            },
        ));
    }

    if let Ok((rest, _)) = char::<&str, nom::error::Error<&str>>('(')(input) {
        let (rest, expr) = parse_bbox_expr(rest, ctx)?;
        let rest = ws_str(rest);
        let (rest, _) =
            char(')')(rest).map_err(|_: nom::Err<nom::error::Error<&str>>| DslError::Parse {
                expected: ") in bbox expression",
                at: snip(rest),
            })?;
        return Ok((rest, expr));
    }

    let (rest, var_name) =
        ident::<nom::error::Error<&str>>(input).map_err(|_| DslError::Parse {
            expected: "bbox expression",
            at: snip(input),
        })?;
    let slot = ctx.slot(var_name)?;
    let rest = ws_str(rest);
    let (rest, _) =
        char('.')(rest).map_err(|_: nom::Err<nom::error::Error<&str>>| DslError::Parse {
            expected: ".coord in bbox expression",
            at: snip(rest),
        })?;
    let (rest, expr) = parse_bbox_coord(rest)?;
    Ok((
        rest,
        ParsedBBoxExpr {
            slot: Some(slot),
            expr,
        },
    ))
}

fn parse_bbox_coord(input: &str) -> Result<(&str, BBoxExpr), DslError> {
    let coords = [
        ("center_x", BBoxExpr::center_x()),
        ("center_y", BBoxExpr::center_y()),
        ("width", BBoxExpr::width()),
        ("height", BBoxExpr::height()),
        ("cx", BBoxExpr::center_x()),
        ("cy", BBoxExpr::center_y()),
        ("x1", BBoxExpr::x1()),
        ("y1", BBoxExpr::y1()),
        ("x2", BBoxExpr::x2()),
        ("y2", BBoxExpr::y2()),
    ];
    for (name, expr) in coords {
        if let Ok((rest, _)) = tag_no_case::<&str, &str, nom::error::Error<&str>>(name)(input) {
            return Ok((rest, expr));
        }
    }
    Err(DslError::Parse {
        expected: "bbox coord (x1, y1, x2, y2, cx, cy, width, height)",
        at: snip(input),
    })
}

fn combine_bbox_exprs(
    lhs: ParsedBBoxExpr,
    rhs: ParsedBBoxExpr,
    subtract: bool,
) -> Result<ParsedBBoxExpr, DslError> {
    let slot = match (lhs.slot, rhs.slot) {
        (Some(a), Some(b)) if a != b => {
            return Err(DslError::UnsupportedComparison(
                "bbox expressions cannot mix multiple variables on the same side",
            ));
        }
        (Some(a), _) => Some(a),
        (_, Some(b)) => Some(b),
        (None, None) => None,
    };

    Ok(ParsedBBoxExpr {
        slot,
        expr: if subtract {
            lhs.expr - rhs.expr
        } else {
            lhs.expr + rhs.expr
        },
    })
}

fn scale_bbox_expr(lhs: ParsedBBoxExpr, rhs: ParsedBBoxExpr) -> Result<ParsedBBoxExpr, DslError> {
    match (lhs.slot, rhs.slot) {
        (None, None) => Ok(ParsedBBoxExpr {
            slot: None,
            expr: BBoxExpr::constant(lhs.expr.offset * rhs.expr.offset),
        }),
        (Some(slot), None) => Ok(ParsedBBoxExpr {
            slot: Some(slot),
            expr: lhs.expr * rhs.expr.offset,
        }),
        (None, Some(slot)) => Ok(ParsedBBoxExpr {
            slot: Some(slot),
            expr: rhs.expr * lhs.expr.offset,
        }),
        (Some(_), Some(_)) => Err(DslError::UnsupportedComparison(
            "bbox expressions only support scalar multiplication",
        )),
    }
}

fn build_analytic_cmp_pred(
    lhs: ParsedBBoxExpr,
    op: &str,
    rhs: ParsedBBoxExpr,
) -> Result<ParsedPred, DslError> {
    let cmp = parse_cmp_op(op)?;

    match (lhs.slot, rhs.slot) {
        (None, None) => Err(DslError::UnsupportedComparison(
            "analytic comparison must reference at least one variable",
        )),
        (Some(slot), None) => Ok(AnalyticalPred {
            group: usize::MAX,
            kind: AnalyticalPredKind::Unary {
                expr: lhs.expr,
                op: cmp,
                threshold: rhs.expr.offset,
                slot: slot,
            },
        }),
        (None, Some(slot)) => Ok(AnalyticalPred {
            group: usize::MAX,
            kind: AnalyticalPredKind::Unary {
                expr: rhs.expr,
                op: invert_cmp(cmp),
                threshold: lhs.expr.offset,
                slot: slot,
            },
        }),
        (Some(slot_l), Some(slot_r)) if slot_l == slot_r => Ok(AnalyticalPred {
            group: usize::MAX,
            kind: AnalyticalPredKind::Unary {
                expr: lhs.expr - rhs.expr,
                op: cmp,
                threshold: 0.0,
                slot: slot_l,
            },
        }),
        (Some(slot_l), Some(slot_r)) => Ok(AnalyticalPred {
            group: usize::MAX,
            kind: AnalyticalPredKind::Binary {
                expr: BinaryBBoxExpr::new(lhs.expr, rhs.expr),
                op: cmp,
                threshold: 0.0,
                slots: [slot_l, slot_r],
            },
        }),
    }
}

fn parse_cmp_op(op: &str) -> Result<CmpOp, DslError> {
    match op {
        ">" => Ok(CmpOp::Gt),
        ">=" => Ok(CmpOp::Gte),
        "<" => Ok(CmpOp::Lt),
        "<=" => Ok(CmpOp::Lte),
        _ => Err(DslError::UnsupportedComparison(
            "invalid relational operator",
        )),
    }
}

fn invert_cmp(op: CmpOp) -> CmpOp {
    match op {
        CmpOp::Gt => CmpOp::Lt,
        CmpOp::Gte => CmpOp::Lte,
        CmpOp::Lt => CmpOp::Gt,
        CmpOp::Lte => CmpOp::Gte,
    }
}

/// Returns the operator string: `"<"`, `">"`, `"<="`, `">="`.
fn parse_rel_op(input: &str) -> Result<(&str, &str), DslError> {
    if let Ok((r, _)) = tag::<&str, &str, nom::error::Error<&str>>("<=")(input) {
        return Ok((r, "<="));
    }
    if let Ok((r, _)) = tag::<&str, &str, nom::error::Error<&str>>("<")(input) {
        return Ok((r, "<"));
    }
    if let Ok((r, _)) = tag::<&str, &str, nom::error::Error<&str>>(">=")(input) {
        return Ok((r, ">="));
    }
    if let Ok((r, _)) = tag::<&str, &str, nom::error::Error<&str>>(">")(input) {
        return Ok((r, ">"));
    }
    Err(DslError::Parse {
        expected: "relational operator (< > <= >=)",
        at: snip(input),
    })
}

// ---------------------------------------------------------------------------
// Lexical helpers
// ---------------------------------------------------------------------------

fn ws_str(input: &str) -> &str {
    input.trim_start_matches(|c: char| c.is_whitespace())
}

/// Consumes a `.center` target suffix, returning the rest of the input.
///
/// Rejects `.center_x` / `.center_y`, which name a single coordinate rather than the centre
/// point, so `a.center_x < 5` still reaches the comparison parser.
fn strip_center_suffix(input: &str) -> Option<&str> {
    let rest = input.strip_prefix('.')?;
    let (rest, _) = tag_no_case::<&str, &str, nom::error::Error<&str>>("center")(rest).ok()?;
    if rest.starts_with(|c: char| c.is_alphanumeric() || c == '_') {
        return None;
    }
    Some(rest)
}

/// Parse an ASCII identifier: `[a-zA-Z_][a-zA-Z0-9_]*`
fn ident<'a, E: nom::error::ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, &'a str, E> {
    take_while1(|c: char| c.is_alphanumeric() || c == '_')(input)
}

/// Parse a value token: ident, integer, or "quoted string".
fn parse_value_token(input: &str) -> Result<(&str, &str), nom::Err<nom::error::Error<&str>>> {
    // quoted string
    if input.starts_with('"') {
        let rest = &input[1..];
        if let Some(end) = rest.find('"') {
            return Ok((&rest[end + 1..], &rest[..end]));
        }
    }
    // ident or integer (anything non-whitespace, non-punctuation up to the next delimiter)
    let end = input
        .find(|c: char| c.is_whitespace() || matches!(c, ',' | ')' | ']' | '(' | '[' | '='))
        .unwrap_or(input.len());
    if end == 0 {
        return Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::TakeWhile1,
        )));
    }
    Ok((&input[end..], &input[..end]))
}

/// Short snippet of remaining input for error messages (max 40 chars).
fn snip(s: &str) -> String {
    let s = s.trim_start();
    if s.len() <= 40 {
        s.to_owned()
    } else {
        format!("{}…", &s[..40])
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use trql_bbox_compiler::{AnalyticalPred, AnalyticalPredKind, BBoxExpr, CmpOp};
    use trql_types::prop::{Schema, TypeDesc, Value};

    /// What `v INSIDE [x1, y1, x2, y2]` must expand to, spelled out independently of the
    /// [`inside_rect`] helper that builds it — otherwise the assertion would just be
    /// `inside_rect == inside_rect` and any change to the expansion would pass silently.
    fn expected_whole_box(x1: f64, y1: f64, x2: f64, y2: f64, slot: Slot) -> AnalyticalPred {
        AnalyticalPred::unary_gte(BBoxExpr::x1(), x1, slot)
            .and(AnalyticalPred::unary_gte(BBoxExpr::y1(), y1, slot))
            .and(AnalyticalPred::unary_lte(BBoxExpr::x2(), x2, slot))
            .and(AnalyticalPred::unary_lte(BBoxExpr::y2(), y2, slot))
    }

    /// The same, for `v.center INSIDE [x1, y1, x2, y2]`.
    fn expected_center(x1: f64, y1: f64, x2: f64, y2: f64, slot: Slot) -> AnalyticalPred {
        AnalyticalPred::unary_gte(BBoxExpr::center_x(), x1, slot)
            .and(AnalyticalPred::unary_gte(BBoxExpr::center_y(), y1, slot))
            .and(AnalyticalPred::unary_lte(BBoxExpr::center_x(), x2, slot))
            .and(AnalyticalPred::unary_lte(BBoxExpr::center_y(), y2, slot))
    }

    /// `(position, name)` for each slot — checks both the tuple position and
    /// the query variable name the slot was written with.
    /// `(slot, declared name)` for each declared variable, in tuple order.
    /// A `Slot` is only a position, so the name comes from the variable table.
    fn named(vars: &VariablesWithConstraints) -> Vec<(Slot, &str)> {
        vars.slots()
            .map(|s| (s, vars.name_of(s).expect("declared")))
            .collect()
    }

    fn cityflow_schema() -> Schema {
        Schema::new(vec![
            (
                "car_type".into(),
                TypeDesc::Enum(vec![
                    "SUV".into(),
                    "Sedan".into(),
                    "Van".into(),
                    "Bus".into(),
                    "Truck".into(),
                ]),
            ),
            (
                "car_color".into(),
                TypeDesc::Enum(vec!["Red".into(), "Blue".into(), "White".into()]),
            ),
        ])
    }

    fn clevrer_schema() -> Schema {
        Schema::new(vec![
            (
                "shape".into(),
                TypeDesc::Enum(vec!["Sphere".into(), "Cube".into(), "Cylinder".into()]),
            ),
            (
                "color".into(),
                TypeDesc::Enum(vec![
                    "Red".into(),
                    "Blue".into(),
                    "Cyan".into(),
                    "Brown".into(),
                    "Gray".into(),
                    "Green".into(),
                    "Yellow".into(),
                    "Purple".into(),
                ]),
            ),
            (
                "material".into(),
                TypeDesc::Enum(vec!["Rubber".into(), "Metal".into()]),
            ),
        ])
    }

    fn empty_schema() -> Schema {
        Schema::new(vec![])
    }

    // ------------------------------------------------------------------
    // Selector
    // ------------------------------------------------------------------

    #[test]
    fn selector_constrained() {
        let schema = cityflow_schema();
        let (selector, _) = parse_query(
            "SELECT (a: car_type=SUV, b: car_type=Sedan) WHERE a.x1 < b.x1",
            &schema,
        )
        .unwrap();
        assert_eq!(selector.num_slots(), 2);
        // Declaration order is tuple order, and each slot keeps the name it was
        // declared with: a#0 = SUV, b#1 = Sedan.
        assert_eq!(named(&selector), [(0, "a"), (1, "b")]);
        for (slot, expected) in [(0, Value::Enum(0)), (1, Value::Enum(1))] {
            let constraints = selector.get_slot_constraints(slot).unwrap();
            assert_eq!(constraints.num_constraints(), 1);
            let constraint = constraints.get_constraint(0).unwrap();
            assert_eq!(constraint.prop, "car_type");
            assert_eq!(constraint.val, expected, "slot {slot}");
        }
    }

    #[test]
    fn selector_unconstrained() {
        let schema = empty_schema();
        let (selector, _) = parse_query("SELECT (a:, b:) WHERE a.x1 < b.x1", &schema).unwrap();
        assert_eq!(selector.num_slots(), 2);
        assert!(selector.get_slot_constraints(0).unwrap().is_empty());
        assert!(selector.get_slot_constraints(0).unwrap().is_empty());
    }

    #[test]
    fn selector_single_slot() {
        let schema = cityflow_schema();
        let (selector, _) =
            parse_query("SELECT (a: car_type=Bus) WHERE a.x1 < 100", &schema).unwrap();
        assert_eq!(selector.num_slots(), 1);
    }

    #[test]
    fn clevrer_selector_constrained() {
        let schema = clevrer_schema();
        let (selector, _) = parse_query(
            "SELECT (a: shape=Cube color=Red material=Metal) WHERE a.x1 < 100",
            &schema,
        )
        .unwrap();

        assert_eq!(selector.num_slots(), 1);
        let constraints = selector.get_slot_constraints(0).unwrap();
        assert_eq!(
            selector.get_slot_constraints(0).unwrap().num_constraints(),
            3
        );

        let constraint1 = constraints.get_constraint(0).unwrap();
        assert_eq!(constraint1.prop, "shape");
        assert_eq!(constraint1.val, Value::Enum(1));

        let constraint2 = constraints.get_constraint(1).unwrap();
        assert_eq!(constraint2.prop, "color");
        assert_eq!(constraint2.val, Value::Enum(0));

        let constraint3 = constraints.get_constraint(2).unwrap();
        assert_eq!(constraint3.prop, "material");
        assert_eq!(constraint3.val, Value::Enum(1));
    }

    // ------------------------------------------------------------------
    // Binary analytic
    // ------------------------------------------------------------------

    #[test]
    fn binary_analytic_less() {
        let schema = empty_schema();
        let (_, expr) = parse_query("SELECT (a:, b:) WHERE a.x1 < b.x1", &schema).unwrap();
        let TrajExpr::AnalyticPred(pred) = expr else {
            panic!()
        };
        let AnalyticalPredKind::Binary { op, slots, .. } = pred.kind else {
            panic!()
        };
        assert_eq!(op, CmpOp::Lt);
        assert_eq!(slots, [0, 1]);
    }

    #[test]
    fn binary_analytic_greater_y1() {
        let schema = empty_schema();
        let (_, expr) = parse_query("SELECT (a:, b:) WHERE a.y1 > b.y1", &schema).unwrap();

        let TrajExpr::AnalyticPred(pred) = expr else {
            panic!()
        };
        let AnalyticalPredKind::Binary { op, slots, .. } = pred.kind else {
            panic!()
        };
        assert_eq!(op, CmpOp::Gt);
        assert_eq!(slots, [0, 1]);
    }

    // ------------------------------------------------------------------
    // Unary analytic
    // ------------------------------------------------------------------

    #[test]
    fn unary_analytic_lt() {
        let schema = empty_schema();
        let (_, expr) = parse_query("SELECT (a:) WHERE a.y1 < 400", &schema).unwrap();

        let TrajExpr::AnalyticPred(pred) = expr else {
            panic!()
        };
        let AnalyticalPredKind::Unary { op, slot, .. } = pred.kind else {
            panic!()
        };
        assert_eq!(op, CmpOp::Lt);
        assert_eq!(slot, 0);
    }

    #[test]
    fn unary_analytic_gte() {
        let schema = empty_schema();
        let (_, expr) = parse_query("SELECT (a:) WHERE a.x2 >= 100", &schema).unwrap();

        let TrajExpr::AnalyticPred(pred) = expr else {
            panic!()
        };
        let AnalyticalPredKind::Unary { op, .. } = pred.kind else {
            panic!()
        };
        assert_eq!(op, CmpOp::Gte);
    }

    // ------------------------------------------------------------------
    // INSIDE
    // ------------------------------------------------------------------

    #[test]
    fn inside_bbox() {
        let schema = empty_schema();
        let (_, expr) =
            parse_query("SELECT (a:) WHERE a INSIDE [0, 0, 100, 200]", &schema).unwrap();

        let TrajExpr::AnalyticPred(pred) = expr else {
            panic!("expected AnalyticPred")
        };
        assert_eq!(pred, expected_whole_box(0.0, 0.0, 100.0, 200.0, 0));
    }

    /// Region bounds are `f64`, so fractional and negative coordinates are legal — a
    /// region may legitimately start off-frame.
    #[test]
    fn inside_accepts_fractional_and_negative_bounds() {
        let (_, expr) = parse_query(
            "SELECT (a:) WHERE a INSIDE [-10.5, 0, 100, 200.25]",
            &empty_schema(),
        )
        .unwrap();

        let TrajExpr::AnalyticPred(pred) = expr else {
            panic!("expected AnalyticPred")
        };
        assert_eq!(pred, expected_whole_box(-10.5, 0.0, 100.0, 200.25, 0));
    }

    #[test]
    fn inside_rejects_inverted_region() {
        let err = parse_query(
            "SELECT (a:) WHERE a INSIDE [100, 0, 50, 200]",
            &empty_schema(),
        )
        .expect_err("an inverted region should not parse");
        assert!(matches!(err, DslError::EmptyRegion { x1, x2, .. } if x1 == 100.0 && x2 == 50.0));
    }

    #[test]
    fn center_inside_tests_the_centre_point() {
        let (_, expr) = parse_query(
            "SELECT (a:) WHERE a.center INSIDE [0, 0, 100, 200]",
            &empty_schema(),
        )
        .unwrap();

        let TrajExpr::AnalyticPred(pred) = expr else {
            panic!("expected AnalyticPred")
        };
        assert_eq!(pred, expected_center(0.0, 0.0, 100.0, 200.0, 0));
    }

    // ------------------------------------------------------------------
    // INSIDE a polygon
    // ------------------------------------------------------------------

    /// Collects every leaf of a predicate tree, so tests can assert on the *shape* of the
    /// expansion without spelling out a deeply nested `And`.
    fn leaves(pred: &AnalyticalPred, out: &mut Vec<AnalyticalPredKind>) {
        match &pred.kind {
            AnalyticalPredKind::And(l, r) | AnalyticalPredKind::Or(l, r) => {
                leaves(l, out);
                leaves(r, out);
            }
            AnalyticalPredKind::Not(i) => leaves(i, out),
            kind => out.push(kind.clone()),
        }
    }

    fn parse_pred(q: &str) -> Result<AnalyticalPred, DslError> {
        let (_, expr) = parse_query(q, &empty_schema())?;
        let TrajExpr::AnalyticPred(pred) = expr else {
            panic!("expected AnalyticPred")
        };
        Ok(pred)
    }

    /// A convex polygon must expand to nothing but half-plane comparisons: one per edge for
    /// the centre form, one per edge per corner for the whole box. That is what keeps the
    /// optimizer, the evaluator, and the SQL backend from ever meeting a polygon.
    #[test]
    fn convex_polygon_expands_to_half_plane_comparisons() {
        let centre =
            parse_pred("SELECT (a:) WHERE a.center INSIDE [[0,0],[10,0],[10,10],[0,10]]").unwrap();
        let mut out = Vec::new();
        leaves(&centre, &mut out);
        assert_eq!(out.len(), 4, "one comparison per edge");
        assert!(
            out.iter()
                .all(|k| matches!(k, AnalyticalPredKind::Unary { .. }))
        );

        let whole = parse_pred("SELECT (a:) WHERE a INSIDE [[0,0],[10,0],[10,10],[0,10]]").unwrap();
        let mut out = Vec::new();
        leaves(&whole, &mut out);
        assert_eq!(out.len(), 16, "four edges × four corners");
        assert!(
            out.iter()
                .all(|k| matches!(k, AnalyticalPredKind::Unary { .. }))
        );
    }

    /// Winding is normalised, so the same region written clockwise compiles identically —
    /// otherwise every half-plane would come out inverted and the query would match nothing.
    #[test]
    fn polygon_winding_does_not_change_the_predicate() {
        let ccw =
            parse_pred("SELECT (a:) WHERE a.center INSIDE [[0,0],[10,0],[10,10],[0,10]]").unwrap();
        let cw =
            parse_pred("SELECT (a:) WHERE a.center INSIDE [[0,10],[10,10],[10,0],[0,0]]").unwrap();
        assert_eq!(ccw, cw);
    }

    /// Only a concave region needs the analytical polygon node.
    #[test]
    fn concave_polygon_centre_becomes_an_in_polygon_node() {
        let pred = parse_pred(
            "SELECT (a:) WHERE a.center INSIDE [[0,0],[10,0],[10,5],[5,5],[5,10],[0,10]]",
        )
        .unwrap();
        assert!(matches!(pred.kind, AnalyticalPredKind::InPolygon { .. }));
    }

    /// Whole-box containment in a concave region is not the conjunction of its corner tests
    /// alone, so the expansion carries an edge-clearance condition on top: one point test
    /// per corner, plus a separating-axis disjunction per edge.
    #[test]
    fn concave_polygon_whole_box_adds_edge_clearance() {
        let pred =
            parse_pred("SELECT (a:) WHERE a INSIDE [[0,0],[10,0],[10,5],[5,5],[5,10],[0,10]]")
                .expect("whole-box containment in a concave region should compile");
        let mut out = Vec::new();
        leaves(&pred, &mut out);

        let corners = out
            .iter()
            .filter(|k| matches!(k, AnalyticalPredKind::InPolygon { .. }))
            .count();
        assert_eq!(corners, 4, "one point test per box corner");

        // Six edges, six separating-axis candidates each — two box axes in each direction
        // plus the edge's own normal from either side.
        let comparisons = out
            .iter()
            .filter(|k| matches!(k, AnalyticalPredKind::Unary { .. }))
            .count();
        assert_eq!(comparisons, 36);
    }

    /// A ring that encloses nothing would match nothing, which is miserable to debug in a
    /// literal that is just a list of numbers — the same reason inverted rectangles are
    /// rejected at parse time.
    #[test]
    fn degenerate_polygons_are_rejected() {
        for q in [
            "SELECT (a:) WHERE a.center INSIDE [[0,0],[1,1]]",
            "SELECT (a:) WHERE a.center INSIDE [[0,0],[1,1],[2,2]]",
        ] {
            assert!(matches!(
                parse_pred(q).expect_err("degenerate ring should not parse"),
                DslError::BadPolygon(_)
            ));
        }
    }

    /// Polygons are declarable in `REGIONS` alongside rectangles, and a named region means
    /// exactly what the inline literal means.
    #[test]
    fn regions_clause_accepts_polygons() {
        let named = parse_pred(
            "REGIONS (zone = [[0,0],[10,0],[10,10],[0,10]]) \
             SELECT (a:) WHERE a.center INSIDE zone",
        )
        .unwrap();
        let inline =
            parse_pred("SELECT (a:) WHERE a.center INSIDE [[0,0],[10,0],[10,10],[0,10]]").unwrap();
        assert_eq!(named, inline);
    }

    /// The rectangle form is told from the polygon form by the second bracket alone, so a
    /// query written before polygons existed must still compile to the four comparisons.
    #[test]
    fn rectangle_literals_are_unaffected_by_the_polygon_form() {
        let pred = parse_pred("SELECT (a:) WHERE a INSIDE [0, 0, 100, 200]").unwrap();
        assert_eq!(pred, expected_whole_box(0.0, 0.0, 100.0, 200.0, 0));
    }

    /// `.center` must not swallow `.center_x`, which is a coordinate and belongs to the
    /// comparison parser.
    #[test]
    fn center_x_is_still_a_coordinate() {
        let (_, expr) = parse_query("SELECT (a:) WHERE a.center_x < 500", &empty_schema()).unwrap();

        let TrajExpr::AnalyticPred(pred) = expr else {
            panic!("expected AnalyticPred")
        };
        let AnalyticalPredKind::Unary { expr, op, .. } = pred.kind else {
            panic!("expected Unary")
        };
        assert_eq!(op, CmpOp::Lt);
        assert_eq!(expr, BBoxExpr::center_x());
    }

    // ------------------------------------------------------------------
    // REGIONS
    // ------------------------------------------------------------------

    /// The whole point of the clause: a named region is sugar, producing byte-for-byte the
    /// tree the inlined rectangle would have produced.
    #[test]
    fn named_region_matches_the_inline_form() {
        let named = parse_query(
            "REGIONS (kerb = [0, 0, 100, 200]) SELECT (a:) WHERE a INSIDE kerb",
            &empty_schema(),
        )
        .unwrap()
        .1;
        let inlined = parse_query(
            "SELECT (a:) WHERE a INSIDE [0, 0, 100, 200]",
            &empty_schema(),
        )
        .unwrap()
        .1;

        let (TrajExpr::AnalyticPred(named), TrajExpr::AnalyticPred(inlined)) = (named, inlined)
        else {
            panic!("expected AnalyticPred")
        };
        assert_eq!(named, inlined);
    }

    #[test]
    fn named_region_works_with_the_centre_form() {
        let (_, expr) = parse_query(
            "REGIONS (crosswalk = [0, 0, 100, 200]) \
             SELECT (a:) WHERE a.center INSIDE crosswalk",
            &empty_schema(),
        )
        .unwrap();

        let TrajExpr::AnalyticPred(pred) = expr else {
            panic!("expected AnalyticPred")
        };
        assert_eq!(pred, expected_center(0.0, 0.0, 100.0, 200.0, 0));
    }

    #[test]
    fn several_regions_resolve_independently() {
        let (_, expr) = parse_query(
            "REGIONS (a_side = [0, 0, 100, 100], b_side = [200, 200, 300, 300]) \
             SELECT (v:) WHERE v INSIDE a_side THEN v INSIDE b_side",
            &empty_schema(),
        )
        .unwrap();

        let TrajExpr::Then { lhs, rhs, .. } = expr else {
            panic!("expected Then")
        };
        let (TrajExpr::AnalyticPred(lhs), TrajExpr::AnalyticPred(rhs)) = (*lhs, *rhs) else {
            panic!("expected AnalyticPred")
        };
        assert_eq!(lhs, expected_whole_box(0.0, 0.0, 100.0, 100.0, 0));
        assert_eq!(rhs, expected_whole_box(200.0, 200.0, 300.0, 300.0, 0));
    }

    #[test]
    fn unknown_region_errors() {
        let err = parse_query(
            "REGIONS (a_side = [0, 0, 100, 100]) SELECT (v:) WHERE v INSIDE b_side",
            &empty_schema(),
        )
        .expect_err("an undeclared region should not parse");
        assert!(matches!(err, DslError::UnknownRegion(name) if name == "b_side"));
    }

    /// With no REGIONS clause there is nothing to resolve against, so a bare name is an
    /// unknown region rather than a confusing syntax error.
    #[test]
    fn region_name_without_a_regions_clause_errors() {
        let err = parse_query("SELECT (v:) WHERE v INSIDE kerb", &empty_schema())
            .expect_err("no regions are declared");
        assert!(matches!(err, DslError::UnknownRegion(name) if name == "kerb"));
    }

    #[test]
    fn duplicate_region_errors() {
        let err = parse_query(
            "REGIONS (r = [0, 0, 10, 10], r = [0, 0, 20, 20]) SELECT (v:) WHERE v INSIDE r",
            &empty_schema(),
        )
        .expect_err("a region declared twice should not parse");
        assert!(matches!(err, DslError::DuplicateRegion(name) if name == "r"));
    }

    #[test]
    fn regions_clause_rejects_an_inverted_region() {
        let err = parse_query(
            "REGIONS (r = [0, 200, 10, 10]) SELECT (v:) WHERE v INSIDE r",
            &empty_schema(),
        )
        .expect_err("an inverted region should not parse");
        assert!(matches!(err, DslError::EmptyRegion { y1, y2, .. } if y1 == 200.0 && y2 == 10.0));
    }

    // ------------------------------------------------------------------
    // DURATION
    // ------------------------------------------------------------------

    #[test]
    fn duration() {
        let schema = empty_schema();
        let (_, expr) =
            parse_query("SELECT (a:, b:) WHERE duration(a.x1 < b.x1, 30)", &schema).unwrap();

        let TrajExpr::Duration { min_frames, inner } = expr else {
            panic!()
        };
        assert_eq!(min_frames, 30);
        assert!(matches!(*inner, TrajExpr::AnalyticPred(_)));
    }

    // ------------------------------------------------------------------
    // THEN
    // ------------------------------------------------------------------

    #[test]
    fn then_no_gap() {
        let schema = empty_schema();
        let (_, expr) = parse_query(
            "SELECT (a:, b:) WHERE duration(a.x1 < b.x1, 30) THEN duration(a.x1 > b.x1, 30)",
            &schema,
        )
        .unwrap();
        let TrajExpr::Then {
            min_gap, max_gap, ..
        } = expr
        else {
            panic!()
        };
        assert_eq!(min_gap, 0);
        assert_eq!(max_gap, None);
    }

    #[test]
    fn then_with_min_gap() {
        let schema = empty_schema();
        let (_, expr) = parse_query(
            "SELECT (a:, b:) WHERE duration(a.x1 < b.x1, 5) THEN[10] duration(a.x1 > b.x1, 5)",
            &schema,
        )
        .unwrap();
        let TrajExpr::Then {
            min_gap, max_gap, ..
        } = expr
        else {
            panic!()
        };
        assert_eq!(min_gap, 10);
        assert_eq!(max_gap, None);
    }

    #[test]
    fn then_with_range_gap() {
        let schema = empty_schema();
        let (_, expr) = parse_query(
            "SELECT (a:, b:) WHERE a.x1 < b.x1 THEN[5..20] a.x1 > b.x1",
            &schema,
        )
        .unwrap();
        let TrajExpr::Then {
            min_gap, max_gap, ..
        } = expr
        else {
            panic!()
        };
        assert_eq!(min_gap, 5);
        assert_eq!(max_gap, Some(20));
    }

    // ------------------------------------------------------------------
    // Error cases
    // ------------------------------------------------------------------

    #[test]
    fn unknown_var() {
        let schema = empty_schema();
        let err = parse_query("SELECT (a:) WHERE b.x1 < 100", &schema)
            .err()
            .unwrap();
        assert!(matches!(err, DslError::UnknownVar(_)));
    }

    #[test]
    fn unknown_prop() {
        let schema = empty_schema();
        let err = parse_query("SELECT (a: nonexistent=foo) WHERE a.x1 < 100", &schema)
            .err()
            .unwrap();
        assert!(matches!(err, DslError::UnknownProp(_)));
    }

    #[test]
    fn unknown_enum_value() {
        let schema = cityflow_schema();
        let err = parse_query("SELECT (a: car_type=Spaceship) WHERE a.x1 < 100", &schema)
            .err()
            .unwrap();
        assert!(matches!(err, DslError::UnknownEnumValue { .. }));
    }

    #[test]
    fn unsupported_binary_shape() {
        let schema = empty_schema();
        let err = parse_query("SELECT (a:, b:) WHERE a.x1 + b.x1 < 100", &schema)
            .err()
            .unwrap();
        assert!(matches!(err, DslError::UnsupportedComparison(_)));
    }

    // ------------------------------------------------------------------
    // CONTAINS / OVERLAP
    // ------------------------------------------------------------------

    #[test]
    fn contains_takes_two_expressions() {
        let schema = empty_schema();
        let (_, expr) = parse_query(
            "SELECT (a:, b:) WHERE CONTAINS(duration(a.x1 < 10, 30), b.x1 > 5)",
            &schema,
        )
        .unwrap();

        let TrajExpr::Contained { lhs, rhs } = expr else {
            panic!()
        };
        assert!(matches!(*lhs, TrajExpr::Duration { .. }));
        assert!(matches!(*rhs, TrajExpr::AnalyticPred(_)));
    }

    #[test]
    fn overlap_takes_two_expressions() {
        let schema = empty_schema();
        let (_, expr) = parse_query(
            "SELECT (a:, b:) WHERE OVERLAP(duration(a.x1 < 10, 30), b.x1 > 5)",
            &schema,
        )
        .unwrap();

        let TrajExpr::Overlap { lhs, rhs } = expr else {
            panic!()
        };
        assert!(matches!(*lhs, TrajExpr::Duration { .. }));
        assert!(matches!(*rhs, TrajExpr::AnalyticPred(_)));
    }

    #[test]
    fn overlap_is_case_insensitive_and_nests() {
        let schema = empty_schema();
        let (_, expr) = parse_query(
            "SELECT (a:, b:, c:) WHERE overlap(a.x1 < 10, OVERLAP(b.x1 < 10, c.x1 < 10))",
            &schema,
        )
        .unwrap();

        let TrajExpr::Overlap { rhs, .. } = expr else {
            panic!()
        };
        assert!(matches!(*rhs, TrajExpr::Overlap { .. }));
    }

    #[test]
    fn overlap_can_be_sequenced_by_then() {
        // The merged span an OVERLAP reports is what a following THEN sequences from, so
        // the two have to compose without parentheses.
        let schema = empty_schema();
        let (_, expr) = parse_query(
            "SELECT (a:, b:) WHERE OVERLAP(a.x1 < 10, b.x1 < 10) THEN[5..20] a.x1 > 50",
            &schema,
        )
        .unwrap();

        let TrajExpr::Then {
            lhs,
            min_gap,
            max_gap,
            ..
        } = expr
        else {
            panic!()
        };
        assert!(matches!(*lhs, TrajExpr::Overlap { .. }));
        assert_eq!(min_gap, 5);
        assert_eq!(max_gap, Some(20));
    }

    #[test]
    fn a_variable_may_be_named_overlap() {
        // The keyword check only fires when `OVERLAP` is not followed by an identifier
        // character, so a variable spelled `overlapping` still parses as a variable.
        let schema = empty_schema();
        let (vars, _) =
            parse_query("SELECT (overlapping:) WHERE overlapping.x1 < 10", &schema).unwrap();
        assert_eq!(vars.name_of(0), Some("overlapping"));
    }

    // ------------------------------------------------------------------
    // Overtake query (integration)
    // ------------------------------------------------------------------

    #[test]
    fn overtake_query() {
        let schema = cityflow_schema();
        let (selector, expr) = parse_query(
            "SELECT (a: car_type=SUV, b: car_type=Sedan) \
             WHERE duration(a.x1 < b.x1, 30) THEN duration(a.x1 > b.x1, 30)",
            &schema,
        )
        .unwrap();
        assert_eq!(selector.num_slots(), 2);

        let TrajExpr::Then {
            min_gap,
            max_gap,
            lhs,
            rhs,
        } = expr
        else {
            panic!()
        };
        assert_eq!(min_gap, 0);
        assert_eq!(max_gap, None);
        assert!(matches!(*lhs, TrajExpr::Duration { .. }));
        assert!(matches!(*rhs, TrajExpr::Duration { .. }));
    }

    #[test]
    fn parses_unary_bboxexpr_comparison() {
        let (_, expr) = parse_query(
            "SELECT (a:) WHERE a.width + 2 * a.x1 >= 100",
            &empty_schema(),
        )
        .unwrap();
        let TrajExpr::AnalyticPred(pred) = expr else {
            panic!("expected unary analytic")
        };
        let AnalyticalPredKind::Unary {
            expr,
            op,
            threshold,
            slot,
        } = pred.kind
        else {
            panic!("expected Unary");
        };
        assert_eq!(slot, 0);
        assert_eq!(op, CmpOp::Gte);
        assert_eq!(threshold, 100.0);
        assert_eq!(expr, BBoxExpr::width() + BBoxExpr::x1() * 2.0);
    }

    #[test]
    fn parses_binary_bboxexpr_comparison() {
        let (_, expr) = parse_query(
            "SELECT (a:, b:) WHERE a.cx + 5 < b.x1 - b.width",
            &empty_schema(),
        )
        .unwrap();
        let TrajExpr::AnalyticPred(pred) = expr else {
            panic!("expected binary analytic")
        };
        let AnalyticalPredKind::Binary {
            expr,
            op,
            threshold,
            slots,
        } = pred.kind
        else {
            panic!("expected Binary");
        };
        assert_eq!(slots, [0, 1]);
        assert_eq!(op, CmpOp::Lt);
        assert_eq!(threshold, 0.0);
        assert_eq!(expr.lhs, BBoxExpr::center_x() + BBoxExpr::constant(5.0));
        assert_eq!(expr.rhs, BBoxExpr::x1() - BBoxExpr::width());
    }

    #[test]
    fn same_slot_comparison_rewrites_to_unary() {
        let (_, expr) = parse_query("SELECT (a:) WHERE a.x2 - a.x1 > 50", &empty_schema()).unwrap();

        let TrajExpr::AnalyticPred(pred) = expr else {
            panic!("expected unary analytic")
        };
        let AnalyticalPredKind::Unary {
            expr,
            threshold,
            slot,
            ..
        } = pred.kind
        else {
            panic!("expected Unary");
        };
        assert_eq!(slot, 0);
        assert_eq!(expr, BBoxExpr::x2() - BBoxExpr::x1());
        assert_eq!(threshold, 50.0);
    }

    #[test]
    fn inside_uses_structured_unary_pred() {
        let (_, expr) = parse_query(
            "SELECT (a:) WHERE a INSIDE [0, 0, 100, 200]",
            &empty_schema(),
        )
        .unwrap();
        let TrajExpr::AnalyticPred(pred) = expr else {
            panic!("expected unary analytic");
        };
        assert_eq!(pred, expected_whole_box(0.0, 0.0, 100.0, 200.0, 0));
    }

    #[test]
    fn duration_and_then_still_parse() {
        let (_, expr) = parse_query(
            "SELECT (a:, b:) WHERE duration(a.x1 < b.x1, 30) THEN [5..10] duration(a.x1 > b.x1, 30)",
            &empty_schema(),
        )
        .unwrap();
        let TrajExpr::Then {
            lhs,
            rhs,
            min_gap,
            max_gap,
        } = expr
        else {
            panic!("expected then");
        };
        assert!(matches!(*lhs, TrajExpr::Duration { .. }));
        assert!(matches!(*rhs, TrajExpr::Duration { .. }));
        assert_eq!(min_gap, 5);
        assert_eq!(max_gap, Some(10));
    }

    #[test]
    fn unknown_var_errors() {
        let err = parse_query("SELECT (a:) WHERE b.x1 < 100", &empty_schema())
            .err()
            .expect("expected parse error");
        assert!(matches!(err, DslError::UnknownVar(name) if name == "b"));
    }

    // ------------------------------------------------------------------
    // Pred-level AND / OR / NOT
    // ------------------------------------------------------------------

    #[test]
    fn pred_and_unary() {
        let (_, expr) = parse_query(
            "SELECT (a:) WHERE a.x1 > 10 AND a.x2 < 500",
            &empty_schema(),
        )
        .unwrap();
        let TrajExpr::AnalyticPred(pred) = expr else {
            panic!("expected AnalyticPred")
        };
        assert!(matches!(pred.kind, AnalyticalPredKind::And(..)));
    }

    #[test]
    fn pred_or_unary() {
        let (_, expr) =
            parse_query("SELECT (a:) WHERE a.x1 < 10 OR a.x1 > 500", &empty_schema()).unwrap();
        let TrajExpr::AnalyticPred(pred) = expr else {
            panic!("expected AnalyticPred")
        };
        assert!(matches!(pred.kind, AnalyticalPredKind::Or(..)));
    }

    #[test]
    fn pred_not_unary() {
        let (_, expr) = parse_query("SELECT (a:) WHERE NOT a.x1 < 10", &empty_schema()).unwrap();
        let TrajExpr::AnalyticPred(pred) = expr else {
            panic!("expected AnalyticPred")
        };
        assert!(matches!(pred.kind, AnalyticalPredKind::Not(..)));
    }

    #[test]
    fn pred_and_binary() {
        let (_, expr) = parse_query(
            "SELECT (a:, b:) WHERE a.x1 < b.x1 AND a.y1 < b.y1",
            &empty_schema(),
        )
        .unwrap();
        let TrajExpr::AnalyticPred(pred) = expr else {
            panic!("expected AnalyticPred")
        };
        assert!(matches!(pred.kind, AnalyticalPredKind::And(..)));
    }

    #[test]
    fn pred_not_binary() {
        let (_, expr) =
            parse_query("SELECT (a:, b:) WHERE NOT a.x1 < b.x1", &empty_schema()).unwrap();
        let TrajExpr::AnalyticPred(pred) = expr else {
            panic!("expected AnalyticPred")
        };
        assert!(matches!(pred.kind, AnalyticalPredKind::Not(..)));
    }

    #[test]
    fn pred_or_in_duration() {
        let (_, expr) = parse_query(
            "SELECT (a:) WHERE DURATION(a.x1 > 10 OR a.x2 < 500, 30)",
            &empty_schema(),
        )
        .unwrap();
        let TrajExpr::Duration { inner, min_frames } = expr else {
            panic!("expected Duration")
        };
        assert_eq!(min_frames, 30);
        assert!(
            matches!(*inner, TrajExpr::AnalyticPred(ref p) if matches!(p.kind, AnalyticalPredKind::Or(..)))
        );
    }

    #[test]
    fn pred_paren_grouping() {
        // NOT (a.x1 > 10 AND a.x2 < 500)
        let (_, expr) = parse_query(
            "SELECT (a:) WHERE NOT (a.x1 > 10 AND a.x2 < 500)",
            &empty_schema(),
        )
        .unwrap();
        let TrajExpr::AnalyticPred(pred) = expr else {
            panic!("expected AnalyticPred")
        };
        let AnalyticalPredKind::Not(inner) = pred.kind else {
            panic!("expected Not")
        };
        assert!(matches!(inner.kind, AnalyticalPredKind::And(..)));
    }

    #[test]
    fn pred_starting_with_paren_keeps_trailing_boolean() {
        // `(A OR B) AND (C OR D)` — the leading paren must not be mistaken for a
        // parenthesised trajectory expression, which would strand the ` AND …`.
        let (_, expr) = parse_query(
            "SELECT (b:, pl:) WHERE DURATION((b.y1>pl.y1+20 OR pl.y1>b.y1+20) \
             AND (b.x1>pl.x1+20 OR pl.x1>b.x1+20), 1)",
            &empty_schema(),
        )
        .unwrap();
        let TrajExpr::Duration { inner, min_frames } = expr else {
            panic!("expected Duration")
        };
        assert_eq!(min_frames, 1);
        let TrajExpr::AnalyticPred(pred) = *inner else {
            panic!("expected AnalyticPred")
        };
        let AnalyticalPredKind::And(lhs, rhs) = pred.kind else {
            panic!("expected And")
        };
        assert!(matches!(lhs.kind, AnalyticalPredKind::Or(..)));
        assert!(matches!(rhs.kind, AnalyticalPredKind::Or(..)));
    }

    #[test]
    fn paren_trajectory_group_still_parses() {
        // Same leading `(`, but the contents are a trajectory expression — this must
        // still fall through to the trajectory paren rule.
        let (_, expr) = parse_query(
            "SELECT (a:, b:) WHERE (a.x1 < b.x1 THEN a.x1 > b.x1)",
            &empty_schema(),
        )
        .unwrap();
        assert!(matches!(expr, TrajExpr::Then { .. }));
    }

    #[test]
    fn cross_slot_and_parses() {
        // a.x1 > b.x1 AND b.x1 > c.x1 — different slot pairs, now valid
        let (_, expr) = parse_query(
            "SELECT (a:, b:, c:) WHERE a.x1 > b.x1 AND b.x1 > c.x1",
            &empty_schema(),
        )
        .unwrap();
        let TrajExpr::AnalyticPred(pred) = expr else {
            panic!("expected AnalyticPred")
        };
        // Top level is And of two Binary preds with different slots
        let AnalyticalPredKind::And(lhs, rhs) = pred.kind else {
            panic!("expected And")
        };
        let AnalyticalPredKind::Binary {
            slots: lhs_slots, ..
        } = lhs.kind
        else {
            panic!("expected Binary")
        };
        let AnalyticalPredKind::Binary {
            slots: rhs_slots, ..
        } = rhs.kind
        else {
            panic!("expected Binary")
        };
        assert_eq!(lhs_slots, [0, 1]);
        assert_eq!(rhs_slots, [1, 2]);
    }
}
