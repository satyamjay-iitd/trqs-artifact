"""Differential test: generated SQL vs. the brute-force oracle.

Builds a small synthetic duckdb.db with hand-chosen trajectories (crossings, a detection
gap, a limited appearance window), then for each query compares the *denotation* -- the
full set of concrete (s, e) intervals -- computed two ways:

  * the compiler's SQL, whose 5-tuples are expanded back into concrete intervals
  * `oracle.py`, which enumerates intervals directly and never uses the 5-tuple algebra

Run from the repository root:
  .trqs-pyenv/bin/python engine/trql-sql/tests/differential.py

Requires the Python duckdb package and Rust/Cargo. Pass --real-db PATH to
add the first 60 frames of a converted SoccerNet clip; otherwise only the
self-contained synthetic cases run.
"""

import argparse
import os
from pathlib import Path
import tempfile
import subprocess
import sys

import duckdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oracle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB = None  # Assigned inside a temporary directory by main().
N_FRAMES = 30


def build_db():
    if os.path.exists(DB):
        os.remove(DB)
    con = duckdb.connect(DB)
    con.execute("CREATE TABLE objects (oid BIGINT PRIMARY KEY, class VARCHAR)")
    con.execute(
        "CREATE TABLE bbox (fid BIGINT, oid BIGINT, x1 DOUBLE, y1 DOUBLE, "
        "x2 DOUBLE, y2 DOUBLE, aid BIGINT)"
    )

    objs = [(0, "a"), (1, "b"), (2, "a"), (3, "b"), (4, "c"), (5, "d"), (6, "e")]
    con.executemany("INSERT INTO objects VALUES (?, ?)", objs)

    rows = []
    for f in range(N_FRAMES):
        # o0 moves right, o1 moves left -- they cross near the middle, giving THEN
        # something to chain across.
        rows.append((f, 0, float(f), 5.0, float(f) + 4, 9.0, None))
        rows.append((f, 1, float(N_FRAMES - 1 - f), 5.0, float(N_FRAMES - 1 - f) + 4, 9.0, None))
        # o2 sits still but disappears for frames 12..15 -- a detection gap, which must
        # split intervals rather than being smoothed over.
        if not (12 <= f <= 15):
            rows.append((f, 2, 10.0, 0.0, 14.0, 6.0, None))
        # o3 is stationary and always present.
        rows.append((f, 3, 20.0, 20.0, 26.0, 28.0, None))
        # o4 only ever appears in the first half.
        if f < 15:
            rows.append((f, 4, float(2 * f), 30.0, float(2 * f) + 3, 34.0, None))
        # o5 only appears from frame 8, so it enters while o4 is already mid-run: a
        # leaf over o4 alone must not be clipped to the frames it shares with o5.
        if f >= 8:
            rows.append((f, 5, 10.0, 40.0, 14.0, 44.0, None))
        # o6 only appears from frame 20, after o4 has left: the two never co-appear.
        if f >= 20:
            rows.append((f, 6, 30.0, 40.0, 34.0, 44.0, None))

    con.executemany("INSERT INTO bbox VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    con.close()


def compile_sql(query):
    out = subprocess.run(
        ["cargo", "run", "-q", "-p", "trql-sql", "--example", "compile", "--",
         query, "--tuples"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"compile failed for {query!r}:\n{out.stderr}")
    return out.stdout.strip().rstrip(";")


# Each case pairs the TrQL text with the oracle's independent description of the same
# query. `g(var, coord)` reads a coordinate; `P(slots, fn)` is a pointwise predicate.
def P(slots, fn):
    return ("pred", slots, fn)


CASES = [
    (
        "unary comparison",
        "SELECT (x: class=a) WHERE x.x1 < 10",
        {"x": "a"},
        P(["x"], lambda g: g("x", "x1") < 10),
    ),
    (
        "detection gap splits intervals",
        "SELECT (x: class=a) WHERE x.x1 > 5",
        {"x": "a"},
        P(["x"], lambda g: g("x", "x1") > 5),
    ),
    (
        "binary comparison",
        "SELECT (x: class=a, y: class=b) WHERE x.x1 < y.x1",
        {"x": "a", "y": "b"},
        P(["x", "y"], lambda g: g("x", "x1") < g("y", "x1")),
    ),
    (
        "AND / OR / NOT",
        "SELECT (x: class=a) WHERE (x.x1 > 4 OR x.y1 > 1) AND NOT x.x2 > 20",
        {"x": "a"},
        P(["x"], lambda g: (g("x", "x1") > 4 or g("x", "y1") > 1)
                           and not g("x", "x2") > 20),
    ),
    (
        "derived coords",
        "SELECT (x: class=a) WHERE x.cx > 8 AND x.width < 5",
        {"x": "a"},
        P(["x"], lambda g: g("x", "cx") > 8 and g("x", "width") < 5),
    ),
    (
        "INSIDE",
        "SELECT (x: class=a) WHERE x INSIDE [0, 0, 20, 10]",
        {"x": "a"},
        P(["x"], lambda g: g("x", "x1") >= 0 and g("x", "y1") >= 0
                           and g("x", "x2") <= 20 and g("x", "y2") <= 10),
    ),
    (
        "DURATION",
        "SELECT (x: class=a, y: class=b) WHERE DURATION(x.x1 < y.x1, 6)",
        {"x": "a", "y": "b"},
        ("duration", P(["x", "y"], lambda g: g("x", "x1") < g("y", "x1")), 6),
    ),
    (
        "DURATION over a gap",
        "SELECT (x: class=a) WHERE DURATION(x.x1 > 5, 8)",
        {"x": "a"},
        ("duration", P(["x"], lambda g: g("x", "x1") > 5), 8),
    ),
    (
        "THEN, shared slots, no gap bound",
        "SELECT (x: class=a, y: class=b) WHERE x.x1 < y.x1 THEN x.x1 > y.x1",
        {"x": "a", "y": "b"},
        ("then",
         P(["x", "y"], lambda g: g("x", "x1") < g("y", "x1")),
         P(["x", "y"], lambda g: g("x", "x1") > g("y", "x1")),
         0, None),
    ),
    (
        "THEN with min/max gap",
        "SELECT (x: class=a, y: class=b) WHERE DURATION(x.x1 < y.x1, 4) "
        "THEN[2..6] DURATION(x.x1 > y.x1, 4)",
        {"x": "a", "y": "b"},
        ("then",
         ("duration", P(["x", "y"], lambda g: g("x", "x1") < g("y", "x1")), 4),
         ("duration", P(["x", "y"], lambda g: g("x", "x1") > g("y", "x1")), 4),
         2, 6),
    ),
    (
        "THEN over disjoint slots (cross join, separate groups)",
        "SELECT (x: class=c, y: class=b) WHERE DURATION(x.x1 < 20, 3) "
        "THEN[0..10] DURATION(y.x1 < 10, 3)",
        {"x": "c", "y": "b"},
        ("then",
         ("duration", P(["x"], lambda g: g("x", "x1") < 20), 3),
         ("duration", P(["y"], lambda g: g("y", "x1") < 10), 3),
         0, 10),
    ),
    (
        "THEN chain of three",
        "SELECT (x: class=a, y: class=b) WHERE x.x1 < y.x1 THEN[1..3] x.x1 > y.x1 "
        "THEN[0..5] x.x1 > 20",
        {"x": "a", "y": "b"},
        ("then",
         ("then",
          P(["x", "y"], lambda g: g("x", "x1") < g("y", "x1")),
          P(["x", "y"], lambda g: g("x", "x1") > g("y", "x1")),
          1, 3),
         P(["x"], lambda g: g("x", "x1") > 20),
         0, 5),
    ),
    (
        "CONTAINS",
        "SELECT (x: class=a) WHERE CONTAINS(x.x1 < 25, x.x1 < 8)",
        {"x": "a"},
        ("contains",
         P(["x"], lambda g: g("x", "x1") < 25),
         P(["x"], lambda g: g("x", "x1") < 8)),
    ),
    (
        "CONTAINS with DURATION inside",
        "SELECT (x: class=a) WHERE CONTAINS(DURATION(x.x1 < 25, 5), DURATION(x.x1 < 8, 3))",
        {"x": "a"},
        ("contains",
         ("duration", P(["x"], lambda g: g("x", "x1") < 25), 5),
         ("duration", P(["x"], lambda g: g("x", "x1") < 8), 3)),
    ),
    (
        # Both operands have c = 1, so the `max(..., 3)` floor on `c_out` is what keeps
        # one- and two-frame results out. Dropping it shows up here and nowhere else.
        "OVERLAP",
        "SELECT (x: class=a) WHERE OVERLAP(x.x1 < 20, x.x1 > 10)",
        {"x": "a"},
        ("overlap",
         P(["x"], lambda g: g("x", "x1") < 20),
         P(["x"], lambda g: g("x", "x1") > 10)),
    ),
    (
        # Drives `c₁ + 1` and `c₂ + 1` past the floor, and reaches the `b.s + b.c - 1`
        # term of `r_out` and the `a.e - a.c + 1` term of `l_out`.
        "OVERLAP with DURATION on both sides",
        "SELECT (x: class=a) WHERE OVERLAP(DURATION(x.x1 < 20, 6), DURATION(x.x1 > 10, 4))",
        {"x": "a"},
        ("overlap",
         ("duration", P(["x"], lambda g: g("x", "x1") < 20), 6),
         ("duration", P(["x"], lambda g: g("x", "x1") > 10), 4)),
    ),
    (
        # A THEN as the left operand leaves `l < e`, which is the only way the `a.e - 1`
        # and `b.l - 1` bounds become observable rather than dominated by `a.l`.
        "OVERLAP whose left operand is a THEN",
        "SELECT (x: class=a) WHERE OVERLAP("
        "DURATION(x.x1 < 15, 6) THEN DURATION(x.x1 > 18, 3), DURATION(x.x1 > 10, 4))",
        {"x": "a"},
        ("overlap",
         ("then",
          ("duration", P(["x"], lambda g: g("x", "x1") < 15), 6),
          ("duration", P(["x"], lambda g: g("x", "x1") > 18), 3),
          0, None),
         ("duration", P(["x"], lambda g: g("x", "x1") > 10), 4)),
    ),
    (
        # The merged span an OVERLAP reports has to feed THEN correctly: it ends at the
        # *right* operand's stop, so a following THEN sequences from there.
        "THEN whose right operand is an OVERLAP",
        "SELECT (x: class=a) WHERE x.x1 < 3 THEN "
        "OVERLAP(DURATION(x.x1 < 20, 5), DURATION(x.x1 > 10, 5))",
        {"x": "a"},
        ("then",
         P(["x"], lambda g: g("x", "x1") < 3),
         ("overlap",
          ("duration", P(["x"], lambda g: g("x", "x1") < 20), 5),
          ("duration", P(["x"], lambda g: g("x", "x1") > 10), 5)),
         0, None),
    ),
    (
        # The mirror of the case above, and the only one that reaches the `b.l - 1` term of
        # `l_out`: it binds only when the *right* operand's start bound is tighter than
        # anything the left contributes, which needs a right-hand THEN whose DURATION
        # threshold equals its own run length (driving `l` down to 1 while `e` stays at 29).
        "OVERLAP whose right operand is a THEN",
        "SELECT (x: class=a) WHERE OVERLAP(x.x1 < 25, "
        "DURATION(x.x1 < 11, 10) THEN DURATION(x.x1 > 20, 3))",
        {"x": "a"},
        ("overlap",
         P(["x"], lambda g: g("x", "x1") < 25),
         ("then",
          ("duration", P(["x"], lambda g: g("x", "x1") < 11), 10),
          ("duration", P(["x"], lambda g: g("x", "x1") > 20), 3),
          0, None)),
    ),
    (
        "OVERLAP over two variables",
        "SELECT (x: class=a, y: class=b) WHERE OVERLAP("
        "DURATION(x.x1 < y.x1 + 10, 4), DURATION(x.x1 > y.x1 - 10, 4))",
        {"x": "a", "y": "b"},
        ("overlap",
         ("duration", P(["x", "y"], lambda g: g("x", "x1") < g("y", "x1") + 10), 4),
         ("duration", P(["x", "y"], lambda g: g("x", "x1") > g("y", "x1") - 10), 4)),
    ),
    # -- Leaves naming different slots of one group -------------------------------------
    #
    # Each leaf reads only the frames where the objects it names are present. Before this
    # was fixed, every leaf was clipped to the frames where the *whole* group co-appears,
    # which cut runs short (CLEVRER CL-Q2/CL-Q3) -- each case below fails that way.
    (
        "OVERLAP whose operands name different objects",
        "SELECT (x: class=c, y: class=d) WHERE OVERLAP("
        "DURATION(x.x1 < 25, 4), DURATION(y.x1 < 50, 4))",
        {"x": "c", "y": "d"},
        ("overlap",
         ("duration", P(["x"], lambda g: g("x", "x1") < 25), 4),
         ("duration", P(["y"], lambda g: g("y", "x1") < 50), 4)),
    ),
    (
        # DURATION bounds the union OVERLAP returns. x and y share only frames
        # 8..14, so a 25-frame result must come from the union, not a shared stretch.
        "DURATION over a mixed OVERLAP bounds the union",
        "SELECT (x: class=c, y: class=d) WHERE DURATION(OVERLAP("
        "DURATION(x.x1 < 25, 4), DURATION(y.x1 < 50, 4)), 25)",
        {"x": "c", "y": "d"},
        ("duration",
         ("overlap",
          ("duration", P(["x"], lambda g: g("x", "x1") < 25), 4),
          ("duration", P(["y"], lambda g: g("y", "x1") < 50), 4)),
         25),
    ),
    (
        "CONTAINS whose outer spell names a third object",
        "SELECT (w: class=d, x: class=a, y: class=b) WHERE CONTAINS("
        "DURATION(w.x1 < 50, 10), DURATION(x.x1 < y.x1, 3))",
        {"w": "d", "x": "a", "y": "b"},
        ("contains",
         ("duration", P(["w"], lambda g: g("w", "x1") < 50), 10),
         ("duration", P(["x", "y"], lambda g: g("x", "x1") < g("y", "x1")), 3)),
    ),
    (
        "mixed-slot OVERLAP keeps the group's objects distinct",
        "SELECT (x: class=a, y: class=a) WHERE OVERLAP("
        "DURATION(x.x1 < 20, 3), DURATION(y.x1 > 5, 3))",
        {"x": "a", "y": "a"},
        ("overlap",
         ("duration", P(["x"], lambda g: g("x", "x1") < 20), 3),
         ("duration", P(["y"], lambda g: g("y", "x1") > 5), 3)),
    ),
    (
        # x and y never share a frame, so although THEN alone would chain them, the group
        # {w, x, y} has no co-appearance and the selector never enumerates it.
        "nested THEN under CONTAINS needs the group to co-appear",
        "SELECT (w: class=a, x: class=c, y: class=e) WHERE CONTAINS("
        "w.x1 < 100, DURATION(x.x1 < 100, 2) THEN DURATION(y.x1 < 100, 2))",
        {"w": "a", "x": "c", "y": "e"},
        ("contains",
         P(["w"], lambda g: g("w", "x1") < 100),
         ("then",
          ("duration", P(["x"], lambda g: g("x", "x1") < 100), 2),
          ("duration", P(["y"], lambda g: g("y", "x1") < 100), 2),
          0, None)),
    ),
    (
        "nested THEN under CONTAINS, co-appearing group",
        "SELECT (w: class=a, x: class=c, y: class=d) WHERE CONTAINS("
        "w.x1 < 100, DURATION(x.x1 < 100, 2) THEN DURATION(y.x1 < 100, 2))",
        {"w": "a", "x": "c", "y": "d"},
        ("contains",
         P(["w"], lambda g: g("w", "x1") < 100),
         ("then",
          ("duration", P(["x"], lambda g: g("x", "x1") < 100), 2),
          ("duration", P(["y"], lambda g: g("y", "x1") < 100), 2),
          0, None)),
    ),
    (
        "nested THEN under DURATION (stays one group)",
        "SELECT (x: class=a, y: class=b) WHERE DURATION(x.x1 < y.x1 THEN x.x1 > y.x1, 10)",
        {"x": "a", "y": "b"},
        ("duration",
         ("then",
          P(["x", "y"], lambda g: g("x", "x1") < g("y", "x1")),
          P(["x", "y"], lambda g: g("x", "x1") > g("y", "x1")),
          0, None),
         10),
    ),
    (
        "unconstrained variables (all classes are candidates)",
        "SELECT (x:, y:) WHERE DURATION(x.y1 < y.y1, 5)",
        {"x": None, "y": None},
        ("duration", P(["x", "y"], lambda g: g("x", "y1") < g("y", "y1")), 5),
    ),
    (
        # Holds when x and y are the same object, so it fails loudly if the generated SQL
        # forgets that two slots of one group may not bind the same object.
        "self-join must be excluded",
        "SELECT (x:, y:) WHERE DURATION(x.x1 < y.x1 + 100, 5)",
        {"x": None, "y": None},
        ("duration", P(["x", "y"], lambda g: g("x", "x1") < g("y", "x1") + 100), 5),
    ),
    (
        # o2 is absent for frames 12..15, so `x1 > 5` gives it the runs [0,11] and [16,29].
        # A threshold of exactly 12 puts [0,11] precisely on the boundary, catching an
        # off-by-one in the length filter.
        "DURATION threshold exactly equals an interval length",
        "SELECT (x: class=a) WHERE DURATION(x.x1 > 5, 12)",
        {"x": "a"},
        ("duration", P(["x"], lambda g: g("x", "x1") > 5), 12),
    ),
    (
        # Likewise on the other side of the boundary.
        "DURATION threshold one above an interval length",
        "SELECT (x: class=a) WHERE DURATION(x.x1 > 5, 13)",
        {"x": "a"},
        ("duration", P(["x"], lambda g: g("x", "x1") > 5), 13),
    ),
    (
        # Exercises the exact gap bounds: the crossing produces adjacent runs, so a gap
        # window that just includes or just excludes them is sensitive to off-by-ones.
        "THEN with a zero-width gap window",
        "SELECT (x: class=a, y: class=b) WHERE x.x1 < y.x1 THEN[0..0] x.x1 > y.x1",
        {"x": "a", "y": "b"},
        ("then",
         P(["x", "y"], lambda g: g("x", "x1") < g("y", "x1")),
         P(["x", "y"], lambda g: g("x", "x1") > g("y", "x1")),
         0, 0),
    ),
    (
        # Right-nested, so the outer THEN's right operand is itself a THEN and therefore has
        # l < e. That is the only situation where the `L2 - c1 - gmin` term in the `l` bound
        # binds; with a flat left-leaning spine every operand is a leaf with l == e.
        "right-nested THEN",
        "SELECT (x: class=a) WHERE DURATION(x.x1 > 3, 3) THEN[0..20] "
        "(DURATION(x.x1 > 8, 3) THEN[0..5] DURATION(x.x1 > 20, 3))",
        {"x": "a"},
        ("then",
         ("duration", P(["x"], lambda g: g("x", "x1") > 3), 3),
         ("then",
          ("duration", P(["x"], lambda g: g("x", "x1") > 8), 3),
          ("duration", P(["x"], lambda g: g("x", "x1") > 20), 3),
          0, 5),
         0, 20),
    ),
    (
        # CONTAINS also yields l < e, reaching the same term by a different route.
        "THEN whose right operand is a CONTAINS",
        "SELECT (x: class=a) WHERE DURATION(x.x1 > 3, 2) THEN[0..25] "
        "CONTAINS(DURATION(x.x1 > 8, 4), DURATION(x.x1 > 20, 2))",
        {"x": "a"},
        ("then",
         ("duration", P(["x"], lambda g: g("x", "x1") > 3), 2),
         ("contains",
          ("duration", P(["x"], lambda g: g("x", "x1") > 8), 4),
          ("duration", P(["x"], lambda g: g("x", "x1") > 20), 2)),
         0, 25),
    ),
    (
        # Reaches the `L2 - c1 - gmin` term in THEN's `l` bound *observably*. That needs a
        # right operand with `e >= l + c`, which the inner THEN supplies: a DURATION whose
        # threshold equals its own run length drives `l` down to near `s` while `e` stays
        # far out. Without this case, dropping that term from the formula goes unnoticed.
        "THEN reaching the L2 term in the start bound",
        "SELECT (x: class=a) WHERE x.x1 < 100 THEN "
        "(DURATION(x.x1 < 11, 10) THEN DURATION(x.x1 > 20, 3))",
        {"x": "a"},
        ("then",
         P(["x"], lambda g: g("x", "x1") < 100),
         ("then",
          ("duration", P(["x"], lambda g: g("x", "x1") < 11), 10),
          ("duration", P(["x"], lambda g: g("x", "x1") > 20), 3),
          0, None),
         0, None),
    ),
    (
        "limited appearance window",
        "SELECT (x: class=c, y: class=b) WHERE x.x1 < y.x1",
        {"x": "c", "y": "b"},
        P(["x", "y"], lambda g: g("x", "x1") < g("y", "x1")),
    ),
]


REAL_SRC = None  # Optional --real-db input.
REAL_DB = None
REAL_FRAMES = 60


def build_real_db():
    """A frame-limited slice of a real clip, so brute force stays tractable."""
    if REAL_SRC is None:
        return False
    src = REAL_SRC
    if not os.path.isfile(src):
        raise FileNotFoundError(src)
    if os.path.exists(REAL_DB):
        os.remove(REAL_DB)
    con = duckdb.connect(REAL_DB)
    con.execute(f"ATTACH '{src}' AS s (READ_ONLY)")
    con.execute(
        "CREATE TABLE objects AS SELECT * FROM s.objects "
        "WHERE class IN ('ball', 'player_left', 'referee')"
    )
    con.execute(
        f"CREATE TABLE bbox AS SELECT * FROM s.bbox WHERE fid < {REAL_FRAMES} "
        "AND oid IN (SELECT oid FROM objects)"
    )
    con.close()
    return True


REAL_CASES = [
    (
        "real: ball near a left player",
        "SELECT (b: class=ball, p: class=player_left) WHERE "
        "DURATION(b.cx < p.cx + 60 AND p.cx < b.cx + 60, 4)",
        {"b": "ball", "p": "player_left"},
        ("duration",
         P(["b", "p"], lambda g: g("b", "cx") < g("p", "cx") + 60
                                 and g("p", "cx") < g("b", "cx") + 60), 4),
    ),
    (
        "real: crossing then separating",
        "SELECT (b: class=ball, p: class=player_left) WHERE "
        "DURATION(b.x1 > p.x1, 3) THEN[0..20] DURATION(b.x1 < p.x1, 3)",
        {"b": "ball", "p": "player_left"},
        ("then",
         ("duration", P(["b", "p"], lambda g: g("b", "x1") > g("p", "x1")), 3),
         ("duration", P(["b", "p"], lambda g: g("b", "x1") < g("p", "x1")), 3),
         0, 20),
    ),
    (
        "real: disjoint groups across a THEN",
        "SELECT (r: class=referee, b: class=ball) WHERE "
        "DURATION(r.y1 < 300, 3) THEN[0..40] DURATION(b.y1 < 900, 3)",
        {"r": "referee", "b": "ball"},
        ("then",
         ("duration", P(["r"], lambda g: g("r", "y1") < 300), 3),
         ("duration", P(["b"], lambda g: g("b", "y1") < 900), 3),
         0, 40),
    ),
    (
        "real: containment",
        "SELECT (p: class=player_left) WHERE "
        "CONTAINS(DURATION(p.y2 > 300, 5), DURATION(p.y2 > 500, 2))",
        {"p": "player_left"},
        ("contains",
         ("duration", P(["p"], lambda g: g("p", "y2") > 300), 5),
         ("duration", P(["p"], lambda g: g("p", "y2") > 500), 2)),
    ),
]


def run_cases():
    build_db()
    boxes, cls, frames = oracle.load(DB)

    passed = failed = 0
    for name, query, variables, node in CASES:
        expected = oracle.run(node, variables, boxes, cls, frames)
        actual = oracle.sql_denotation(DB, compile_sql(query))
        # The oracle keys tuples by variable order, which is declaration order -- the same
        # order the SQL projects its oid columns.
        if oracle.compare(name, expected, actual):
            passed += 1
        else:
            failed += 1
            print(f"        query: {query}")

    if build_real_db():
        print(f"\n-- real clip, first {REAL_FRAMES} frames --")
        rboxes, rcls, rframes = oracle.load(REAL_DB)
        for name, query, variables, node in REAL_CASES:
            expected = oracle.run(node, variables, rboxes, rcls, rframes)
            actual = oracle.sql_denotation(REAL_DB, compile_sql(query))
            if oracle.compare(name, expected, actual):
                passed += 1
            else:
                failed += 1
                print(f"        query: {query}")
    else:
        print("\n(skipped optional real-clip cases: supply --real-db PATH)")

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


def main():
    global DB, REAL_DB, REAL_SRC
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-db", type=Path,
                        help="converted SoccerNet duckdb.db with ball/player_left/referee classes")
    args = parser.parse_args()
    REAL_SRC = str(args.real_db.resolve()) if args.real_db else None
    with tempfile.TemporaryDirectory(prefix="trql-sql-differential-") as tmp:
        DB = os.path.join(tmp, "synthetic.db")
        REAL_DB = os.path.join(tmp, "real.db")
        return run_cases()


if __name__ == "__main__":
    sys.exit(main())
