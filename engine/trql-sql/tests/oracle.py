"""Brute-force reference for the TrQL->SQL compiler.

Deliberately naive: it represents every node's value as an explicit *set of concrete
(s, e) intervals* and computes each operator straight from its definition. Nothing here
uses the 5-tuple (s, e, c, l, r) encoding the compiler emits, so agreement between the two
is real evidence rather than a shared bug.

Reads the same duckdb.db the generated SQL reads.
"""

import itertools
import sys

import duckdb


# -- data -------------------------------------------------------------------------------

def load(db):
    con = duckdb.connect(db, read_only=True)
    boxes = {}          # (fid, oid) -> (x1, y1, x2, y2)
    for fid, oid, x1, y1, x2, y2 in con.execute(
        "SELECT fid, oid, x1, y1, x2, y2 FROM bbox"
    ).fetchall():
        boxes[(fid, oid)] = (x1, y1, x2, y2)
    cls = dict(con.execute("SELECT oid, class FROM objects").fetchall())
    frames = sorted({f for f, _ in boxes})
    con.close()
    return boxes, cls, frames


def coord(box, name):
    x1, y1, x2, y2 = box
    return {
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "cx": (x1 + x2) / 2.0, "cy": (y1 + y2) / 2.0,
        "width": x2 - x1, "height": y2 - y1,
    }[name]


# -- interval sets ----------------------------------------------------------------------
#
# A node's value is a set of (s, e) pairs. A leaf contributes every sub-interval of every
# maximal run where the predicate holds -- that is exactly what `IntervalSet.all_between`
# means, spelled out.

def leaf(truth, frames):
    """truth: dict fid -> bool, over the frames where the tuple co-appears."""
    out = set()
    present = [f for f in frames if f in truth]
    run = []
    runs = []
    for f in present:
        if truth[f] and (not run or f == run[-1] + 1):
            run.append(f)
        elif truth[f]:
            runs.append(run)
            run = [f]
        else:
            if run:
                runs.append(run)
            run = []
    if run:
        runs.append(run)

    for r in runs:
        for i in range(len(r)):
            for j in range(i, len(r)):
                out.add((r[i], r[j]))
    return out


def duration(xs, k):
    return {(s, e) for (s, e) in xs if e - s + 1 >= k}


def then(xs, ys, gmin, gmax):
    out = set()
    for (s1, e1) in xs:
        for (s2, e2) in ys:
            gap = s2 - e1 - 1
            if gap < gmin:
                continue
            if gmax is not None and gap > gmax:
                continue
            out.add((s1, e2))
    return out


def contains(xs, ys):
    out = set()
    for (s1, e1) in xs:
        for (s2, e2) in ys:
            if s1 < s2 and e2 < e1:
                out.add((s1, e1))
                break
    return out


def overlap(xs, ys):
    """Allen's *overlaps*: the second starts strictly inside the first and ends strictly
    after it. The result is the merged span, so unlike `contains` it takes `e2`."""
    out = set()
    for (s1, e1) in xs:
        for (s2, e2) in ys:
            if s1 < s2 <= e1 < e2:
                out.add((s1, e2))
    return out


def expand(s, e, c, l, r):
    """The set of concrete intervals a 5-tuple denotes."""
    return {
        (ss, ee)
        for ss in range(s, l + 1)
        for ee in range(max(r, ss), e + 1)
        if ee - ss + 1 >= c
    }


# -- query evaluation -------------------------------------------------------------------
#
# Queries are given as python closures over a `get(var, coord)` accessor so the oracle
# never has to parse TrQL -- another way it stays independent of the code under test.

def evaluate(node, binding, boxes, frames, gslots):
    """`gslots` are the whole group's slots.

    A leaf is clipped to the frames where the objects *it names* co-appear, not the whole
    group's -- the algebra clips a leaf to its group's bounds only when it names every slot
    of the group, and otherwise to its own slots' co-appearance. Under `CONTAINS` and
    `OVERLAP` the two differ: the operands need not be live at the same time, so a run of
    one object may start before another of the group ever appears. A leaf naming no slot
    falls back to the group.
    """
    kind = node[0]

    if kind == "pred":
        _, slots, fn = node
        own = slots or gslots
        truth = {}
        for f in frames:
            bs = {v: boxes.get((f, binding[v])) for v in own}
            if any(b is None for b in bs.values()):
                continue        # not co-appearing: the tuple has no value on this frame
            truth[f] = fn(lambda v, c: coord(bs[v], c))
        return leaf(truth, frames)

    if kind == "duration":
        return duration(evaluate(node[1], binding, boxes, frames, gslots), node[2])
    if kind == "then":
        return then(
            evaluate(node[1], binding, boxes, frames, gslots),
            evaluate(node[2], binding, boxes, frames, gslots),
            node[3], node[4],
        )
    if kind == "contains":
        return contains(
            evaluate(node[1], binding, boxes, frames, gslots),
            evaluate(node[2], binding, boxes, frames, gslots),
        )
    if kind == "overlap":
        return overlap(
            evaluate(node[1], binding, boxes, frames, gslots),
            evaluate(node[2], binding, boxes, frames, gslots),
        )
    raise ValueError(kind)


def group_slots(node):
    """Slots referenced by a subtree."""
    if node[0] == "pred":
        return set(node[1])
    if node[0] == "duration":
        return group_slots(node[1])
    return group_slots(node[1]) | group_slots(node[2])


def coappears(slots, binding, boxes, frames):
    return any(all((f, binding[v]) in boxes for v in slots) for f in frames)


def spine(node):
    """Flattens the top-level THEN spine into (groups, gaps), mirroring the algebra."""
    if node[0] == "then":
        lg, lgap = spine(node[1])
        rg, rgap = spine(node[2])
        return lg + rg, lgap + [(node[3], node[4])] + rgap
    return [node], []


def run(node, variables, boxes, cls, frames):
    """variables: {name: class or None}. Returns {binding_tuple: set of (s,e)}."""
    names = list(variables)
    candidates = {
        v: [o for o, c in cls.items() if variables[v] is None or c == variables[v]]
        for v in names
    }

    groups, gaps = spine(node)

    results = {}
    for combo in itertools.product(*(candidates[v] for v in names)):
        binding = dict(zip(names, combo))

        # Objects bound to two slots of the *same* group would be a self-join.
        if any(
            len({binding[v] for v in group_slots(g)}) != len(group_slots(g))
            for g in groups
        ):
            continue

        # The selector only enumerates tuples whose every group shares at least one frame.
        # Where a group's leaves all name its whole slot set this is implied by any
        # interval at all; where they name different subsets it is a real constraint.
        if not all(coappears(group_slots(g), binding, boxes, frames) for g in groups):
            continue

        value = evaluate(groups[0], binding, boxes, frames, group_slots(groups[0]))
        for g, (gmin, gmax) in zip(groups[1:], gaps):
            value = then(
                value, evaluate(g, binding, boxes, frames, group_slots(g)), gmin, gmax
            )

        if value:
            results[combo] = value
    return results


def sql_denotation(db, sql):
    """Runs the compiler's SQL and expands its 5-tuples back into concrete intervals."""
    con = duckdb.connect(db, read_only=True)
    rows = con.execute(sql).fetchall()
    con.close()
    out = {}
    for row in rows:
        *oids, s, e, c, l, r = row
        out.setdefault(tuple(oids), set()).update(expand(s, e, c, l, r))
    return out


def compare(name, expected, actual):
    if expected == actual:
        total = sum(len(v) for v in expected.values())
        print(f"  PASS  {name}  ({len(expected)} tuples, {total} intervals)")
        return True

    print(f"  FAIL  {name}")
    for key in sorted(set(expected) | set(actual), key=str):
        exp, act = expected.get(key, set()), actual.get(key, set())
        if exp != act:
            print(f"        tuple {key}")
            missing, extra = sorted(exp - act), sorted(act - exp)
            if missing:
                print(f"          missing {len(missing)}: {missing[:6]}")
            if extra:
                print(f"          extra   {len(extra)}: {extra[:6]}")
    return False


if __name__ == "__main__":
    print(duckdb.__version__, file=sys.stderr)
