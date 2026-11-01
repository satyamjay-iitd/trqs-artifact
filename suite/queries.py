#!/usr/bin/env python3
"""The 24-query evaluation workload.

Six queries per dataset over soccernet, i24_motion, clevrer_merged and nuscenes,
spanning 1 to 4 bound objects and exercising THEN, DURATION, CONTAINS and
OVERLAP. Each `Query` carries everything the paper table needs (arity, operators,
prose) beside the DSL itself, so `tables/workload.tex` is generated rather than
maintained by hand.

Two properties of the language shape how these are written, and both are
load-bearing:

*No distance primitive.* `AnalyticalPred` compares affine combinations of one
frame's coordinates, so "within N metres of" is not expressible. Proximity is
written as a bounded coordinate difference (`a.cx < b.cx - d`), and "player p has
the ball" as the ball's centre lying inside p's box.

*No variable-inequality constraint.* Two slots in the same appearance group can
never bind the same object -- the selector rejects that as a self-join -- but two
slots separated by a top-level THEN sit in different groups and may. So a query
that needs two *distinct* same-kind objects either keeps them inside one group
(CONTAINS, OVERLAP, DURATION all do) or distinguishes them by a property. That is
why SN-Q3 is a turnover between teams rather than a pass within one: the pass
version admits a player passing to themselves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from frames import Frame
from scenes import scene_classes, scene_regions


@dataclass
class Query:
    id: str
    dataset: str
    objects: int
    ops: list[str]
    desc: str
    #: The DSL, or a `(dataset, sub) -> DSL` builder for a query whose regions
    #: are per-sub-dataset. nuScenes needs the second form: every scene sits on a
    #: different patch of the map raster, so a region written once would be off
    #: the edge of nearly every scene.
    dsl: str | Callable[[str, str], str]
    #: Sub-datasets to run on; `None` means every sub-dataset of the family.
    subs: list[str] | None = None
    #: Object classes the query constrains on. A sub-dataset that lacks one is
    #: skipped rather than run: the engine rejects an unknown enum value as a
    #: parse error, so an absent class fails the run instead of matching nothing.
    requires: list[str] = field(default_factory=list)
    notes: str = ""

    def dsl_for(self, sub: str) -> str:
        """This query's DSL as it should run against `sub`."""
        return self.dsl if isinstance(self.dsl, str) else self.dsl(self.dataset, sub)

    def runs_on(self, sub: str) -> bool:
        """Whether `sub` carries every class this query constrains on."""
        if not self.requires:
            return True
        return set(self.requires) <= scene_classes(self.dataset, sub)


# --------------------------------------------------------------------------
# SoccerNet -- 1920x1080 image pixels, 25 fps, 750-frame clips, y starts at 167.
# --------------------------------------------------------------------------
SN = Frame()  # plain pixels
LEFT, MID, RIGHT = "[0,167,640,1080]", "[640,167,1280,1080]", "[1280,167,1920,1080]"


def _has_ball(ball: str, player: str) -> str:
    """`player` is in possession: the ball's centre lies inside its box."""
    return (f"({ball}.cx > {player}.x1 AND {ball}.cx < {player}.x2 AND "
            f"{ball}.cy > {player}.y1 AND {ball}.cy < {player}.y2)")


SOCCERNET = [
    Query(
        "SN-Q1", "soccernet", 1, ["THEN"],
        "The ball crosses the frame: left third to right third within 3 s.",
        f"REGIONS (l = {LEFT}, r = {RIGHT}) "
        "SELECT (b: class=ball) "
        "WHERE b.center INSIDE l THEN[0..75] b.center INSIDE r",
    ),
    Query(
        "SN-Q2", "soccernet", 2, ["DURATION", "THEN"],
        "Two opposing players swap left/right order, each side held for 1 s.",
        "SELECT (a: class=player_left, b: class=player_right) "
        "WHERE DURATION(a.cx < b.cx, 25) THEN DURATION(a.cx > b.cx, 25)",
    ),
    Query(
        "SN-Q3", "soccernet", 3, ["THEN"],
        "Turnover: a left-team player holds the ball, then a right-team player "
        "does within 2 s.",
        "SELECT (bl: class=ball, p1: class=player_left, p2: class=player_right) "
        f"WHERE {_has_ball('bl', 'p1')} THEN[0..50] {_has_ball('bl', 'p2')}",
        notes="Opposing classes make p1 != p2 by construction; see module docstring.",
    ),
    Query(
        "SN-Q4", "soccernet", 3, ["CONTAINS", "DURATION"],
        "The referee moves between an opposing pair, strictly inside a 2 s spell "
        "in which that pair holds its left-right ordering.",
        "SELECT (a: class=player_left, b: class=player_right, r: class=referee) "
        "WHERE CONTAINS(DURATION(a.cx < b.cx, 50), "
        "DURATION(r.cx > a.cx AND r.cx < b.cx, 10))",
        notes="Possession is not the outer interval here because it cannot be: a "
              "ball centre stays inside a player's box for at most ~5 frames "
              "(measured), so no CONTAINS can be built on top of it.",
    ),
    Query(
        "SN-Q5", "soccernet", 2, ["OVERLAP", "DURATION"],
        "Two same-team players' 1 s spells in the right third genuinely overlap "
        "-- neither contains the other.",
        f"REGIONS (box = {RIGHT}) "
        "SELECT (a: class=player_left, b: class=player_left) "
        "WHERE OVERLAP(DURATION(a.center INSIDE box, 25), "
        "DURATION(b.center INSIDE box, 25))",
        notes="One appearance group, so a != b is enforced by the selector.",
    ),
    Query(
        "SN-Q6", "soccernet", 4, ["THEN", "CONTAINS", "DURATION"],
        "Attack develops: the referee steps between an opposing pair during a 2 s "
        "spell of their ordering, and the ball then reaches the right third.",
        f"REGIONS (box = {RIGHT}) "
        "SELECT (a: class=player_left, b: class=player_right, r: class=referee, "
        "bl: class=ball) "
        "WHERE CONTAINS(DURATION(a.cx < b.cx, 50), "
        "DURATION(r.cx > a.cx AND r.cx < b.cx, 10)) "
        "THEN[0..75] DURATION(bl.center INSIDE box, 10)",
        notes="Highest arity in the suite; all four slots carry distinct classes, "
              "so the tuple is distinct by construction even across the THEN.",
    ),
]


# --------------------------------------------------------------------------
# CLEVRER -- 480x320 canvas, 25 fps. `clevrer_merged` concatenates the 10,000
# training videos into one database as 128-frame blocks, so every THEN gap is
# bounded below 128: an object lives in exactly one block, which already makes a
# cross-video match impossible, but a bounded window says so in the query rather
# than relying on the layout.
# --------------------------------------------------------------------------
CL_LEFT, CL_MID, CL_RIGHT = "[0,0,160,320]", "[160,0,320,320]", "[320,0,480,320]"

#: Two boxes intersect -- the closest thing to "touching" the language can state,
#: since there is no distance primitive.
_BOXES_TOUCH = "a.x1 < b.x2 AND a.x2 > b.x1 AND a.y1 < b.y2 AND a.y2 > b.y1"

#: `|a.cx - b.cx| > 45` and its negation. The absolute-difference form the draft
#: workload table prints does not parse -- a bbox expression may not mix two
#: variables on one side -- so the runnable spelling is a pair of one-sided
#: bounds combined with OR / AND.
_FAR = "a.cx > b.cx + 45 OR a.cx < b.cx - 45"
_NEAR = "a.cx <= b.cx + 45 AND a.cx >= b.cx - 45"

CLEVRER = [
    Query(
        "CL-Q1", "clevrer_merged", 2, ["DURATION", "THEN"],
        "Collision proxy: two objects' boxes intersect, then separate with a "
        "fully clear of b.",
        f"SELECT (a:, b:) WHERE DURATION({_BOXES_TOUCH}, 3) "
        "THEN[0..60] DURATION(a.x2 < b.x1, 3)",
    ),
    Query(
        "CL-Q2", "clevrer_merged", 3, ["CONTAINS", "DURATION"],
        "Two objects meet strictly inside a spell where a third sits in the "
        "centre band.",
        f"REGIONS (c = {CL_MID}) SELECT (a:, b:, w:) "
        f"WHERE CONTAINS(DURATION(w.center INSIDE c, 25), "
        f"DURATION({_BOXES_TOUCH}, 3))",
    ),
    Query(
        "CL-Q3", "clevrer_merged", 2, ["OVERLAP", "DURATION"],
        "Two objects' spells in the centre band genuinely overlap -- neither "
        "contains the other.",
        f"REGIONS (c = {CL_MID}) SELECT (a:, b:) "
        "WHERE OVERLAP(DURATION(a.center INSIDE c, 10), "
        "DURATION(b.center INSIDE c, 10))",
    ),
    Query(
        "CL-Q4", "clevrer_merged", 1, ["THEN"],
        "One object traverses the canvas: left band, centre band, right band.",
        f"REGIONS (l = {CL_LEFT}, c = {CL_MID}, r = {CL_RIGHT}) SELECT (a:) "
        "WHERE a.center INSIDE l THEN[0..60] a.center INSIDE c "
        "THEN[0..60] a.center INSIDE r",
    ),
    Query(
        "CL-Q5", "clevrer_merged", 2, ["DURATION", "THEN"],
        "TQ1D retargeted: a metal cube and a rubber sphere separate, approach, "
        "then separate again.",
        "SELECT (a: class=\"gray metal cube\", b: class=\"red rubber sphere\") "
        f"WHERE DURATION({_FAR}, 5) THEN[0..60] DURATION({_NEAR}, 1) "
        f"THEN[0..60] DURATION({_FAR}, 1)",
        notes="Same shape as the draft table's Q4 so old and new numbers compare, "
              "but written in the syntax the parser accepts.",
    ),
    Query(
        "CL-Q6", "clevrer_merged", 4, ["CONTAINS", "THEN", "DURATION"],
        "Arity stress: two objects meet inside a third's centre-band spell, then "
        "a fourth reaches the right band.",
        f"REGIONS (c = {CL_MID}, r = {CL_RIGHT}) SELECT (a:, b:, w:, z:) "
        f"WHERE CONTAINS(DURATION(w.center INSIDE c, 25), "
        f"DURATION({_BOXES_TOUCH}, 3)) "
        "THEN[0..60] DURATION(z.center INSIDE r, 10)",
        notes="BROKEN -- do not use this timing. `z` is introduced only after the "
              "top-level THEN, so it sits in a second appearance group that never "
              "has to co-appear with a/b/w. On clevrer_merged, where 10,000 videos "
              "are concatenated as 128-frame blocks, that lets a match pair objects "
              "from *different videos*: 126,322 of 247,640 result spans (51%) do. "
              "Bounding the THEN gap below 128 does not help -- any positive gap "
              "crosses near a block boundary. Rewrite with all four slots in one "
              "appearance group, which forces co-appearance and hence one video.",
    ),
]


# --------------------------------------------------------------------------
# i24_motion -- roadway feet, 25 fps, 830,366 objects over 280M boxes on one
# 4 h sub-dataset. The scale story of the suite.
#
# Lane geometry is measured, not assumed: a 2 ft histogram of centre-y over the
# first hour (79.3M boxes) shows two carriageways of four travel lanes plus a
# shoulder/HOV band, spaced ~12.5 ft, with clean valleys between them. Lane
# centres come out at y = 15, 27, 41, 53, 65 ft eastbound and their negatives
# westbound; the bands below are the midpoint boundaries.
#
# Bands are written in FEET and converted by the sub-dataset's own recorded
# frame (`frames.Frame`), because the database stores centi-feet shifted
# non-negative -- lane 1 eastbound is 21100..22200 in stored units, which is not
# a number anyone can review.
# --------------------------------------------------------------------------
I24 = Frame.of("i24_motion", "post1")
X0, X1 = 309000.0, 333000.0  # full 7 km of instrumented freeway, in feet

#: (name, y_lo_ft, y_hi_ft) per travel lane, ordered from the median outwards.
I24_LANES_EB = [("eb1", 11, 22), ("eb2", 22, 34.5), ("eb3", 34.5, 46.5),
                ("eb4", 46.5, 60), ("eb5", 60, 72)]
I24_LANES_WB = [("wb1", -23, -12), ("wb2", -35, -23), ("wb3", -47, -35),
                ("wb4", -61, -47), ("wb5", -72, -61)]


def _i24_regions(*names: str) -> str:
    """A REGIONS clause for the named lanes, feet in, stored units out."""
    bands = {n: (lo, hi) for n, lo, hi in I24_LANES_EB + I24_LANES_WB}
    parts = [f"{n} = {I24.rect(X0, bands[n][0], X1, bands[n][1])}" for n in names]
    return "REGIONS (" + ", ".join(parts) + ") "


#: The eastbound carriageway as a whole: everything on the positive-y side of
#: the median. Used to keep two-vehicle queries within one direction of travel.
EB_ALL = I24.rect(X0, 11, X1, 72)

def _follows(a: str, b: str, max_gap_ft: float | None = None) -> str:
    """`a` is behind `b`, optionally within `max_gap_ft` of following distance.

    Written on the box *edges*, not the centres: a semi is 70.3 ft long against a
    sedan's 17.7 ft (measured), so a centre-to-centre bound means something
    different for every pair of classes -- and a "20 ft" centre separation
    between a semi and a van is not close following, it is the two boxes
    overlapping. `a.x2` to `b.x1` is the gap between a's front bumper and b's
    rear bumper whatever the vehicles are.
    """
    behind = f"{a}.x2 < {b}.x1"
    if max_gap_ft is None:
        return behind
    return f"{behind} AND {a}.x2 > {b}.x1 - {round(max_gap_ft * I24.scale)}"

# Every multi-object i24 query constrains its slots by vehicle class, and that is
# not decoration -- it is what makes them runnable. The selector enumerates
# candidate tuples *before* any predicate filters them, so cost is linear in the
# product of the per-slot cardinalities. Measured on a 10-minute slice, same
# query, classes varied: 13.6k pairs -> 1.4 ms, 585k -> 95 ms, 3.5M -> 584 ms,
# i.e. ~0.17 us per candidate pair regardless. Unconstrained on the full post1
# that product is 830,366^2 ~ 7e11 pairs, which extrapolates to ~32 hours and was
# confirmed by a run that was still going at 15 minutes.
#
# Adding a locality predicate ("both within 200 ft") does not help and the same
# reasoning says why: a vehicle four miles away is still an enumerated candidate.
# Only cardinality helps. `semi` is 8% of the database and `van`/`truck` under
# 2%, so a class-constrained pair is 2-3 orders of magnitude cheaper -- and "a
# semi overtakes a pickup" is the interesting query anyway.

#: i24 queries run on the 10-minute slices `slice_i24.py` cuts, not on the 4-hour
#: `post1`. Two reasons, and only the first is about cost: (1) a multi-object query
#: on post1 enumerates a candidate count that is the product of two 10^5-scale
#: cardinalities, which the table above prices at hours; (2) reporting a
#: distribution over 24 slices is how soccernet and nuscenes are reported, so i24
#: stops being the one dataset summarised by a single number. The full-post1
#: single-object figure is kept separately, as the scale row.
I24_SLICES = [f"post1_m{i:02d}" for i in range(24)]

I24_MOTION = [
    Query(
        "I24-Q1", "i24_motion", 1, ["DURATION", "THEN"],
        "A semi changes lanes: 2 s settled in eastbound lane 1, then 2 s "
        "settled in lane 2 within the next 2 s.",
        _i24_regions("eb1", "eb2") + "SELECT (v: class=semi) "
        "WHERE DURATION(v.center INSIDE eb1, 50) "
        "THEN[0..50] DURATION(v.center INSIDE eb2, 50)",
        subs=I24_SLICES,
    ),
    Query(
        "I24-Q2", "i24_motion", 2, ["DURATION", "THEN"],
        "Overtake within the eastbound carriageway: a semi trails a pickup for "
        "2 s, then leads it for 2 s.",
        f"REGIONS (eb = {EB_ALL}) SELECT (a: class=semi, b: class=pickup) "
        "WHERE DURATION(a.cx < b.cx AND a.center INSIDE eb AND b.center INSIDE eb, 50) "
        "THEN[0..250] DURATION(a.cx > b.cx AND a.center INSIDE eb AND b.center INSIDE eb, 50)",
        subs=I24_SLICES,
    ),
    Query(
        "I24-Q3", "i24_motion", 2, ["CONTAINS", "DURATION"],
        "Cut-in: a van's 2 s spell in eastbound lane 2 sits strictly inside a 4 s "
        "spell of a semi following it.",
        _i24_regions("eb2") + "SELECT (a: class=semi, b: class=van) "
        f"WHERE CONTAINS(DURATION({_follows('a', 'b')}, 100), "
        "DURATION(b.center INSIDE eb2, 50))",
        subs=I24_SLICES,
    ),
    Query(
        "I24-Q4", "i24_motion", 3, ["OVERLAP", "DURATION"],
        "Platoon forms and breaks: a semi's and a van's 2 s spells in eastbound "
        "lane 1 overlap without either containing the other, while a pickup holds "
        "lane 2 throughout.",
        _i24_regions("eb1", "eb2") + "SELECT (a: class=semi, b: class=van, c: class=pickup) "
        "WHERE OVERLAP(DURATION(a.center INSIDE eb1 AND c.center INSIDE eb2, 50), "
        "DURATION(b.center INSIDE eb1 AND c.center INSIDE eb2, 50))",
        subs=I24_SLICES,
    ),
    Query(
        "I24-Q5", "i24_motion", 3, ["THEN", "DURATION"],
        "Weaving: a semi settles in lane 1 behind a van, then in lane 2 behind a "
        "pickup.",
        _i24_regions("eb1", "eb2") + "SELECT (v: class=semi, a: class=van, c: class=pickup) "
        f"WHERE DURATION(v.center INSIDE eb1 AND {_follows('v', 'a')}, 50) "
        f"THEN[0..250] DURATION(v.center INSIDE eb2 AND {_follows('v', 'c')}, 50)",
        subs=I24_SLICES,
    ),
    Query(
        "I24-Q6", "i24_motion", 2, ["DURATION"],
        "Tailgating: a sedan follows a midsize at under 50 ft of clear road for "
        "3 s straight, both in eastbound lane 1.",
        _i24_regions("eb1") + "SELECT (a: class=sedan, b: class=midsize) "
        f"WHERE DURATION({_follows('a', 'b', 50)} "
        "AND a.center INSIDE eb1 AND b.center INSIDE eb1, 75)",
        subs=I24_SLICES,
        notes="Rebuilt after the first version returned 0 on all 24 slices, for "
              "three independent reasons, each measured: (1) it bounded the "
              "centre separation at 20 ft between a 70 ft semi and a 21 ft van, "
              "which is not tailgating but interpenetration; (2) it asked for 10 s "
              "of continuous co-presence when a semi's median trajectory is 3.2 s "
              "and only 211 of 2 066 reach 10 s; (3) only 2 523 frames of a slice "
              "have a semi and a van in lane 1 together at all. sedan and midsize "
              "are the two commonest classes and the ones that actually tailgate.",
    ),
]


# --------------------------------------------------------------------------
# nuScenes -- 850 scenes, 12 fps, ~233 frames (20 s) each, annotations already
# in the map frame (metres x 100) rather than the moving camera's. That is the
# point of including it: the queries below are ego-independent, so the moving
# camera costs nothing at query time.
#
# Two things are per-scene rather than global. Regions: each scene sits on its
# own patch of a city-sized raster, so `near`/`mid`/`far` are cut from the
# scene's own extent along its dominant axis of motion (see `scenes.py`).
# Classes: a scene carries only what was in view, and the engine rejects an
# unknown enum value as a *parse error*, so each query declares what it needs
# and the driver skips scenes without it.
# --------------------------------------------------------------------------
CAR, TRUCK = "vehicle.car", "vehicle.truck"
PED = "human.pedestrian.adult"
CONE = "movable_object.trafficcone"

SEC = 12  # frames per second


def _ns(build):
    """Wrap a builder so it receives the scene's generated regions."""
    return lambda dataset, sub: build(scene_regions(dataset, sub))


NUSCENES = [
    Query(
        "NS-Q1", "nuscenes", 2, ["THEN", "DURATION"],
        "A car passes a traffic cone: 1 s on one side of it, then 1 s on the "
        "other, within 10 s.",
        _ns(lambda r: (
            f'SELECT (c: class="{CAR}", k: class="{CONE}") '
            f"WHERE DURATION(c.cx < k.cx, {SEC}) "
            f"THEN[0..{10 * SEC}] DURATION(c.cx > k.cx, {SEC})"
        )),
        requires=[CAR, CONE],
        notes="Purely relational, so it does not depend on the scene's generated "
              "bands -- a cone is roadside furniture and often sits in whichever "
              "band the car never leaves.",
    ),
    Query(
        "NS-Q2", "nuscenes", 3, ["CONTAINS", "DURATION"],
        "A pedestrian crosses in front of a car strictly inside a 2 s spell in "
        "which a truck holds station in the mid band.",
        _ns(lambda r: (
            f'REGIONS (mid = {r["mid"]}) '
            f'SELECT (c: class="{CAR}", p: class="{PED}", t: class="{TRUCK}") '
            f"WHERE CONTAINS(DURATION(t.center INSIDE mid, {2 * SEC}), "
            f"DURATION(p.cx > c.x1 AND p.cx < c.x2, {SEC // 2}))"
        )),
        requires=[CAR, PED, TRUCK],
    ),
    Query(
        "NS-Q3", "nuscenes", 2, ["OVERLAP", "DURATION"],
        "A truck's and a car's 1 s spells in the mid band genuinely overlap -- "
        "neither contains the other.",
        _ns(lambda r: (
            f'REGIONS (mid = {r["mid"]}) '
            f'SELECT (t: class="{TRUCK}", c: class="{CAR}") '
            f"WHERE OVERLAP(DURATION(t.center INSIDE mid, {SEC}), "
            f"DURATION(c.center INSIDE mid, {SEC}))"
        )),
        requires=[CAR, TRUCK],
    ),
    Query(
        "NS-Q4", "nuscenes", 1, ["THEN"],
        "A car advances a full band along the scene's dominant axis: half a "
        "second settled in the near band, then in the mid band.",
        _ns(lambda r: (
            f'REGIONS (near = {r["near"]}, mid = {r["mid"]}) '
            f'SELECT (c: class="{CAR}") '
            f"WHERE DURATION(c.center INSIDE near, {SEC // 2}) "
            f"THEN[0..{20 * SEC}] DURATION(c.center INSIDE mid, {SEC // 2})"
        )),
        requires=[CAR],
        notes="Two adjacent bands rather than all three: the bands are cut from "
              "the union of every object's extent, including parked cars far "
              "from the ego track, so one vehicle crossing all three in a 20 s "
              "scene is rare (19 of 120 scenes) where crossing one is not (44).",
    ),
    Query(
        "NS-Q5", "nuscenes", 3, ["THEN", "DURATION"],
        "A car overtakes a truck and then a pedestrian, each ordering held for "
        "half a second.",
        _ns(lambda r: (
            f'SELECT (c: class="{CAR}", t: class="{TRUCK}", p: class="{PED}") '
            f"WHERE DURATION(c.cx < t.cx, {SEC // 2}) "
            f"THEN[0..{10 * SEC}] DURATION(c.cx > p.cx, {SEC // 2})"
        )),
        requires=[CAR, TRUCK, PED],
    ),
    Query(
        "NS-Q6", "nuscenes", 2, ["DURATION"],
        "Sustained co-presence: a car and a truck are both in the mid band, car "
        "ahead, for 3 s straight.",
        _ns(lambda r: (
            f'REGIONS (mid = {r["mid"]}) '
            f'SELECT (c: class="{CAR}", t: class="{TRUCK}") '
            f"WHERE DURATION(c.center INSIDE mid AND t.center INSIDE mid "
            f"AND c.cx > t.cx, {3 * SEC})"
        )),
        requires=[CAR, TRUCK],
    ),
]

SUITE = SOCCERNET + I24_MOTION + CLEVRER + NUSCENES
