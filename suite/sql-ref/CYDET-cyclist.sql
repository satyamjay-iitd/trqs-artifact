-- Reference SQL for
--
--   SELECT (p: class=person, b: class=bicycle)
--   WHERE p.x1 < b.x2 AND b.x1 < p.x2 AND p.y1 < b.y2 AND b.y1 < p.y2
--
-- A person's box and a bicycle's box intersect: Rekall's cyclist-detection
-- tutorial query (tutorial 01, cell 34), ported to TrQL and to SQL.
--
-- WHAT REKALL WRITES. Their query is
--
--   person.join(bikes,
--       predicate = and_pred(Bounds3D.T(equal()),
--                            Bounds3D.X(overlaps()),
--                            Bounds3D.Y(overlaps())),
--       merge_op  = lambda i1, i2: Interval(i1['bounds'].span(i2['bounds'])),
--       window    = 0.0)
--
-- `T(equal())` is "the same frame", since their intervals are one frame long
-- (`t1 = f/fps`, `t2 = (f+1)/fps`); here that is the join on `fid`. `X` and `Y`
-- carry `overlaps()`, which on a non-degenerate box is exactly the strict
-- interval overlap written above -- `rekall/predicates.py:147` is a four-way
-- disjunction that reduces to `a1 < b2 AND b1 < a2` whenever `x1 < x2` on both
-- arguments, which `build_rekall_cydet.py` asserts of the data it writes.
--
-- NO DURATION. The tutorial imposes no minimum, so a single frame of overlap is
-- an answer. This is the easy case for every encoding, which is the point: it is
-- their own published example, not one chosen to suit us.
--
-- ORIENTATION. The two slots carry different classes, so each person-bicycle
-- pair appears once and in one order, and there are no self-pairs. Rekall's
-- `person.join(bikes)` is asymmetric in the same way. Neither side
-- double-counts.
--
-- Written against the DSL's denotation, sharing nothing with the engine's
-- interval-set algebra:
--
--   [P] = { (s,e) : P holds at every frame in [s,e] }
--
-- so the answer per pair is the maximal runs of consecutive frames on which the
-- boxes intersect -- which is what the engine reports after `maximal()`.
WITH pairs AS (
    SELECT bp.fid AS fid,
           bp.oid AS o0,
           bb.oid AS o1
    FROM bbox bp
    JOIN objects jp ON jp.oid = bp.oid AND jp.class = 'person'
    JOIN bbox bb ON bb.fid = bp.fid
    JOIN objects jb ON jb.oid = bb.oid AND jb.class = 'bicycle'
    WHERE bp.x1 < bb.x2 AND bb.x1 < bp.x2
      AND bp.y1 < bb.y2 AND bb.y1 < bp.y2
)

-- Maximal runs of consecutive frames per pair. `fid - ROW_NUMBER()` is constant
-- exactly along a run of consecutive frames, so grouping on it yields the runs.
SELECT o0 AS "p", o1 AS "b", MIN(fid) AS s, MAX(fid) AS e
FROM (SELECT o0, o1, fid,
             fid - ROW_NUMBER() OVER (PARTITION BY o0, o1 ORDER BY fid) AS run
      FROM pairs) t
GROUP BY o0, o1, run
ORDER BY o0, o1, s, e;
