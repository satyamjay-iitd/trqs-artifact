-- Reference SQL for
--
--   SELECT (a: class=car, b: class=car)
--   WHERE DURATION(a.cx > b.cx + 200 OR a.cx < b.cx - 200
--               OR a.cy > b.cy + 200 OR a.cy < b.cy - 200, 15)
--    THEN DURATION(a.cx <= b.cx + 70 AND a.cx >= b.cx - 70
--              AND a.cy <= b.cy + 70 AND a.cy >= b.cy - 70, 30)
--
-- Two vehicles converge: more than 200 px apart on some axis for ~0.6 s, then
-- within 70 px on both axes for ~1.2 s. Run unchanged on a parking lot (VIRAT,
-- 1280x720 at 23.97 fps) and on a motorway junction (DETRAC, 960x540 at 25 fps),
-- which is what makes the two measurable against each other: the encodings
-- differ, the question asked of them does not.
--
-- WHY VEHICLE-TO-VEHICLE. DETRAC annotates vehicles only, so a query naming a
-- person cannot run on it at all. Both slots therefore take `class=car`, which
-- both datasets have.
--
-- SELF-PAIRS. `b1.oid <> b0.oid` is an optimisation, not a semantic: for a = b
-- every difference is 0, so the first phase is unsatisfiable and such a pair
-- could never reach the output.
--
-- ORDERED PAIRS. The predicates are symmetric, so a converging pair matches in
-- both slot orders and appears twice. Both sides report both, so the comparison
-- is unaffected.
--
-- WHY THE DISTANCE TESTS LOOK LIKE THAT. The DSL has no absolute value, so
-- |d| <= 80 becomes a conjunction and |d| > 200 a disjunction -- with AND the
-- far test would be unsatisfiable and with OR the near test a tautology. A
-- comparison may not mix variables on one side, so the offset moves across the
-- operator: `a.cx > b.cx + 200`.
--
-- Written against the DSL's denotation, sharing nothing with the engine's
-- interval-set algebra:
--
--   [P]             = { (s,e) : P holds at every frame in [s,e] }
--   [DURATION(E,k)] = { (s,e) in [E] : e-s+1 >= k }
--   [A THEN B]      = { (s1,e2) : (s1,e1) in [A], (s2,e2) in [B], s2 > e1 }
--
-- Greedy earliest-fit for the THEN, sound for a plain (unbounded) gap: a phase
-- constrains the next only through its end frame, and ending earlier is never
-- worse, so if any packing works the earliest one does.
WITH pairs AS (
    SELECT b0.fid                AS fid,
           b0.oid                AS o0,
           b1.oid                AS o1,
           (b0.x1 + b0.x2) / 2.0 - (b1.x1 + b1.x2) / 2.0 AS dx,
           (b0.y1 + b0.y2) / 2.0 - (b1.y1 + b1.y2) / 2.0 AS dy
    FROM bbox b0
    JOIN objects j0 ON j0.oid = b0.oid AND j0.class = 'car'
    JOIN bbox b1 ON b1.fid = b0.fid AND b1.oid <> b0.oid
    JOIN objects j1 ON j1.oid = b1.oid AND j1.class = 'car'
),

-- Maximal runs of each phase, within the pair's co-appearance. `fid -
-- ROW_NUMBER()` is constant exactly along consecutive frames, so grouping on it
-- yields the runs; HAVING applies the DURATION minimum.
r1 AS (
    SELECT o0, o1, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT o0, o1, fid,
                 fid - ROW_NUMBER() OVER (PARTITION BY o0, o1 ORDER BY fid) AS run
          FROM pairs
          WHERE dx > 200 OR -dx > 200 OR dy > 200 OR -dy > 200) t
    GROUP BY o0, o1, run HAVING MAX(fid) - MIN(fid) + 1 >= 15
),
r2 AS (
    SELECT o0, o1, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT o0, o1, fid,
                 fid - ROW_NUMBER() OVER (PARTITION BY o0, o1 ORDER BY fid) AS run
          FROM pairs
          WHERE dx <= 70 AND -dx <= 70 AND dy <= 70 AND -dy <= 70) t
    GROUP BY o0, o1, run HAVING MAX(fid) - MIN(fid) + 1 >= 30
),

-- Interval sets are downward closed, so it is not enough to ask whether the two
-- maximal runs happen to sequence. The first phase may start at A.s and needs 15
-- frames, so the second can begin no earlier than A.s + 15; inside B's run that
-- is GREATEST(B.s, A.s + 15), and 30 frames must still fit before B.e.
raw AS (
    SELECT A.o0 AS o0, A.o1 AS o1, A.s AS s, B.e AS e
    FROM r1 A
    JOIN r2 B ON B.o0 = A.o0 AND B.o1 = A.o1
    WHERE GREATEST(B.s, A.s + 15) + 30 - 1 <= B.e
),

-- The engine reports merged, non-overlapping spans per pair.
ordered AS (
    SELECT o0, o1, s, e,
           MAX(e) OVER (PARTITION BY o0, o1 ORDER BY s, e
                        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_e
    FROM raw
),
grouped AS (
    SELECT o0, o1, s, e,
           SUM(CASE WHEN prev_e IS NULL OR s > prev_e THEN 1 ELSE 0 END)
               OVER (PARTITION BY o0, o1 ORDER BY s, e ROWS UNBOUNDED PRECEDING) AS grp
    FROM ordered
)
SELECT o0 AS "a", o1 AS "b", MIN(s) AS s, MAX(e) AS e
FROM grouped
GROUP BY o0, o1, grp
ORDER BY o0, o1, s, e;
