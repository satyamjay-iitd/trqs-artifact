-- Reference SQL for
--
--   SELECT (p: class=person, c: class=car)
--   WHERE DURATION(p.cx > c.cx + 300 OR p.cx < c.cx - 300
--               OR p.cy > c.cy + 300 OR p.cy < c.cy - 300, 25)
--    THEN DURATION(p.cx <= c.cx + 100 AND p.cx >= c.cx - 100
--              AND p.cy <= c.cy + 100 AND p.cy >= c.cy - 100, 50)
--
-- A person approaches a parked vehicle and stays at it: more than 300 px away on
-- some axis for ~0.8 s, then within 100 px on both axes for ~1.7 s. 29.97 fps, so
-- 25 frames = 0.83 s and 50 = 1.67 s. 1920x1080 imagery.
--
-- WHY THE DISTANCE TESTS LOOK LIKE THAT. Two constraints compose. The DSL has no
-- absolute value, so |d| <= 100 must become a conjunction and |d| > 300 a
-- disjunction — with AND for the far test it would be unsatisfiable, with OR for
-- the near test a tautology. And a comparison may not mix variables on one side
-- (`p.cx - c.cx > 300` is rejected: "bbox expressions cannot mix multiple
-- variables on the same side"), so the offset moves across the operator and the
-- bound is written `p.cx > c.cx + 300`. This query states the same thing as the
-- `dx`/`dy` columns below; SQL is under no such restriction and says it directly.
--
-- NEAR IS BOTH AXES, FAR IS EITHER. Deliberate, and they are not negations of one
-- another: there is a band between 100 and 300 px belonging to neither, so the
-- query asks for a real approach rather than for any frame-to-frame jitter across
-- a single threshold.
--
-- PURELY RELATIONAL. The 24 merged segments share one camera (1920x1080, 29.97
-- fps, `merged_inconsistent` lists only `has_events`), so an absolute region
-- *would* be portable here — unlike the 315-source VIRAT merge, which disagrees on
-- `fps`, `width` and `height`. A comparison between two objects is safe either
-- way, since both sit in the same frame.
--
-- GAPS. 4 of 125 objects carry an interior gap and the smallest is 4 frames, so
-- with `max_gap=1` every gap splits the appearance and the `pairs` join below
-- breaks its runs at exactly the same places the engine does. Had any gap been a
-- single frame the engine would have bridged it while this query split on it, and
-- the two would disagree.
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
    JOIN objects j0 ON j0.oid = b0.oid AND j0.class = 'person'
    JOIN bbox b1 ON b1.fid = b0.fid
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
          WHERE dx > 300 OR -dx > 300 OR dy > 300 OR -dy > 300) t
    GROUP BY o0, o1, run HAVING MAX(fid) - MIN(fid) + 1 >= 25
),
r2 AS (
    SELECT o0, o1, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT o0, o1, fid,
                 fid - ROW_NUMBER() OVER (PARTITION BY o0, o1 ORDER BY fid) AS run
          FROM pairs
          WHERE dx <= 100 AND -dx <= 100 AND dy <= 100 AND -dy <= 100) t
    GROUP BY o0, o1, run HAVING MAX(fid) - MIN(fid) + 1 >= 50
),

-- Interval sets are downward closed, so it is not enough to ask whether the two
-- maximal runs happen to sequence. The first phase may start at A.s and needs 25
-- frames, so the second can begin no earlier than A.s + 25; inside B's run that is
-- GREATEST(B.s, A.s + 25), and 50 frames must still fit before B.e.
raw AS (
    SELECT A.o0 AS o0, A.o1 AS o1, A.s AS s, B.e AS e
    FROM r1 A
    JOIN r2 B ON B.o0 = A.o0 AND B.o1 = A.o1
    WHERE GREATEST(B.s, A.s + 25) + 50 - 1 <= B.e
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
SELECT o0 AS "p", o1 AS "c", MIN(s) AS s, MAX(e) AS e
FROM grouped
GROUP BY o0, o1, grp
ORDER BY o0, o1, s, e;
