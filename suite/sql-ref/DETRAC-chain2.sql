-- Reference SQL for
--
--   SELECT (v: class=car)
--   WHERE DURATION(v.cy < 100, 10)
--    THEN DURATION(v.cy > 250, 10)
--
-- A car crosses the frame top to bottom through 0 intermediate band(s).
--
-- WHY THE CHAIN. `DETRAC-chain2` asks only that the car was at the top and later
-- at the bottom; the longer chains additionally require it to be *seen* in each
-- band between. On ground truth all three return the identical answer — a real
-- trajectory passes through every band it crosses — so any difference on
-- detector output is occlusion and nothing else. That is the point: the longer
-- the chain, the more chances a missing detection has to break it.
--
-- SQL cannot recover from that. A band the detector missed has no rows, so the
-- run is absent and the chain fails. The engine can: with `max_gap` above 1 it
-- bridges the hole, and the fitted polynomial supplies a position through the
-- missing frames, which may satisfy the band. These queries exist to measure
-- that difference, so the reference deliberately has no equivalent mechanism.
--
-- COORDINATES ARE TRUNCATED, and this query must truncate the same way. See
-- `DETRAC-dwell.sql`.
--
-- Written against the DSL's denotation, sharing nothing with the engine's
-- interval-set algebra:
--
--   [P]             = { (s,e) : P holds at every frame in [s,e] }
--   [DURATION(E,k)] = { (s,e) in [E] : e-s+1 >= k }
--   [A THEN B]      = { (s1,e2) : (s1,e1) in [A], (s2,e2) in [B], s2 > e1 }
--
-- Greedy earliest-fit, unrolled over the chain: each phase starts as early as the
-- one before it allows, which is sound for a plain (unbounded) gap because a
-- phase constrains the next only through its end frame.
WITH boxes AS (
    SELECT b.fid AS fid,
           b.oid AS oid,
           (floor(b.y1) + floor(b.y2)) / 2.0 AS cy
    FROM bbox b
    JOIN objects o ON o.oid = b.oid AND o.class = 'car'
),
r1 AS (
    SELECT oid, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT oid, fid,
                 fid - ROW_NUMBER() OVER (PARTITION BY oid ORDER BY fid) AS run
          FROM boxes WHERE cy < 100) t
    GROUP BY oid, run HAVING MAX(fid) - MIN(fid) + 1 >= 10
),
r2 AS (
    SELECT oid, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT oid, fid,
                 fid - ROW_NUMBER() OVER (PARTITION BY oid ORDER BY fid) AS run
          FROM boxes WHERE cy > 250) t
    GROUP BY oid, run HAVING MAX(fid) - MIN(fid) + 1 >= 10
),
raw AS (
    SELECT A.oid AS oid, A.s AS s, B.e AS e
    FROM r1 A
    JOIN r2 B ON B.oid = A.oid
    WHERE GREATEST(B.s, A.s + 10) + 10 - 1 <= B.e
),
ordered AS (
    SELECT oid, s, e,
           MAX(e) OVER (PARTITION BY oid ORDER BY s, e
                        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_e
    FROM raw
),
grouped AS (
    SELECT oid, s, e,
           SUM(CASE WHEN prev_e IS NULL OR s > prev_e THEN 1 ELSE 0 END)
               OVER (PARTITION BY oid ORDER BY s, e ROWS UNBOUNDED PRECEDING) AS grp
    FROM ordered
)
SELECT oid AS "v", MIN(s) AS s, MAX(e) AS e
FROM grouped
GROUP BY oid, grp
ORDER BY oid, s, e;
