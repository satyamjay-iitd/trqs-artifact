-- Reference SQL for
--
--   SELECT (v: class=car)
--   WHERE DURATION(v.cy < 1000, 15) THEN DURATION(v.cy > 1050, 15)
--
-- A car crosses the frame downward: 0.5 s above y=1000, then 0.5 s below y=1050.
--
-- CONSTANTS ARE PER SEQUENCE. VisDrone is aerial 2688x1512 footage, so the
-- thresholds here are nothing like `DETRAC-*`'s: cars sit at centre-y 744..1318
-- on this clip, against 37..236 on DETRAC's 960x540 street view. The query
-- *shapes* are deliberately identical to the DETRAC ones so the two datasets'
-- accuracy numbers compare; only the constants are retuned.
--
-- COORDINATES ARE TRUNCATED, and this query must truncate the same way. VisDrone
-- stores bbox corners as DOUBLE and the engine reads them with
-- `get_f64(...) as u32`, which truncates toward zero. See `DETRAC-dwell.sql`.
--
-- CLASS. `car` is the one label VisDrone and COCO agree on.
--
-- Written against the DSL's denotation, sharing nothing with the engine's
-- interval-set algebra:
--
--   [P]             = { (s,e) : P holds at every frame in [s,e] }
--   [DURATION(E,k)] = { (s,e) in [E] : e-s+1 >= k }
--   [A THEN B]      = { (s1,e2) : (s1,e1) in [A], (s2,e2) in [B], s2 > e1 }
WITH boxes AS (
    SELECT b.fid AS fid,
           b.oid AS oid,
           (floor(b.y1) + floor(b.y2)) / 2.0 AS cy
    FROM bbox b
    JOIN objects o ON o.oid = b.oid AND o.class = 'car'
),

-- Maximal runs of each phase. `fid - ROW_NUMBER()` is constant exactly along
-- consecutive frames, so grouping on it yields the runs; HAVING applies the
-- DURATION minimum. A frame where the object is absent breaks the run, which is
-- what `max_gap = 1` does too.
r1 AS (
    SELECT oid, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT oid, fid,
                 fid - ROW_NUMBER() OVER (PARTITION BY oid ORDER BY fid) AS run
          FROM boxes WHERE cy < 1000) t
    GROUP BY oid, run HAVING MAX(fid) - MIN(fid) + 1 >= 15
),
r2 AS (
    SELECT oid, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT oid, fid,
                 fid - ROW_NUMBER() OVER (PARTITION BY oid ORDER BY fid) AS run
          FROM boxes WHERE cy > 1050) t
    GROUP BY oid, run HAVING MAX(fid) - MIN(fid) + 1 >= 15
),

-- Interval sets are downward closed, so it is not enough to ask whether the two
-- maximal runs happen to sequence. The first phase may start at A.s and needs 25
-- frames, so the second can begin no earlier than A.s + 25; inside B's run that
-- is GREATEST(B.s, A.s + 25), and 25 frames must still fit before B.e.
raw AS (
    SELECT A.oid AS oid, A.s AS s, B.e AS e
    FROM r1 A
    JOIN r2 B ON B.oid = A.oid
    WHERE GREATEST(B.s, A.s + 15) + 15 - 1 <= B.e
),

-- The engine reports merged, non-overlapping spans per object.
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
