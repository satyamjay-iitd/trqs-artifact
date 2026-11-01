-- Reference SQL for
--
--   SELECT (a: class=car, b: class=car)
--   WHERE DURATION(a.cy < b.cy, 25) THEN DURATION(a.cy > b.cy, 25)
--
-- Two cars swap along-frame order: `a` is above `b` for ~0.8 s, then below it.
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
WITH pairs AS (
    SELECT b0.fid                AS fid,
           b0.oid                AS o0,
           b1.oid                AS o1,
           (floor(b0.y1) + floor(b0.y2)) / 2.0 AS a_cy,
           (floor(b1.y1) + floor(b1.y2)) / 2.0 AS b_cy
    FROM bbox b0
    JOIN objects j0 ON j0.oid = b0.oid AND j0.class = 'car'
    JOIN bbox b1 ON b1.fid = b0.fid
    JOIN objects j1 ON j1.oid = b1.oid AND j1.class = 'car'
    WHERE b0.oid <> b1.oid
),

-- Runs of each phase within the pair's co-appearance: `pairs` holds only frames
-- where both cars exist, so a frame either is missing breaks the run.
r1 AS (
    SELECT o0, o1, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT o0, o1, fid,
                 fid - ROW_NUMBER() OVER (PARTITION BY o0, o1 ORDER BY fid) AS run
          FROM pairs WHERE a_cy < b_cy) t
    GROUP BY o0, o1, run HAVING MAX(fid) - MIN(fid) + 1 >= 25
),
r2 AS (
    SELECT o0, o1, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT o0, o1, fid,
                 fid - ROW_NUMBER() OVER (PARTITION BY o0, o1 ORDER BY fid) AS run
          FROM pairs WHERE a_cy > b_cy) t
    GROUP BY o0, o1, run HAVING MAX(fid) - MIN(fid) + 1 >= 25
),

-- Interval sets are downward closed, so it is not enough to ask whether the two
-- maximal runs happen to sequence. The first phase may start at A.s and needs 25
-- frames, so the second can begin no earlier than A.s + 25; inside B's run that
-- is GREATEST(B.s, A.s + 25), and 25 frames must still fit before B.e.
raw AS (
    SELECT A.o0 AS o0, A.o1 AS o1, A.s AS s, B.e AS e
    FROM r1 A
    JOIN r2 B ON B.o0 = A.o0 AND B.o1 = A.o1
    WHERE GREATEST(B.s, A.s + 25) + 25 - 1 <= B.e
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
