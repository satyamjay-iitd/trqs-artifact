-- Reference SQL for
--
--   SELECT (v: class=car)
--   WHERE DURATION(v.cy > 900, 50)
--
-- A car spends ~1.7 s (50 frames at 30 fps) in the lower half of the frame.
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
WITH boxes AS (
    SELECT b.fid                AS fid,
           b.oid                AS oid,
           (floor(b.y1) + floor(b.y2)) / 2.0 AS cy
    FROM bbox b
    JOIN objects o ON o.oid = b.oid AND o.class = 'car'
),

-- Maximal runs of the predicate. `fid - ROW_NUMBER()` is constant exactly along
-- consecutive frames, so grouping on it yields the runs; HAVING applies the
-- DURATION minimum. A frame where the object is absent breaks the run, which is
-- what `max_gap = 1` does too.
runs AS (
    SELECT oid, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT oid, fid,
                 fid - ROW_NUMBER() OVER (PARTITION BY oid ORDER BY fid) AS run
          FROM boxes WHERE cy > 900) t
    GROUP BY oid, run HAVING MAX(fid) - MIN(fid) + 1 >= 50
)
SELECT oid AS "v", s, e
FROM runs
ORDER BY oid, s, e;
