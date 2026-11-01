-- Reference SQL for
--
--   SELECT (p: class=pedestrian)
--   WHERE DURATION(p.cy > 300, 50)
--
-- A pedestrian spends ~1.7 s (50 frames at 30 fps) in the lower half of the frame.
--
-- CLASS IS `pedestrian`, not `car`: MOT17 annotates people only, so the DETRAC
-- and VisDrone versions of these shapes do not transfer. COCO's `person`
-- (class 0) is what the detector must be filtered to.
--
-- CONSTANTS ARE PER SEQUENCE. MOT17-04 is a 1920x1080 elevated street view;
-- pedestrians sit at centre-y 14..752 (median 195). The query *shapes* match the
-- DETRAC and VisDrone ones so the three datasets' accuracy numbers compare; only
-- the class and the constants change.
--
-- NEGATIVE COORDINATES ARE CLAMPED TO 0, and this query clamps them too.
--
-- MOT17 boxes run off the frame: 20.9% of MOT17-04's have a negative corner,
-- down to -107. `trql-storage`'s converter reads corners with
-- `get_f64(...) as u32`, and Rust saturates a negative float cast to 0 — so the
-- engine sees 0 where the annotation says -8, and a centre computed from it is
-- wrong by up to ~50 px.
--
-- `greatest(floor(x), 0)` reproduces that exactly (saturate below, truncate
-- above). This is mirroring a *storage limitation*, not a semantic choice: the
-- engine cannot currently represent a box outside the frame, and a reference
-- written against the true coordinates disagrees with it — on MOT17-pass that
-- showed up as 7 tuples SQL found and 11 the engine found, with shifted spans.
-- Fixing it properly means signed coordinates or an offset in storage; until
-- then this query answers the question the engine is actually being asked.
--
-- COORDINATES ARE TRUNCATED, and this query must truncate the same way. MOT17
-- stores bbox corners as DOUBLE and the engine reads them with
-- `get_f64(...) as u32`, which truncates toward zero. See `DETRAC-dwell.sql`.
--
-- Written against the DSL's denotation, sharing nothing with the engine's
-- interval-set algebra:
--
--   [P]             = { (s,e) : P holds at every frame in [s,e] }
--   [DURATION(E,k)] = { (s,e) in [E] : e-s+1 >= k }
WITH boxes AS (
    SELECT b.fid                AS fid,
           b.oid                AS oid,
           (greatest(floor(b.y1), 0) + greatest(floor(b.y2), 0)) / 2.0 AS cy
    FROM bbox b
    JOIN objects o ON o.oid = b.oid AND o.class = 'pedestrian'
),

-- Maximal runs of the predicate. `fid - ROW_NUMBER()` is constant exactly along
-- consecutive frames, so grouping on it yields the runs; HAVING applies the
-- DURATION minimum. A frame where the object is absent breaks the run, which is
-- what `max_gap = 1` does too.
runs AS (
    SELECT oid, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT oid, fid,
                 fid - ROW_NUMBER() OVER (PARTITION BY oid ORDER BY fid) AS run
          FROM boxes WHERE cy > 300) t
    GROUP BY oid, run HAVING MAX(fid) - MIN(fid) + 1 >= 50
)
SELECT oid AS "v", s, e
FROM runs
ORDER BY oid, s, e;
