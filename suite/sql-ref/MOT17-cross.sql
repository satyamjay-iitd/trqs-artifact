-- Reference SQL for
--
--   SELECT (p: class=pedestrian)
--   WHERE DURATION(p.cy < 250, 15) THEN DURATION(p.cy > 350, 15)
--
-- A pedestrian walks toward the camera: 0.5 s above y=250, then 0.5 s below y=350.
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
--   [A THEN B]      = { (s1,e2) : (s1,e1) in [A], (s2,e2) in [B], s2 > e1 }
WITH boxes AS (
    SELECT b.fid AS fid,
           b.oid AS oid,
           (greatest(floor(b.y1), 0) + greatest(floor(b.y2), 0)) / 2.0 AS cy
    FROM bbox b
    JOIN objects o ON o.oid = b.oid AND o.class = 'pedestrian'
),

-- Maximal runs of each phase. `fid - ROW_NUMBER()` is constant exactly along
-- consecutive frames, so grouping on it yields the runs; HAVING applies the
-- DURATION minimum. A frame where the object is absent breaks the run, which is
-- what `max_gap = 1` does too.
r1 AS (
    SELECT oid, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT oid, fid,
                 fid - ROW_NUMBER() OVER (PARTITION BY oid ORDER BY fid) AS run
          FROM boxes WHERE cy < 250) t
    GROUP BY oid, run HAVING MAX(fid) - MIN(fid) + 1 >= 15
),
r2 AS (
    SELECT oid, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT oid, fid,
                 fid - ROW_NUMBER() OVER (PARTITION BY oid ORDER BY fid) AS run
          FROM boxes WHERE cy > 350) t
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
