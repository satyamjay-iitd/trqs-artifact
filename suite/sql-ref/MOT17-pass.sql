-- Reference SQL for
--
--   SELECT (a: class=pedestrian, b: class=pedestrian)
--   WHERE DURATION(a.cy < b.cy, 25) THEN DURATION(a.cy > b.cy, 25)
--
-- Two pedestrians swap depth order: `a` is further from the camera than `b` for ~0.8 s, then nearer.
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
WITH pairs AS (
    SELECT b0.fid                AS fid,
           b0.oid                AS o0,
           b1.oid                AS o1,
           (greatest(floor(b0.y1), 0) + greatest(floor(b0.y2), 0)) / 2.0 AS a_cy,
           (greatest(floor(b1.y1), 0) + greatest(floor(b1.y2), 0)) / 2.0 AS b_cy
    FROM bbox b0
    JOIN objects j0 ON j0.oid = b0.oid AND j0.class = 'pedestrian'
    JOIN bbox b1 ON b1.fid = b0.fid
    JOIN objects j1 ON j1.oid = b1.oid AND j1.class = 'pedestrian'
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
