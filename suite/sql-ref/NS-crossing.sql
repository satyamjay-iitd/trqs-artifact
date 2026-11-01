-- Reference SQL for
--
--   REGIONS (crossing = [37500, 46210, 45000, 51210])
--   SELECT (c: class="vehicle.car", p: class="human.pedestrian.adult")
--   WHERE CONTAINS(DURATION(c.center INSIDE crossing, 60),
--                  DURATION(p.center INSIDE crossing, 12))
--
-- A car dwells at the crossing for >= 5 s and a pedestrian is present there for
-- >= 1 s strictly inside that dwell -- the car was already waiting when the
-- pedestrian arrived and still waiting when they left. `CONTAINS` is Allen's
-- *contains*; the result spans the OUTER interval, so a row reports the car'"'"'s wait.
--
-- The crossing is the busiest 25 m cell for BOTH classes: cell (16,19) of a 25 m
-- grid, 116 distinct cars and 164 distinct pedestrians, widened to its busy
-- neighbours -- x 375..450 m, y 1606..1656 m (75 x 50 m).
--
-- TWO THINGS THE OBVIOUS VERSION GETS WRONG, both caught by the exactness check
-- (it reported engine 2962 tuples against 237, precision 0.074):
--
-- 1. The operand interval sets are DOWNWARD CLOSED, so it is not enough to ask
--    whether the pedestrian'"'"'s maximal run nests inside the car'"'"'s. Any
--    sub-interval of the run that is still >= 12 frames also belongs to the set.
--    Car 6901 and pedestrian 6902 share the identical run [17473,17704]: the
--    maximal runs do not strictly nest, but [17480,17690] does, so the pair
--    matches. Handled below by asking whether a qualifying inner interval FITS in
--    the room the outer leaves, not whether the whole run happens to fit.
--
-- 2. Both operands are evaluated over the pair'"'"'s CO-APPEARANCE, not over each
--    object'"'"'s whole life. Pedestrian 8828 covers 82 frames across a 151-frame
--    span -- three interior gaps -- so with max_gap=1 its appearance splits, the
--    co-appearance window with car 8823 splits with it, and a nesting that holds
--    over the unclipped runs need not hold inside any one window. 691 of 43,360
--    objects here carry an interior gap, so this is not a corner case. Handled by
--    deriving both runs from a `pairs` join on `fid`, which admits only frames
--    where BOTH objects exist -- the same shape the other reference files use, and
--    the step the first version of this file omitted.
--
-- COORDINATES. nuScenes stores a Y-UP frame (`stored_y = y_delta - metres*scale`,
-- `y_delta` = map raster height), so `frames.py` re-sorts the corners; written
-- without the flip the region lands off the map. It is meaningful across all 467
-- merged scenes only because they share one frame (`merged_inconsistent` absent).
--
-- Denotation:
--   [P]              = { (s,e) : P holds at every frame in [s,e] }
--   [DURATION(E,k)]  = { (s,e) in [E] : e-s+1 >= k }
--   [CONTAINS(A,B)]  = { (s1,e1) in [A] : exists (s2,e2) in [B], s1 < s2 <= e2 < e1 }
--
-- For a run pair the widest outer interval is the whole car run: widening the
-- outer only makes room, so if any qualifying pair nests, the maximal outer does.
-- The inner must then fit in [C.s+1, C.e-1] intersected with the pedestrian run,
-- which is the length test in `raw`.
WITH pairs AS (
    SELECT b0.fid                AS fid,
           b0.oid                AS o0,
           b1.oid                AS o1,
           (b0.x1 + b0.x2) / 2.0 AS c_cx,
           (b0.y1 + b0.y2) / 2.0 AS c_cy,
           (b1.x1 + b1.x2) / 2.0 AS p_cx,
           (b1.y1 + b1.y2) / 2.0 AS p_cy
    FROM bbox b0
    JOIN objects j0 ON j0.oid = b0.oid AND j0.class = 'vehicle.car'
    JOIN bbox b1 ON b1.fid = b0.fid
    JOIN objects j1 ON j1.oid = b1.oid AND j1.class = 'human.pedestrian.adult'
),
-- Runs of each phase WITHIN the pair'"'"'s co-appearance: `pairs` already holds only
-- frames where both exist, so a frame either object is missing breaks the run.
car AS (
    SELECT o0, o1, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT o0, o1, fid,
                 fid - ROW_NUMBER() OVER (PARTITION BY o0, o1 ORDER BY fid) AS run
          FROM pairs
          WHERE c_cx BETWEEN 37500 AND 45000 AND c_cy BETWEEN 46210 AND 51210) t
    GROUP BY o0, o1, run HAVING MAX(fid) - MIN(fid) + 1 >= 60
),
ped AS (
    SELECT o0, o1, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT o0, o1, fid,
                 fid - ROW_NUMBER() OVER (PARTITION BY o0, o1 ORDER BY fid) AS run
          FROM pairs
          WHERE p_cx BETWEEN 37500 AND 45000 AND p_cy BETWEEN 46210 AND 51210) t
    GROUP BY o0, o1, run HAVING MAX(fid) - MIN(fid) + 1 >= 1
),
raw AS (
    SELECT C.o0 AS o0, C.o1 AS o1, C.s AS s, C.e AS e
    FROM car C
    JOIN ped P ON P.o0 = C.o0 AND P.o1 = C.o1
    -- Does a >= 12-frame inner interval fit strictly inside the car'"'"'s dwell?
    WHERE LEAST(P.e, C.e - 1) - GREATEST(P.s, C.s + 1) + 1 >= 12
),
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
SELECT o0 AS "c", o1 AS "p", MIN(s) AS s, MAX(e) AS e
FROM grouped GROUP BY o0, o1, grp ORDER BY o0, o1, s, e;
