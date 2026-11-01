-- Reference SQL for
--
--   REGIONS (junction = [187500, 127510, 192500, 130010])
--   SELECT (c: class="vehicle.car")
--   WHERE DURATION(c.center INSIDE junction, 120)
--
-- Cars that linger at the busiest junction in boston-seaport for >= 10 s.
--
-- "Stopping" is a DWELL-TIME PROXY, not a velocity test: the language compares
-- affine combinations of ONE frame's coordinates, so cross-frame displacement --
-- and therefore speed -- is not expressible. The proxy is calibrated to the box:
-- crossing 50 m at urban speed (~8 m/s) takes ~6 s, so >= 10 s inside it means an
-- average speed under 5 m/s. Measured dwell at this junction: median 4.1 s,
-- p90 15.9 s, max 19.7 s (capped by the ~20 s scene length).
--
-- The junction is the busiest 50 x 25 m patch on the map by distinct cars: cells
-- (75,51) and (76,51) of a 25 m grid, holding 223 and 225 distinct cars.
--
-- COORDINATES. The region is x 1875..1925 m, y 818..843 m on the boston-seaport
-- map, converted by `frames.py`. nuScenes stores a Y-UP frame --
-- `stored_y = y_delta - metres*scale` with y_delta the map raster height -- so the
-- natural-lower edge becomes the stored-upper one. Writing the metres straight in
-- without the flip would put this region 3.4 km away, on water.
--
-- An absolute region is only meaningful here because all 467 merged scenes share
-- one coordinate frame (`merged_inconsistent` is absent from the metadata). The
-- same query would be meaningless on soccertrack_merged, whose sources disagree.
--
-- Written against the DSL's denotation:
--   [P]             = { (s,e) : P holds at every frame in [s,e] }
--   [DURATION(E,k)] = { (s,e) in [E] : e-s+1 >= k }
--
-- One object, so there is no self-join: this is a scan, an island pass, and a
-- length filter.
WITH c AS (
    SELECT b.fid AS fid, b.oid AS oid,
           (b.x1 + b.x2) / 2.0 AS cx,
           (b.y1 + b.y2) / 2.0 AS cy
    FROM bbox b
    JOIN objects o ON o.oid = b.oid AND o.class = 'vehicle.car'
),
-- Maximal runs inside the junction. `fid - ROW_NUMBER()` is constant exactly
-- along consecutive frames, so grouping on it yields the runs; any frame the car
-- spends outside the box starts a new one.
runs AS (
    SELECT oid, MIN(fid) AS s, MAX(fid) AS e
    FROM (
        SELECT oid, fid,
               fid - ROW_NUMBER() OVER (PARTITION BY oid ORDER BY fid) AS run
        FROM c
        WHERE cx >= 187500.0 AND cx <= 192500.0
          AND cy >= 127510.0 AND cy <= 130010.0
    ) t
    GROUP BY oid, run
    HAVING MAX(fid) - MIN(fid) + 1 >= 120
)
SELECT oid AS "c", s, e FROM runs ORDER BY oid, s, e;
