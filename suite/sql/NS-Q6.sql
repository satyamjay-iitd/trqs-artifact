-- NS-Q6 (nuscenes, 2 objects, DURATION)
-- Sustained co-presence: a car and a truck are both in the mid band, car ahead, for 3 s straight.
-- DSL: REGIONS (mid = [96134, 132240, 100679, 143608]) SELECT (c: class="vehicle.car", t: class="vehicle.truck") WHERE DURATION(c.center INSIDE mid AND t.center INSIDE mid AND c.cx > t.cx, 36)
-- compiled for sub-dataset: scene-0001

WITH g0_frames AS (
    SELECT
        b0.fid AS fid,
        b0.oid AS o0,
        CAST(b0.x1 AS DOUBLE) AS s0_x1,
        CAST(b0.y1 AS DOUBLE) AS s0_y1,
        CAST(b0.x2 AS DOUBLE) AS s0_x2,
        CAST(b0.y2 AS DOUBLE) AS s0_y2,
        b1.oid AS o1,
        CAST(b1.x1 AS DOUBLE) AS s1_x1,
        CAST(b1.y1 AS DOUBLE) AS s1_y1,
        CAST(b1.x2 AS DOUBLE) AS s1_x2,
        CAST(b1.y2 AS DOUBLE) AS s1_y2
    FROM bbox b0
    JOIN objects j0 ON j0.oid = b0.oid AND j0.class = 'vehicle.car'
    JOIN bbox b1 ON b1.fid = b0.fid
    JOIN objects j1 ON j1.oid = b1.oid AND j1.class = 'vehicle.truck'
    WHERE b0.oid <> b1.oid
),
n0 AS (
    SELECT o0, o1,
           MIN(fid) AS s,
           MAX(fid) AS e,
           1 AS c,
           MAX(fid) AS l,
           MIN(fid) AS r
    FROM (
        SELECT o0, o1, fid,
               fid - ROW_NUMBER() OVER (PARTITION BY o0, o1 ORDER BY fid) AS island
        FROM g0_frames
        WHERE (((((((0.5 * s0_x1 + 0.5 * s0_x2) >= 96134.0) AND ((0.5 * s0_y1 + 0.5 * s0_y2) >= 132240.0)) AND ((0.5 * s0_x1 + 0.5 * s0_x2) <= 100679.0)) AND ((0.5 * s0_y1 + 0.5 * s0_y2) <= 143608.0)) AND (((((0.5 * s1_x1 + 0.5 * s1_x2) >= 96134.0) AND ((0.5 * s1_y1 + 0.5 * s1_y2) >= 132240.0)) AND ((0.5 * s1_x1 + 0.5 * s1_x2) <= 100679.0)) AND ((0.5 * s1_y1 + 0.5 * s1_y2) <= 143608.0))) AND ((0.5 * s0_x1 + 0.5 * s0_x2) - (0.5 * s1_x1 + 0.5 * s1_x2) > 0.0))
    ) t
    GROUP BY o0, o1, island
),
n1 AS (
    SELECT o0, o1,
           s, e,
           GREATEST(c, 36) AS c,
           l, r
    FROM n0
    WHERE e - s + 1 >= GREATEST(c, 36)
)
SELECT o0 AS "c", o1 AS "t", s, e
FROM n1
ORDER BY o0, o1, s, e;
