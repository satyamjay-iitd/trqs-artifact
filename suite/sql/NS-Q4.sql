-- NS-Q4 (nuscenes, 1 objects, THEN)
-- A car advances a full band along the scene's dominant axis: half a second settled in the near band, then in the mid band.
-- DSL: REGIONS (near = [91590, 132240, 96134, 143608], mid = [96134, 132240, 100679, 143608]) SELECT (c: class="vehicle.car") WHERE DURATION(c.center INSIDE near, 6) THEN[0..240] DURATION(c.center INSIDE mid, 6)
-- compiled for sub-dataset: scene-0001

WITH g0_frames AS (
    SELECT
        b0.fid AS fid,
        b0.oid AS o0,
        CAST(b0.x1 AS DOUBLE) AS s0_x1,
        CAST(b0.y1 AS DOUBLE) AS s0_y1,
        CAST(b0.x2 AS DOUBLE) AS s0_x2,
        CAST(b0.y2 AS DOUBLE) AS s0_y2
    FROM bbox b0
    JOIN objects j0 ON j0.oid = b0.oid AND j0.class = 'vehicle.car'
),
n0 AS (
    SELECT o0,
           MIN(fid) AS s,
           MAX(fid) AS e,
           1 AS c,
           MAX(fid) AS l,
           MIN(fid) AS r
    FROM (
        SELECT o0, fid,
               fid - ROW_NUMBER() OVER (PARTITION BY o0 ORDER BY fid) AS island
        FROM g0_frames
        WHERE (((((0.5 * s0_x1 + 0.5 * s0_x2) >= 91590.0) AND ((0.5 * s0_y1 + 0.5 * s0_y2) >= 132240.0)) AND ((0.5 * s0_x1 + 0.5 * s0_x2) <= 96134.0)) AND ((0.5 * s0_y1 + 0.5 * s0_y2) <= 143608.0))
    ) t
    GROUP BY o0, island
),
n1 AS (
    SELECT o0,
           s, e,
           GREATEST(c, 6) AS c,
           l, r
    FROM n0
    WHERE e - s + 1 >= GREATEST(c, 6)
),
g1_frames AS (
    SELECT
        b0.fid AS fid,
        b0.oid AS o0,
        CAST(b0.x1 AS DOUBLE) AS s0_x1,
        CAST(b0.y1 AS DOUBLE) AS s0_y1,
        CAST(b0.x2 AS DOUBLE) AS s0_x2,
        CAST(b0.y2 AS DOUBLE) AS s0_y2
    FROM bbox b0
    JOIN objects j0 ON j0.oid = b0.oid AND j0.class = 'vehicle.car'
),
n2 AS (
    SELECT o0,
           MIN(fid) AS s,
           MAX(fid) AS e,
           1 AS c,
           MAX(fid) AS l,
           MIN(fid) AS r
    FROM (
        SELECT o0, fid,
               fid - ROW_NUMBER() OVER (PARTITION BY o0 ORDER BY fid) AS island
        FROM g1_frames
        WHERE (((((0.5 * s0_x1 + 0.5 * s0_x2) >= 96134.0) AND ((0.5 * s0_y1 + 0.5 * s0_y2) >= 132240.0)) AND ((0.5 * s0_x1 + 0.5 * s0_x2) <= 100679.0)) AND ((0.5 * s0_y1 + 0.5 * s0_y2) <= 143608.0))
    ) t
    GROUP BY o0, island
),
n3 AS (
    SELECT o0,
           s, e,
           GREATEST(c, 6) AS c,
           l, r
    FROM n2
    WHERE e - s + 1 >= GREATEST(c, 6)
),
n4 AS (
    SELECT DISTINCT o0, s, e, c, l, r
    FROM (
        SELECT a.o0 AS o0,
           a.s AS s,
           b.e AS e,
           a.c + b.c + 0 AS c,
           LEAST(a.l, a.e - a.c + 1, b.l - 0 - a.c) AS l,
           GREATEST(b.r, a.r + b.c + 0, b.s + b.c - 1) AS r
        FROM n1 a
        JOIN n3 b ON a.o0 = b.o0
        WHERE b.l - 0 >= 1
          AND b.l - 0 - a.c >= 0
          AND a.r <= b.l - 0 - 1
          AND b.s <= a.e + 240 + 1
    ) t
    WHERE c >= 1 AND s <= e AND s <= l AND r <= e AND e - s + 1 >= c
)
SELECT o0 AS "c", s, e
FROM n4
ORDER BY o0, s, e;
