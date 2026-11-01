-- NS-Q3 (nuscenes, 2 objects, OVERLAP+DURATION)
-- A truck's and a car's 1 s spells in the mid band genuinely overlap -- neither contains the other.
-- DSL: REGIONS (mid = [96134, 132240, 100679, 143608]) SELECT (t: class="vehicle.truck", c: class="vehicle.car") WHERE OVERLAP(DURATION(t.center INSIDE mid, 12), DURATION(c.center INSIDE mid, 12))
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
    JOIN objects j0 ON j0.oid = b0.oid AND j0.class = 'vehicle.truck'
    JOIN bbox b1 ON b1.fid = b0.fid
    JOIN objects j1 ON j1.oid = b1.oid AND j1.class = 'vehicle.car'
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
        WHERE (((((0.5 * s0_x1 + 0.5 * s0_x2) >= 96134.0) AND ((0.5 * s0_y1 + 0.5 * s0_y2) >= 132240.0)) AND ((0.5 * s0_x1 + 0.5 * s0_x2) <= 100679.0)) AND ((0.5 * s0_y1 + 0.5 * s0_y2) <= 143608.0))
    ) t
    GROUP BY o0, o1, island
),
n1 AS (
    SELECT o0, o1,
           s, e,
           GREATEST(c, 12) AS c,
           l, r
    FROM n0
    WHERE e - s + 1 >= GREATEST(c, 12)
),
n2 AS (
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
        WHERE (((((0.5 * s1_x1 + 0.5 * s1_x2) >= 96134.0) AND ((0.5 * s1_y1 + 0.5 * s1_y2) >= 132240.0)) AND ((0.5 * s1_x1 + 0.5 * s1_x2) <= 100679.0)) AND ((0.5 * s1_y1 + 0.5 * s1_y2) <= 143608.0))
    ) t
    GROUP BY o0, o1, island
),
n3 AS (
    SELECT o0, o1,
           s, e,
           GREATEST(c, 12) AS c,
           l, r
    FROM n2
    WHERE e - s + 1 >= GREATEST(c, 12)
),
n4 AS (
    SELECT DISTINCT o0, o1, s, e, c, l, r
    FROM (
        SELECT a.o0 AS o0,
           a.o1 AS o1,
           a.s AS s,
           b.e AS e,
           GREATEST(a.c + 1, b.c + 1, 3) AS c,
           LEAST(a.l, a.e - a.c + 1, a.e - 1, b.l - 1) AS l,
           GREATEST(b.r, a.r + 1, b.s + b.c - 1, b.s + 1) AS r
        FROM n1 a
        JOIN n3 b ON a.o0 = b.o0 AND a.o1 = b.o1
        WHERE b.s <= a.e
          AND a.s < a.e
          AND a.s < b.l
    ) t
    WHERE c >= 1 AND s <= e AND s <= l AND r <= e AND e - s + 1 >= c
)
SELECT o0 AS "t", o1 AS "c", s, e
FROM n4
ORDER BY o0, o1, s, e;
