-- NS-Q5 (nuscenes, 3 objects, THEN+DURATION)
-- A car overtakes a truck and then a pedestrian, each ordering held for half a second.
-- DSL: SELECT (c: class="vehicle.car", t: class="vehicle.truck", p: class="human.pedestrian.adult") WHERE DURATION(c.cx < t.cx, 6) THEN[0..120] DURATION(c.cx > p.cx, 6)
-- compiled for sub-dataset: scene-0002

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
        WHERE ((0.5 * s0_x1 + 0.5 * s0_x2) - (0.5 * s1_x1 + 0.5 * s1_x2) < 0.0)
    ) t
    GROUP BY o0, o1, island
),
n1 AS (
    SELECT o0, o1,
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
        CAST(b0.y2 AS DOUBLE) AS s0_y2,
        b2.oid AS o2,
        CAST(b2.x1 AS DOUBLE) AS s2_x1,
        CAST(b2.y1 AS DOUBLE) AS s2_y1,
        CAST(b2.x2 AS DOUBLE) AS s2_x2,
        CAST(b2.y2 AS DOUBLE) AS s2_y2
    FROM bbox b0
    JOIN objects j0 ON j0.oid = b0.oid AND j0.class = 'vehicle.car'
    JOIN bbox b2 ON b2.fid = b0.fid
    JOIN objects j2 ON j2.oid = b2.oid AND j2.class = 'human.pedestrian.adult'
    WHERE b0.oid <> b2.oid
),
n2 AS (
    SELECT o0, o2,
           MIN(fid) AS s,
           MAX(fid) AS e,
           1 AS c,
           MAX(fid) AS l,
           MIN(fid) AS r
    FROM (
        SELECT o0, o2, fid,
               fid - ROW_NUMBER() OVER (PARTITION BY o0, o2 ORDER BY fid) AS island
        FROM g1_frames
        WHERE ((0.5 * s0_x1 + 0.5 * s0_x2) - (0.5 * s2_x1 + 0.5 * s2_x2) > 0.0)
    ) t
    GROUP BY o0, o2, island
),
n3 AS (
    SELECT o0, o2,
           s, e,
           GREATEST(c, 6) AS c,
           l, r
    FROM n2
    WHERE e - s + 1 >= GREATEST(c, 6)
),
n4 AS (
    SELECT DISTINCT o0, o1, o2, s, e, c, l, r
    FROM (
        SELECT a.o0 AS o0,
           a.o1 AS o1,
           b.o2 AS o2,
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
          AND b.s <= a.e + 120 + 1
    ) t
    WHERE c >= 1 AND s <= e AND s <= l AND r <= e AND e - s + 1 >= c
)
SELECT o0 AS "c", o1 AS "t", o2 AS "p", s, e
FROM n4
ORDER BY o0, o1, o2, s, e;
