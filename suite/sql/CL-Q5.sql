-- CL-Q5 (clevrer_merged, 2 objects, DURATION+THEN)
-- TQ1D retargeted: a metal cube and a rubber sphere separate, approach, then separate again.
-- DSL: SELECT (a: class="gray metal cube", b: class="red rubber sphere") WHERE DURATION(a.cx > b.cx + 45 OR a.cx < b.cx - 45, 5) THEN[0..60] DURATION(a.cx <= b.cx + 45 AND a.cx >= b.cx - 45, 1) THEN[0..60] DURATION(a.cx > b.cx + 45 OR a.cx < b.cx - 45, 1)
-- compiled for sub-dataset: video_00000-9999

WITH g0_frames AS (
    SELECT
        b0.fid AS fid,
        b0.oid AS o0,
        b0.x1 AS s0_x1,
        b0.y1 AS s0_y1,
        b0.x2 AS s0_x2,
        b0.y2 AS s0_y2,
        b1.oid AS o1,
        b1.x1 AS s1_x1,
        b1.y1 AS s1_y1,
        b1.x2 AS s1_x2,
        b1.y2 AS s1_y2
    FROM bbox b0
    JOIN objects j0 ON j0.oid = b0.oid AND j0.class = 'gray metal cube'
    JOIN bbox b1 ON b1.fid = b0.fid
    JOIN objects j1 ON j1.oid = b1.oid AND j1.class = 'red rubber sphere'
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
        WHERE (((0.5 * s0_x1 + 0.5 * s0_x2) - (0.5 * s1_x1 + 0.5 * s1_x2 + 45.0) > 0.0) OR ((0.5 * s0_x1 + 0.5 * s0_x2) - (0.5 * s1_x1 + 0.5 * s1_x2 + -45.0) < 0.0))
    ) t
    GROUP BY o0, o1, island
),
n1 AS (
    SELECT o0, o1,
           s, e,
           GREATEST(c, 5) AS c,
           l, r
    FROM n0
    WHERE e - s + 1 >= GREATEST(c, 5)
),
g1_frames AS (
    SELECT
        b0.fid AS fid,
        b0.oid AS o0,
        b0.x1 AS s0_x1,
        b0.y1 AS s0_y1,
        b0.x2 AS s0_x2,
        b0.y2 AS s0_y2,
        b1.oid AS o1,
        b1.x1 AS s1_x1,
        b1.y1 AS s1_y1,
        b1.x2 AS s1_x2,
        b1.y2 AS s1_y2
    FROM bbox b0
    JOIN objects j0 ON j0.oid = b0.oid AND j0.class = 'gray metal cube'
    JOIN bbox b1 ON b1.fid = b0.fid
    JOIN objects j1 ON j1.oid = b1.oid AND j1.class = 'red rubber sphere'
    WHERE b0.oid <> b1.oid
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
        FROM g1_frames
        WHERE (((0.5 * s0_x1 + 0.5 * s0_x2) - (0.5 * s1_x1 + 0.5 * s1_x2 + 45.0) <= 0.0) AND ((0.5 * s0_x1 + 0.5 * s0_x2) - (0.5 * s1_x1 + 0.5 * s1_x2 + -45.0) >= 0.0))
    ) t
    GROUP BY o0, o1, island
),
n3 AS (
    SELECT o0, o1,
           s, e,
           GREATEST(c, 1) AS c,
           l, r
    FROM n2
    WHERE e - s + 1 >= GREATEST(c, 1)
),
n4 AS (
    SELECT DISTINCT o0, o1, s, e, c, l, r
    FROM (
        SELECT a.o0 AS o0,
           a.o1 AS o1,
           a.s AS s,
           b.e AS e,
           a.c + b.c + 0 AS c,
           LEAST(a.l, a.e - a.c + 1, b.l - 0 - a.c) AS l,
           GREATEST(b.r, a.r + b.c + 0, b.s + b.c - 1) AS r
        FROM n1 a
        JOIN n3 b ON a.o0 = b.o0 AND a.o1 = b.o1
        WHERE b.l - 0 >= 1
          AND b.l - 0 - a.c >= 0
          AND a.r <= b.l - 0 - 1
          AND b.s <= a.e + 60 + 1
    ) t
    WHERE c >= 1 AND s <= e AND s <= l AND r <= e AND e - s + 1 >= c
),
g2_frames AS (
    SELECT
        b0.fid AS fid,
        b0.oid AS o0,
        b0.x1 AS s0_x1,
        b0.y1 AS s0_y1,
        b0.x2 AS s0_x2,
        b0.y2 AS s0_y2,
        b1.oid AS o1,
        b1.x1 AS s1_x1,
        b1.y1 AS s1_y1,
        b1.x2 AS s1_x2,
        b1.y2 AS s1_y2
    FROM bbox b0
    JOIN objects j0 ON j0.oid = b0.oid AND j0.class = 'gray metal cube'
    JOIN bbox b1 ON b1.fid = b0.fid
    JOIN objects j1 ON j1.oid = b1.oid AND j1.class = 'red rubber sphere'
    WHERE b0.oid <> b1.oid
),
n5 AS (
    SELECT o0, o1,
           MIN(fid) AS s,
           MAX(fid) AS e,
           1 AS c,
           MAX(fid) AS l,
           MIN(fid) AS r
    FROM (
        SELECT o0, o1, fid,
               fid - ROW_NUMBER() OVER (PARTITION BY o0, o1 ORDER BY fid) AS island
        FROM g2_frames
        WHERE (((0.5 * s0_x1 + 0.5 * s0_x2) - (0.5 * s1_x1 + 0.5 * s1_x2 + 45.0) > 0.0) OR ((0.5 * s0_x1 + 0.5 * s0_x2) - (0.5 * s1_x1 + 0.5 * s1_x2 + -45.0) < 0.0))
    ) t
    GROUP BY o0, o1, island
),
n6 AS (
    SELECT o0, o1,
           s, e,
           GREATEST(c, 1) AS c,
           l, r
    FROM n5
    WHERE e - s + 1 >= GREATEST(c, 1)
),
n7 AS (
    SELECT DISTINCT o0, o1, s, e, c, l, r
    FROM (
        SELECT a.o0 AS o0,
           a.o1 AS o1,
           a.s AS s,
           b.e AS e,
           a.c + b.c + 0 AS c,
           LEAST(a.l, a.e - a.c + 1, b.l - 0 - a.c) AS l,
           GREATEST(b.r, a.r + b.c + 0, b.s + b.c - 1) AS r
        FROM n4 a
        JOIN n6 b ON a.o0 = b.o0 AND a.o1 = b.o1
        WHERE b.l - 0 >= 1
          AND b.l - 0 - a.c >= 0
          AND a.r <= b.l - 0 - 1
          AND b.s <= a.e + 60 + 1
    ) t
    WHERE c >= 1 AND s <= e AND s <= l AND r <= e AND e - s + 1 >= c
)
SELECT o0 AS "a", o1 AS "b", s, e
FROM n7
ORDER BY o0, o1, s, e;
