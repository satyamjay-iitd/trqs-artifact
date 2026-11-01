-- I24-Q1 (i24_motion, 1 objects, DURATION+THEN)
-- A semi changes lanes: 2 s settled in eastbound lane 1, then 2 s settled in lane 2 within the next 2 s.
-- DSL: REGIONS (eb1 = [0, 21100, 2400000, 22200], eb2 = [0, 22200, 2400000, 23450]) SELECT (v: class=semi) WHERE DURATION(v.center INSIDE eb1, 50) THEN[0..50] DURATION(v.center INSIDE eb2, 50)
-- compiled for sub-dataset: post1_m00

WITH g0_frames AS (
    SELECT
        b0.fid AS fid,
        b0.oid AS o0,
        b0.x1 AS s0_x1,
        b0.y1 AS s0_y1,
        b0.x2 AS s0_x2,
        b0.y2 AS s0_y2
    FROM bbox b0
    JOIN objects j0 ON j0.oid = b0.oid AND j0.class = 'semi'
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
        WHERE (((((0.5 * s0_x1 + 0.5 * s0_x2) >= 0.0) AND ((0.5 * s0_y1 + 0.5 * s0_y2) >= 21100.0)) AND ((0.5 * s0_x1 + 0.5 * s0_x2) <= 2400000.0)) AND ((0.5 * s0_y1 + 0.5 * s0_y2) <= 22200.0))
    ) t
    GROUP BY o0, island
),
n1 AS (
    SELECT o0,
           s, e,
           GREATEST(c, 50) AS c,
           l, r
    FROM n0
    WHERE e - s + 1 >= GREATEST(c, 50)
),
g1_frames AS (
    SELECT
        b0.fid AS fid,
        b0.oid AS o0,
        b0.x1 AS s0_x1,
        b0.y1 AS s0_y1,
        b0.x2 AS s0_x2,
        b0.y2 AS s0_y2
    FROM bbox b0
    JOIN objects j0 ON j0.oid = b0.oid AND j0.class = 'semi'
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
        WHERE (((((0.5 * s0_x1 + 0.5 * s0_x2) >= 0.0) AND ((0.5 * s0_y1 + 0.5 * s0_y2) >= 22200.0)) AND ((0.5 * s0_x1 + 0.5 * s0_x2) <= 2400000.0)) AND ((0.5 * s0_y1 + 0.5 * s0_y2) <= 23450.0))
    ) t
    GROUP BY o0, island
),
n3 AS (
    SELECT o0,
           s, e,
           GREATEST(c, 50) AS c,
           l, r
    FROM n2
    WHERE e - s + 1 >= GREATEST(c, 50)
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
          AND b.s <= a.e + 50 + 1
    ) t
    WHERE c >= 1 AND s <= e AND s <= l AND r <= e AND e - s + 1 >= c
)
SELECT o0 AS "v", s, e
FROM n4
ORDER BY o0, s, e;
