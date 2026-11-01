-- SN-Q1 (soccernet, 1 objects, THEN)
-- The ball crosses the frame: left third to right third within 3 s.
-- DSL: REGIONS (l = [0,167,640,1080], r = [1280,167,1920,1080]) SELECT (b: class=ball) WHERE b.center INSIDE l THEN[0..75] b.center INSIDE r
-- compiled for sub-dataset: test_SNMOT-116

WITH g0_frames AS (
    SELECT
        b0.fid AS fid,
        b0.oid AS o0,
        CAST(b0.x1 AS DOUBLE) AS s0_x1,
        CAST(b0.y1 AS DOUBLE) AS s0_y1,
        CAST(b0.x2 AS DOUBLE) AS s0_x2,
        CAST(b0.y2 AS DOUBLE) AS s0_y2
    FROM bbox b0
    JOIN objects j0 ON j0.oid = b0.oid AND j0.class = 'ball'
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
        WHERE (((((0.5 * s0_x1 + 0.5 * s0_x2) >= 0.0) AND ((0.5 * s0_y1 + 0.5 * s0_y2) >= 167.0)) AND ((0.5 * s0_x1 + 0.5 * s0_x2) <= 640.0)) AND ((0.5 * s0_y1 + 0.5 * s0_y2) <= 1080.0))
    ) t
    GROUP BY o0, island
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
    JOIN objects j0 ON j0.oid = b0.oid AND j0.class = 'ball'
),
n1 AS (
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
        WHERE (((((0.5 * s0_x1 + 0.5 * s0_x2) >= 1280.0) AND ((0.5 * s0_y1 + 0.5 * s0_y2) >= 167.0)) AND ((0.5 * s0_x1 + 0.5 * s0_x2) <= 1920.0)) AND ((0.5 * s0_y1 + 0.5 * s0_y2) <= 1080.0))
    ) t
    GROUP BY o0, island
),
n2 AS (
    SELECT DISTINCT o0, s, e, c, l, r
    FROM (
        SELECT a.o0 AS o0,
           a.s AS s,
           b.e AS e,
           a.c + b.c + 0 AS c,
           LEAST(a.l, a.e - a.c + 1, b.l - 0 - a.c) AS l,
           GREATEST(b.r, a.r + b.c + 0, b.s + b.c - 1) AS r
        FROM n0 a
        JOIN n1 b ON a.o0 = b.o0
        WHERE b.l - 0 >= 1
          AND b.l - 0 - a.c >= 0
          AND a.r <= b.l - 0 - 1
          AND b.s <= a.e + 75 + 1
    ) t
    WHERE c >= 1 AND s <= e AND s <= l AND r <= e AND e - s + 1 >= c
)
SELECT o0 AS "b", s, e
FROM n2
ORDER BY o0, s, e;
