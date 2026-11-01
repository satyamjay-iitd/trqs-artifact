-- CL-Q4 (clevrer_merged, 1 objects, THEN)
-- One object traverses the canvas: left band, centre band, right band.
-- DSL: REGIONS (l = [0,0,160,320], c = [160,0,320,320], r = [320,0,480,320]) SELECT (a:) WHERE a.center INSIDE l THEN[0..60] a.center INSIDE c THEN[0..60] a.center INSIDE r
-- compiled for sub-dataset: video_00000-9999

WITH g0_frames AS (
    SELECT
        b0.fid AS fid,
        b0.oid AS o0,
        b0.x1 AS s0_x1,
        b0.y1 AS s0_y1,
        b0.x2 AS s0_x2,
        b0.y2 AS s0_y2
    FROM bbox b0
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
        WHERE (((((0.5 * s0_x1 + 0.5 * s0_x2) >= 0.0) AND ((0.5 * s0_y1 + 0.5 * s0_y2) >= 0.0)) AND ((0.5 * s0_x1 + 0.5 * s0_x2) <= 160.0)) AND ((0.5 * s0_y1 + 0.5 * s0_y2) <= 320.0))
    ) t
    GROUP BY o0, island
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
        WHERE (((((0.5 * s0_x1 + 0.5 * s0_x2) >= 160.0) AND ((0.5 * s0_y1 + 0.5 * s0_y2) >= 0.0)) AND ((0.5 * s0_x1 + 0.5 * s0_x2) <= 320.0)) AND ((0.5 * s0_y1 + 0.5 * s0_y2) <= 320.0))
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
        b0.y2 AS s0_y2
    FROM bbox b0
),
n3 AS (
    SELECT o0,
           MIN(fid) AS s,
           MAX(fid) AS e,
           1 AS c,
           MAX(fid) AS l,
           MIN(fid) AS r
    FROM (
        SELECT o0, fid,
               fid - ROW_NUMBER() OVER (PARTITION BY o0 ORDER BY fid) AS island
        FROM g2_frames
        WHERE (((((0.5 * s0_x1 + 0.5 * s0_x2) >= 320.0) AND ((0.5 * s0_y1 + 0.5 * s0_y2) >= 0.0)) AND ((0.5 * s0_x1 + 0.5 * s0_x2) <= 480.0)) AND ((0.5 * s0_y1 + 0.5 * s0_y2) <= 320.0))
    ) t
    GROUP BY o0, island
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
        FROM n2 a
        JOIN n3 b ON a.o0 = b.o0
        WHERE b.l - 0 >= 1
          AND b.l - 0 - a.c >= 0
          AND a.r <= b.l - 0 - 1
          AND b.s <= a.e + 60 + 1
    ) t
    WHERE c >= 1 AND s <= e AND s <= l AND r <= e AND e - s + 1 >= c
)
SELECT o0 AS "a", s, e
FROM n4
ORDER BY o0, s, e;
