-- I24-Q2 (i24_motion, 2 objects, DURATION+THEN)
-- Overtake within the eastbound carriageway: a trails b for 2 s, then leads it for 2 s.
-- DSL: REGIONS (eb = [0, 21100, 2400000, 27200]) SELECT (a:, b:) WHERE DURATION(a.cx < b.cx AND a.center INSIDE eb AND b.center INSIDE eb, 50) THEN[0..250] DURATION(a.cx > b.cx AND a.center INSIDE eb AND b.center INSIDE eb, 50)
-- compiled for sub-dataset: post1

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
    JOIN bbox b1 ON b1.fid = b0.fid
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
        WHERE ((((0.5 * s0_x1 + 0.5 * s0_x2) - (0.5 * s1_x1 + 0.5 * s1_x2) < 0.0) AND (((((0.5 * s0_x1 + 0.5 * s0_x2) >= 0.0) AND ((0.5 * s0_y1 + 0.5 * s0_y2) >= 21100.0)) AND ((0.5 * s0_x1 + 0.5 * s0_x2) <= 2400000.0)) AND ((0.5 * s0_y1 + 0.5 * s0_y2) <= 27200.0))) AND (((((0.5 * s1_x1 + 0.5 * s1_x2) >= 0.0) AND ((0.5 * s1_y1 + 0.5 * s1_y2) >= 21100.0)) AND ((0.5 * s1_x1 + 0.5 * s1_x2) <= 2400000.0)) AND ((0.5 * s1_y1 + 0.5 * s1_y2) <= 27200.0)))
    ) t
    GROUP BY o0, o1, island
),
n1 AS (
    SELECT o0, o1,
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
        b0.y2 AS s0_y2,
        b1.oid AS o1,
        b1.x1 AS s1_x1,
        b1.y1 AS s1_y1,
        b1.x2 AS s1_x2,
        b1.y2 AS s1_y2
    FROM bbox b0
    JOIN bbox b1 ON b1.fid = b0.fid
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
        WHERE ((((0.5 * s0_x1 + 0.5 * s0_x2) - (0.5 * s1_x1 + 0.5 * s1_x2) > 0.0) AND (((((0.5 * s0_x1 + 0.5 * s0_x2) >= 0.0) AND ((0.5 * s0_y1 + 0.5 * s0_y2) >= 21100.0)) AND ((0.5 * s0_x1 + 0.5 * s0_x2) <= 2400000.0)) AND ((0.5 * s0_y1 + 0.5 * s0_y2) <= 27200.0))) AND (((((0.5 * s1_x1 + 0.5 * s1_x2) >= 0.0) AND ((0.5 * s1_y1 + 0.5 * s1_y2) >= 21100.0)) AND ((0.5 * s1_x1 + 0.5 * s1_x2) <= 2400000.0)) AND ((0.5 * s1_y1 + 0.5 * s1_y2) <= 27200.0)))
    ) t
    GROUP BY o0, o1, island
),
n3 AS (
    SELECT o0, o1,
           s, e,
           GREATEST(c, 50) AS c,
           l, r
    FROM n2
    WHERE e - s + 1 >= GREATEST(c, 50)
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
          AND b.s <= a.e + 250 + 1
    ) t
    WHERE c >= 1 AND s <= e AND s <= l AND r <= e AND e - s + 1 >= c
)
SELECT o0 AS "a", o1 AS "b", s, e
FROM n4
ORDER BY o0, o1, s, e;
