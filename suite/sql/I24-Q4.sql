-- I24-Q4 (i24_motion, 3 objects, OVERLAP+DURATION)
-- Platoon forms and breaks: two vehicles' 2 s spells in eastbound lane 1 overlap without either containing the other, while a third is in lane 2.
-- DSL: REGIONS (eb1 = [0, 21100, 2400000, 22200], eb2 = [0, 22200, 2400000, 23450]) SELECT (a:, b:, c:) WHERE OVERLAP(DURATION(a.center INSIDE eb1 AND c.center INSIDE eb2, 50), DURATION(b.center INSIDE eb1 AND c.center INSIDE eb2, 50))
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
        b1.y2 AS s1_y2,
        b2.oid AS o2,
        b2.x1 AS s2_x1,
        b2.y1 AS s2_y1,
        b2.x2 AS s2_x2,
        b2.y2 AS s2_y2
    FROM bbox b0
    JOIN bbox b1 ON b1.fid = b0.fid
    JOIN bbox b2 ON b2.fid = b0.fid
    WHERE b0.oid <> b1.oid
      AND b0.oid <> b2.oid
      AND b1.oid <> b2.oid
),
n0 AS (
    SELECT o0, o1, o2,
           MIN(fid) AS s,
           MAX(fid) AS e,
           1 AS c,
           MAX(fid) AS l,
           MIN(fid) AS r
    FROM (
        SELECT o0, o1, o2, fid,
               fid - ROW_NUMBER() OVER (PARTITION BY o0, o1, o2 ORDER BY fid) AS island
        FROM g0_frames
        WHERE ((((((0.5 * s0_x1 + 0.5 * s0_x2) >= 0.0) AND ((0.5 * s0_y1 + 0.5 * s0_y2) >= 21100.0)) AND ((0.5 * s0_x1 + 0.5 * s0_x2) <= 2400000.0)) AND ((0.5 * s0_y1 + 0.5 * s0_y2) <= 22200.0)) AND (((((0.5 * s2_x1 + 0.5 * s2_x2) >= 0.0) AND ((0.5 * s2_y1 + 0.5 * s2_y2) >= 22200.0)) AND ((0.5 * s2_x1 + 0.5 * s2_x2) <= 2400000.0)) AND ((0.5 * s2_y1 + 0.5 * s2_y2) <= 23450.0)))
    ) t
    GROUP BY o0, o1, o2, island
),
n1 AS (
    SELECT o0, o1, o2,
           s, e,
           GREATEST(c, 50) AS c,
           l, r
    FROM n0
    WHERE e - s + 1 >= GREATEST(c, 50)
),
n2 AS (
    SELECT o0, o1, o2,
           MIN(fid) AS s,
           MAX(fid) AS e,
           1 AS c,
           MAX(fid) AS l,
           MIN(fid) AS r
    FROM (
        SELECT o0, o1, o2, fid,
               fid - ROW_NUMBER() OVER (PARTITION BY o0, o1, o2 ORDER BY fid) AS island
        FROM g0_frames
        WHERE ((((((0.5 * s1_x1 + 0.5 * s1_x2) >= 0.0) AND ((0.5 * s1_y1 + 0.5 * s1_y2) >= 21100.0)) AND ((0.5 * s1_x1 + 0.5 * s1_x2) <= 2400000.0)) AND ((0.5 * s1_y1 + 0.5 * s1_y2) <= 22200.0)) AND (((((0.5 * s2_x1 + 0.5 * s2_x2) >= 0.0) AND ((0.5 * s2_y1 + 0.5 * s2_y2) >= 22200.0)) AND ((0.5 * s2_x1 + 0.5 * s2_x2) <= 2400000.0)) AND ((0.5 * s2_y1 + 0.5 * s2_y2) <= 23450.0)))
    ) t
    GROUP BY o0, o1, o2, island
),
n3 AS (
    SELECT o0, o1, o2,
           s, e,
           GREATEST(c, 50) AS c,
           l, r
    FROM n2
    WHERE e - s + 1 >= GREATEST(c, 50)
),
n4 AS (
    SELECT DISTINCT o0, o1, o2, s, e, c, l, r
    FROM (
        SELECT a.o0 AS o0,
           a.o1 AS o1,
           a.o2 AS o2,
           a.s AS s,
           b.e AS e,
           GREATEST(a.c + 1, b.c + 1, 3) AS c,
           LEAST(a.l, a.e - a.c + 1, a.e - 1, b.l - 1) AS l,
           GREATEST(b.r, a.r + 1, b.s + b.c - 1, b.s + 1) AS r
        FROM n1 a
        JOIN n3 b ON a.o0 = b.o0 AND a.o1 = b.o1 AND a.o2 = b.o2
        WHERE b.s <= a.e
          AND a.s < a.e
          AND a.s < b.l
    ) t
    WHERE c >= 1 AND s <= e AND s <= l AND r <= e AND e - s + 1 >= c
)
SELECT o0 AS "a", o1 AS "b", o2 AS "c", s, e
FROM n4
ORDER BY o0, o1, o2, s, e;
