-- CL-Q2 (clevrer_merged, 3 objects, CONTAINS+DURATION)
-- Two objects meet strictly inside a spell where a third sits in the centre band.
-- DSL: REGIONS (c = [160,0,320,320]) SELECT (a:, b:, w:) WHERE CONTAINS(DURATION(w.center INSIDE c, 25), DURATION(a.x1 < b.x2 AND a.x2 > b.x1 AND a.y1 < b.y2 AND a.y2 > b.y1, 3))
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
        WHERE (((((0.5 * s2_x1 + 0.5 * s2_x2) >= 160.0) AND ((0.5 * s2_y1 + 0.5 * s2_y2) >= 0.0)) AND ((0.5 * s2_x1 + 0.5 * s2_x2) <= 320.0)) AND ((0.5 * s2_y1 + 0.5 * s2_y2) <= 320.0))
    ) t
    GROUP BY o0, o1, o2, island
),
n1 AS (
    SELECT o0, o1, o2,
           s, e,
           GREATEST(c, 25) AS c,
           l, r
    FROM n0
    WHERE e - s + 1 >= GREATEST(c, 25)
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
        WHERE (((((s0_x1) - (s1_x2) < 0.0) AND ((s0_x2) - (s1_x1) > 0.0)) AND ((s0_y1) - (s1_y2) < 0.0)) AND ((s0_y2) - (s1_y1) > 0.0))
    ) t
    GROUP BY o0, o1, o2, island
),
n3 AS (
    SELECT o0, o1, o2,
           s, e,
           GREATEST(c, 3) AS c,
           l, r
    FROM n2
    WHERE e - s + 1 >= GREATEST(c, 3)
),
n4 AS (
    SELECT DISTINCT o0, o1, o2, s, e, c, l, r
    FROM (
        SELECT a.o0 AS o0,
           a.o1 AS o1,
           a.o2 AS o2,
           a.s AS s,
           a.e AS e,
           GREATEST(a.c, b.c + 2) AS c,
           LEAST(a.l, b.l - 1, b.e - b.c) AS l,
           GREATEST(a.r, b.r + 1, b.s + b.c) AS r
        FROM n1 a
        JOIN n3 b ON a.o0 = b.o0 AND a.o1 = b.o1 AND a.o2 = b.o2
        WHERE a.s < b.l
          AND b.c <= b.e
    ) t
    WHERE c >= 1 AND s <= e AND s <= l AND r <= e AND e - s + 1 >= c
)
SELECT o0 AS "a", o1 AS "b", o2 AS "w", s, e
FROM n4
ORDER BY o0, o1, o2, s, e;
