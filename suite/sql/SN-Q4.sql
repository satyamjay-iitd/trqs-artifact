-- SN-Q4 (soccernet, 3 objects, CONTAINS+DURATION)
-- The referee moves between an opposing pair, strictly inside a 2 s spell in which that pair holds its left-right ordering.
-- DSL: SELECT (a: class=player_left, b: class=player_right, r: class=referee) WHERE CONTAINS(DURATION(a.cx < b.cx, 50), DURATION(r.cx > a.cx AND r.cx < b.cx, 10))
-- compiled for sub-dataset: test_SNMOT-116

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
        CAST(b1.y2 AS DOUBLE) AS s1_y2,
        b2.oid AS o2,
        CAST(b2.x1 AS DOUBLE) AS s2_x1,
        CAST(b2.y1 AS DOUBLE) AS s2_y1,
        CAST(b2.x2 AS DOUBLE) AS s2_x2,
        CAST(b2.y2 AS DOUBLE) AS s2_y2
    FROM bbox b0
    JOIN objects j0 ON j0.oid = b0.oid AND j0.class = 'player_left'
    JOIN bbox b1 ON b1.fid = b0.fid
    JOIN objects j1 ON j1.oid = b1.oid AND j1.class = 'player_right'
    JOIN bbox b2 ON b2.fid = b0.fid
    JOIN objects j2 ON j2.oid = b2.oid AND j2.class = 'referee'
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
        WHERE ((0.5 * s0_x1 + 0.5 * s0_x2) - (0.5 * s1_x1 + 0.5 * s1_x2) < 0.0)
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
        WHERE (((0.5 * s2_x1 + 0.5 * s2_x2) - (0.5 * s0_x1 + 0.5 * s0_x2) > 0.0) AND ((0.5 * s2_x1 + 0.5 * s2_x2) - (0.5 * s1_x1 + 0.5 * s1_x2) < 0.0))
    ) t
    GROUP BY o0, o1, o2, island
),
n3 AS (
    SELECT o0, o1, o2,
           s, e,
           GREATEST(c, 10) AS c,
           l, r
    FROM n2
    WHERE e - s + 1 >= GREATEST(c, 10)
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
SELECT o0 AS "a", o1 AS "b", o2 AS "r", s, e
FROM n4
ORDER BY o0, o1, o2, s, e;
