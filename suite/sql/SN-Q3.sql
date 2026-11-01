-- SN-Q3 (soccernet, 3 objects, THEN)
-- Turnover: a left-team player holds the ball, then a right-team player does within 2 s.
-- DSL: SELECT (bl: class=ball, p1: class=player_left, p2: class=player_right) WHERE (bl.cx > p1.x1 AND bl.cx < p1.x2 AND bl.cy > p1.y1 AND bl.cy < p1.y2) THEN[0..50] (bl.cx > p2.x1 AND bl.cx < p2.x2 AND bl.cy > p2.y1 AND bl.cy < p2.y2)
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
        CAST(b1.y2 AS DOUBLE) AS s1_y2
    FROM bbox b0
    JOIN objects j0 ON j0.oid = b0.oid AND j0.class = 'ball'
    JOIN bbox b1 ON b1.fid = b0.fid
    JOIN objects j1 ON j1.oid = b1.oid AND j1.class = 'player_left'
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
        WHERE (((((0.5 * s0_x1 + 0.5 * s0_x2) - (s1_x1) > 0.0) AND ((0.5 * s0_x1 + 0.5 * s0_x2) - (s1_x2) < 0.0)) AND ((0.5 * s0_y1 + 0.5 * s0_y2) - (s1_y1) > 0.0)) AND ((0.5 * s0_y1 + 0.5 * s0_y2) - (s1_y2) < 0.0))
    ) t
    GROUP BY o0, o1, island
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
    JOIN objects j0 ON j0.oid = b0.oid AND j0.class = 'ball'
    JOIN bbox b2 ON b2.fid = b0.fid
    JOIN objects j2 ON j2.oid = b2.oid AND j2.class = 'player_right'
    WHERE b0.oid <> b2.oid
),
n1 AS (
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
        WHERE (((((0.5 * s0_x1 + 0.5 * s0_x2) - (s2_x1) > 0.0) AND ((0.5 * s0_x1 + 0.5 * s0_x2) - (s2_x2) < 0.0)) AND ((0.5 * s0_y1 + 0.5 * s0_y2) - (s2_y1) > 0.0)) AND ((0.5 * s0_y1 + 0.5 * s0_y2) - (s2_y2) < 0.0))
    ) t
    GROUP BY o0, o2, island
),
n2 AS (
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
        FROM n0 a
        JOIN n1 b ON a.o0 = b.o0
        WHERE b.l - 0 >= 1
          AND b.l - 0 - a.c >= 0
          AND a.r <= b.l - 0 - 1
          AND b.s <= a.e + 50 + 1
    ) t
    WHERE c >= 1 AND s <= e AND s <= l AND r <= e AND e - s + 1 >= c
)
SELECT o0 AS "bl", o1 AS "p1", o2 AS "p2", s, e
FROM n2
ORDER BY o0, o1, o2, s, e;
