-- I24-Q6 (i24_motion, 2 objects, DURATION)
-- Tailgating: a follows b at under 20 ft for 10 s straight, both in eastbound lane 1.
-- DSL: REGIONS (eb1 = [0, 21100, 2400000, 22200]) SELECT (a:, b:) WHERE DURATION(a.cx > b.cx - 2000 AND a.cx < b.cx AND a.center INSIDE eb1 AND b.center INSIDE eb1, 250)
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
        WHERE (((((0.5 * s0_x1 + 0.5 * s0_x2) - (0.5 * s1_x1 + 0.5 * s1_x2 + -2000.0) > 0.0) AND ((0.5 * s0_x1 + 0.5 * s0_x2) - (0.5 * s1_x1 + 0.5 * s1_x2) < 0.0)) AND (((((0.5 * s0_x1 + 0.5 * s0_x2) >= 0.0) AND ((0.5 * s0_y1 + 0.5 * s0_y2) >= 21100.0)) AND ((0.5 * s0_x1 + 0.5 * s0_x2) <= 2400000.0)) AND ((0.5 * s0_y1 + 0.5 * s0_y2) <= 22200.0))) AND (((((0.5 * s1_x1 + 0.5 * s1_x2) >= 0.0) AND ((0.5 * s1_y1 + 0.5 * s1_y2) >= 21100.0)) AND ((0.5 * s1_x1 + 0.5 * s1_x2) <= 2400000.0)) AND ((0.5 * s1_y1 + 0.5 * s1_y2) <= 22200.0)))
    ) t
    GROUP BY o0, o1, island
),
n1 AS (
    SELECT o0, o1,
           s, e,
           GREATEST(c, 250) AS c,
           l, r
    FROM n0
    WHERE e - s + 1 >= GREATEST(c, 250)
)
SELECT o0 AS "a", o1 AS "b", s, e
FROM n1
ORDER BY o0, o1, s, e;
