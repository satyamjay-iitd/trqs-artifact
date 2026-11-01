-- Reference SQL for
--
--   SELECT (s: class=semi, t: class=truck, v: class=van)
--   WHERE DURATION(s.cy > 21100 AND s.cy < 22200
--              AND t.cy > 21100 AND t.cy < 22200
--              AND v.cy > 21100 AND v.cy < 22200, 50)
--
-- A semi, a truck and a van are all in eastbound lane 1 at the same time for
-- >= 50 frames (2 s at 25 fps), none of them leaving the lane. One group of
-- three variables: a single DURATION over a per-frame conjunction, so there is
-- no temporal operator and no gap to reason about.
--
-- Lane band in stored centi-feet (feet = (stored - 20000)/100): eb1 is
-- 21100..22200, the same band I24-lane-truck uses for its first phase.
--
-- As in NS-standoff.sql, the predicate is evaluated over the TRIPLE'S
-- CO-APPEARANCE: a frame in which any of the three is missing cannot satisfy the
-- conjunction and breaks the run, which the join on `fid` enforces. The answer
-- set is every sub-interval of length >= 50 of a qualifying run, and `maximal()`
-- reports the runs themselves, so island detection with a HAVING on run length
-- is exactly the answer.
--
-- The band filter is pushed into each per-class CTE rather than applied after
-- the join: unfiltered this is a 3-way self-join over 280M rows, where filtering
-- first leaves 126K semi, 33K truck and 1.2M van rows.
WITH s AS (
    SELECT b.fid, b.oid FROM bbox b JOIN objects j ON j.oid = b.oid AND j.class = 'semi'
    WHERE (b.y1 + b.y2) / 2.0 > 21100 AND (b.y1 + b.y2) / 2.0 < 22200
),
t AS (
    SELECT b.fid, b.oid FROM bbox b JOIN objects j ON j.oid = b.oid AND j.class = 'truck'
    WHERE (b.y1 + b.y2) / 2.0 > 21100 AND (b.y1 + b.y2) / 2.0 < 22200
),
v AS (
    SELECT b.fid, b.oid FROM bbox b JOIN objects j ON j.oid = b.oid AND j.class = 'van'
    WHERE (b.y1 + b.y2) / 2.0 > 21100 AND (b.y1 + b.y2) / 2.0 < 22200
),
triples AS (
    SELECT s.fid AS fid, s.oid AS o0, t.oid AS o1, v.oid AS o2
    FROM s JOIN t ON t.fid = s.fid JOIN v ON v.fid = s.fid
),
runs AS (
    SELECT o0, o1, o2, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT o0, o1, o2, fid,
                 fid - ROW_NUMBER() OVER (PARTITION BY o0, o1, o2 ORDER BY fid) AS run
          FROM triples) x
    GROUP BY o0, o1, o2, run
    HAVING MAX(fid) - MIN(fid) + 1 >= 50
)
SELECT o0 AS "s", o1 AS "t", o2 AS "v", s, e
FROM runs ORDER BY o0, o1, o2, s, e;
