-- Reference SQL for
--
--   REGIONS (crossing = [37500, 46210, 45000, 51210])
--   SELECT (c: class="vehicle.car", t: class="vehicle.truck",
--           p: class="human.pedestrian.adult")
--   WHERE DURATION(c.center INSIDE crossing AND t.center INSIDE crossing
--                  AND p.center INSIDE crossing, 25)
--
-- A car, a truck and a pedestrian are all inside the crossing at the same time
-- for >= 25 frames (~2 s at 12 fps) without any of them leaving it. One group of
-- three variables, so the whole query is a single DURATION over a per-frame
-- conjunction -- no temporal operator and no gap to reason about.
--
-- Two things this shares with NS-crossing.sql:
--
-- 1. The predicate is evaluated over the TRIPLE'S CO-APPEARANCE, not over each
--    object's whole life. A frame in which any one of the three is missing is
--    not a frame the conjunction can hold on, and it breaks the run. The
--    `triples` CTE therefore joins all three on `fid`, which admits only frames
--    where all three exist.
--
-- 2. The region is the same busiest 25 m cell as NS-crossing, widened to its
--    busy neighbours: x 375..450 m, y 1606..1656 m in stored units.
--
-- Unlike NS-crossing there is no downward-closure subtlety to handle. The
-- answer set is every sub-interval of length >= 25 of a qualifying run, and
-- `maximal()` reports the runs themselves, so island detection with a HAVING on
-- the run length is exactly the answer.
WITH triples AS (
    SELECT b0.fid AS fid, b0.oid AS o0, b1.oid AS o1, b2.oid AS o2
    FROM bbox b0
    JOIN objects j0 ON j0.oid = b0.oid AND j0.class = 'vehicle.car'
    JOIN bbox b1 ON b1.fid = b0.fid
    JOIN objects j1 ON j1.oid = b1.oid AND j1.class = 'vehicle.truck'
    JOIN bbox b2 ON b2.fid = b0.fid
    JOIN objects j2 ON j2.oid = b2.oid AND j2.class = 'human.pedestrian.adult'
    WHERE (b0.x1 + b0.x2) / 2.0 BETWEEN 37500 AND 45000
      AND (b0.y1 + b0.y2) / 2.0 BETWEEN 46210 AND 51210
      AND (b1.x1 + b1.x2) / 2.0 BETWEEN 37500 AND 45000
      AND (b1.y1 + b1.y2) / 2.0 BETWEEN 46210 AND 51210
      AND (b2.x1 + b2.x2) / 2.0 BETWEEN 37500 AND 45000
      AND (b2.y1 + b2.y2) / 2.0 BETWEEN 46210 AND 51210
),
runs AS (
    SELECT o0, o1, o2, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT o0, o1, o2, fid,
                 fid - ROW_NUMBER() OVER (PARTITION BY o0, o1, o2 ORDER BY fid) AS run
          FROM triples) t
    GROUP BY o0, o1, o2, run
    HAVING MAX(fid) - MIN(fid) + 1 >= 25
)
SELECT o0 AS "c", o1 AS "t", o2 AS "p", s, e
FROM runs ORDER BY o0, o1, o2, s, e;
