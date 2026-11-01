-- Reference SQL for
--
--   SELECT (v: class=car)
--   WHERE DURATION(v.cy > 100, 50)
--
-- A vehicle that stays in the lower part of the frame for two seconds at 25 fps.
-- One object, one predicate: the simplest shape in the suite, and the one that
-- isolates what a frame-expanded store costs on its own, with no pair join to
-- share the blame.
--
-- Written against the DSL's denotation:
--
--   [P]             = { (s,e) : P holds at every frame in [s,e] }
--   [DURATION(E,k)] = { (s,e) in [E] : e-s+1 >= k }
--
-- `fid - ROW_NUMBER()` is constant exactly along consecutive frames, so grouping
-- on it yields the maximal runs; HAVING applies the DURATION minimum. Maximal
-- runs of one predicate are already disjoint, so no merging step is needed.
WITH v AS (
    SELECT b.oid AS oid, b.fid AS fid, (b.y1 + b.y2) / 2.0 AS cy
    FROM bbox b
    JOIN objects j ON j.oid = b.oid AND j.class = 'car'
)
SELECT oid AS "v", MIN(fid) AS s, MAX(fid) AS e
FROM (SELECT oid, fid,
             fid - ROW_NUMBER() OVER (PARTITION BY oid ORDER BY fid) AS run
      FROM v WHERE cy > 100) t
GROUP BY oid, run
HAVING MAX(fid) - MIN(fid) + 1 >= 50
ORDER BY 1, 2, 3;
