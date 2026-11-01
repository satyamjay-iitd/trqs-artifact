-- Reference SQL for
--
--   SELECT (b: class=biker, p: class=pedestrian, q: class=pedestrian)
--   WHERE DURATION(b.cx > p.cx AND b.cx < q.cx AND q.cx < p.cx + 200, 90)
--
-- A biker threads between two pedestrians walking within 200 px of each other,
-- holding that position for 3 s (Stanford Drone is ~30 fps, so 90 frames).
--
-- `p` and `q` sit in one appearance group, so the engine can never bind both to
-- the same object; `j1.oid <> j2.oid` is how that shows up here. The pair is
-- ordered by the predicate itself (p is left of q), so a triple is reported once.
--
-- Written against the DSL's denotation, sharing nothing with the engine's
-- interval-set algebra:
--
--   [P]             = { (s,e) : P holds at every frame in [s,e] }
--   [DURATION(E,k)] = { (s,e) in [E] : e-s+1 >= k }
--
-- One DURATION over a conjunction, so the answer is just its maximal runs: any
-- sub-interval of a run is in the set too, and the engine reports merged spans.
WITH triples AS (
    SELECT b0.fid                AS fid,
           b0.oid                AS o0,
           b1.oid                AS o1,
           b2.oid                AS o2,
           (b0.x1 + b0.x2) / 2.0 AS b_cx,
           (b1.x1 + b1.x2) / 2.0 AS p_cx,
           (b2.x1 + b2.x2) / 2.0 AS q_cx
    FROM bbox b0
    JOIN objects j0 ON j0.oid = b0.oid AND j0.class = 'biker'
    JOIN bbox b1 ON b1.fid = b0.fid
    JOIN objects j1 ON j1.oid = b1.oid AND j1.class = 'pedestrian'
    JOIN bbox b2 ON b2.fid = b0.fid
    JOIN objects j2 ON j2.oid = b2.oid AND j2.class = 'pedestrian'
    WHERE j1.oid <> j2.oid
),

-- Maximal runs. `fid - ROW_NUMBER()` is constant exactly along consecutive
-- frames, so grouping on it yields the runs; HAVING applies the DURATION minimum.
runs AS (
    SELECT o0, o1, o2, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT o0, o1, o2, fid,
                 fid - ROW_NUMBER() OVER (PARTITION BY o0, o1, o2 ORDER BY fid) AS run
          FROM triples
          WHERE b_cx > p_cx AND b_cx < q_cx AND q_cx < p_cx + 200) t
    GROUP BY o0, o1, o2, run HAVING MAX(fid) - MIN(fid) + 1 >= 90
)
SELECT o0 AS "b", o1 AS "p", o2 AS "q", s, e
FROM runs
ORDER BY o0, o1, o2, s, e;
