-- Reference SQL for
--
--   SELECT (c: class=car, p: class=pedestrian)
--   WHERE OVERLAP(DURATION(c.cx < p.cx, 60), DURATION(c.cy < p.cy, 60))
--
-- A car passes a pedestrian on a campus path: left of them for 2 s, and that
-- spell properly overlaps a 2 s spell of being above them, so the car closes in
-- on one axis while already committed on the other. ~30 fps, so 60 frames = 2 s.
--
-- WHY car AND NOT biker. The same shape over `biker` (the earlier SD-approach)
-- runs over the dataset's largest pair set, 4,210 bikers against 5,232
-- pedestrians. Vehicle-pedestrian encounters are the ones worth finding on a
-- campus, and `car`'s 316 objects make the candidate set ~13x smaller.
--
-- Written against the DSL's denotation, sharing nothing with the engine's
-- interval-set algebra:
--
--   [P]             = { (s,e) : P holds at every frame in [s,e] }
--   [DURATION(E,k)] = { (s,e) in [E] : e-s+1 >= k }
--   [OVERLAP(A,B)]  = { (s1,e2) : (s1,e1) in [A], (s2,e2) in [B],
--                                 s1 < s2 <= e1 < e2 }
--
-- Allen's *overlaps*, proper and ordered: neither interval contains the other,
-- and the result spans the union.
--
-- Maximal runs are not enough on their own here, because [DURATION] is
-- downward closed: the witness sub-intervals (s1,e1) and (s2,e2) may be shorter
-- than the runs that contain them. For maximal runs A=[a1,a2] and B=[b1,b2],
-- the widest result is [a1,b2] -- s1 is smallest and e2 largest there, and every
-- other choice lies inside it -- and it is attainable exactly when some witness
-- pair fits:
--
--   s2 >= max(b1, a1+1)                 (s2 in B, and strictly after s1=a1)
--   s2 <= b2-k+1                        ((s2,b2) is at least k long)
--   max(a1+k-1, s2) <= min(a2, b2-1)    (an e1 exists: at least k from a1,
--                                        inside A, at or after s2, before e2)
--
-- Taking the smallest admissible s2 makes both conditions as easy as possible,
-- so testing that one value decides feasibility.
WITH pairs AS (
    SELECT b0.fid                AS fid,
           b0.oid                AS o0,
           b1.oid                AS o1,
           (b0.x1 + b0.x2) / 2.0 AS c_cx,
           (b1.x1 + b1.x2) / 2.0 AS p_cx,
           (b0.y1 + b0.y2) / 2.0 AS c_cy,
           (b1.y1 + b1.y2) / 2.0 AS p_cy
    FROM bbox b0
    JOIN objects j0 ON j0.oid = b0.oid AND j0.class = 'car'
    JOIN bbox b1 ON b1.fid = b0.fid
    JOIN objects j1 ON j1.oid = b1.oid AND j1.class = 'pedestrian'
),
r1 AS (
    SELECT o0, o1, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT o0, o1, fid,
                 fid - ROW_NUMBER() OVER (PARTITION BY o0, o1 ORDER BY fid) AS run
          FROM pairs WHERE c_cx < p_cx) t
    GROUP BY o0, o1, run HAVING MAX(fid) - MIN(fid) + 1 >= 60
),
r2 AS (
    SELECT o0, o1, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT o0, o1, fid,
                 fid - ROW_NUMBER() OVER (PARTITION BY o0, o1 ORDER BY fid) AS run
          FROM pairs WHERE c_cy < p_cy) t
    GROUP BY o0, o1, run HAVING MAX(fid) - MIN(fid) + 1 >= 60
),
raw AS (
    SELECT A.o0 AS o0, A.o1 AS o1, A.s AS s, B.e AS e
    FROM r1 A
    JOIN r2 B ON B.o0 = A.o0 AND B.o1 = A.o1
    WHERE GREATEST(B.s, A.s + 1) <= B.e - 60 + 1
      AND GREATEST(A.s + 60 - 1, GREATEST(B.s, A.s + 1)) <= LEAST(A.e, B.e - 1)
),

-- The engine reports merged, non-overlapping spans per pair.
ordered AS (
    SELECT o0, o1, s, e,
           MAX(e) OVER (PARTITION BY o0, o1 ORDER BY s, e
                        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_e
    FROM raw
),
grouped AS (
    SELECT o0, o1, s, e,
           SUM(CASE WHEN prev_e IS NULL OR s > prev_e THEN 1 ELSE 0 END)
               OVER (PARTITION BY o0, o1 ORDER BY s, e ROWS UNBOUNDED PRECEDING) AS grp
    FROM ordered
)
SELECT o0 AS "c", o1 AS "p", MIN(s) AS s, MAX(e) AS e
FROM grouped
GROUP BY o0, o1, grp
ORDER BY o0, o1, s, e;
