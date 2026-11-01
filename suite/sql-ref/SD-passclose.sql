-- Reference SQL for
--
--   SELECT (b: class=biker, p: class=pedestrian)
--   WHERE CONTAINS(OVERLAP(DURATION(b.cx < p.cx, 60), DURATION(b.cy < p.cy, 60)),
--                  DURATION(b.cx <= p.cx + 100 AND b.cx >= p.cx - 100
--                       AND b.cy <= p.cy + 100 AND b.cy >= p.cy - 100, 30))
--
-- SD-approach with a proximity requirement: a biker passes a pedestrian
-- diagonally -- left of them for 2 s, overlapping a 2 s spell of being above
-- them -- and at some point strictly inside that pass comes within 100 px on
-- both axes for a full second. ~30 fps, so 60 frames = 2 s and 30 = 1 s.
--
-- WHY THE PROXIMITY CLAUSE. The OVERLAP alone constrains only *relative*
-- position, so a biker on the far side of the frame qualifies while it is
-- left-of and above. Requiring closeness cuts the answer from 23,206 to 3,305
-- intervals on the 60-video merged set.
--
-- WHY CONTAINS AND NOT AND. In this algebra AND is interval *equality*, which
-- would demand the close spell occupy exactly the interval of the whole pass.
-- CONTAINS asks only for strict containment, which is what "at some point
-- during" means.
--
-- WHY THE DISTANCE TEST LOOKS LIKE THAT. The DSL has no absolute value, so
-- |d| <= 100 becomes a conjunction, and a comparison may not mix variables on
-- one side, so the offset moves across the operator: `b.cx <= p.cx + 100`.
--
-- Written against the DSL's denotation, sharing nothing with the engine's
-- interval-set algebra:
--
--   [P]              = { (s,e) : P holds at every frame in [s,e] }
--   [DURATION(E,k)]  = { (s,e) in [E] : e-s+1 >= k }
--   [OVERLAP(A,B)]   = { (s1,e2) : (s1,e1) in [A], (s2,e2) in [B],
--                                  s1 < s2 <= e1 < e2 }
--   [CONTAINS(A,B)]  = { (s1,e1) in [A] : (s2,e2) in [B], s1 < s2, e2 < e1 }
--
-- THE OVERLAP LAYER is SD-approach's, unchanged: for maximal runs A=[a1,a2]
-- and B=[b1,b2] the widest result is [a1,b2], attainable exactly when some
-- witness pair fits, and taking the smallest admissible s2 decides it.
--
-- THE CONTAINS LAYER needs only that widest interval. [CONTAINS] filters
-- members of [OVERLAP], and strict containment is monotone in the container:
-- if some (S,E) in [OVERLAP] drawn from this run pair strictly contains a
-- proximity interval, then [a1,b2] does too, since a1 <= S and E <= b2. So the
-- run pair contributes exactly when [a1,b2] qualifies, and [a1,b2] is then the
-- maximal answer it contributes.
--
-- [DURATION(C,30)] is downward closed, so a proximity run [c1,c2] supplies a
-- witness strictly inside [S,E] exactly when the usable window
--
--   [ max(c1, S+1), min(c2, E-1) ]
--
-- is at least 30 frames long. Runs shorter than 30 cannot supply one at all,
-- which is what the HAVING prunes.
WITH pairs AS (
    SELECT b0.fid                AS fid,
           b0.oid                AS o0,
           b1.oid                AS o1,
           (b0.x1 + b0.x2) / 2.0 AS b_cx,
           (b1.x1 + b1.x2) / 2.0 AS p_cx,
           (b0.y1 + b0.y2) / 2.0 AS b_cy,
           (b1.y1 + b1.y2) / 2.0 AS p_cy
    FROM bbox b0
    JOIN objects j0 ON j0.oid = b0.oid AND j0.class = 'biker'
    JOIN bbox b1 ON b1.fid = b0.fid
    JOIN objects j1 ON j1.oid = b1.oid AND j1.class = 'pedestrian'
),

-- Maximal runs of each phase. `fid - ROW_NUMBER()` is constant exactly along
-- consecutive frames, so grouping on it yields the runs; HAVING applies the
-- DURATION minimum.
r1 AS (
    SELECT o0, o1, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT o0, o1, fid,
                 fid - ROW_NUMBER() OVER (PARTITION BY o0, o1 ORDER BY fid) AS run
          FROM pairs WHERE b_cx < p_cx) t
    GROUP BY o0, o1, run HAVING MAX(fid) - MIN(fid) + 1 >= 60
),
r2 AS (
    SELECT o0, o1, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT o0, o1, fid,
                 fid - ROW_NUMBER() OVER (PARTITION BY o0, o1 ORDER BY fid) AS run
          FROM pairs WHERE b_cy < p_cy) t
    GROUP BY o0, o1, run HAVING MAX(fid) - MIN(fid) + 1 >= 60
),
r3 AS (
    SELECT o0, o1, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT o0, o1, fid,
                 fid - ROW_NUMBER() OVER (PARTITION BY o0, o1 ORDER BY fid) AS run
          FROM pairs
          WHERE b_cx <= p_cx + 100 AND b_cx >= p_cx - 100
            AND b_cy <= p_cy + 100 AND b_cy >= p_cy - 100) t
    GROUP BY o0, o1, run HAVING MAX(fid) - MIN(fid) + 1 >= 30
),
raw AS (
    SELECT A.o0 AS o0, A.o1 AS o1, A.s AS s, B.e AS e
    FROM r1 A
    JOIN r2 B ON B.o0 = A.o0 AND B.o1 = A.o1
    WHERE GREATEST(B.s, A.s + 1) <= B.e - 60 + 1
      AND GREATEST(A.s + 60 - 1, GREATEST(B.s, A.s + 1)) <= LEAST(A.e, B.e - 1)
      AND EXISTS (
            SELECT 1 FROM r3 C
            WHERE C.o0 = A.o0 AND C.o1 = A.o1
              AND GREATEST(C.s, A.s + 1) + 30 - 1 <= LEAST(C.e, B.e - 1)
          )
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
SELECT o0 AS "b", o1 AS "p", MIN(s) AS s, MAX(e) AS e
FROM grouped
GROUP BY o0, o1, grp
ORDER BY o0, o1, s, e;
