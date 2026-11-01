-- Reference SQL for
--
--   SELECT (a:, b:)
--   WHERE DURATION(a.cx < b.cx, 5)
--    THEN DURATION(a.cy < 160, 1)
--    THEN DURATION(a.cx > b.cx, 5)
--
-- An overtake with a midline touch in between: `a` trails `b` for 5 frames, is in
-- the upper half of the canvas at some point, then leads `b` for 5 frames.
--
-- Derived from the draft paper's CLEVRER Q5 (TQ2D) by dropping the middle phase's
-- proximity conjunct `|a.cx - b.cx| <= 45`. See TQ2D.sql for that version; the two
-- files differ in exactly one line, the r2 WHERE.
--
-- THEN is left-associative, so this is (P1 THEN P2) THEN P3.
--
-- Written against the DSL's denotation, sharing nothing with the engine's
-- interval-set algebra:
--
--   [P]             = { (s,e) : P holds at every frame in [s,e] }
--   [DURATION(E,k)] = { (s,e) in [E] : e-s+1 >= k }
--   [A THEN B]      = { (s1,e2) : (s1,e1) in [A], (s2,e2) in [B], s2 > e1 }
--
-- The method is greedy earliest-fit, not a formula over the interval encoding:
-- pack each phase as early as its run allows, and ask whether the next phase still
-- fits after it. That is optimal because a phase constrains the next one ONLY
-- through its end frame, and ending earlier is never worse -- so if any packing
-- works, the earliest one does. The same packing was validated against
-- brute-force enumeration of every split point on TQ2D (217 rows vs 217).
--
-- The answer span is [P1_run.s, P3_run.e]: starting earlier or ending later is
-- impossible, and both extremes are achievable whenever the triple is feasible,
-- since widening either end only enlarges the room the split points have.

WITH pairs AS (
    SELECT b0.fid                AS fid,
           b0.oid                AS o0,
           b1.oid                AS o1,
           (b0.x1 + b0.x2) / 2.0 AS a_cx,
           (b0.y1 + b0.y2) / 2.0 AS a_cy,
           (b1.x1 + b1.x2) / 2.0 AS b_cx
    FROM bbox b0
    JOIN bbox b1 ON b1.fid = b0.fid
    WHERE b0.oid <> b1.oid
),

-- Maximal runs of each phase's predicate. `fid - ROW_NUMBER()` is constant
-- exactly along consecutive frames, so grouping on it yields the runs; any
-- missing frame starts a new one. HAVING applies the DURATION minimum.
r1 AS (
    SELECT o0, o1, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT o0, o1, fid,
                 fid - ROW_NUMBER() OVER (PARTITION BY o0, o1 ORDER BY fid) AS run
          FROM pairs WHERE a_cx < b_cx) t
    GROUP BY o0, o1, run HAVING MAX(fid) - MIN(fid) + 1 >= 5
),
r2 AS (
    SELECT o0, o1, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT o0, o1, fid,
                 fid - ROW_NUMBER() OVER (PARTITION BY o0, o1 ORDER BY fid) AS run
          FROM pairs
          WHERE a_cy < 160.0) t
    GROUP BY o0, o1, run HAVING MAX(fid) - MIN(fid) + 1 >= 1
),
r3 AS (
    SELECT o0, o1, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT o0, o1, fid,
                 fid - ROW_NUMBER() OVER (PARTITION BY o0, o1 ORDER BY fid) AS run
          FROM pairs WHERE a_cx > b_cx) t
    GROUP BY o0, o1, run HAVING MAX(fid) - MIN(fid) + 1 >= 5
),

-- Earliest-fit packing across the three runs.
fit AS (
    SELECT A.o0, A.o1, A.s AS s, C.e AS e,
           A.s + 5 - 1                              AS e1,   -- earliest end of phase 1
           GREATEST(B.s, A.s + 5 - 1 + 1)           AS s2,   -- earliest start of phase 2
           GREATEST(B.s, A.s + 5 - 1 + 1) + 1 - 1   AS e2,   -- earliest end of phase 2
           B.e AS b_end, C.s AS c_start, C.e AS c_end
    FROM r1 A
    JOIN r2 B ON B.o0 = A.o0 AND B.o1 = A.o1
    JOIN r3 C ON C.o0 = A.o0 AND C.o1 = A.o1
),
raw AS (
    SELECT o0, o1, s, e
    FROM fit
    WHERE s2 + 1 - 1 <= b_end                     -- phase 2 still fits in its run
      AND GREATEST(c_start, e2 + 1) + 5 - 1 <= c_end  -- phase 3 still fits in its run
),

-- The engine reports merged, non-overlapping spans per object pair; different
-- run triples can produce overlapping answers. Merge on overlap or touching.
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
SELECT o0 AS "a", o1 AS "b", MIN(s) AS s, MAX(e) AS e
FROM grouped
GROUP BY o0, o1, grp
ORDER BY o0, o1, s, e;
