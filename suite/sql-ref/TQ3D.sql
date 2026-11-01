-- Reference SQL for
--
--   SELECT (a:, b:)
--   WHERE DURATION(a.cy > b.cy AND a.cx < 240, 15)
--    THEN DURATION(a.cx < 240 AND a.cx > b.cx AND a.cy < 160, 5)
--
-- Hand-written to match the DSL's *denotation*, deliberately sharing nothing
-- with the engine's implementation. `trql-sql` emits the (s, e, c, l, r)
-- interval-set encoding with its GREATEST/LEAST closed forms, which is our own
-- formulation -- comparing the engine against that tests the transcription, not
-- the semantics. This file tests the semantics.
--
-- The denotation being implemented:
--
--   [P]            = { (s,e) : P holds at every frame in [s,e] }
--   [DURATION(E,k)]= { (s,e) in [E] : e-s+1 >= k }
--   [A THEN B]     = { (s1,e2) : (s1,e1) in [A], (s2,e2) in [B], s2 > e1 }
--
-- and the answer reported per object pair is the maximal spans of that set,
-- overlapping ones merged -- which is what `IntervalSet::maximal()` returns.
--
-- The one subtlety, and the reason the obvious SQL is wrong: THEN quantifies
-- over ALL qualifying sub-intervals of each side, not just whole islands. The
-- natural `JOIN ... WHERE B.s > A.e` uses only e1 = A.e and s2 = B.s, so it
-- misses every case where the two predicates' islands overlap in time -- there
-- one can still cut the A part short and start the B part late. Verified: with
-- P1 true on [0,100] and P2 on [50,150], the engine returns [0,150] and the
-- naive join returns nothing.
--
-- So the split point is searched for explicitly, by enumerating it.

WITH pairs AS (
    -- Every ordered pair of distinct objects co-appearing in a frame: the SQL
    -- counterpart of an appearance group.
    SELECT b0.fid                     AS fid,
           b0.oid                     AS o0,
           b1.oid                     AS o1,
           (b0.x1 + b0.x2) / 2.0      AS a_cx,
           (b0.y1 + b0.y2) / 2.0      AS a_cy,
           (b1.x1 + b1.x2) / 2.0      AS b_cx,
           (b1.y1 + b1.y2) / 2.0      AS b_cy
    FROM bbox b0
    JOIN bbox b1 ON b1.fid = b0.fid
    WHERE b0.oid <> b1.oid
),

-- ---- phase 1: DURATION(a.cy > b.cy AND a.cx < 240, 15) --------------------
-- A frame run where the predicate holds without interruption. `fid - ROW_NUMBER()`
-- is constant exactly along consecutive frames, so grouping on it yields the
-- maximal runs; any missing frame starts a new one.
p1_runs AS (
    SELECT o0, o1, MIN(fid) AS s, MAX(fid) AS e
    FROM (
        SELECT o0, o1, fid,
               fid - ROW_NUMBER() OVER (PARTITION BY o0, o1 ORDER BY fid) AS run
        FROM pairs
        WHERE a_cy > b_cy AND a_cx < 240.0
    ) t
    GROUP BY o0, o1, run
    HAVING MAX(fid) - MIN(fid) + 1 >= 15          -- DURATION's minimum length
),

-- ---- phase 2: DURATION(a.cx < 240 AND a.cx > b.cx AND a.cy < 160, 5) ------
p2_runs AS (
    SELECT o0, o1, MIN(fid) AS s, MAX(fid) AS e
    FROM (
        SELECT o0, o1, fid,
               fid - ROW_NUMBER() OVER (PARTITION BY o0, o1 ORDER BY fid) AS run
        FROM pairs
        WHERE a_cx < 240.0 AND a_cx > b_cx AND a_cy < 160.0
    ) t
    GROUP BY o0, o1, run
    HAVING MAX(fid) - MIN(fid) + 1 >= 5
),

-- ---- THEN ----------------------------------------------------------------
-- For a run pair (A, B) the widest candidate answer is [A.s, B.e]: starting
-- earlier or ending later is impossible, and both are achievable whenever the
-- pairing is legal at all, since moving A's start earlier or B's end later only
-- widens the range the split point may fall in.
--
-- The pairing is legal iff some split exists: an end e1 for the A part that
-- leaves it >= 15 frames, and a start s2 for the B part that leaves it >= 5,
-- with s2 strictly after e1 (a plain THEN is a gap of 0 or more). Rather than
-- solve for that, enumerate the candidates and let the database look.
raw AS (
    SELECT A.o0 AS o0, A.o1 AS o1, A.s AS s, B.e AS e
    FROM p1_runs A
    JOIN p2_runs B ON B.o0 = A.o0 AND B.o1 = A.o1
    WHERE EXISTS (
        SELECT 1
        FROM generate_series(A.s + 15 - 1, A.e)     AS t1(e1)
        JOIN generate_series(B.s, B.e - 5 + 1)      AS t2(s2)
          ON t2.s2 - t1.e1 - 1 >= 0                 -- gap >= min_gap (= 0)
    )
),

-- ---- merge overlapping answers per object pair ---------------------------
-- Different run pairs can yield overlapping spans; the engine reports merged,
-- non-overlapping spans. Note it merges only when they overlap or touch
-- (`s <= prev_end`), NOT when merely adjacent (`s = prev_end + 1`).
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
