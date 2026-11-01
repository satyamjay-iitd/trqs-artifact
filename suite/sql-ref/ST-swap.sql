-- Reference SQL for
--
--   SELECT (a: class=playerA, b: class=playerB)
--   WHERE DURATION(a.cx < b.cx, 25) THEN DURATION(a.cx > b.cx, 25)
--
-- An opposing pair swapping their along-pitch order: `a` (team A) is behind `b`
-- (team B) for 1 s, then ahead of it for 1 s. 25 fps, so 25 frames = 1 s.
--
-- PURELY RELATIONAL, and that is a requirement, not a style choice. The 20 merged
-- matches disagree on `x_delta`/`y_delta` (spread 15.75 m and 9.72 m on a
-- 105x68 m pitch), so `stored = metres*100 + delta` puts the pitch in a different
-- place per source block and no absolute region is portable across the merge.
-- A comparison between two objects is safe because both share one match's frame,
-- so the deltas cancel. See `merged_inconsistent` in the dataset's metadata.json.
--
-- Written against the DSL's denotation, sharing nothing with the engine's
-- interval-set algebra:
--
--   [P]             = { (s,e) : P holds at every frame in [s,e] }
--   [DURATION(E,k)] = { (s,e) in [E] : e-s+1 >= k }
--   [A THEN B]      = { (s1,e2) : (s1,e1) in [A], (s2,e2) in [B], s2 > e1 }
--
-- Greedy earliest-fit for the THEN, sound for a plain (unbounded) gap: a phase
-- constrains the next only through its end frame, and ending earlier is never
-- worse, so if any packing works the earliest one does.
WITH pairs AS (
    SELECT b0.fid                AS fid,
           b0.oid                AS o0,
           b1.oid                AS o1,
           (b0.x1 + b0.x2) / 2.0 AS a_cx,
           (b1.x1 + b1.x2) / 2.0 AS b_cx
    FROM bbox b0
    JOIN objects j0 ON j0.oid = b0.oid AND j0.class = 'playerA'
    JOIN bbox b1 ON b1.fid = b0.fid
    JOIN objects j1 ON j1.oid = b1.oid AND j1.class = 'playerB'
),

-- Maximal runs of each phase. `fid - ROW_NUMBER()` is constant exactly along
-- consecutive frames, so grouping on it yields the runs; HAVING applies the
-- DURATION minimum.
r1 AS (
    SELECT o0, o1, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT o0, o1, fid,
                 fid - ROW_NUMBER() OVER (PARTITION BY o0, o1 ORDER BY fid) AS run
          FROM pairs WHERE a_cx < b_cx) t
    GROUP BY o0, o1, run HAVING MAX(fid) - MIN(fid) + 1 >= 25
),
r2 AS (
    SELECT o0, o1, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT o0, o1, fid,
                 fid - ROW_NUMBER() OVER (PARTITION BY o0, o1 ORDER BY fid) AS run
          FROM pairs WHERE a_cx > b_cx) t
    GROUP BY o0, o1, run HAVING MAX(fid) - MIN(fid) + 1 >= 25
),
raw AS (
    SELECT A.o0 AS o0, A.o1 AS o1, A.s AS s, B.e AS e
    FROM r1 A
    JOIN r2 B ON B.o0 = A.o0 AND B.o1 = A.o1
    WHERE GREATEST(B.s, A.s + 25) + 25 - 1 <= B.e
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
SELECT o0 AS "a", o1 AS "b", MIN(s) AS s, MAX(e) AS e
FROM grouped
GROUP BY o0, o1, grp
ORDER BY o0, o1, s, e;
