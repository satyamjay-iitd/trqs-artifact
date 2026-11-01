-- Reference SQL for
--
--   SELECT (a: class=car, b: class=car)
--   WHERE DURATION(a.cy < b.cy, 25) THEN DURATION(a.cy > b.cy, 25)
--
-- Two cars swap depth order: `a` is further up the frame than `b` for 1 s, then
-- nearer than it for 1 s. On a fixed traffic camera, image-y is distance along
-- the road, so this is one vehicle drawing level with and passing another.
--
-- BOTH SLOTS ARE THE SAME CLASS, so the join must exclude a car pairing with
-- itself — `b0.oid <> b1.oid` below. `ST-swap`, the query this is shaped after,
-- needed no such guard because its two slots drew from different teams. The
-- engine rejects a slot sharing an object with another in the same group, so
-- without the guard SQL would report matches the engine cannot.
--
-- Two objects on purpose: this is the sensitive half of the detection-accuracy
-- experiment. Both tracks must survive detection, tracking and id alignment for
-- a match to appear, so it should degrade further on predicted input than the
-- one-object `DETRAC-dwell`.
--
-- CLASS. `car` is the one label DETRAC and COCO agree on; see `DETRAC-dwell.sql`.
--
-- COORDINATES ARE TRUNCATED, and this query must truncate the same way. DETRAC
-- stores bbox corners as DOUBLE (592.75, 378.8); the engine reads them with
-- `get_f64(...) as u32`, which truncates toward zero, so it sees 592 and 378.
-- Comparing against the exact values instead disagrees on interval edges — it
-- cost `DETRAC-dwell` an IoU of 0.9931 against an otherwise identical answer.
-- Every other dataset in the suite stores integers, so this is the first query
-- where it matters.
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
           (floor(b0.y1) + floor(b0.y2)) / 2.0 AS a_cy,
           (floor(b1.y1) + floor(b1.y2)) / 2.0 AS b_cy
    FROM bbox b0
    JOIN objects j0 ON j0.oid = b0.oid AND j0.class = 'car'
    JOIN bbox b1 ON b1.fid = b0.fid
    JOIN objects j1 ON j1.oid = b1.oid AND j1.class = 'car'
    WHERE b0.oid <> b1.oid
),

-- Runs of each phase within the pair's co-appearance: `pairs` holds only frames
-- where both cars exist, so a frame either is missing breaks the run.
r1 AS (
    SELECT o0, o1, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT o0, o1, fid,
                 fid - ROW_NUMBER() OVER (PARTITION BY o0, o1 ORDER BY fid) AS run
          FROM pairs WHERE a_cy < b_cy) t
    GROUP BY o0, o1, run HAVING MAX(fid) - MIN(fid) + 1 >= 25
),
r2 AS (
    SELECT o0, o1, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT o0, o1, fid,
                 fid - ROW_NUMBER() OVER (PARTITION BY o0, o1 ORDER BY fid) AS run
          FROM pairs WHERE a_cy > b_cy) t
    GROUP BY o0, o1, run HAVING MAX(fid) - MIN(fid) + 1 >= 25
),

-- Interval sets are downward closed, so it is not enough to ask whether the two
-- maximal runs happen to sequence. The first phase may start at A.s and needs 25
-- frames, so the second can begin no earlier than A.s + 25; inside B's run that
-- is GREATEST(B.s, A.s + 25), and 25 frames must still fit before B.e.
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
