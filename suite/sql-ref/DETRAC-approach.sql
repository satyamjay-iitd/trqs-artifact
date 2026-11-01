-- Reference SQL for
--
--   SELECT (v: class=car)
--   WHERE DURATION(v.cy < 100, 25) THEN DURATION(v.cy > 200, 25)
--
-- One car drives toward the camera: 1 s high in the frame, then 1 s low in it.
-- On a fixed traffic camera image-y is distance along the road, so this is
-- approach rather than an arbitrary pair of bands.
--
-- WHY THIS QUERY EXISTS. `DETRAC-dwell` is one object and one DURATION;
-- `DETRAC-pass` is two objects *and* a THEN. Between those two alone, a drop in
-- accuracy on detector output cannot be attributed to arity rather than to
-- temporal composition. This is one object with a THEN, which separates them:
-- whatever it loses is composition, and whatever `DETRAC-pass` loses beyond it
-- is arity.
--
-- The two bands do not touch (100..200 belongs to neither), so a match needs a
-- real traversal rather than jitter across one threshold.
--
-- COORDINATES ARE TRUNCATED, and this query must truncate the same way. DETRAC
-- stores bbox corners as DOUBLE (592.75, 378.8); the engine reads them with
-- `get_f64(...) as u32`, which truncates toward zero, so it sees 592 and 378.
-- See `DETRAC-dwell.sql` — comparing against the exact values disagrees on
-- interval edges.
--
-- CLASS. `car` is the one label DETRAC and COCO agree on; see `DETRAC-dwell.sql`.
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
WITH boxes AS (
    SELECT b.fid AS fid,
           b.oid AS oid,
           (floor(b.y1) + floor(b.y2)) / 2.0 AS cy
    FROM bbox b
    JOIN objects o ON o.oid = b.oid AND o.class = 'car'
),

-- Maximal runs of each phase. `fid - ROW_NUMBER()` is constant exactly along
-- consecutive frames, so grouping on it yields the runs; HAVING applies the
-- DURATION minimum. A frame where the object is absent breaks the run, which is
-- what `max_gap = 1` does too.
r1 AS (
    SELECT oid, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT oid, fid,
                 fid - ROW_NUMBER() OVER (PARTITION BY oid ORDER BY fid) AS run
          FROM boxes WHERE cy < 100) t
    GROUP BY oid, run HAVING MAX(fid) - MIN(fid) + 1 >= 25
),
r2 AS (
    SELECT oid, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT oid, fid,
                 fid - ROW_NUMBER() OVER (PARTITION BY oid ORDER BY fid) AS run
          FROM boxes WHERE cy > 200) t
    GROUP BY oid, run HAVING MAX(fid) - MIN(fid) + 1 >= 25
),

-- Interval sets are downward closed, so it is not enough to ask whether the two
-- maximal runs happen to sequence. The first phase may start at A.s and needs 25
-- frames, so the second can begin no earlier than A.s + 25; inside B's run that
-- is GREATEST(B.s, A.s + 25), and 25 frames must still fit before B.e.
raw AS (
    SELECT A.oid AS oid, A.s AS s, B.e AS e
    FROM r1 A
    JOIN r2 B ON B.oid = A.oid
    WHERE GREATEST(B.s, A.s + 25) + 25 - 1 <= B.e
),

-- The engine reports merged, non-overlapping spans per object.
ordered AS (
    SELECT oid, s, e,
           MAX(e) OVER (PARTITION BY oid ORDER BY s, e
                        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_e
    FROM raw
),
grouped AS (
    SELECT oid, s, e,
           SUM(CASE WHEN prev_e IS NULL OR s > prev_e THEN 1 ELSE 0 END)
               OVER (PARTITION BY oid ORDER BY s, e ROWS UNBOUNDED PRECEDING) AS grp
    FROM ordered
)
SELECT oid AS "v", MIN(s) AS s, MAX(e) AS e
FROM grouped
GROUP BY oid, grp
ORDER BY oid, s, e;
