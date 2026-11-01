-- Reference SQL for
--
--   SELECT (v: class=car)
--   WHERE DURATION(v.cy < 100, 10)
--    THEN DURATION(v.cy > 120 AND v.cy < 180, 10)
--    THEN DURATION(v.cy > 250, 10)
--
-- One vehicle crossing the frame top to bottom, through a band in the middle:
-- one object, three phases. Where DETRAC-dwell isolates the per-frame scan,
-- this one isolates temporal composition -- three gaps-and-islands passes and
-- two sequencing joins here, three interval sets and two THEN operators there.
--
--   [P]             = { (s,e) : P holds at every frame in [s,e] }
--   [DURATION(E,k)] = { (s,e) in [E] : e-s+1 >= k }
--   [A THEN B]      = { (s1,e2) : (s1,e1) in [A], (s2,e2) in [B], s2 > e1 }
--
-- Greedy earliest-fit, sound for a plain (unbounded) gap: a phase constrains the
-- next only through its end frame, and ending earlier is never worse. Phase 1
-- may start at A.s and needs 10 frames, so phase 2 starts no earlier than
-- A.s + 10 -- inside B's run that is GREATEST(B.s, A.s + 10) -- and phase 3 no
-- earlier than 10 frames after that.
WITH v AS (
    SELECT b.oid AS oid, b.fid AS fid, (b.y1 + b.y2) / 2.0 AS cy
    FROM bbox b
    JOIN objects j ON j.oid = b.oid AND j.class = 'car'
),
r1 AS (
    SELECT oid, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT oid, fid, fid - ROW_NUMBER() OVER (PARTITION BY oid ORDER BY fid) AS run
          FROM v WHERE cy < 100) t
    GROUP BY oid, run HAVING MAX(fid) - MIN(fid) + 1 >= 10
),
r2 AS (
    SELECT oid, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT oid, fid, fid - ROW_NUMBER() OVER (PARTITION BY oid ORDER BY fid) AS run
          FROM v WHERE cy > 120 AND cy < 180) t
    GROUP BY oid, run HAVING MAX(fid) - MIN(fid) + 1 >= 10
),
r3 AS (
    SELECT oid, MIN(fid) AS s, MAX(fid) AS e
    FROM (SELECT oid, fid, fid - ROW_NUMBER() OVER (PARTITION BY oid ORDER BY fid) AS run
          FROM v WHERE cy > 250) t
    GROUP BY oid, run HAVING MAX(fid) - MIN(fid) + 1 >= 10
),
raw AS (
    SELECT A.oid AS oid, A.s AS s, C.e AS e
    FROM r1 A
    JOIN r2 B ON B.oid = A.oid
    JOIN r3 C ON C.oid = A.oid
    WHERE GREATEST(B.s, A.s + 10) + 10 - 1 <= B.e
      AND GREATEST(C.s, GREATEST(B.s, A.s + 10) + 10) + 10 - 1 <= C.e
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
FROM grouped GROUP BY oid, grp ORDER BY 1, 2, 3;
