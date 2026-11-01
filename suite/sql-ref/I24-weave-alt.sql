-- i24, single object: a truck weaves lane1 -> lane2 -> lane1 -> lane2, 25 frames
-- settled in each phase, chained by plain (unbounded) THEN.
--
-- Phases 1 and 3 both draw from the lane-1 runs and 2 and 4 from lane-2 runs. They
-- cannot collapse onto the same run: runs are maximal, so a single uninterrupted
-- lane-1 run cannot contain a lane-2 excursion between two of its own phases.
--
-- Greedy earliest-fit down the chain (sound for an unbounded gap): a phase
-- constrains the next only through its end frame, so if any packing works the
-- earliest one does.
WITH o AS (
    SELECT b.fid AS fid, b.oid AS oid, (b.y1 + b.y2) / 2.0 AS cy
    FROM bbox b JOIN objects j ON j.oid = b.oid AND j.class = 'truck'
),
lane1 AS (SELECT oid, MIN(fid) s, MAX(fid) e FROM (
        SELECT oid, fid, fid - ROW_NUMBER() OVER (PARTITION BY oid ORDER BY fid) run
        FROM o WHERE cy > 21100.0 AND cy < 22200.0) t
       GROUP BY oid, run HAVING MAX(fid) - MIN(fid) + 1 >= 25),
lane2 AS (SELECT oid, MIN(fid) s, MAX(fid) e FROM (
        SELECT oid, fid, fid - ROW_NUMBER() OVER (PARTITION BY oid ORDER BY fid) run
        FROM o WHERE cy > 22200.0 AND cy < 23450.0) t
       GROUP BY oid, run HAVING MAX(fid) - MIN(fid) + 1 >= 25),
raw AS (
    SELECT A.oid AS oid, A.s AS s, D.e AS e FROM lane1 A
    JOIN lane2 B ON B.oid = A.oid
    JOIN lane1 C ON C.oid = A.oid
    JOIN lane2 D ON D.oid = A.oid
    WHERE GREATEST(B.s, A.s + 25) + 25 - 1 <= B.e
      AND GREATEST(C.s, GREATEST(B.s, A.s + 25) + 25) + 25 - 1 <= C.e
      AND GREATEST(D.s, GREATEST(C.s, GREATEST(B.s, A.s + 25) + 25) + 25) + 25 - 1 <= D.e
),
ordered AS (SELECT oid, s, e, MAX(e) OVER (PARTITION BY oid ORDER BY s, e
              ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) prev_e FROM raw),
grouped AS (SELECT oid, s, e, SUM(CASE WHEN prev_e IS NULL OR s > prev_e THEN 1 ELSE 0 END)
              OVER (PARTITION BY oid ORDER BY s, e ROWS UNBOUNDED PRECEDING) grp FROM ordered)
SELECT oid AS "v", MIN(s) AS s, MAX(e) AS e FROM grouped GROUP BY oid, grp ORDER BY oid, s, e;
