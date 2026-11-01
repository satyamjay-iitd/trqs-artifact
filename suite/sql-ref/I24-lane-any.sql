-- i24 lane change, single object: 2 s settled in eastbound lane 1, then 2 s in lane 2.
-- Lane bands are the measured histogram valleys, in stored centi-feet
-- (feet = (stored - delta)/scale): lane 1 = 11..22 ft, lane 2 = 22..34.5 ft.
--
-- One object, so there is no self-join at all -- this is the purest test of
-- SQL's cost being proportional to the table it must scan.
--
-- THEN is a plain (unbounded) gap. Greedy earliest-fit: a phase constrains the
-- next only through its end frame, and ending earlier is never worse, so if any
-- packing works the earliest one does.
WITH o AS (
    SELECT b.fid AS fid, b.oid AS oid, (b.y1 + b.y2) / 2.0 AS cy
    FROM bbox b
    
),
r1 AS (
    SELECT oid, MIN(fid) AS s, MAX(fid) AS e FROM (
        SELECT oid, fid, fid - ROW_NUMBER() OVER (PARTITION BY oid ORDER BY fid) AS run
        FROM o WHERE cy > 21100.0 AND cy < 22200.0) t
    GROUP BY oid, run HAVING MAX(fid) - MIN(fid) + 1 >= 50
),
r2 AS (
    SELECT oid, MIN(fid) AS s, MAX(fid) AS e FROM (
        SELECT oid, fid, fid - ROW_NUMBER() OVER (PARTITION BY oid ORDER BY fid) AS run
        FROM o WHERE cy > 22200.0 AND cy < 23450.0) t
    GROUP BY oid, run HAVING MAX(fid) - MIN(fid) + 1 >= 50
),
raw AS (
    SELECT A.oid AS oid, A.s AS s, B.e AS e
    FROM r1 A JOIN r2 B ON B.oid = A.oid
    WHERE GREATEST(B.s, A.s + 50 - 1 + 1) + 50 - 1 <= B.e
),
ordered AS (
    SELECT oid, s, e, MAX(e) OVER (PARTITION BY oid ORDER BY s, e
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_e FROM raw
),
grouped AS (
    SELECT oid, s, e, SUM(CASE WHEN prev_e IS NULL OR s > prev_e THEN 1 ELSE 0 END)
            OVER (PARTITION BY oid ORDER BY s, e ROWS UNBOUNDED PRECEDING) AS grp FROM ordered
)
SELECT oid AS "v", MIN(s) AS s, MAX(e) AS e FROM grouped GROUP BY oid, grp ORDER BY oid, s, e;
