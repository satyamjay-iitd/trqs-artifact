-- i24, single object: a truck passes four consecutive 200 ft stretches of road in
-- order. Guaranteed to match for any truck tracked through that 800 ft window --
-- vehicles travel monotonically, so the four phases occur in sequence by physics.
--
-- Four DISTINCT predicates is the point: each one costs DuckDB its own island pass
-- over the scanned rows, while the engine composes four interval sets for almost
-- nothing.
--
-- Bands in stored centi-feet (feet = (stored - (-30,900,000))/100):
--   b1 1,100,000..1,120,000  = 320,000..320,200 ft
--   b2 1,120,000..1,140,000  b3 1,140,000..1,160,000  b4 1,160,000..1,180,000
--
-- Greedy earliest-fit down the chain, sound for the unbounded gap of a plain THEN.
WITH o AS (
    SELECT b.fid AS fid, b.oid AS oid, (b.x1 + b.x2) / 2.0 AS cx
    FROM bbox b JOIN objects j ON j.oid = b.oid AND j.class = 'truck'
),
b1 AS (SELECT oid, MIN(fid) s, MAX(fid) e FROM (
        SELECT oid, fid, fid - ROW_NUMBER() OVER (PARTITION BY oid ORDER BY fid) run
        FROM o WHERE cx > 1100000.0 AND cx < 1120000.0) t
       GROUP BY oid, run HAVING MAX(fid) - MIN(fid) + 1 >= 25),
b2 AS (SELECT oid, MIN(fid) s, MAX(fid) e FROM (
        SELECT oid, fid, fid - ROW_NUMBER() OVER (PARTITION BY oid ORDER BY fid) run
        FROM o WHERE cx > 1120000.0 AND cx < 1140000.0) t
       GROUP BY oid, run HAVING MAX(fid) - MIN(fid) + 1 >= 25),
b3 AS (SELECT oid, MIN(fid) s, MAX(fid) e FROM (
        SELECT oid, fid, fid - ROW_NUMBER() OVER (PARTITION BY oid ORDER BY fid) run
        FROM o WHERE cx > 1140000.0 AND cx < 1160000.0) t
       GROUP BY oid, run HAVING MAX(fid) - MIN(fid) + 1 >= 25),
b4 AS (SELECT oid, MIN(fid) s, MAX(fid) e FROM (
        SELECT oid, fid, fid - ROW_NUMBER() OVER (PARTITION BY oid ORDER BY fid) run
        FROM o WHERE cx > 1160000.0 AND cx < 1180000.0) t
       GROUP BY oid, run HAVING MAX(fid) - MIN(fid) + 1 >= 25),
raw AS (
    SELECT A.oid AS oid, A.s AS s, D.e AS e FROM b1 A
    JOIN b2 B ON B.oid = A.oid
    JOIN b3 C ON C.oid = A.oid
    JOIN b4 D ON D.oid = A.oid
    WHERE GREATEST(B.s, A.s + 25) + 24 <= B.e
      AND GREATEST(C.s, GREATEST(B.s, A.s + 25) + 25) + 24 <= C.e
      AND GREATEST(D.s, GREATEST(C.s, GREATEST(B.s, A.s + 25) + 25) + 25) + 24 <= D.e
),
ordered AS (SELECT oid, s, e, MAX(e) OVER (PARTITION BY oid ORDER BY s, e
              ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) prev_e FROM raw),
grouped AS (SELECT oid, s, e, SUM(CASE WHEN prev_e IS NULL OR s > prev_e THEN 1 ELSE 0 END)
              OVER (PARTITION BY oid ORDER BY s, e ROWS UNBOUNDED PRECEDING) grp FROM ordered)
SELECT oid AS "v", MIN(s) AS s, MAX(e) AS e FROM grouped GROUP BY oid, grp ORDER BY oid, s, e;
