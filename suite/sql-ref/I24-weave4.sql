-- i24, single object: a truck crosses four lanes in sequence, 2 s settled in each.
-- Four DURATION phases chained by plain (unbounded) THEN.
--
-- Lane bands, stored centi-feet (feet = (stored - 20000)/100):
--   eb1 21100..22200 (11..22 ft)   eb2 22200..23450 (22..34.5 ft)
--   eb3 23450..24650 (34.5..46.5)  eb4 24650..26000 (46.5..60 ft)
--
-- Greedy earliest-fit down the chain: a phase constrains the next only through
-- its end frame, and ending earlier is never worse, so if any packing works the
-- earliest one does. (Sound only for an unbounded gap, which is what plain THEN is.)
WITH o AS (
    SELECT b.fid AS fid, b.oid AS oid, (b.y1 + b.y2) / 2.0 AS cy
    FROM bbox b
    JOIN objects j ON j.oid = b.oid AND j.class = 'truck'
),
r1 AS (SELECT oid, MIN(fid) s, MAX(fid) e FROM (
        SELECT oid, fid, fid - ROW_NUMBER() OVER (PARTITION BY oid ORDER BY fid) run
        FROM o WHERE cy > 21100.0 AND cy < 22200.0) t
       GROUP BY oid, run HAVING MAX(fid) - MIN(fid) + 1 >= 50),
r2 AS (SELECT oid, MIN(fid) s, MAX(fid) e FROM (
        SELECT oid, fid, fid - ROW_NUMBER() OVER (PARTITION BY oid ORDER BY fid) run
        FROM o WHERE cy > 22200.0 AND cy < 23450.0) t
       GROUP BY oid, run HAVING MAX(fid) - MIN(fid) + 1 >= 50),
r3 AS (SELECT oid, MIN(fid) s, MAX(fid) e FROM (
        SELECT oid, fid, fid - ROW_NUMBER() OVER (PARTITION BY oid ORDER BY fid) run
        FROM o WHERE cy > 23450.0 AND cy < 24650.0) t
       GROUP BY oid, run HAVING MAX(fid) - MIN(fid) + 1 >= 50),
r4 AS (SELECT oid, MIN(fid) s, MAX(fid) e FROM (
        SELECT oid, fid, fid - ROW_NUMBER() OVER (PARTITION BY oid ORDER BY fid) run
        FROM o WHERE cy > 24650.0 AND cy < 26000.0) t
       GROUP BY oid, run HAVING MAX(fid) - MIN(fid) + 1 >= 50),
fit AS (
    SELECT A.oid AS oid, A.s AS s, D.e AS e,
           A.s + 49                                    AS e1,
           GREATEST(B.s, A.s + 50)                     AS s2,
           GREATEST(B.s, A.s + 50) + 49                AS e2,
           B.e AS be, C.s AS cs, C.e AS ce, D.s AS ds, D.e AS de
    FROM r1 A JOIN r2 B ON B.oid = A.oid
              JOIN r3 C ON C.oid = A.oid
              JOIN r4 D ON D.oid = A.oid
),
raw AS (
    SELECT oid, s, e FROM fit
    WHERE e2 <= be
      AND GREATEST(cs, e2 + 1) + 49 <= ce
      AND GREATEST(ds, GREATEST(cs, e2 + 1) + 49 + 1) + 49 <= de
),
ordered AS (SELECT oid, s, e, MAX(e) OVER (PARTITION BY oid ORDER BY s, e
              ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) prev_e FROM raw),
grouped AS (SELECT oid, s, e, SUM(CASE WHEN prev_e IS NULL OR s > prev_e THEN 1 ELSE 0 END)
              OVER (PARTITION BY oid ORDER BY s, e ROWS UNBOUNDED PRECEDING) grp FROM ordered)
SELECT oid AS "v", MIN(s) AS s, MAX(e) AS e FROM grouped GROUP BY oid, grp ORDER BY oid, s, e;
