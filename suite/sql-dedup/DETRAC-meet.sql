-- The same query as sql-ref/DETRAC-meet.sql, over the frame-deduplicated copy of the
-- clip (`papers/vldb27/suite/build_dedup.py`), which keeps a row only where it
-- says something the previous frame did not.
--
--   SELECT (a: class=car, b: class=car)
--   WHERE DURATION(a.cx > b.cx + 200 OR a.cx < b.cx - 200
--               OR a.cy > b.cy + 200 OR a.cy < b.cy - 200, 15)
--    THEN DURATION(a.cx <= b.cx + 70 AND a.cx >= b.cx - 70
--              AND a.cy <= b.cy + 70 AND a.cy >= b.cy - 70, 30)
--
-- WHAT CHANGES. The reference joins the two objects frame by frame, which is
-- where the frame-expanded encoding spends everything: a parked car contributes
-- one row per frame and the join re-tests a predicate whose operands have not
-- moved. Here a row already carries the interval it holds over, so co-appearance
-- is an interval overlap and the predicate is tested once per pair of runs.
-- Nothing else moves: the phases, the thresholds, the DURATION minima and the
-- THEN are the reference's, and the answer is required to be identical to it.
--
-- WHY THE BUCKETS. Written as the overlap itself (`c.s <= p.e AND c.e >= p.s`)
-- the join is a band join, which DuckDB runs several times slower than the
-- reference. Tiling the frame axis into 64-frame buckets and equi-joining on the
-- bucket gives the optimiser back the equality that frame expansion handed it
-- for free. An interval is repeated once per bucket it spans, which is
-- redundancy again, but coarse, and `a.bucket = GREATEST(...) >> 6` keeps one
-- row per overlapping pair -- the bucket where the intersection starts.
--
-- WHY `unnest(range(...))` AND NOT `generate_series`. They compute the same
-- buckets; the table-function form carries ~0.6 us per row of per-invocation
-- overhead, which on a clip with tens of thousands of surviving rows exceeds the
-- whole frame-expanded join.
--
-- WHY THE RUNS STILL NEED COALESCING. The intersections of one object's runs
-- with another's are disjoint and ordered, but consecutive ones may both satisfy
-- a phase, and the DURATION minimum is over the whole stretch. The reference
-- gets this for free: there, `fid - ROW_NUMBER()` groups consecutive frames.
--
-- SELF-PAIRS. `b.oid <> a.oid` is an optimisation: for a = b every difference is
-- 0, so the first phase is unsatisfiable.
WITH seg AS (
    SELECT s.oid, s.fid AS s, s.e_fid AS e,
           (s.x1 + s.x2) / 2.0 AS cx, (s.y1 + s.y2) / 2.0 AS cy
    FROM bbox s JOIN objects j ON j.oid = s.oid AND j.class = 'car'
),
ab AS (SELECT seg.*, unnest(range(seg.s >> 6, (seg.e >> 6) + 1)) AS bucket FROM seg),
pairs AS (
    SELECT GREATEST(a.s, b.s) AS s,
           LEAST(a.e, b.e)    AS e,
           a.oid              AS o0,
           b.oid              AS o1,
           a.cx - b.cx        AS dx,
           a.cy - b.cy        AS dy
    FROM ab a JOIN ab b ON b.bucket = a.bucket AND b.oid <> a.oid
    WHERE a.s <= b.e AND b.s <= a.e
      AND a.bucket = GREATEST(a.s, b.s) >> 6
),
far AS (
    SELECT o0, o1, s, e,
           max(e) OVER (PARTITION BY o0, o1 ORDER BY s
                        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_e
    FROM pairs WHERE dx > 200 OR -dx > 200 OR dy > 200 OR -dy > 200
),
r1 AS (
    SELECT o0, o1, min(s) AS s, max(e) AS e
    FROM (SELECT o0, o1, s, e,
                 sum(CASE WHEN prev_e IS NULL OR s > prev_e + 1 THEN 1 ELSE 0 END)
                     OVER (PARTITION BY o0, o1 ORDER BY s ROWS UNBOUNDED PRECEDING) AS island
          FROM far) t
    GROUP BY o0, o1, island HAVING max(e) - min(s) + 1 >= 15
),
near AS (
    SELECT o0, o1, s, e,
           max(e) OVER (PARTITION BY o0, o1 ORDER BY s
                        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_e
    FROM pairs WHERE dx <= 70 AND -dx <= 70 AND dy <= 70 AND -dy <= 70
),
r2 AS (
    SELECT o0, o1, min(s) AS s, max(e) AS e
    FROM (SELECT o0, o1, s, e,
                 sum(CASE WHEN prev_e IS NULL OR s > prev_e + 1 THEN 1 ELSE 0 END)
                     OVER (PARTITION BY o0, o1 ORDER BY s ROWS UNBOUNDED PRECEDING) AS island
          FROM near) t
    GROUP BY o0, o1, island HAVING max(e) - min(s) + 1 >= 30
),
raw AS (
    SELECT A.o0 AS o0, A.o1 AS o1, A.s AS s, B.e AS e
    FROM r1 A JOIN r2 B ON B.o0 = A.o0 AND B.o1 = A.o1
    WHERE GREATEST(B.s, A.s + 15) + 30 - 1 <= B.e
),
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
