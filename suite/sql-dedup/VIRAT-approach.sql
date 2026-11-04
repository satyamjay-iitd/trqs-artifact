-- The same query as sql-ref/VIRAT-approach.sql, over the frame-deduplicated copy
-- of the clip (`suite/build_dedup.py`), which keeps a row only
-- where it says something the previous frame did not.
--
--   SELECT (p: class=person, c: class=car)
--   WHERE DURATION(p.cx > c.cx + 300 OR p.cx < c.cx - 300
--               OR p.cy > c.cy + 300 OR p.cy < c.cy - 300, 25)
--    THEN DURATION(p.cx <= c.cx + 100 AND p.cx >= c.cx - 100
--              AND p.cy <= c.cy + 100 AND p.cy >= c.cy - 100, 50)
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
-- the join is a band join, which DuckDB runs at 320 ms against the reference's
-- 53 ms on the dynamic clip. Tiling the frame axis into 64-frame buckets and
-- equi-joining on the bucket gives the optimiser back the equality that frame
-- expansion handed it for free. An interval is repeated once per bucket it
-- spans, which is redundancy again, but coarse -- a bucket holds 64 frames --
-- and `pb.bucket = GREATEST(...) >> 6` keeps one row per overlapping pair, the
-- bucket where the intersection starts, so a long interval pays for its span
-- without duplicating an answer. The width is not delicate: 8 to 256 frames all
-- land within 5 % on both clips.
--
-- WHY `unnest(range(...))` AND NOT `generate_series`. They compute the same
-- buckets, and on the dynamic clip the table-function form costs 30 ms against
-- this one's 11 ms -- ~0.6 us per row of per-invocation overhead, over 47 k
-- surviving rows. That overhead alone exceeded the whole frame-expanded join,
-- so the choice decides whether this baseline looks competitive or hopeless.
--
-- WHY THE RUNS STILL NEED COALESCING. The intersections of one object's runs
-- with another's are disjoint and ordered, but consecutive ones may both satisfy
-- a phase -- the person moved while the car stayed put -- and the DURATION
-- minimum is over the whole stretch. The gaps-and-islands step merges intervals
-- that touch (`s <= prev_e + 1`) before the minimum applies. The reference gets
-- this for free: there, `fid - ROW_NUMBER()` groups consecutive frames.
--
-- A RUN NEVER SPANS AN ABSENCE, so the overlap tests co-appearance exactly as
-- the reference's `b1.fid = b0.fid` does.
WITH seg AS (
    SELECT oid, fid AS s, e_fid AS e,
           (x1 + x2) / 2.0 AS cx, (y1 + y2) / 2.0 AS cy
    FROM bbox
),
p AS (SELECT s.* FROM seg s JOIN objects j ON j.oid = s.oid AND j.class = 'person'),
c AS (SELECT s.* FROM seg s JOIN objects j ON j.oid = s.oid AND j.class = 'car'),
pb AS (SELECT p.*, unnest(range(p.s >> 6, (p.e >> 6) + 1)) AS bucket FROM p),
cb AS (SELECT c.*, unnest(range(c.s >> 6, (c.e >> 6) + 1)) AS bucket FROM c),
pairs AS (
    SELECT GREATEST(pb.s, cb.s) AS s,
           LEAST(pb.e, cb.e)    AS e,
           pb.oid               AS o0,
           cb.oid               AS o1,
           pb.cx - cb.cx        AS dx,
           pb.cy - cb.cy        AS dy
    FROM pb JOIN cb ON cb.bucket = pb.bucket
    WHERE pb.s <= cb.e AND cb.s <= pb.e
      -- one row per overlapping pair: the bucket holding the intersection's start
      AND pb.bucket = GREATEST(pb.s, cb.s) >> 6
),
far AS (
    SELECT o0, o1, s, e,
           max(e) OVER (PARTITION BY o0, o1 ORDER BY s
                        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_e
    FROM pairs WHERE dx > 300 OR -dx > 300 OR dy > 300 OR -dy > 300
),
r1 AS (
    SELECT o0, o1, min(s) AS s, max(e) AS e
    FROM (SELECT o0, o1, s, e,
                 sum(CASE WHEN prev_e IS NULL OR s > prev_e + 1 THEN 1 ELSE 0 END)
                     OVER (PARTITION BY o0, o1 ORDER BY s ROWS UNBOUNDED PRECEDING) AS island
          FROM far) t
    GROUP BY o0, o1, island HAVING max(e) - min(s) + 1 >= 25
),
near AS (
    SELECT o0, o1, s, e,
           max(e) OVER (PARTITION BY o0, o1 ORDER BY s
                        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_e
    FROM pairs WHERE dx <= 100 AND -dx <= 100 AND dy <= 100 AND -dy <= 100
),
r2 AS (
    SELECT o0, o1, min(s) AS s, max(e) AS e
    FROM (SELECT o0, o1, s, e,
                 sum(CASE WHEN prev_e IS NULL OR s > prev_e + 1 THEN 1 ELSE 0 END)
                     OVER (PARTITION BY o0, o1 ORDER BY s ROWS UNBOUNDED PRECEDING) AS island
          FROM near) t
    GROUP BY o0, o1, island HAVING max(e) - min(s) + 1 >= 50
),
raw AS (
    SELECT A.o0 AS o0, A.o1 AS o1, A.s AS s, B.e AS e
    FROM r1 A JOIN r2 B ON B.o0 = A.o0 AND B.o1 = A.o1
    WHERE GREATEST(B.s, A.s + 25) + 50 - 1 <= B.e
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
SELECT o0 AS "p", o1 AS "c", MIN(s) AS s, MAX(e) AS e
FROM grouped GROUP BY o0, o1, grp ORDER BY o0, o1, s, e;
