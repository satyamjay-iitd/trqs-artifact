# /// script
# requires-python = ">=3.13"
# dependencies = ["duckdb>=1.5.6", "numpy>=2.1"]
# ///
"""Rank every VIRAT clip by how much its objects actually move."""
import duckdb, json
from pathlib import Path
import numpy as np

from paths import OUTPUT, PROCESSED

ROOT = PROCESSED / "virat"
SQL = """
WITH s AS (SELECT oid, fid, x1,y1,x2,y2,(x1+x2)/2.0 cx,(y1+y2)/2.0 cy FROM bbox),
d AS (SELECT fid-lag(fid) OVER w df,
             cx-lag(cx) OVER w dcx, cy-lag(cy) OVER w dcy,
             (x1=lag(x1) OVER w AND y1=lag(y1) OVER w
              AND x2=lag(x2) OVER w AND y2=lag(y2) OVER w) same
      FROM s WINDOW w AS (PARTITION BY oid ORDER BY fid)),
e AS (SELECT sqrt(dcx*dcx+dcy*dcy) step, same FROM d WHERE df=1)
SELECT count(*), avg(CASE WHEN same THEN 1.0 ELSE 0.0 END),
       quantile_cont(step,0.5), quantile_cont(step,0.9), quantile_cont(step,0.99), max(step)
FROM e
"""
CLS = "SELECT class, count(*) FROM objects GROUP BY 1"
out = []
for d in sorted(ROOT.iterdir()):
    db = d / "duckdb.db"
    if not db.is_file():
        continue
    try:
        c = duckdb.connect(str(db), read_only=True)
        n, same, p50, p90, p99, mx = c.execute(SQL).fetchone()
        if not n:
            continue
        cls = dict(c.execute(CLS).fetchall())
        rows = c.execute("SELECT count(*) FROM bbox").fetchone()[0]
        frames = c.execute("SELECT max(fid)+1 FROM bbox").fetchone()[0]
        c.close()
    except Exception as exc:
        print(f"skip {d.name}: {exc}")
        continue
    out.append(dict(clip=d.name, steps=n, rows=rows, frames=frames,
                    same=float(same), p50=float(p50 or 0), p90=float(p90 or 0),
                    p99=float(p99 or 0), max=float(mx or 0),
                    person=cls.get("person", 0), car=cls.get("car", 0),
                    objs=sum(cls.values())))
OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT / "virat_scan.json").write_text(json.dumps(out, indent=1))
print(f"scanned {len(out)} clips\n")
usable = [r for r in out if r["person"] >= 2 and r["car"] >= 2 and r["rows"] >= 20000]
usable.sort(key=lambda r: r["same"])
print(f"{'clip':<42}{'rows':>9}{'held':>7}{'p50':>7}{'p90':>7}{'p99':>7}{'ppl':>5}{'car':>5}{'dedup':>7}")
for r in usable[:20]:
    print(f"{r['clip']:<42}{r['rows']:>9,}{100*r['same']:>6.1f}%{r['p50']:>7.2f}{r['p90']:>7.2f}"
          f"{r['p99']:>7.2f}{r['person']:>5}{r['car']:>5}{1/(1-r['same']):>6.1f}x")
