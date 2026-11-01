#!/usr/bin/env python3
"""Time \\sys against its own compiled SQL, one core each.

    ./head_to_head.py --dataset i24_motion --sub post1_m00
    ./head_to_head.py --dataset soccernet --sub test_SNMOT-116

Both sides get one thread: `trql-cli` is single-threaded, and a DuckDB left on
its default (one thread per core) measures parallelism rather than the two
evaluation strategies. `--threads` reports the parallel figure instead; label it
as such if you use it.

\\sys is timed two ways because both matter: `query_ms` is what `trql-cli`
reports for evaluation alone, `wall_s` includes loading the `.trql`, which for
the undivided i24 post1 is 12 s of the answer. DuckDB's wall time includes
opening the database, which is mmap'd and cheap.

Rows are compared, not just counted -- a timing against a wrong answer is not a
measurement. Any disagreement is reported alongside the times.
"""
from __future__ import annotations

import argparse
import subprocess
import pathlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compile_sql import compile_dsl, engine_oids, run_sql  # noqa: E402
from frames import PROCESSED, ROOT  # noqa: E402
from queries import SUITE  # noqa: E402

import json  # noqa: E402
import tempfile  # noqa: E402


def time_engine(dataset: str, sub: str, dsl: str) -> tuple[float, float, set]:
    """Returns `(wall_s, query_s, spans)`.

    Both numbers are reported because they answer different questions. `wall_s`
    is one-shot cost: start the process, load the `.trql`, answer, exit. `query_s`
    is what `trql-cli` reports for evaluation alone, which is the honest number
    when a loaded database serves many queries -- and for a 50 MB clevrer artifact
    the load is ~0.9 s, more than the query itself on four of the six.

    DuckDB has no equivalent split: it mmaps the file and reads what it touches,
    so its load is inside its query time and cannot be separated the same way.
    Compare `wall` to `wall`; read `query` as the ceiling the engine approaches
    if the artifact is already resident.
    """
    with tempfile.NamedTemporaryFile(suffix=".json") as out:
        t0 = time.time()
        proc = subprocess.run(
            [str(ROOT / "target" / "release" / "trql-cli"),
             "--dataset", dataset, "--sub-dataset", sub,
             "--format", "json", "run", dsl, out.name],
            capture_output=True, text=True, cwd=ROOT,
        )
        wall = time.time() - t0
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip()[:300])
        query_s = json.loads(proc.stdout)["elapsed_ms"] / 1000.0
        results = json.loads(pathlib.Path(out.name).read_text())["results"]
    oids = engine_oids(dataset, sub)
    spans = {(*[oids[o] for o in objs], iv["s"], iv["e"])
             for objs, ivs in results for iv in ivs}
    return wall, query_s, spans


def time_sql(sql: str, dataset: str, sub: str, threads: int) -> tuple[float, set]:
    t0 = time.time()
    rows = run_sql(sql, dataset, sub, threads=threads)
    return time.time() - t0, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sub", required=True)
    ap.add_argument("--query", nargs="+")
    ap.add_argument("--threads", type=int, default=1, help="DuckDB threads (default 1)")
    args = ap.parse_args()

    todo = [q for q in SUITE if q.dataset == args.dataset]
    if args.query:
        todo = [q for q in todo if q.id in args.query]

    print(f"{args.dataset}/{args.sub}   TrQS release build, 1 core   "
          f"DuckDB threads={args.threads}\n")
    print(f"{'id':9s} {'TrQS wall':>10s} {'(query)':>9s} {'DuckDB':>9s} "
          f"{'wall/wall':>10s} {'query/duck':>11s} {'rows':>8s}  agree")
    for q in todo:
        dsl = q.dsl_for(args.sub)
        try:
            e_wall, e_query, e_rows = time_engine(q.dataset, args.sub, dsl)
            sql = compile_dsl(dsl)
            d_wall, d_rows = time_sql(sql, q.dataset, args.sub, args.threads)
        except Exception as exc:
            print(f"{q.id:9s} ERROR {str(exc)[:90]}", flush=True)
            continue
        agree = "yes" if e_rows == d_rows else f"NO (+{len(e_rows-d_rows)}/-{len(d_rows-e_rows)})"
        r_wall = d_wall / e_wall if e_wall else float("nan")
        r_query = d_wall / e_query if e_query else float("nan")
        print(f"{q.id:9s} {e_wall:9.2f}s {e_query:8.2f}s {d_wall:8.2f}s "
              f"{r_wall:9.2f}x {r_query:10.2f}x {len(e_rows):8d}  {agree}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
