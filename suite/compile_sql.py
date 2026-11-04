#!/usr/bin/env python3
"""Compile each suite query to DuckDB SQL and, optionally, run it.

    ./compile_sql.py                        # write sql/<ID>.sql for every query
    ./compile_sql.py --check --limit 5      # also run each and diff against the engine

The SQL listings double as the appendix material `05_evaluation.tex` gestures at.

`--check` is the equivalence test (T3.1). It compares the *engine's* result rows to the
SQL's, as sets of `(objects…, s, e)` spans. Two divergences are expected and are reported
rather than treated as failures:

  * The engine fits polynomials across detection gaps up to `max_gap`, while the SQL breaks
    an island at any missing frame. Any object with an interior gap can differ.
  * The engine reads boxes as `u32`; a database whose boxes are DOUBLE is rounded on the
    way in, so a predicate sitting exactly on a boundary can fall either way.

A query whose regions are generated per sub-dataset compiles per sub-dataset too, so its
listing is written for the first scene it runs on and named accordingly.
"""
from __future__ import annotations

import argparse
import functools
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import OUTPUT, PROCESSED, ROOT  # noqa: E402
from queries import SUITE  # noqa: E402
from run_suite import CLI, targets  # noqa: E402

SQL_DIR = OUTPUT / "sql"


def compile_dsl(dsl: str) -> str:
    compiler = ROOT / "engine" / "target" / "release" / "examples" / "compile"
    command = [str(compiler), dsl] if compiler.is_file() else [
        "cargo", "run", "--locked", "--release", "-q", "--manifest-path",
        str(ROOT / "engine" / "Cargo.toml"), "-p", "trql-sql", "--example", "compile", "--", dsl]
    proc = subprocess.run(
        command,
        capture_output=True, text=True, cwd=ROOT,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[:400])
    return proc.stdout


#: Prepended to every SQL run. `threads=1` is the point: \sys evaluates on one
#: core, so a DuckDB left on its default (one thread per core -- 27-29 here)
#: measures parallelism rather than the two evaluation strategies. Report the
#: parallel figure separately if you want it, but the head-to-head is 1-vs-1.
SQL_PRELUDE = "SET threads=1;\n"


def run_sql(sql: str, dataset: str, sub: str, threads: int = 1) -> set[tuple]:
    db = PROCESSED / dataset / sub / "duckdb.db"
    proc = subprocess.run(
        ["duckdb", "-readonly", "-json", str(db)],
        input=f"SET threads={int(threads)};\n" + sql,
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[:400])
    out = proc.stdout.strip()
    rows = json.loads(out) if out else []
    return {tuple(r[k] for k in r) for r in rows}


@functools.lru_cache(maxsize=None)
def engine_oids(dataset: str, sub: str) -> list[int]:
    """The db `oid` behind each of the engine's object indices.

    The engine numbers objects densely from 0 in the order
    `trql-storage`'s `read_objects` reads them (`... ORDER BY b.oid`), while the
    SQL reports the `oid` column itself. Those coincide only when the source
    happens to number its objects 0..N-1 -- which soccernet, nuscenes and clevrer
    all do, and an i24 *slice* does not, because slicing keeps each vehicle's
    original id so a match can be traced across slices.

    Without this map the two sides disagree on every row while describing exactly
    the same spans, which looks like a compiler bug and is not one.
    """
    con = duckdb.connect(str(PROCESSED / dataset / sub / "duckdb.db"), read_only=True)
    try:
        return [r[0] for r in con.execute(
            "SELECT DISTINCT oid FROM bbox ORDER BY oid").fetchall()]
    finally:
        con.close()


def run_engine(dataset: str, sub: str, dsl: str) -> set[tuple]:
    """The engine's spans, flattened to the same `(oid…, s, e)` shape as the SQL."""
    with tempfile.NamedTemporaryFile(suffix=".json") as out:
        proc = subprocess.run(
            [str(CLI), "--data-dir", str(PROCESSED), "--dataset", dataset, "--sub-dataset", sub,
             "--format", "json", "run", dsl, out.name],
            capture_output=True, text=True, cwd=ROOT,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip()[:400])
        results = json.loads(Path(out.name).read_text())["results"]
    oids = engine_oids(dataset, sub)
    spans = set()
    for objs, ivs in results:
        real = tuple(oids[o] for o in objs)
        for iv in ivs:
            spans.add((*real, iv["s"], iv["e"]))
    return spans


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", nargs="+")
    ap.add_argument("--dataset", action="append")
    ap.add_argument("--check", action="store_true", help="run the SQL and diff it")
    ap.add_argument("--strict", action="store_true",
                    help="exit nonzero when --check finds different answers")
    ap.add_argument("--limit", type=int, default=1,
                    help="sub-datasets to check per query (default 1)")
    args = ap.parse_args()

    SQL_DIR.mkdir(parents=True, exist_ok=True)
    todo = SUITE
    if args.dataset:
        todo = [q for q in todo if q.dataset in args.dataset]
    if args.query:
        todo = [q for q in todo if q.id in args.query]

    bad = 0
    for q in todo:
        subs = targets(q, args.limit)
        if not subs:
            print(f"{q.id:8s} no runnable sub-dataset"); continue
        sub = subs[0]
        try:
            sql = compile_dsl(q.dsl_for(sub))
        except Exception as e:
            print(f"{q.id:8s} COMPILE FAILED: {e}"); bad += 1; continue
        (SQL_DIR / f"{q.id}.sql").write_text(
            f"-- {q.id} ({q.dataset}, {q.objects} objects, {'+'.join(q.ops)})\n"
            f"-- {q.desc}\n-- DSL: {q.dsl_for(sub)}\n"
            f"-- compiled for sub-dataset: {sub}\n\n{sql}"
        )
        if not args.check:
            print(f"{q.id:8s} compiled -> sql/{q.id}.sql")
            continue

        for sub in subs:
            try:
                sql_i = compile_dsl(q.dsl_for(sub))
                a, b = run_engine(q.dataset, sub, q.dsl_for(sub)), run_sql(sql_i, q.dataset, sub)
            except Exception as e:
                print(f"{q.id:8s} {sub:20s} RUN FAILED: {e}"); bad += 1; continue
            if a == b:
                print(f"{q.id:8s} {sub:20s} agree ({len(a)} spans)")
            else:
                print(f"{q.id:8s} {sub:20s} DIFFER engine={len(a)} sql={len(b)} "
                      f"only-engine={len(a - b)} only-sql={len(b - a)}")
                if args.strict:
                    bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
