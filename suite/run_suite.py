#!/usr/bin/env python3
"""Run the evaluation workload and aggregate it.

`trql-cli` measures one sub-dataset per invocation, but two of the four datasets
are not one sub-dataset: soccernet is 106 clips and nuscenes 850 scenes. This
driver is the loop around that -- for each query it runs every sub-dataset the
query can run on, then reports the distribution rather than whichever clip
happened to be first.

    ./run_suite.py                          # everything
    ./run_suite.py --dataset soccernet      # one family
    ./run_suite.py --query SN-Q2 CL-Q3      # named queries
    ./run_suite.py --limit 20               # sample 20 sub-datasets per family
    ./run_suite.py --bench --results-db ../../plotter/dbs/vldb27.db

Two modes:

*run* (default) executes each query once per sub-dataset and records match count
and query time. This is the cheap pass -- use it to check a query still matches
after an edit.

*bench* (`--bench`) loads the selected database once, executes the query
`--run-repeat` times, samples peak RSS, and appends to the SQLite results
store. `--db-repeat` is the dataset replication factor.

Query ids are namespaced by dataset (`SN-Q1`, `I24-Q1`, `CL-Q1`, `NS-Q1`)
because the results schema keys runs on `queries.name` alone: an un-namespaced
`Q1` from soccernet and from i24 would collapse into one row.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import CLI, OUTPUT, PROCESSED, ROOT  # noqa: E402
from queries import SUITE, Query  # noqa: E402

def sub_datasets(dataset: str) -> list[str]:
    """Every sub-dataset of a family that has a built `.trql`, in name order."""
    return sorted(
        p.parent.name
        for p in (PROCESSED / dataset).glob("*/10_0.trql")
    )


def targets(q: Query, limit: int | None) -> list[str]:
    subs = q.subs if q.subs is not None else sub_datasets(q.dataset)
    subs = [s for s in subs if (PROCESSED / q.dataset / s / "10_0.trql").is_file() and q.runs_on(s)]
    return subs[:limit] if limit else subs


def run_one(q: Query, sub: str) -> dict:
    """One `trql-cli run`: match count and query time for this sub-dataset."""
    with tempfile.NamedTemporaryFile(suffix=".json") as out:
        proc = subprocess.run(
            [str(CLI), "--data-dir", str(PROCESSED), "--dataset", q.dataset, "--sub-dataset", sub,
             "--format", "json", "run", q.dsl_for(sub), out.name],
            capture_output=True, text=True, cwd=ROOT,
        )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    r = json.loads(proc.stdout)
    return {"sub": sub, "matches": r["matches"], "ms": r["elapsed_ms"]}


def bench_one(q: Query, sub: str, db_repeat: int, run_repeat: int,
              results_db: Path | None) -> dict:
    """One `trql-cli bench`: timings, peak RSS, and the results-db append."""
    cmd = [str(CLI), "--data-dir", str(PROCESSED), "--dataset", q.dataset, "--sub-dataset", sub,
           "--format", "json", "bench", q.id, q.dsl_for(sub),
           str(db_repeat), str(run_repeat)]
    if results_db:
        cmd += ["--results-db", str(results_db)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    r = json.loads(proc.stdout)
    return {"sub": sub, "matches": r["rows"], "ms": r["avg_ms"],
            "peak_mem_mb": r["peak_mem_mb"], "storage_mb": r["storage_mb"],
            "objects": r["objects"]}


def summarise(q: Query, per_sub: list[dict], skipped: int, failed: list) -> dict:
    times = sorted(d["ms"] for d in per_sub)
    matches = [d["matches"] for d in per_sub]
    s = {
        "id": q.id, "dataset": q.dataset, "objects": q.objects, "ops": q.ops,
        "desc": q.desc,
        "sub_datasets_run": len(per_sub),
        "sub_datasets_skipped_missing_class": skipped,
        "total_matches": sum(matches),
        "sub_datasets_with_a_match": sum(1 for m in matches if m > 0),
        "per_sub": per_sub,
    }
    if times:
        s["median_ms"] = statistics.median(times)
        # P99 of a 106-clip sample is the second-slowest; report the max too so
        # nothing hides behind the percentile on the small families.
        s["p99_ms"] = times[min(len(times) - 1, int(0.99 * len(times)))]
        s["max_ms"] = times[-1]
    if failed:
        s["failures"] = failed
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", help="restrict to these families")
    ap.add_argument("--query", nargs="+", help="restrict to these query ids")
    ap.add_argument("--limit", type=int, help="sample at most N sub-datasets per query")
    ap.add_argument("--bench", action="store_true", help="measure via `trql-cli bench`")
    ap.add_argument("--db-repeat", type=int, default=1, help="dataset replication factor")
    ap.add_argument("--run-repeat", type=int, default=3)
    ap.add_argument("--results-db", type=Path)
    ap.add_argument("--out", type=Path, default=OUTPUT / "results.json")
    args = ap.parse_args()

    if not CLI.is_file():
        print(f"!! {CLI} not built -- run `cargo build --release --manifest-path engine/Cargo.toml -p trql-cli`")
        return 1

    todo = SUITE
    if args.dataset:
        todo = [q for q in todo if q.dataset in args.dataset]
    if args.query:
        todo = [q for q in todo if q.id in args.query]
    if not todo:
        print("!! no queries selected")
        return 1

    out = []
    for q in todo:
        subs = targets(q, args.limit)
        all_subs = [s for s in (q.subs if q.subs is not None else sub_datasets(q.dataset))
                    if (PROCESSED / q.dataset / s / "10_0.trql").is_file()]
        skipped = len(all_subs) - len([s for s in all_subs if q.runs_on(s)])
        per_sub, failed = [], []
        t0 = time.time()
        for sub in subs:
            try:
                per_sub.append(
                    bench_one(q, sub, args.db_repeat, args.run_repeat, args.results_db)
                    if args.bench else run_one(q, sub)
                )
            except Exception as e:
                failed.append({"sub": sub, "error": str(e)[:300]})
        s = summarise(q, per_sub, skipped, failed)
        out.append(s)
        wall = time.time() - t0
        line = (f"{q.id:8s} {q.dataset:15s} {q.objects}obj "
                f"{'+'.join(q.ops):26s} {s['total_matches']:9d} matches  "
                f"{s['sub_datasets_with_a_match']:3d}/{len(per_sub):3d} subs  ")
        if per_sub:
            line += f"med {s['median_ms']:8.1f} ms  max {s['max_ms']:9.1f} ms  "
        line += f"wall {wall:6.1f} s"
        if skipped:
            line += f"  ({skipped} skipped: class absent)"
        if failed:
            line += f"  !! {len(failed)} FAILED"
        print(line, flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 1 if any("failures" in s for s in out) or not any(s["sub_datasets_run"] for s in out) else 0


if __name__ == "__main__":
    sys.exit(main())
