#!/usr/bin/env python3
"""Smoke-run one DSL query against one sub-dataset and print the match count.

Authoring aid, not part of the measured pipeline: `run_suite.py` is what
produces the numbers. This exists so a query can be checked for "does it parse
and does it match anything" in one line while the suite is being written.

    ./probe.py soccernet test_SNMOT-116 "SELECT (b: class=ball) WHERE ..."
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CLI = ROOT / "target" / "release" / "trql-cli"


def probe(dataset: str, sub: str, query: str) -> tuple[int, float, list]:
    with tempfile.NamedTemporaryFile(suffix=".json") as out:
        proc = subprocess.run(
            [str(CLI), "--dataset", dataset, "--sub-dataset", sub,
             "--format", "json", "run", query, out.name],
            capture_output=True, text=True, cwd=ROOT,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
        report = json.loads(proc.stdout)
        rows = json.loads(Path(out.name).read_text())["results"]
    return report["matches"], report["elapsed_ms"], rows


if __name__ == "__main__":
    dataset, sub, query = sys.argv[1], sys.argv[2], sys.argv[3]
    n, ms, rows = probe(dataset, sub, query)
    print(f"{n} matches in {ms:.1f} ms")
    for r in rows[:5]:
        print("   ", r)
