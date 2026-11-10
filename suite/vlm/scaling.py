# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Every VLM cell in one table, plus what a second query costs each system.

Three things are being separated here, and they are easy to confuse:

* **Window length.** 30 s against 10 min on the same query. Quote the 10 min
  cell: five ground-truth turns quantise recall into steps of 0.2, so the 30 s
  numbers look better than they are.
* **Query.** Q1 (turn right out of the queue lane) against Q2 (enter from the
  top of the frame and turn right), both on the 10 min window. They select
  disjoint cars. Q2's manoeuvre is shorter and in a corner, so its accuracy is
  not comparable with Q1's.
* **Amortization.** What the *k*-th query costs. The engine pays ingestion once
  and then milliseconds; the model re-reads the video every time. This is the
  only one of the three that the figure's cost panel claims.

    uv run suite/vlm/scaling.py
"""

from __future__ import annotations

import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import VLM

_HERE = VLM

#: `(window, query, results file)`.
CELLS = [("30 s", "Q1", "results-vlm.json"),
         ("30 s", "Q2", "results-vlm-top.json"),
         ("10 min", "Q1", "results-vlm-600s.json"),
         ("10 min", "Q2", "results-vlm-top-600s.json"),
         ("15 min", "Q1", "results-vlm-900s.json"),
         ("15 min", "Q2", "results-vlm-top-900s.json")]
#: Detection + tracking + fitting, in seconds, per window of `ytstream/uav_0006`
#: from 180 s. Recorded once with YOLO11x and BoT-SORT and reported as stated;
#: the artifact does not re-run the detector. Same values as the plotter's
#: `benchplot.plots.vlm.INGEST_S`.
INGEST_S = {"30s": 5.65, "600s": 49.43, "900s": 73.01}

#: `(window, INGEST_S key, results files)` for the amortization block. Each is
#: one video and one window, so the ingestion is genuinely paid once.
AMORT = [("30 s", "30s", ["results-vlm.json", "results-vlm-top.json"]),
         ("10 min", "600s", ["results-vlm-600s.json", "results-vlm-top-600s.json"]),
         ("15 min", "900s", ["results-vlm-900s.json", "results-vlm-top-900s.json"])]


def _f1(tp: int, fp: int, fn: int) -> float:
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 0.0


def load(fname: str) -> dict | None:
    p = _HERE / fname
    return json.loads(p.read_text()) if p.is_file() else None


def table() -> None:
    hdr = (f"{'window':>7} {'q':>3} {'mode':9} {'turns':>6} {'cars':>6} {'ans':>5} "
           f"{'TP':>4} {'FP':>4} {'prec':>6} {'rec':>6} {'F1':>6} {'lat':>8} {'tokens':>9}")
    print(hdr)
    print("-" * len(hdr))
    for window, q, fname in CELLS:
        res = load(fname)
        if res is None:
            print(f"{window:>7} {q:>3} (no {fname} yet)")
            continue
        gt, n = res["ground_truth"], res["objects_in_window"]
        n_gt = len(gt["turns"])
        for r in res["runs"]:
            s = r.get("score")
            if s is None:
                why = r.get("parse_error") or r.get("error", "failed")
                print(f"{window:>7} {q:>3} {r['mode']:9} {n_gt:>6} {n:>6}   -- {why[:60]}")
                continue
            tok = r.get("usage", {}).get("totalTokenCount", 0)
            print(f"{window:>7} {q:>3} {r['mode']:9} {n_gt:>6} {n:>6} "
                  f"{s['answers']:>5} {s['true_positives']:>4} {s['false_positives']:>4} "
                  f"{s['precision']:>6.3f} {s['recall']:>6.3f} "
                  f"{_f1(s['true_positives'], s['false_positives'], n_gt - s['true_positives']):>6.3f} "
                  f"{r['latency_ms'] / 1000:>7.1f}s {tok:>9,}")
        print(f"{window:>7} {q:>3} {'TrQS':9} {n_gt:>6} {n:>6} {n_gt:>5} {n_gt:>4} "
              f"{0:>4} {1.0:>6.3f} {1.0:>6.3f} {1.0:>6.3f} "
              f"{res['trqs_ms'] / 1000:>7.3f}s {'-':>9}")
        print()
    print("TrQS is exact by construction: the TrQL query defines the event, so its")
    print("rows report latency and expressibility, not accuracy.\n")


def amortization() -> None:
    for window, cell, files in AMORT:
        runs = [load(f) for f in files]
        if any(r is None for r in runs):
            print(f"amortization: {window} cell incomplete, skipping")
            continue
        _one(window, INGEST_S[cell], runs)


def _one(window: str, ingest: float, runs: list[dict]) -> None:

    per: dict[str, list[float]] = {"static": [], "agentic": [], "trqs": []}
    for res in runs:
        for r in res["runs"]:
            per[r["mode"]].append(r["latency_ms"] / 1000.0)
        per["trqs"].append(res["trqs_ms"] / 1000.0)

    n = len(runs)
    print(f"\n{window} window — cumulative cost of k queries on one video "
          f"(k<={n} measured, beyond that the mean extrapolated):\n")
    print(f"  {'k':>3} {'TrQS':>12} {'static':>12} {'agentic':>12}   "
          f"{'vs static':>10} {'vs agentic':>11}")
    for k in (1, 2, 5, 10):
        row = {}
        for sys_, vals in per.items():
            mean = sum(vals) / len(vals)
            base = ingest if sys_ == "trqs" else 0.0
            row[sys_] = base + sum(vals[i] if i < len(vals) else mean
                                   for i in range(k))
        mark = " " if k <= n else "*"
        print(f"  {k:>3}{mark}{row['trqs']:>11.2f}s {row['static']:>11.1f}s "
              f"{row['agentic']:>11.1f}s   {row['static'] / row['trqs']:>9.1f}x "
              f"{row['agentic'] / row['trqs']:>10.1f}x")
    print(f"\n  ingestion {ingest:.1f}s is paid once; TrQS then adds "
          f"{', '.join(f'{v * 1000:.1f}ms' for v in per['trqs'])} per query.")
    print("  * extrapolated from the mean measured per-query cost, not measured.")


def main() -> None:
    table()
    amortization()


if __name__ == "__main__":
    main()
