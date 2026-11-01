# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "duckdb>=1.5.6",
#     "tqdm",
#     "python-constraint==1.4.0",
#     "rekallpy",
# ]
#
# [tool.uv.sources]
# rekallpy = { path = "/home/satyam/dev/frameql/baselines/rekall/rekallpy" }
# ///
"""Rekall, DuckDB and TrQS on Rekall's own cyclist-detection query.

Rekall (Fu et al., Stanford TR 2019) is the closest published system to the
interval algebra this paper evaluates: spatiotemporal labels composed with joins
under Allen-style predicates. Its own evaluation is about authoring effort and
accuracy and reports no query-execution cost -- "the Rekall implementation
contains minimal optimization ... final queries ran in less than thirty
seconds". This script measures that cost, on its own tutorial query and its own
released detections, beside the two encodings the paper already compares.

    uv run papers/vldb27/suite/bench_rekall.py --tile 1 2 5 10

Three encodings of one question. Rekall holds a materialised `Interval` per
detection and joins them in Python; DuckDB holds one row per object per frame
and joins them relationally; TrQS holds fitted polynomial segments and evaluates
the predicate analytically. All three read the same boxes out of the same
`duckdb.db`, written by `build_rekall_cydet.py`.

WHAT IS TIMED, AND WHAT IS NOT

Excluded from every number, because it is the same work for everyone: the
detector, and reading the rows out of DuckDB.

Reported separately, never summed into a query time: each system's cost to get
from those rows into the form it queries. For Rekall that is
`IntervalSetMapping` construction -- `Interval` and `Bounds3D` allocation plus
the `sorted()` in `IntervalSet.__init__` -- a transient in-process Python object
that every session must rebuild and that has no on-disk form. For TrQS it is
`dataset build`, a compile step producing a reloadable `.trql`. For DuckDB it is
nothing: the rows are already the queryable form. These sit in different places
in a system's lifecycle, so adding any of them into a query time would be an
argument by arithmetic.

Timed: the `join` alone for Rekall (best-of-N, with the mapping already
materialised -- the analogue of DuckDB's "connection already open"), and for the
other two whatever `trql-cli bench --sql` times, which is the protocol every
other number in the paper is measured under.

TWO REKALL CONFIGURATIONS, AND WHY BOTH

`IntervalSet.join` copies a list slice of the other set per outer interval
(`interval_set.py:236`, `other.get_intervals()[start_index:]`), which is
quadratic in set size and has nothing to do with the query. Timing only that
would be reporting an implementation artifact, so the run covers:

  whole -- one `IntervalSet` for the video, the tutorial's own shape.
  frame -- an `IntervalSetMapping` keyed by frame. `IntervalSetMapping`'s binary
           methods join per matching key (`interval_set_mapping.py:293`), so
           each join sees one frame's boxes instead of the whole video's. Same
           operators, same predicate, same `window=0.0`; only the domain
           decomposition changes, and keying a mapping by domain is Rekall's own
           idiom.

`frame` is the number to quote. `whole` is reported beside it so the gap between
them is visible rather than assumed. The two answers must be identical --
`T(equal())` cannot match across frames -- and the script asserts it.

`window=0.0` is passed on every call, as the tutorial does. Left to default,
`IntervalSet._get_optimization_window` (`interval_set.py:63`) would use 1/100 of
the set's temporal span, ~67 frames here, and the `whole` number would be a
measurement of that default rather than of the query.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import duckdb
from rekall import Bounds3D, Interval, IntervalSet, IntervalSetMapping
from rekall.predicates import and_pred, equal, overlaps

_REPO = Path("/home/satyam/dev/frameql")
_PROCESSED = _REPO / "datasets" / "processed"
_SUITE = _REPO / "papers" / "vldb27" / "suite"
_TMP = Path("/home/satyam/.claude/jobs/c5f85dd3/tmp")
_CLI = _REPO / "target" / "release" / "trql-cli"

_QUERY_ID = "CYDET-cyclist"
_SUB = "0001"
# `raw` is Rekall's data as published -- no track ids, so every detection is a
# one-frame trajectory and TrQS's segment model has nothing to compress. That is
# its floor, and it is reported. `tracked` is the same boxes with identity added
# by `build_rekall_cydet.py`'s tracker, which is the regime TrQS is built for.
_VARIANTS = ("raw", "tracked")
_MAX_GAP = 1

# The tutorial's own predicate, built once: `and_pred` and the `Bounds3D.X/Y/T`
# casts allocate wrappers per call, so constructing it inside the timed region
# would charge Rekall for work the tutorial does at authoring time.
_PREDICATE = and_pred(
    Bounds3D.T(equal()), Bounds3D.X(overlaps()), Bounds3D.Y(overlaps())
)


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
def tiled(src: Path, times: int, out: Path) -> Path:
    """`times` copies of `src`, laid out exactly as `sqlrun::tiled_duckdb` does.

    Replica i shifts frames by i * (max_fid + 1) and objects by
    i * (max_oid + 1), which is what `trql-cli --db-repeat` does on both of its
    sides -- so the database Rekall sees here is the one the engine and DuckDB
    were measured on.
    """
    if times == 1:
        return src
    for stale in (out, out.with_suffix(".db.wal")):
        stale.unlink(missing_ok=True)
    conn = duckdb.connect(str(out))
    conn.execute(f"ATTACH '{src}' AS src (READ_ONLY)")
    stride = conn.execute("SELECT COALESCE(MAX(fid), -1) + 1 FROM src.bbox").fetchone()[0]
    oid_stride = conn.execute(
        "SELECT COALESCE(MAX(oid), -1) + 1 FROM src.objects"
    ).fetchone()[0]
    conn.execute(
        f"""CREATE TABLE objects AS
              SELECT o.* EXCLUDE (oid), o.oid + t.i * {oid_stride} AS oid
              FROM src.objects o CROSS JOIN generate_series(0, {times - 1}) t(i)"""
    )
    conn.execute(
        f"""CREATE TABLE bbox AS
              SELECT b.* EXCLUDE (oid, fid),
                     b.oid + t.i * {oid_stride} AS oid,
                     b.fid + t.i * {stride}     AS fid
              FROM src.bbox b CROSS JOIN generate_series(0, {times - 1}) t(i)"""
    )
    conn.execute("DETACH src")
    conn.close()
    return out


def load_rows(db: Path) -> tuple[list, list, list[int]]:
    """The person boxes, the bicycle boxes, and the engine's object-index map.

    Index `i` of a TrQS tuple is `oids[i]` in the database
    (`sqlrun::engine_oids`), which is what lets a Rekall answer be written as the
    ground truth the engine is scored against.
    """
    conn = duckdb.connect(str(db), read_only=True)
    conn.execute("SET threads=1")
    rows = {}
    for cls in ("person", "bicycle"):
        rows[cls] = conn.execute(
            """SELECT b.fid, b.oid, b.x1, b.y1, b.x2, b.y2
                 FROM bbox b JOIN objects o ON o.oid = b.oid AND o.class = ?
                ORDER BY b.fid, b.oid""",
            [cls],
        ).fetchall()
    oids = [r[0] for r in conn.execute("SELECT DISTINCT oid FROM bbox ORDER BY oid").fetchall()]
    conn.close()
    return rows["person"], rows["bicycle"], oids


# --------------------------------------------------------------------------- #
# Rekall
# --------------------------------------------------------------------------- #
def _interval(row) -> Interval:
    fid, oid, x1, y1, x2, y2 = row
    # Frame-index time and pixel coordinates, where the tutorial uses seconds and
    # width-normalised bounds. Every predicate in this query compares two
    # coordinates of the same kind, so applying one order-preserving rescaling to
    # both operands leaves it invariant; integers additionally remove the only
    # float-equality test in the query (`T(equal())`) and put Rekall on exactly
    # the integers `trql-storage` ingests.
    return Interval(Bounds3D(t1=fid, t2=fid + 1, x1=x1, x2=x2, y1=y1, y2=y2), payload=oid)


def build_ism(person: list, bikes: list, keyed_by_frame: bool):
    """Rekall's queryable form, in one of the two decompositions."""
    if not keyed_by_frame:
        return (
            IntervalSetMapping({0: IntervalSet([_interval(r) for r in person])}),
            IntervalSetMapping({0: IntervalSet([_interval(r) for r in bikes])}),
        )
    out = []
    for rows in (person, bikes):
        by_frame: dict[int, list] = {}
        for r in rows:
            by_frame.setdefault(r[0], []).append(_interval(r))
        out.append(IntervalSetMapping({f: IntervalSet(v) for f, v in by_frame.items()}))
    return tuple(out)


def _merge(i1: Interval, i2: Interval) -> Interval:
    # The tutorial's `merge_op` is `Interval(i1['bounds'].span(i2['bounds']))`,
    # which discards payload -- it can, because its boxes have no identity. The
    # pair of oids is carried here because without it the answer cannot be
    # compared with the other two systems at all. It costs one tuple per *output*
    # interval, and `--verbatim-merge` times the tutorial's version so that is
    # not taken on trust.
    return Interval(i1["bounds"].span(i2["bounds"]), payload=(i1["payload"], i2["payload"]))


def _merge_verbatim(i1: Interval, i2: Interval) -> Interval:
    return Interval(i1["bounds"].span(i2["bounds"]))


def run_rekall(person_ism, bikes_ism, merge_op) -> tuple[float, object]:
    """Best-of-one `join`, timed alone. `window=0.0`, as the tutorial passes."""
    t0 = time.perf_counter()
    out = person_ism.join(
        bikes_ism, predicate=_PREDICATE, merge_op=merge_op, window=0.0
    )
    # `join` materialises its result, but touch the size inside the region so no
    # laziness can hide work outside the clock.
    n = sum(len(s) for s in out.get_grouped_intervals().values())
    ms = (time.perf_counter() - t0) * 1000.0
    return ms, (out, n)


def _deep_size(obj, seen: set[int]) -> int:
    """Bytes held by `obj` and everything reachable from it, counted once.

    `sys.getsizeof` is shallow, so a traversal is needed: an `Interval` is a
    small object pointing at a `Bounds3D`, which points at its own `__dict__` of
    six floats, and an `IntervalSet` at a list of those. Containers are walked
    by type and arbitrary objects through `__dict__`, which covers every class
    Rekall builds here -- none of them uses `__slots__`.
    """
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)
    total = sys.getsizeof(obj)
    if isinstance(obj, (str, bytes, bytearray, int, float, type(None))):
        return total
    if isinstance(obj, dict):
        for k, v in obj.items():
            total += _deep_size(k, seen) + _deep_size(v, seen)
        return total
    if isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            total += _deep_size(v, seen)
        return total
    d = getattr(obj, "__dict__", None)
    if d is not None:
        total += _deep_size(d, seen)
    return total


def ism_memory_mb(person: list, bikes: list) -> dict[str, float]:
    """Bytes each `IntervalSetMapping` holds, by traversing it.

    Resident-set measurement does not work for this, by three routes.
    `ru_maxrss` is a high-water mark that never falls, so after one large
    mapping every later one reads as free. A before/after difference in current
    RSS under-reports, because a freed mapping leaves the allocator holding
    pages the next one fills without growing the resident set -- the whole-video
    configuration first came out at 10 MB for 393 k intervals that way. And
    measuring a fresh child process against a control does not separate the two
    either, since loading the rows out of DuckDB raises the watermark above what
    the build then needs.

    Traversing the object graph sidesteps all of it and is what the claim
    actually needs: how much the materialised-label representation costs to
    hold, exclusive of interpreter and allocator overhead -- so it is a floor on
    the resident cost, not an estimate of it.
    """
    out = {}
    for name, by_frame in (("whole", False), ("frame", True)):
        p_ism, b_ism = build_ism(person, bikes, by_frame)
        seen: set[int] = set()
        out[name] = (
            _deep_size(p_ism, seen) + _deep_size(b_ism, seen)
        ) / (1024.0 * 1024.0)
        del p_ism, b_ism
        gc.collect()
    return out


def _interval_tutorial(row, fps: float, width: int, height: int) -> Interval:
    """One detection in the tutorial's own units: seconds, frame-relative boxes.

    `cyclist_tutorial_helpers.py:get_maskrcnn_bboxes` verbatim, minus the
    `SpatialType_Bbox` payload (which only vgrid reads) and plus the oid the
    answers are compared on.
    """
    fid, oid, x1, y1, x2, y2 = row
    return Interval(
        Bounds3D(
            t1=fid / fps,
            t2=(fid + 1) / fps,
            x1=x1 / width,
            x2=x2 / width,
            y1=y1 / height,
            y2=y2 / height,
        ),
        payload=oid,
    )


def verify_encoding(db: Path, meta: dict) -> None:
    """The tutorial's units and ours answer the same question on the same boxes.

    Every predicate in this query compares two coordinates of the same kind, so
    the pixel/frame encoding the benchmark uses should be invariant to the
    tutorial's rescaling -- `should` being the reason this check exists rather
    than a paragraph asserting it. Run once; it is not timed and not reported.
    """
    person, bikes, _ = load_rows(db)
    fps, w, h = meta["fps"], meta["width"], meta["height"]

    p_ism, b_ism = build_ism(person, bikes, True)
    _, (out, _n) = run_rekall(p_ism, b_ism, _merge)
    ours = triples(out)

    tut = IntervalSetMapping(
        {0: IntervalSet([_interval_tutorial(r, fps, w, h) for r in person])}
    ).join(
        IntervalSetMapping(
            {0: IntervalSet([_interval_tutorial(r, fps, w, h) for r in bikes])}
        ),
        predicate=_PREDICATE,
        merge_op=_merge,
        window=0.0,
    )
    tut_triples = {
        (i["payload"][0], i["payload"][1], round(i["t1"] * fps))
        for st in tut.get_grouped_intervals().values()
        for i in st.get_intervals()
    }
    if tut_triples != ours:
        raise SystemExit(
            "the tutorial's seconds/normalised encoding and the benchmark's "
            f"frame/pixel encoding disagree: {len(tut_triples - ours)} only in "
            f"the tutorial's, {len(ours - tut_triples)} only in ours, "
            f"of {len(tut_triples)} and {len(ours)}"
        )
    print(
        f"encoding check: {len(ours):,} (pair, frame) rows, identical under the "
        "tutorial's seconds/frame-relative encoding and the benchmark's "
        "frame/pixel one"
    )


def triples(out) -> set[tuple[int, int, int]]:
    """A Rekall answer as `(person oid, bicycle oid, frame)`.

    The finest projection the three systems share, and insensitive to each one's
    interval-merging convention: `t1` is the frame, since every input interval is
    one frame long and `T(equal())` cannot span two.
    """
    return {
        (i["payload"][0], i["payload"][1], int(i["t1"]))
        for s in out.get_grouped_intervals().values()
        for i in s.get_intervals()
    }


# --------------------------------------------------------------------------- #
# DuckDB and TrQS
# --------------------------------------------------------------------------- #
def duckdb_triples(db: Path) -> set[tuple[int, int, int]]:
    """The reference answer, per frame, for the equality gate only (not timed).

    `sql-ref/CYDET-cyclist.sql` projects maximal runs; this asks the same
    question one frame at a time, so a Rekall answer can be compared to it
    without either side's run-merging getting in the way.
    """
    conn = duckdb.connect(str(db), read_only=True)
    conn.execute("SET threads=1")
    rows = conn.execute(
        """SELECT bp.oid, bb.oid, bp.fid
             FROM bbox bp JOIN objects jp ON jp.oid = bp.oid AND jp.class = 'person'
             JOIN bbox bb ON bb.fid = bp.fid
             JOIN objects jb ON jb.oid = bb.oid AND jb.class = 'bicycle'
            WHERE bp.x1 < bb.x2 AND bb.x1 < bp.x2
              AND bp.y1 < bb.y2 AND bb.y1 < bp.y2"""
    ).fetchall()
    conn.close()
    return {(p, b, f) for p, b, f in rows}


def write_gt(path: Path, trs: set[tuple[int, int, int]], oids: list[int]) -> None:
    """A Rekall answer as the CSV `bench --gt` scores the engine against.

    In engine-index space, since the engine reports indices rather than database
    oids. `QueryOutput::from_csv` merges a tuple's intervals on read, so one row
    per (pair, frame) becomes the maximal-run form the engine produces.
    """
    index = {oid: i for i, oid in enumerate(oids)}
    lines = ["oid0,oid1,start,end"]
    lines += [f"{index[p]},{index[b]},{f},{f}" for p, b, f in sorted(trs)]
    path.write_text("\n".join(lines) + "\n")


def run_trql(variant: str, tile: int, err: int, gt: Path | None, runs: int) -> dict:
    """`trql-cli bench`, which times TrQS and the reference SQL under one roof."""
    cmd = [
        str(_CLI),
        "--dataset", f"rekall_cydet_{variant}",
        "--sub-dataset", _SUB,
        "--max-gap", str(_MAX_GAP),
        "--max-error-px", str(err),
        "bench", _QUERY_ID,
        (_SUITE / "queries" / f"{_QUERY_ID}.trql").read_text().strip(),
        str(tile), str(runs),
        "--sql", str(_SUITE / "sql-ref" / f"{_QUERY_ID}.sql"),
        "--bench-db", str(_REPO / "bench.db"),
    ]
    if gt is not None:
        cmd += ["--gt", str(gt)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    report = (
        _PROCESSED
        / f"rekall_cydet_{variant}"
        / _SUB
        / "benchmark"
        / f"{_QUERY_ID}_{_MAX_GAP}_{err}_x{tile}.json"
    )
    return json.loads(report.read_text())


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def report(path: Path) -> None:
    """Print a results file as the table the experiment reports."""
    rows = json.loads(path.read_text())
    for variant in _VARIANTS:
        rs = [r for r in rows if r["variant"] == variant]
        if not rs:
            continue
        print(f"\n{variant.upper()}  (eps=0; neither Rekall nor DuckDB sees a budget)")
        print(f"{'boxes':>10} {'segments':>10} | {'Rekall whole':>13} "
              f"{'Rekall frame':>13} {'DuckDB':>9} {'TrQS':>8} | "
              f"{'vs frame':>9} {'vs DuckDB':>10} {'answers':>8}")
        for r in sorted((x for x in rs if x["max_error_px"] == 0),
                        key=lambda x: x["scale"]):
            rk = r["rekall"]
            print(f"{r['boxes']:>10,.0f} {r['segments']:>10,} | "
                  f"{rk['whole']['ms_query']:>13,.1f} {rk['frame']['ms_query']:>13,.1f} "
                  f"{r['ms_duckdb']:>9,.1f} {r['ms_trqs']:>8,.1f} | "
                  f"{rk['frame']['ms_query'] / r['ms_trqs']:>8.1f}x "
                  f"{r['ms_duckdb'] / r['ms_trqs']:>9.1f}x {r['answer_rows']:>8,}")

        top = max(x["scale"] for x in rs)
        print(f"  epsilon sweep, TrQS only, at x{top}:")
        for r in sorted((x for x in rs if x["scale"] == top),
                        key=lambda x: x["max_error_px"]):
            print(f"    eps={r['max_error_px']:<3} segments {r['segments']:>10,}  "
                  f"{r['ms_trqs']:>8.1f} ms  {r['mb_trqs']:>6.0f} MB  "
                  f"P/R/IoU {r['precision']:.4f}/{r['recall']:.4f}/{r['mean_iou']:.4f}")

        print("  getting into queryable form -- reported, never summed into a query time:")
        for r in sorted((x for x in rs if x["max_error_px"] == 0),
                        key=lambda x: x["scale"]):
            rk = r["rekall"]
            print(f"    x{r['scale']:<4} Rekall mapping  "
                  f"whole {rk['whole']['ms_build']:>7,.0f} ms / "
                  f"{rk['whole']['mb_objects']:>6,.0f} MB held   "
                  f"frame {rk['frame']['ms_build']:>7,.0f} ms / "
                  f"{rk['frame']['mb_objects']:>6,.0f} MB held")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", type=int, nargs="+", default=[1, 2, 5, 10],
                    help="tiling factors (--db-repeat)")
    ap.add_argument("--errs", type=int, nargs="+", default=[0, 1, 2, 4],
                    help="TrQS error budgets; only eps=0 is gated on exactness")
    ap.add_argument("--variants", nargs="+", default=list(_VARIANTS), choices=_VARIANTS)
    ap.add_argument("--runs", type=int, default=3, help="timed repeats")
    ap.add_argument("--rekall-budget-ms", type=float, default=5000.0,
                    help="stop repeating a Rekall configuration once one run costs more")
    ap.add_argument("--verbatim-merge", action="store_true",
                    help="also time Rekall with the tutorial's payload-dropping merge_op")
    ap.add_argument("--verify-encoding", action="store_true",
                    help="check our frame/pixel encoding against the tutorial's "
                         "seconds/frame-relative one, then exit")
    ap.add_argument("--report", action="store_true",
                    help="print `--out` as a table and exit, measuring nothing")
    ap.add_argument("--out", type=Path, default=_SUITE / "results-rekall.json")
    args = ap.parse_args()

    if args.report:
        report(args.out)
        return

    if args.verify_encoding:
        for variant in args.variants:
            sub = _PROCESSED / f"rekall_cydet_{variant}" / _SUB
            print(f"{variant}: ", end="")
            verify_encoding(
                sub / "duckdb.db", json.loads((sub / "metadata.json").read_text())
            )
        return

    results = []
    for variant in args.variants:
        src = _PROCESSED / f"rekall_cydet_{variant}" / _SUB / "duckdb.db"
        for tile in args.tile:
            db = tiled(src, tile, _TMP / f"cydet-{variant}-x{tile}.db")
            person, bikes, oids = load_rows(db)
            ref = duckdb_triples(db)

            rk: dict[str, dict] = {}
            answers: dict[str, set] = {}
            mb = ism_memory_mb(person, bikes)
            for name, by_frame in (("whole", False), ("frame", True)):
                t0 = time.perf_counter()
                p_ism, b_ism = build_ism(person, bikes, by_frame)
                build_ms = (time.perf_counter() - t0) * 1000.0
                # Best-of-N, but stop repeating once one run costs more than
                # the budget: at the top tiling factors the whole-video
                # decomposition is tens of seconds and three of them buy
                # nothing. `runs` records what was actually taken, so a
                # best-of-1 point is visible as one.
                best, out, n, taken = None, None, None, 0
                while taken < args.runs:
                    ms, (out, n) = run_rekall(p_ism, b_ism, _merge)
                    best = ms if best is None else min(best, ms)
                    taken += 1
                    if ms > args.rekall_budget_ms:
                        break
                answers[name] = triples(out)
                rk[name] = {
                    "ms_query": best,
                    "ms_build": build_ms,
                    "answer_intervals": n,
                    "answer_frames": len(answers[name]),
                    "mb_objects": mb[name],
                    "runs": taken,
                }
                if args.verbatim_merge:
                    ms, _ = run_rekall(p_ism, b_ism, _merge_verbatim)
                    rk[name]["ms_query_verbatim_merge"] = ms
                del p_ism, b_ism, out
                gc.collect()

            # Gate 1: the two decompositions are the same query, so the same
            # answer -- a difference means the frame keying changed semantics.
            if answers["whole"] != answers["frame"]:
                raise SystemExit(
                    f"{variant} x{tile}: Rekall's whole-video and frame-keyed "
                    "answers differ; the decomposition is not semantics-preserving"
                )
            # Gate 2: Rekall against the relational answer, per frame. Nothing
            # below this line is believable without it.
            if answers["frame"] != ref:
                only_rk = len(answers["frame"] - ref)
                only_sql = len(ref - answers["frame"])
                raise SystemExit(
                    f"{variant} x{tile}: Rekall and SQL disagree -- "
                    f"{only_rk} only in Rekall, {only_sql} only in SQL, "
                    f"of {len(ref)} reference (pair, frame) rows"
                )

            gt = _TMP / f"cydet-{variant}-x{tile}.gt.csv"
            write_gt(gt, answers["frame"], oids)

            for err in args.errs:
                rep = run_trql(variant, tile, err, gt, args.runs)
                sql = rep["sql"]
                # `metrics` is the scoring against `--gt`; absent means the CSV
                # never reached the comparison, which would silently skip gate 3.
                m = rep.get("metrics")
                if m is None:
                    raise SystemExit(
                        f"{variant} x{tile} eps={err}: no `metrics` in the report -- "
                        "`--gt` was not scored"
                    )
                # Gate 3, at eps=0 only: the engine reproduces Rekall's answer
                # exactly. Above eps=0 the polynomial fit is lossy by
                # construction, so the same numbers are a measurement, not a gate.
                if err == 0 and not (
                    sql["exact"]
                    and m["precision"] == 1.0
                    and m["recall"] == 1.0
                    and m["mean_iou"] == 1.0
                    and m["interval_mismatches"] == 0
                ):
                    raise SystemExit(
                        f"{variant} x{tile} eps=0: TrQS does not reproduce "
                        f"Rekall's answer exactly ({m})"
                    )
                results.append({
                    "variant": variant,
                    "scale": tile,
                    "max_error_px": err,
                    "boxes": rep["db_stats"]["num_objs"] * rep["db_stats"]["avg_traj_len"],
                    "objects": rep["db_stats"]["num_objs"],
                    "segments": rep["db_stats"]["total_segments"],
                    "answer_rows": rep["rows"],
                    "ms_trqs": rep["avg_ms"],
                    "mb_trqs": rep["peak_mem_mb"],
                    "ms_duckdb": sql["elapsed_ms"],
                    "mb_duckdb": sql["peak_mem_mb"],
                    "trqs_exact": sql["exact"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "mean_iou": m["mean_iou"],
                    "rekall": rk,
                    "rekall_frames": len(ref),
                })
                r = results[-1]
                print(
                    f"{variant:8s} x{tile:<3d} eps={err:<2d}  "
                    f"boxes {r['boxes']:>9,.0f}  seg {r['segments']:>8,}  "
                    f"TrQS {r['ms_trqs']:9.1f}  DuckDB {r['ms_duckdb']:9.1f}  "
                    f"Rekall[frame] {rk['frame']['ms_query']:10.1f}  "
                    f"[whole] {rk['whole']['ms_query']:10.1f}  "
                    f"P/R {r['precision']:.3f}/{r['recall']:.3f}"
                )

            if db != src:
                db.unlink(missing_ok=True)

    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    report(args.out)


if __name__ == "__main__":
    main()
