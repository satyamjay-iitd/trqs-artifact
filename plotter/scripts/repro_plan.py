"""Which `trql-cli bench` runs re-measure the points a set of figures draws.

`reproduce_paper.sh` calls this; it is not usually run by hand.

The manifest already says, figure by figure, which (query, budget, scale) points
are drawn. This reads it with benchplot's own manifest reader and applies the
same selection the renderer does, against the *recorded* `bench.db` opened
read-only -- so the plan is exactly the set of runs behind the published
figures, not a list copied out of them that could drift. The recorded database
also supplies what the manifest does not: each point's sub-dataset and gap.

A plot kind may narrow its points further through its own options (`p15` draws
only the first of each query's `budgets`; `speedup` and `epsilon` draw only the
named `budgets` / `eps`). Those are applied here too, so no run is planned for a
point no figure reads -- which matters when an unread point is I24-Motion at
eps=0, an 11.7 GB run.

Each point's TrQL comes from `suite/queries/`, matched by *content* against
the text recorded with the run (whitespace ignored). A query recorded under a
different id from its file (`ST-swap-img` runs `ST-swap.trql`) is found that way
without an alias table, and a point whose TrQL no longer exists in the
repository is reported rather than re-run with something else. Its reference
SQL is `suite/sql-ref/<same name>.sql` as it stands now: the SQL is the
correctness oracle, and a revised oracle (NS-crossing's, after the reference
was brought in line with the paper's semantics) is reported as `revised` rather
than skipped.

Figure 9's ablations (`degree_ablation`, `interval_ablation`) compare engine
*builds*, so their runs are not in `bench.db`; their manifest entries carry the
matrix instead (`[plots.measure]`). `--ablations` prints that plan, one line per
(variant, budget):

    plot query family sub err gap runs variant features default_features output fit trql sql status

`features` is comma-separated (`-` for none); variants with the same `fit` share
fitted files; `status` is `todo`, or `done` when
the output is already in `--ablation-dir/<plot>/`.

Output is tab-separated, one line per distinct point, header first:

    plots query family sub err gap scale trql sql status sql_vs_recorded

`status` is `todo`, `done` (already in `--skip-done-in`), or `no-query-file`;
`sql_vs_recorded` is `same` or `revised` (comments and whitespace ignored).

    uv run --locked python scripts/repro_plan.py --plots p15 epsilon compression
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

from benchplot.manifest import read_manifest
from benchplot.render import Renderer
from benchplot.spec import ManifestError

HERE = Path(__file__).resolve().parent
PLOTTER = HERE.parent
ROOT = PLOTTER.parent

# The bench.db-backed figures of the paper (see the plotter README's table):
# 4a compression, 5 p15, 8 epsilon. The other paper figures read their own
# recorded inputs, not `bench.db`.
PAPER_PLOTS = ("compression", "p15", "epsilon")


def _norm_trql(text: str) -> str:
    return " ".join(text.split())


def _norm_sql(text: str) -> str:
    # `--` comments only; the reference SQL has no `--` inside string literals.
    return " ".join(re.sub(r"--[^\n]*", "", text).split())


def _index(directory: Path, suffix: str, norm) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for path in sorted(directory.glob(f"*{suffix}")):
        out.setdefault(norm(path.read_text()), []).append(path)
    return out


def _pick(candidates: list[Path], name: str) -> Path:
    """Prefer the file whose stem is a prefix of the recorded id, else the first."""
    for path in sorted(candidates, key=lambda p: -len(p.stem)):
        if name.startswith(path.stem):
            return path
    return candidates[0]


def _recorded_text(ref: Path) -> dict[tuple[str, str], tuple[str, str]]:
    with sqlite3.connect(f"file:{ref}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT q.name, d.name, q.trql_query_text, q.sql_query_text "
            "FROM Query q JOIN Dataset d ON d.id = q.dataset_id"
        ).fetchall()
    return {(q, d): (t, s) for q, d, t, s in rows}


def _done(db: Path | None) -> set[tuple[str, str, float, int, int]]:
    if db is None or not db.is_file():
        return set()
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT q.name, d.name, r.trql_error_px, r.trql_gap, COALESCE(r.scale, 1) "
            "FROM Run r JOIN Query q ON q.id = r.query_id "
            "JOIN Dataset d ON d.id = q.dataset_id "
            "WHERE r.sql_time IS NOT NULL AND r.trql_time > 0"
        ).fetchall()
    return {(q, d, float(e), int(g), int(s)) for q, d, e, g, s in rows}


def _narrow(points, plot):
    """Apply the kind-specific budget filters that `render` applies inside the kind."""
    opts = plot.options
    budgets = getattr(opts, "budgets", None) or {}
    eps = getattr(opts, "eps", None) or {}
    keep = []
    for _, row in points.iterrows():
        q, e = row["query"], float(row["err"])
        if plot.kind == "p15":
            if q in budgets and e != float(budgets[q][0]):
                continue
        else:
            if q in eps and e not in {float(v) for v in eps[q]}:
                continue
            if q in budgets and e not in {float(v) for v in budgets[q]}:
                continue
        keep.append(row)
    return keep


def _split_dataset(name: str, scale: int) -> tuple[str, str]:
    suffix = f" x{scale}"
    if scale > 1 and name.endswith(suffix):
        name = name[: -len(suffix)]
    family, _, sub = name.partition("/")
    return family, sub


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, default=PLOTTER / "manifest.toml")
    ap.add_argument(
        "--reference-db",
        type=Path,
        default=ROOT / "bench.db",
        help="Recorded database the plan is read from (opened read-only).",
    )
    ap.add_argument("--plots", nargs="+", default=list(PAPER_PLOTS))
    ap.add_argument(
        "--skip-done-in",
        type=Path,
        help="New bench.db; points already measured there are marked `done`.",
    )
    ap.add_argument(
        "--ablations",
        action="store_true",
        help="Print the plan of the ablation plots among --plots instead.",
    )
    ap.add_argument(
        "--ablation-dir",
        type=Path,
        help="With --ablations: outputs already there are marked `done`.",
    )
    args = ap.parse_args()

    ref = args.reference_db.resolve()
    try:
        plots = read_manifest(args.manifest, db=ref)
    except ManifestError as exc:
        print(f"repro_plan: {exc}", file=sys.stderr)
        return 2
    by_name = {p.name: p for p in plots}
    unknown = [n for n in args.plots if n not in by_name]
    if unknown:
        print(
            f"repro_plan: no plot named {', '.join(unknown)}; "
            f"the manifest holds {', '.join(by_name)}",
            file=sys.stderr,
        )
        return 2

    from benchplot.plots import KINDS

    if args.ablations:
        return _ablation_plan([by_name[n] for n in args.plots], args.ablation_dir)

    renderer = Renderer()
    points: dict[tuple[str, str, float, int, int], list[str]] = {}
    for name in args.plots:
        plot = by_name[name]
        if hasattr(plot.options, "measure"):
            continue  # an ablation: planned by --ablations, not from bench.db
        if not KINDS[plot.kind].needs_runs:
            print(
                f"repro_plan: {name} ({plot.kind}) reads its own recorded inputs, "
                "not bench.db; nothing to re-run for it",
                file=sys.stderr,
            )
            continue
        try:
            selected = renderer.points(plot)
        except ManifestError as exc:
            print(f"repro_plan: {exc}", file=sys.stderr)
            return 2
        for row in _narrow(selected, plot):
            key = (
                row["query"],
                row["dataset"],
                float(row["err"]),
                int(row["gap"]),
                int(row["scale"]),
            )
            points.setdefault(key, []).append(name)

    recorded = _recorded_text(ref)
    trql_index = _index(ROOT / "suite" / "queries", ".trql", _norm_trql)
    done = _done(args.skip_done_in.resolve() if args.skip_done_in else None)

    print("plots\tquery\tfamily\tsub\terr\tgap\tscale\ttrql\tsql\tstatus\tsql_vs_recorded")
    for key in sorted(points):
        query, dataset, err, gap, scale = key
        family, sub = _split_dataset(dataset, scale)
        trql_text, sql_text = recorded[(query, dataset)]
        trql = trql_index.get(_norm_trql(trql_text))
        trql_file = _pick(trql, query) if trql else None
        # The TrQL is the experiment's identity and must match what was measured.
        # The reference SQL is whatever `sql-ref/` holds for that query now: it is
        # a correctness oracle that may have been revised since the recording.
        sql_file = ROOT / "suite" / "sql-ref" / f"{trql_file.stem}.sql" if trql_file else None
        if sql_file is not None and not sql_file.is_file():
            sql_file = None
        trql_path = trql_file.relative_to(ROOT) if trql_file else "-"
        sql_path = sql_file.relative_to(ROOT) if sql_file else "-"
        sql_note = "-"
        if sql_file is not None:
            same = _norm_sql(sql_file.read_text()) == _norm_sql(sql_text)
            sql_note = "same" if same else "revised"
        if not trql_file or not sql_file:
            status = "no-query-file"
        elif key in done:
            status = "done"
        else:
            status = "todo"
        print(
            "\t".join(
                [
                    ",".join(points[key]),
                    query,
                    family,
                    sub,
                    f"{err:g}",
                    str(gap),
                    str(scale),
                    str(trql_path),
                    str(sql_path),
                    status,
                    sql_note,
                ]
            )
        )
    return 0


def _ablation_done(directory: Path | None, plot: str, query: str, variant, err: float) -> bool:
    if directory is None:
        return False
    where = directory / plot
    if variant.output == "report":
        return (where / f"{query}-{variant.name}.json").is_file()
    db = where / f"{variant.name}.db"
    if not db.is_file():
        return False
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        n = conn.execute(
            "SELECT count(*) FROM Run r JOIN Query q ON q.id = r.query_id "
            "WHERE q.name = ? AND r.trql_error_px = ? AND r.trql_time > 0",
            (query, err),
        ).fetchone()[0]
    return n > 0


def _ablation_plan(plots, directory: Path | None) -> int:
    print("plot\tquery\tfamily\tsub\terr\tgap\truns\tvariant\tfeatures\t"
          "default_features\toutput\tfit\ttrql\tsql\tstatus")
    for plot in plots:
        m = getattr(plot.options, "measure", None)
        if m is None:
            continue
        family, _, sub = m.dataset.partition("/")
        trql = ROOT / "suite" / "queries" / f"{m.query}.trql"
        sql = ROOT / "suite" / "sql-ref" / f"{m.query}.sql"
        trql_s = str(trql.relative_to(ROOT)) if trql.is_file() else "-"
        sql_s = str(sql.relative_to(ROOT)) if (m.sql and sql.is_file()) else "-"
        for v in m.variants:
            for err in m.budgets:
                if trql_s == "-" or (m.sql and sql_s == "-"):
                    status = "no-query-file"
                elif _ablation_done(directory, plot.name, m.query, v, err):
                    status = "done"
                else:
                    status = "todo"
                print("\t".join([
                    plot.name, m.query, family, sub, f"{err:g}", str(m.max_gap),
                    str(m.runs), v.name, ",".join(v.features) or "-",
                    "true" if v.default_features else "false", v.output, v.fit,
                    trql_s, sql_s, status,
                ]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
