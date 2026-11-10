"""Turning [`Plot`]s into files.

The selection keys (`queries`, `exclude`, `errs`, `keep_errs`, `aggregate`) mean the same
thing for every kind of plot, so they are applied here, once, and a kind only
ever sees the rows it is meant to draw.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from benchplot.db import collapse_repeats, load_runs
from benchplot.plots import KINDS
from benchplot.spec import ManifestError, Plot, Selection


@dataclass(frozen=True)
class Rendered:
    """What one drawn plot turned out to be, for the CLI to report."""

    plot: Plot
    path: Path
    summary: str


class Renderer:
    """Draws plots, reading each `bench.db` at most once.

    A manifest is typically many figures over one database; without the cache
    every figure would re-open and re-read it.
    """

    def __init__(self) -> None:
        self._runs: dict[Path, pd.DataFrame] = {}

    def runs(self, db: Path, where: str | None = None) -> pd.DataFrame:
        if db not in self._runs:
            try:
                self._runs[db] = load_runs(db)
            except (FileNotFoundError, ValueError) as exc:
                raise ManifestError(f"{where}: {exc}" if where else str(exc)) from exc
        return self._runs[db]

    def points(self, plot: Plot) -> pd.DataFrame:
        """The rows `plot` draws: repeats collapsed, then the selection applied."""
        where = f'plot "{plot.name}"'
        runs = self.runs(plot.selection.db, where)
        points = collapse_repeats(runs, how=plot.selection.aggregate)
        return _select(points, plot.selection, where=where)

    def render(self, plot: Plot) -> Rendered:
        kind = KINDS[plot.kind]
        if kind.needs_runs:
            points = self.points(plot)
            summary = (
                f"{points['query'].nunique()} queries, {len(points)} points, "
                f"{plot.selection.aggregate} of {len(self._runs[plot.selection.db])} runs"
            )
            missing = missing_queries(points, plot.selection)
            if missing:
                summary += f"; marked missing: {', '.join(missing)}"
        else:
            # The kind reads its own source; the runs database is not opened at
            # all, so a manifest may draw this figure on a machine without one.
            points = pd.DataFrame()
            summary = plot.kind
        if kind.describe is not None:
            summary = kind.describe(plot)

        try:
            path = kind.render(points, plot)
        except (FileNotFoundError, ValueError) as exc:
            raise ManifestError(f'plot "{plot.name}": {exc}') from exc
        return Rendered(plot=plot, path=path, summary=summary)


def missing_queries(points: pd.DataFrame, sel: Selection) -> list[str]:
    """Named queries with no runs, in manifest order; empty unless `mark_missing`."""
    if not (sel.mark_missing and sel.queries):
        return []
    have = set(points["query"])
    return [q for q in sel.queries if q not in have]


def _select(points: pd.DataFrame, sel: Selection, *, where: str) -> pd.DataFrame:
    """Apply `queries` / `exclude` / `errs` / `keep_errs`, loudly if nothing is left.

    A plot that silently came out empty is worse than one that did not draw:
    the manifest is checked in, and a renamed query should say so rather than
    leave a blank figure in the paper.
    """
    known = set(points["query"])
    absent: set[str] = set()
    if sel.queries:
        # Naming queries is also how the manifest fixes their order, so an
        # unknown name is a mistake rather than a filter that matched nothing.
        unknown = sorted(set(sel.queries) - known)
        if unknown and sel.mark_missing:
            # Drawn as marked gaps by the kind; there must still be something.
            if len(unknown) == len(set(sel.queries)):
                raise ManifestError(
                    f"{where}: none of the named queries has runs; the database "
                    f"holds {', '.join(sorted(known))}"
                )
            absent = set(unknown)
            known |= absent
            unknown = []
        if unknown:
            raise ManifestError(
                f"{where}: queries has unknown name{'s' if len(unknown) > 1 else ''} "
                f"{', '.join(unknown)}; the database holds {', '.join(sorted(known))}"
            )
        points = points[points["query"].isin(sel.queries)]
    elif sel.exclude:
        points = points[~points["query"].isin(sel.exclude)]
        if points.empty:
            raise ManifestError(f"{where}: exclude dropped every query")

    if sel.errs:
        budgets = sorted(points["err"].unique())
        points = points[points["err"].isin(sel.errs)]
        if points.empty:
            raise ManifestError(
                f"{where}: no run at errs {', '.join(f'{v:g}' for v in sel.errs)}; "
                f"these queries were measured at {', '.join(f'{v:g}' for v in budgets)}"
            )

    for query, dropped in sel.drop_errs.items():
        if query in absent:
            continue
        if query not in known:
            raise ManifestError(
                f"{where}: drop_errs names unknown query {query}; the database "
                f"holds {', '.join(sorted(known))}"
            )
        before = len(points)
        points = points[~(points["query"].eq(query) & points["err"].isin(dropped))]
        if len(points) == before:
            raise ManifestError(
                f"{where}: drop_errs {query} = "
                f"[{', '.join(f'{v:g}' for v in dropped)}] matched no run"
            )

    for query, kept in sel.keep_errs.items():
        if query in absent:
            continue
        if query not in known:
            raise ManifestError(
                f"{where}: keep_errs names unknown query {query}; the database "
                f"holds {', '.join(sorted(known))}"
            )
        in_query = points[points["query"].eq(query)]
        absent = sorted(set(kept) - set(in_query["err"]))
        if absent:
            raise ManifestError(
                f"{where}: keep_errs {query} asks for "
                f"[{', '.join(f'{v:g}' for v in absent)}], which was never "
                f"measured; it has {', '.join(f'{v:g}' for v in sorted(in_query['err']))}"
            )
        points = points[~points["query"].eq(query) | points["err"].isin(kept)]

    if sel.scales:
        measured = sorted(points["scale"].unique())
        points = points[points["scale"].isin(sel.scales)]
        if points.empty:
            raise ManifestError(
                f"{where}: no run at scales {', '.join(f'{v:g}' for v in sel.scales)}; "
                f"these queries were measured at {', '.join(f'{v:g}' for v in measured)}"
            )

    if points.empty:
        raise ManifestError(f"{where}: nothing left to draw after the selection")
    return points
