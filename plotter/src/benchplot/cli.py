"""`benchplot` — plots for the VLDB'27 evaluation, from a `trql-cli bench.db`.

Every figure is described in `manifest.toml`; this is the thin layer that reads
it, draws what it asks for, and says what it wrote. Plot *settings* belong in
the manifest, which is checked in with the paper — the flags here are only the
one-off overrides (another database, another output directory, a subset).
"""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from typing import Annotated, NoReturn, Optional

import typer

from benchplot.db import load_runs
from benchplot.manifest import DEFAULT_MANIFEST, Defaults, read_defaults, read_manifest
from benchplot.plots import KINDS
from benchplot.render import Renderer
from benchplot.spec import ManifestError, Plot

app = typer.Typer(
    add_completion=False,
    help="Draw the evaluation's figures as described by a manifest.toml.",
    no_args_is_help=True,
)

ManifestArgument = Annotated[
    Path,
    typer.Argument(help="The manifest describing the figures.", show_default=True),
]
ManifestOption = Annotated[
    Path,
    typer.Option("--manifest", "-m", help="The manifest to take [defaults] from."),
]
DbOption = Annotated[
    Optional[Path],
    typer.Option("--db", help="Override the manifest's bench.db for this run."),
]


@app.command()
def render(
    manifest: ManifestArgument = DEFAULT_MANIFEST,
    only: Annotated[
        Optional[list[str]],
        typer.Option(
            "--only",
            "-n",
            help="Draw only these plots, by manifest name. Repeatable. Default: all.",
        ),
    ] = None,
    db: DbOption = None,
    out_dir: Annotated[
        Optional[Path],
        typer.Option("--out-dir", help="Override the manifest's output directory."),
    ] = None,
    source_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--source-dir",
            help="For plots that read their own measurements (Fig 9's ablations, "
            "the VLM figures): read them from this directory instead of the "
            "manifest's `source` (and `stages`).",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Say what would be drawn, draw nothing."),
    ] = False,
) -> None:
    """Draw every plot in the manifest."""
    plots = _plots(manifest, db=db, out_dir=out_dir)

    if only:
        named = {p.name for p in plots}
        unknown = sorted(set(only) - named)
        if unknown:
            _die(
                f"no plot named {', '.join(unknown)} in {manifest}; it holds "
                f"{', '.join(sorted(named))}"
            )
        plots = [p for p in plots if p.name in set(only)]

    if not plots:
        typer.echo(f"{manifest}: every plot is disabled — nothing to draw")
        return

    if source_dir is not None:
        plots = [_with_source(p, source_dir.resolve()) for p in plots]

    if dry_run:
        for plot in plots:
            typer.echo(f"{plot.name:<20} {plot.kind:<10} -> {_short(plot.out)}")
        return

    renderer = Renderer()
    for plot in plots:
        try:
            done = renderer.render(plot)
        except ManifestError as exc:
            _die(str(exc))
        typer.echo(f"{done.plot.name:<20} {_short(done.path)}  ({done.summary})")


def _with_source(plot: Plot, source: Path) -> Plot:
    """`plot` reading its measurements from `source`, or a clear refusal."""
    names = {f.name for f in fields(plot.options)} if plot.options is not None else set()
    if "source" not in names:
        _die(f'--source-dir: plot "{plot.name}" ({plot.kind}) reads bench.db, not a '
             "source directory; select plots with --only")
    changes = {k: source for k in ("source", "stages") if k in names}
    return replace(plot, options=replace(plot.options, **changes))


@app.command(name="plots")
def list_plots(manifest: ManifestArgument = DEFAULT_MANIFEST) -> None:
    """Show what the manifest asks for, without touching the database."""
    for plot in _plots(manifest):
        kind = KINDS[plot.kind]
        if not kind.needs_runs:
            # None of the selection keys reach a kind that reads no runs, so
            # printing them here would describe a filter that does nothing.
            note = kind.describe(plot) if kind.describe else "no runs"
            typer.echo(
                f"{plot.name:<20} {plot.kind:<12} -> {_short(plot.out):<24} {note}"
            )
            continue

        sel = plot.selection
        scope = []
        if sel.queries:
            scope.append(f"queries={len(sel.queries)}")
        elif sel.exclude:
            scope.append(f"-{','.join(sorted(sel.exclude))}")
        if sel.errs:
            scope.append("errs=" + ",".join(f"{v:g}" for v in sel.errs))
        if sel.scales:
            scope.append("scales=" + ",".join(f"{v:g}" for v in sel.scales))
        typer.echo(
            f"{plot.name:<20} {plot.kind:<12} -> {_short(plot.out):<24} "
            f"{sel.aggregate} {' '.join(scope)}"
        )


@app.command(name="list")
def list_runs(
    manifest: ManifestOption = DEFAULT_MANIFEST,
    db: DbOption = None,
) -> None:
    """Show what the database holds, so a plot can be scoped before it is drawn."""
    if db is None:
        db = _defaults(manifest).db
    try:
        runs = load_runs(db)
    except (FileNotFoundError, ValueError) as exc:
        _die(str(exc))

    summary = (
        runs.groupby(["query", "dataset"])
        .agg(
            runs=("run_id", "count"),
            budgets=("err", lambda s: ", ".join(f"{v:g}" for v in sorted(s.unique()))),
            best=("speedup", "max"),
        )
        .sort_values("best", ascending=False)
    )
    for (name, dataset), row in summary.iterrows():
        typer.echo(
            f"{name:<16} {row['runs']:>3} runs  best {row['best']:>7.1f}x  "
            f"err=[{row['budgets']}]  {dataset}"
        )


@app.command()
def kinds() -> None:
    """Show the plot kinds a manifest can ask for, and the keys each one takes."""
    for name, kind in sorted(KINDS.items()):
        typer.secho(name, bold=True)
        typer.echo(f"  {kind.summary}")
        for key, help_text in kind.option_help:
            typer.echo(f"    {key:<12} {help_text}")
        typer.echo("")
    typer.echo(
        "Every kind also takes: name, kind, enabled, title, db, out, out_dir,\n"
        "format, aggregate, queries, exclude, errs — see the README."
    )


def _plots(manifest: Path, **overrides: Optional[Path]) -> list[Plot]:
    try:
        return read_manifest(manifest, **overrides)  # type: ignore[arg-type]
    except ManifestError as exc:
        _die(str(exc))


def _defaults(manifest: Path) -> Defaults:
    try:
        return read_defaults(manifest)
    except ManifestError as exc:
        _die(str(exc))


def _short(path: Path) -> str:
    """A path relative to the cwd when that is shorter — these are all nearby."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _die(message: str) -> NoReturn:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
