"""Reading `manifest.toml` — the list of figures the paper wants.

The manifest is the single description of every plot in the evaluation: one
`[defaults]` table for what the figures share (which `bench.db`, where images
go, how repeated runs collapse) and one `[[plots]]` table per figure, which
names a kind and overrides whatever it needs to.

Keys are read by name and leftovers are an error, so a misspelled option fails
loudly instead of silently drawing the default chart.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from benchplot.db import Aggregate
from benchplot.plots import KINDS
from benchplot.spec import AGGREGATES, ManifestError, Plot, Selection, Table

DEFAULT_MANIFEST = Path("manifest.toml")

# Relative to the manifest's own directory; the repository root is where the
# benchmark scripts write.
DEFAULT_DB = "../../bench.db"
DEFAULT_OUT_DIR = "plots"
DEFAULT_FORMAT = "png"
FORMATS = ("png", "pdf", "svg", "eps")

# Keys a `[[plots]]` table may set that are not the kind's own. Listed so the
# "unknown key" message can tell a typo in a common key from a typo in a
# kind-specific one.
COMMON_KEYS = (
    "name",
    "kind",
    "enabled",
    "title",
    "db",
    "out",
    "out_dir",
    "format",
    "aggregate",
    "queries",
    "exclude",
    "errs",
    "scales",
)


@dataclass(frozen=True)
class Defaults:
    """`[defaults]`: what a `[[plots]]` table inherits unless it says otherwise."""

    db: Path
    out_dir: Path
    format: str
    aggregate: str
    queries: tuple[str, ...] | None
    exclude: tuple[str, ...]
    errs: tuple[float, ...] | None
    scales: tuple[float, ...] | None


def read_defaults(path: Path) -> Defaults:
    """Just `[defaults]`, for commands that want the shared `bench.db` and nothing else."""
    root, base = _load(path)
    defaults = _read_defaults(root.table("defaults"), base)
    root.tables("plots")  # read so the unused-key check below does not trip on it
    root.reject_unread()
    return defaults


def read_manifest(
    path: Path,
    *,
    db: Path | None = None,
    out_dir: Path | None = None,
) -> list[Plot]:
    """Every enabled plot in `path`, in manifest order.

    `db` and `out_dir` override the manifest for the whole run — the one-off
    "same figures, other database" case, which otherwise means editing a file
    that is checked in. Plots with `enabled = false` are left out here, so the
    caller never has to think about them.
    """
    root, base = _load(path)
    defaults = _read_defaults(root.table("defaults"), base)
    if db is not None:
        defaults = replace(defaults, db=db.resolve())
    if out_dir is not None:
        defaults = replace(defaults, out_dir=out_dir.resolve())

    tables = root.tables("plots")
    if not tables:
        raise ManifestError(f"{path}: no [[plots]] tables — nothing to draw")
    root.reject_unread()

    plots = [_read_plot(t, defaults, base) for t in tables]

    seen: dict[str, str] = {}
    for table, plot in zip(tables, plots):
        if plot is None:
            continue
        if plot.name in seen:
            raise ManifestError(
                f'{table.where}: the name "{plot.name}" is already used by '
                f"{seen[plot.name]}; names pick output files and --only, so they "
                "must be unique"
            )
        seen[plot.name] = table.where

    return [p for p in plots if p is not None]


def _load(path: Path) -> tuple[Table, Path]:
    """The parsed manifest and the directory its relative paths are against."""
    if not path.is_file():
        raise ManifestError(f"no manifest at {path}")
    try:
        raw = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ManifestError(f"{path}: {exc}") from exc
    base = path.parent.resolve()
    return Table(raw, str(path), base), base


def _read_defaults(table: Table, base: Path) -> Defaults:
    out = Defaults(
        db=cast(Path, table.path("db", base, (base / DEFAULT_DB).resolve())),
        out_dir=cast(Path, table.path("out_dir", base, (base / DEFAULT_OUT_DIR).resolve())),
        format=table.choice("format", FORMATS, DEFAULT_FORMAT),
        aggregate=table.choice("aggregate", AGGREGATES, "latest"),
        queries=table.strings("queries"),
        exclude=table.strings("exclude", ()) or (),
        errs=table.numbers("errs"),
        scales=table.numbers("scales"),
    )
    table.reject_unread()
    return out


def _read_plot(table: Table, defaults: Defaults, base: Path) -> Plot | None:
    """One `[[plots]]` table, or None when it is switched off."""
    name = table.required_string("name")
    kind = table.required_string("kind")
    table.relabel(f'{table.where} "{name}" ({kind})')
    if kind not in KINDS:
        table.fail("kind", f"unknown kind {kind!r}; known: {', '.join(sorted(KINDS))}")
    # Read, but acted on at the end: a switched-off plot is still validated, so
    # it is not broken by the time someone switches it back on.
    enabled = table.boolean("enabled", True)

    out_dir = cast(Path, table.path("out_dir", base, defaults.out_dir))
    fmt = table.choice("format", FORMATS, defaults.format)

    plot = Plot(
        name=name,
        kind=kind,
        out=table.path("out", base) or out_dir / f"{name}.{fmt}",
        title=table.string("title"),
        selection=Selection(
            db=cast(Path, table.path("db", base, defaults.db)),
            aggregate=cast(
                Aggregate, table.choice("aggregate", AGGREGATES, defaults.aggregate)
            ),
            queries=table.strings("queries", defaults.queries),
            exclude=frozenset(table.strings("exclude", defaults.exclude) or ()),
            errs=table.numbers("errs", defaults.errs),
            drop_errs=table.number_list_map("drop_errs"),
            keep_errs=table.number_list_map("keep_errs"),
            scales=table.numbers("scales", defaults.scales),
        ),
        # The kind takes its own keys from the same table, so whatever neither it
        # nor the common keys above claimed is a typo.
        options=KINDS[kind].parse(table),
    )
    _reject_unknown(table, kind)
    return plot if enabled else None


def _reject_unknown(table: Table, kind: str) -> None:
    extra = table.unread()
    if not extra:
        return
    known = sorted(set(COMMON_KEYS) | {opt for opt, _ in KINDS[kind].option_help})
    table.fail(
        None,
        f"unknown key{'s' if len(extra) > 1 else ''}: {', '.join(extra)}. "
        f"A {kind} plot takes: {', '.join(known)}",
    )
