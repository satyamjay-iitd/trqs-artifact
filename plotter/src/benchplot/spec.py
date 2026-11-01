"""What a plot *is*, independent of any particular kind of plot.

A figure in the paper is one `[[plots]]` table in the manifest. Everything every
kind of plot needs — where the data comes from, which runs to draw, where the
image goes — is a [`Plot`]; everything only one kind needs is its `options`,
parsed by that kind from the same table.

This module is imported by both the manifest reader and the plot kinds, and
imports neither, which is what keeps the registry from being a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, NoReturn, Protocol

import pandas as pd

from benchplot.db import Aggregate

AGGREGATES: tuple[str, ...] = ("latest", "median", "min", "best")


class ManifestError(Exception):
    """A manifest that cannot be turned into plots.

    The message carries its own location prefix (`plots[2] "speedup".err`), so
    the CLI prints it as-is rather than guessing where the problem was.
    """


class Table:
    """A TOML table read key by key.

    Reads are recorded, so once the common keys and the kind's own keys have
    both been taken from a plot table, whatever is left is a typo — which is the
    one manifest error that would otherwise pass silently and give you a chart
    that ignored the setting you wrote.
    """

    def __init__(self, raw: dict[str, Any], where: str, base: Path | None = None) -> None:
        self.where = where
        # The manifest's own directory, which every relative path in it is
        # against. Carried on the table so a plot kind reading a path of its own
        # resolves it the same way the common keys do.
        self.base = base or Path.cwd()
        self._raw = raw
        self._read: set[str] = set()

    def fail(self, key: str | None, message: str) -> NoReturn:
        at = f"{self.where}.{key}" if key else self.where
        raise ManifestError(f"{at}: {message}")

    def has(self, key: str) -> bool:
        return key in self._raw

    def _take(self, key: str, default: Any) -> Any:
        self._read.add(key)
        return self._raw.get(key, default)

    def string(self, key: str, default: str | None = None) -> str | None:
        value = self._take(key, default)
        if value is None or isinstance(value, str):
            return value
        self.fail(key, f"expected a string, got {_kind_of(value)}")

    def required_string(self, key: str) -> str:
        value = self.string(key)
        if value is None:
            self.fail(key, "is required")
        return value

    def choice(self, key: str, allowed: tuple[str, ...], default: str) -> str:
        value = self.string(key, default)
        assert value is not None
        if value not in allowed:
            self.fail(key, f"expected one of {', '.join(allowed)}, got {value!r}")
        return value

    def boolean(self, key: str, default: bool) -> bool:
        value = self._take(key, default)
        if isinstance(value, bool):
            return value
        self.fail(key, f"expected true or false, got {_kind_of(value)}")

    def number(self, key: str, default: float | None = None) -> float | None:
        value = self._take(key, default)
        if value is None:
            return None
        # `isinstance(True, int)` is True, so booleans are rejected explicitly.
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        self.fail(key, f"expected a number, got {_kind_of(value)}")

    def strings(self, key: str, default: tuple[str, ...] | None = None) -> tuple[str, ...] | None:
        value = self._take(key, None)
        if value is None:
            return default
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            self.fail(key, f"expected a list of strings, got {_kind_of(value)}")
        return tuple(value)

    def numbers(self, key: str, default: tuple[float, ...] | None = None) -> tuple[float, ...] | None:
        value = self._take(key, None)
        if value is None:
            return default
        ok = isinstance(value, list) and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in value
        )
        if not ok:
            self.fail(key, f"expected a list of numbers, got {_kind_of(value)}")
        return tuple(float(v) for v in value)

    def table(self, key: str) -> "Table":
        """A sub-table, empty when absent — `[defaults]` is optional."""
        value = self._take(key, {})
        if not isinstance(value, dict):
            self.fail(key, f"expected a table, got {_kind_of(value)}")
        return Table(value, f"{self.where} [{key}]", self.base)

    def string_map(self, key: str) -> dict[str, str]:
        """A table of string -> string, empty when absent.

        Its keys are data rather than settings -- query ids, here -- so they are
        not checked against a known set the way a plot table's keys are.
        """
        value = self._take(key, {})
        if not isinstance(value, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in value.items()
        ):
            self.fail(key, "expected a table of string = string")
        return dict(value)

    def number_list_map(self, key: str) -> dict[str, tuple[float, ...]]:
        """A table of string -> list of numbers, empty when absent.

        Like `string_map`, its keys are data (query ids) rather than settings,
        so they are not checked against a known set.
        """
        value = self._take(key, {})
        ok = isinstance(value, dict) and all(
            isinstance(k, str)
            and isinstance(v, list)
            and all(isinstance(n, (int, float)) and not isinstance(n, bool) for n in v)
            for k, v in value.items()
        )
        if not ok:
            self.fail(key, "expected a table of string = [numbers]")
        return {k: tuple(float(n) for n in v) for k, v in value.items()}

    def tables(self, key: str) -> list["Table"]:
        """An array of tables, written `[[key]]`."""
        value = self._take(key, [])
        if not isinstance(value, list) or not all(isinstance(v, dict) for v in value):
            self.fail(key, f"expected an array of tables, written [[{key}]]")
        return [Table(v, f"{self.where} {key}[{i}]", self.base) for i, v in enumerate(value)]

    def relabel(self, where: str) -> "Table":
        """The same table under a better name, once a key has told us what it is."""
        self.where = where
        return self

    def path(self, key: str, base: Path | None = None, default: Path | None = None) -> Path | None:
        """A path, resolved against `base` — by default the manifest's directory.

        Relative to the manifest rather than to the shell's cwd, so `uv run
        benchplot render` writes the same files from anywhere in the tree.
        """
        base = base or self.base
        value = self.string(key)
        if value is None:
            return default
        return (base / value).resolve() if not Path(value).is_absolute() else Path(value)

    def unread(self) -> list[str]:
        return sorted(set(self._raw) - self._read)

    def reject_unread(self) -> None:
        extra = self.unread()
        if extra:
            self.fail(None, f"unknown key{'s' if len(extra) > 1 else ''}: {', '.join(extra)}")


def _kind_of(value: Any) -> str:
    return {
        bool: "true/false",
        int: "a number",
        float: "a number",
        str: "a string",
        list: "a list",
        dict: "a table",
    }.get(type(value), type(value).__name__)


@dataclass(frozen=True)
class Selection:
    """Which runs a plot draws — the keys every kind understands."""

    db: Path
    aggregate: Aggregate
    queries: tuple[str, ...] | None
    exclude: frozenset[str]
    errs: tuple[float, ...] | None
    # Budgets to drop for named queries only, e.g. a lossless point that is
    # measurable but not an operating point anyone would choose. `errs` says
    # which budgets the figure draws at all; this says which of them a
    # particular query does not draw.
    drop_errs: dict[str, tuple[float, ...]]
    # The inverse, for named queries only: exactly these budgets and no others.
    # A drop list has to be revisited every time the query is measured at a new
    # budget, which is how new runs quietly walked into figures they were never
    # meant to be in; a keep list pins the figure instead.
    keep_errs: dict[str, tuple[float, ...]]
    # Tiling factors (`--db-repeat`) to draw. A small dataset measures the
    # engine's fixed per-query cost rather than its algebra, so a figure about
    # speed often wants the tiled runs and nothing else.
    scales: tuple[float, ...] | None


@dataclass(frozen=True)
class Plot:
    """One figure: a kind, the runs behind it, the file it writes."""

    name: str
    kind: str
    out: Path
    title: str | None
    selection: Selection
    options: Any

    @property
    def order(self) -> list[str] | None:
        """The query order the manifest asked for, if it named queries at all."""
        return list(self.selection.queries) if self.selection.queries else None


class Render(Protocol):
    def __call__(self, points: pd.DataFrame, plot: Plot) -> Path: ...


@dataclass(frozen=True)
class PlotKind:
    """A registered kind of plot: how to read its options and how to draw it.

    `parse` is handed the whole `[[plots]]` table after the common keys have
    been taken out of it, and returns whatever object `render` wants back as
    `plot.options`. `option_help` is what `benchplot kinds` prints.
    """

    name: str
    summary: str
    parse: Callable[[Table], Any]
    render: Render
    option_help: tuple[tuple[str, str], ...] = ()
    # False for a kind that brings its own data — measured by something other
    # than `trql-cli bench`, and so not in `bench.db` at all. The renderer then
    # loads nothing, applies no selection, and hands `render` an empty frame;
    # the kind reads whatever file its own options name.
    needs_runs: bool = True
    # What the CLI prints after the output path. Defaults to the runs the plot
    # drew, which means nothing for a kind that reads no runs.
    describe: Callable[[Plot], str] | None = None
