"""Speedup and accuracy against SQL: one group per query, one bar per error budget."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from benchplot.spec import Plot, PlotKind, Table

# Groups are ragged — i24 artifacts stop at 25 centi-feet, nuScenes runs to 50 cm —
# so a budget is identified by colour rather than by its slot in the group.
_PALETTE = plt.get_cmap("tab20")

_GROUP_WIDTH = 0.8


@dataclass(frozen=True)
class _Bar:
    """One measured `(query, err)` point and where it sits on the shared x-axis."""

    x: float
    width: float
    row: pd.Series
    #: Position within its query's group, left to right. When each query is
    #: drawn at its own budgets this, not `err`, is what the colours mean.
    rank: int


def _layout(
    df: pd.DataFrame,
    queries: list[str],
    budgets: dict[str, list[float]] | None = None,
) -> list[_Bar]:
    """Bar positions, computed once so every panel lands on the same x.

    With `budgets` the group is laid out by *name* -- one bar per entry of
    `rank_labels`, in that order -- rather than by the budgets that happen to
    have been measured. Names that resolve to a budget already drawn are not
    drawn again: the second bar would be a copy of the first in every panel, and
    the group says more by being short. It keeps the colour of the first name
    that reached it, and the budget printed on it is what disambiguates.
    """
    bars: list[_Bar] = []
    # One width for the whole figure, set by the busiest group. A query drawn at
    # fewer budgets than its neighbours then gets a narrower group centred on its
    # tick, rather than a few very fat bars that read as a larger measurement.
    widest = (
        max(len(set(budgets.get(q, ()))) for q in queries)
        if budgets
        else max(len(df[df["query"] == q]) for q in queries)
    )
    width = _GROUP_WIDTH / widest
    for gi, query in enumerate(queries):
        rows: list[tuple[int, pd.Series]]
        if budgets:
            in_query = df[df["query"] == query]
            wanted = budgets.get(query)
            if not wanted:
                raise ValueError(f"no budgets given for {query}")
            rows = []
            missing = []
            drawn: set[float] = set()
            for rank, err in enumerate(wanted):
                hit = in_query[in_query["err"] == err]
                if hit.empty:
                    missing.append(err)
                elif err not in drawn:
                    drawn.add(err)
                    rows.append((rank, hit.iloc[0]))
            if missing:
                raise ValueError(
                    f"{query}: no run at max_error_px "
                    + ", ".join(f"{m:g}" for m in missing)
                )
        else:
            ordered = df[df["query"] == query].sort_values("err")
            rows = list(enumerate(row for _, row in ordered.iterrows()))
        left = gi - width * len(rows) / 2
        for bi, (rank, row) in enumerate(rows):
            bars.append(_Bar(left + width * (bi + 0.5), width, row, rank))
    return bars


def _is_exact(row: pd.Series) -> bool:
    """Whether this run reproduced the reference answer, tuple for tuple.

    Read off the two numbers the accuracy panel plots, so a bar is outlined
    exactly when both of its markers sit on 1.0 and there is nothing to read
    between the panels.
    """
    return bool(row.get("precision") == 1.0 and row.get("recall") == 1.0)


def _bar_label(speedup: float) -> str:
    """The number standing on a bar: what it is worth against the baseline."""
    return f"{speedup:.0f}×" if speedup >= 10 else f"{speedup:.1f}×"


def _budget_label(err: float, name_err: bool = False) -> str:
    """The budget that bar was measured at, as it rides above the speedup.

    Colour names a percentile of the dataset's own motion, which is not a
    quantity a reader can check against anything. The budget it resolves to is,
    so when the figure is drawn that way it rides along on the bar -- in
    parentheses and in grey, since the speedup is the number being reported and
    this one qualifies it.

    Only the first bar of a group spells out `\u03b5=`; the rest carry the bare
    number. Repeating the symbol on every bar is three characters of rotated
    text apiece, and rotated text is height, which is what runs the tall bars
    into the top of the panel.
    """
    return f"(\u03b5={err:g})" if name_err else f"({err:g})"


def _headroom(
    ax, values: list[float], log: bool, frac: float = 0.14, label_pt: float = 0.0
) -> None:
    """Reserve room above the tallest bar for the label standing on top of it.

    The labels are offset in points, so autoscaling — which only sees the bars —
    leaves the tallest one running through the top spine. `frac` is the share of
    the panel height kept clear. A fixed share only works while the panel is a
    fixed size: shrink the figure for one column and the same label eats a much
    larger share, so `label_pt` — the label's own extent in points — raises
    `frac` to whatever this panel actually needs.
    """
    ax.autoscale_view()
    if label_pt:
        panel_pt = ax.get_position().height * ax.get_figure().get_figheight() * 72
        if panel_pt > 0:
            frac = max(frac, (label_pt + 5) / panel_pt)
    lo, _ = ax.get_ylim()
    hi = max(values)
    if not log:
        ax.set_ylim(lo, hi + (hi - lo) * frac)
        return
    if lo <= 0:  # a log axis should never give this, but bars start at zero
        lo = min((v for v in values if v > 0), default=1.0) / 10
    span = math.log10(hi) - math.log10(lo)
    ax.set_ylim(lo, 10 ** (math.log10(hi) + span * frac))


def _stack_labels(ax, pairs: list, base_pt: float = 3.0, gap_pt: float = 1.5) -> None:
    """Sit each pair's second label directly on top of its first.

    The two are one label split in two so that only the budget is grey, which
    means the upper one has to be offset by however tall the lower one came out
    -- and rotated upright, a label's height is its length in characters. That
    is a font metric, so it is measured rather than estimated.
    """
    if not pairs:
        return
    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_points = 72.0 / fig.dpi
    for lower, upper in pairs:
        height = lower.get_window_extent(renderer).height * to_points
        upper.xyann = (0, base_pt + height + gap_pt)


def _fit_labels(ax, labels: list, log: bool, pad_px: float = 2.0) -> None:
    """Grow the y-axis until no annotation sticks out of the panel.

    Raising the limit moves the bars down, which moves their labels with them,
    so this converges in a few passes rather than solving for the limit.
    """
    if not labels:
        return
    fig = ax.get_figure()
    for _ in range(8):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        panel = ax.get_window_extent(renderer)
        over = max(
            (lab.get_window_extent(renderer).y1 + pad_px - panel.y1 for lab in labels),
            default=0.0,
        )
        if over <= 0 or panel.height <= 0:
            return
        frac = min(over / panel.height, 0.5) * 1.15
        lo, hi = ax.get_ylim()
        if log:
            span = math.log10(hi) - math.log10(lo)
            ax.set_ylim(lo, 10 ** (math.log10(hi) + span * frac))
        else:
            ax.set_ylim(lo, hi + (hi - lo) * frac)


def _clear_legend(ax, legend, log: bool) -> None:
    """Grow the y-axis until the legend, pinned to the panel's top edge, sits
    clear of every bar label under it.

    Raising the limit lowers the bars and their labels while the legend stays
    where it is, so this converges in a few passes, as `_fit_labels` does.
    """
    fig = ax.get_figure()
    for _ in range(200):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        box = legend.get_window_extent(renderer)
        if not any(t.get_window_extent(renderer).overlaps(box) for t in ax.texts):
            return
        lo, hi = ax.get_ylim()
        if log:
            ax.set_ylim(lo, 10 ** (math.log10(hi) + 0.1 * (math.log10(hi) - math.log10(lo)) / 5))
        else:
            ax.set_ylim(lo, hi + 0.02 * (hi - lo))


def _ylabel(ax, text: str, type_scale: float, compact: bool = False) -> None:
    """Set the axis label, smaller and tighter when the panel is small.

    In one column the label and its tick labels are the whole left margin, and
    that margin is width the plot itself could use, so both come in and the
    label sits closer to the ticks. `compact` is the other case: a wide figure
    drawn shallow, where a label set at full size is taller than the panel it
    names and runs into its neighbour.
    """
    if type_scale == 1.0 and not compact:
        ax.set_ylabel(text, fontsize=9)
    else:
        ax.set_ylabel(text, fontsize=7.4 * type_scale, labelpad=2, linespacing=0.95)


def _draw_speedup(
    ax,
    bars: list[_Bar],
    colour_of,
    log: bool,
    type_scale: float = 1.0,
    err_labels: bool = False,
    compact: bool = False,
) -> None:
    labels: list = []
    stacked: list = []
    for bar in bars:
        ax.bar(
            bar.x,
            bar.row["speedup"],
            width=bar.width * 0.92,
            color=colour_of(bar),
            # Circles mark a bar whose answer is the reference answer. A hatch
            # is tiled in canvas space and clipped to the patch, so the pattern
            # has to be fine enough to sit inside one bar -- a coarse one runs
            # on across the gaps and reads as a mark over the whole group.
            hatch="////" if _is_exact(bar.row) else None,
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
        # A label carrying its budget as well as its speedup is half again as
        # long, and rotated upright a label's length is height. In one column
        # there is no room for both at full size, so the type comes down.
        common = dict(
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.5 * type_scale * (0.74 if err_labels else 1.0),
            rotation=90,
            zorder=4,
        )
        speed = ax.annotate(
            _bar_label(bar.row["speedup"]),
            (bar.x, bar.row["speedup"]),
            xytext=(0, 3),
            **common,
        )
        labels.append(speed)
        if err_labels:
            budget = ax.annotate(
                _budget_label(bar.row["err"], name_err=bar.rank == 0),
                (bar.x, bar.row["speedup"]),
                # Corrected below, once the text above it has been measured.
                xytext=(0, 3 + 6.5 * type_scale * len(_bar_label(bar.row["speedup"]))),
                color="0.45",
                **common,
            )
            labels.append(budget)
            stacked.append((speed, budget))

    # Parity: below this line the engine is slower than single-threaded DuckDB.
    ax.axhline(1.0, color="crimson", linewidth=1.0, linestyle="--", zorder=2)

    if log:
        # Speedups here span 3x to 852x; on a linear axis every query but the
        # fastest collapses onto the baseline.
        ax.set_yscale("log")
    _ylabel(ax, "Speedup over DuckDB" if type_scale == 1.0 and not compact else "Speedup", type_scale, compact)
    # Rotated upright, a bar label is as tall as it is long; 0.65 em per
    # character is a little over the digit advance in the default face, which
    # leaves the widest label ("1,005x") a hair of slack rather than a clip.
    label_fs = 6.5 * type_scale * (0.74 if err_labels else 1.0)
    longest = max(
        (
            len(_bar_label(b.row["speedup"]))
            + (len(_budget_label(b.row["err"], b.rank == 0)) if err_labels else 0)
            for b in bars
        ),
        default=0,
    )
    _headroom(
        ax,
        [b.row["speedup"] for b in bars],
        log,
        label_pt=longest * label_fs * 0.65 + 3,
    )
    _stack_labels(ax, stacked)
    # The estimate above is a character count times a nominal advance, which is
    # close but not exact -- and a label that overshoots by two points still
    # prints over the panel above. Measure what was actually drawn and grow the
    # axis until every label is inside it.
    _fit_labels(ax, labels, log)


def _draw_accuracy(
    ax, bars: list[_Bar], colour_of, type_scale: float = 1.0, compact: bool = False
) -> None:
    """Precision and recall at each budget, under the bar they belong to.

    Both metrics sit at 1.0 for most points, so they are drawn as markers with a
    small horizontal offset rather than stacked exactly on top of each other, and
    the y-range is clipped to the data instead of the full [0, 1].
    """
    for bar in bars:
        colour = colour_of(bar)
        off = bar.width / 5
        ax.plot(
            bar.x - off,
            bar.row["precision"],
            marker="o",
            markersize=3.4,
            color=colour,
            markeredgecolor="black",
            markeredgewidth=0.3,
            linestyle="none",
            zorder=3,
        )
        ax.plot(
            bar.x + off,
            bar.row["recall"],
            marker="^",
            markersize=3.8,
            markerfacecolor="none",
            markeredgecolor=colour,
            markeredgewidth=1.0,
            linestyle="none",
            zorder=3,
        )

    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle=":", zorder=2)

    lo = min(
        min(b.row["precision"] for b in bars),
        min(b.row["recall"] for b in bars),
    )
    # A hair of headroom above 1.0 so markers at exactly 1.0 are not clipped by
    # the spine, and a floor that keeps the spread visible instead of showing a
    # flat line at the top of a [0, 1] axis.
    span = max(1.0 - lo, 0.005)
    ax.set_ylim(lo - span * 0.15, 1.0 + span * 0.15)
    # Rotated upright, this label is longer than its own panel is tall once the
    # figure is narrow, and it was running up into the panel above. Two lines
    # spend a little width — which a narrow figure has — to buy that height back.
    _ylabel(
        ax,
        "Precision / recall" if type_scale == 1.0 and not compact else "Precision /\nrecall",
        type_scale,
        compact,
    )


def _draw_memory(
    ax,
    bars: list[_Bar],
    colour_of,
    queries: list[str],
    log: bool,
    type_scale: float = 1.0,
    compact: bool = False,
) -> None:
    """Engine memory per budget, against DuckDB's flat per-query figure.

    The two are *different quantities* — the engine's is whole-process RSS and so
    includes its eagerly-loaded `.trql`, DuckDB's is its own `duckdb_memory()`
    accounting of a database it demand-pages. They share an axis because the
    question is which system needs more to answer the query, but the difference
    between them is not a like-for-like subtraction, and the legend says so.
    """
    for bar in bars:
        mem = bar.row.get("trql_mem_mb")
        if pd.notna(mem):
            ax.bar(
                bar.x,
                mem,
                width=bar.width * 0.92,
                color=colour_of(bar),
                hatch="////" if _is_exact(bar.row) else None,
                edgecolor="black",
                linewidth=0.4,
                zorder=3,
            )

    # DuckDB never sees the error budget, so its figure is one value per query
    # rather than one per bar — a rule across the group, not a bar beside them.
    for gi, query in enumerate(queries):
        vals = [b.row.get("sql_mem_mb") for b in bars if b.row["query"] == query]
        vals = [v for v in vals if pd.notna(v)]
        if not vals:
            continue
        ax.hlines(
            sum(vals) / len(vals),
            gi - _GROUP_WIDTH / 2,
            gi + _GROUP_WIDTH / 2,
            color="crimson",
            linewidth=1.0,
            zorder=4,
            linestyle="--"
        )

    if log:
        # 29 MB to 11.7 GB across the suite — 400x, so linear would show only i24.
        ax.set_yscale("log")
    _ylabel(ax, "Peak memory (MB)" if type_scale == 1.0 and not compact else "Peak mem. (MB)", type_scale, compact)


def plot_speedup(
    df: pd.DataFrame,
    out_path: Path,
    *,
    order: list[str] | None = None,
    log: bool = True,
    accuracy: bool = True,
    memory: bool = True,
    labels: dict[str, str] | None = None,
    width: float | None = None,
    height: float | None = None,
    rank_labels: tuple[str, ...] | None = None,
    budgets: dict[str, list[float]] | None = None,
    err_labels: bool = False,
    legend_inside: bool = False,
    title: str | None = None,
) -> Path:
    """Writes the chart to `out_path` and returns it.

    `df` is expected to hold one row per `(query, err)` — see
    [`benchplot.db.collapse_repeats`]. What a budget cost in accuracy is left to
    the accuracy panel, which says by how much: nothing in the bars themselves
    marks an inexact answer, so read the two panels together.
    """
    if df.empty:
        raise ValueError("nothing to plot")

    if order:
        missing = [q for q in order if q not in set(df["query"])]
        if missing:
            raise ValueError(f"query not in the database: {', '.join(missing)}")
        queries = order
    else:
        # Widest span first: the eye lands on the queries the error budget moves most.
        span = df.groupby("query")["speedup"].max().sort_values(ascending=False)
        queries = list(span.index)

    errs = sorted(df["err"].unique())
    # `rank_labels` means every query was swept at its own budgets -- a
    # percentile of its own motion, say -- so the budgets do not line up across
    # groups and colouring by their value would give a legend of two dozen
    # entries that says nothing. Colour by position within the group instead,
    # and let the labels name what the positions mean.
    if rank_labels:
        # Indexed by position: `Set1` is qualitative, and a fraction of 1.0 lands
        # on its last entry, which is grey.
        palette = {i: _PALETTE(i) for i in range(len(rank_labels))}
        colour_of = lambda bar: palette[bar.rank]
    else:
        palette = {e: _PALETTE(i) for i, e in enumerate(errs)}
        colour_of = lambda bar: palette[bar.row["err"]]
    # `budgets` lays the group out by name, so a bar's rank is the position of
    # its label rather than its position in the group, and the two differ
    # exactly where a query's budgets coincide.
    bars = _layout(df, queries, budgets)

    # A panel whose column is entirely null is dropped rather than drawn empty:
    # runs recorded before memory was measured carry NULL, not 0.
    has_memory = "trql_mem_mb" in df.columns and df["trql_mem_mb"].notna().any()
    memory = memory and has_memory

    width = width or max(8.0, 1.9 * len(queries))
    # A narrow figure is headed for a single column, where TeX scales it down
    # much less, so the type has to come down instead of the figure.
    narrow = width <= 4.5
    # Every size in the figure rides this: drawn narrow, the file is scaled down
    # far less by TeX, so the type has to be smaller on the canvas to land at
    # the same size on the page.
    type_scale = 0.95 if narrow else 1.0
    # Speedup, then memory, then accuracy: the two panels that answer "what did
    # it cost" sit together, and the accuracy markers — which need the least
    # height — take the bottom, next to the tick labels.
    # The accuracy panel gets a little more of a narrow figure: its y-label is
    # as long as the others and has a much shorter panel to sit in.
    ratios = (
        [2.2]
        + ([1.3] if memory else [])
        + ([1.0 if narrow else 0.8] if accuracy else [])
    )
    # Kept shallow on purpose: the figure spans both columns, so every inch of
    # height is a large fraction of the page. The two lower panels only have to
    # separate values that sit orders of magnitude apart (memory) or within a
    # few percent of 1.0 (accuracy), and neither needs the room the bars do.
    heights = height or {1: 3.6, 2: 5.0, 3: 6.2}[len(ratios)]
    if height is None and narrow:
        heights *= width / 6.0
    fig, axes = plt.subplots(
        len(ratios),
        1,
        figsize=(width, heights),
        sharex=True,
        height_ratios=ratios,
        gridspec_kw={"hspace": 0.08},
        squeeze=False,
    )
    panels = [ax for (ax,) in axes]

    ax_speed = panels.pop(0)
    ax_mem = panels.pop(0) if memory else None
    ax_acc = panels.pop(0) if accuracy else None

    # A wide figure drawn shallow: full-size axis labels are taller than their
    # panels and collide, so every panel takes the short label and the smaller
    # type, exactly as a one-column figure does.
    compact = heights < 5.0 and not narrow
    _draw_speedup(ax_speed, bars, colour_of, log, type_scale, err_labels, compact)
    if ax_mem is not None:
        _draw_memory(ax_mem, bars, colour_of, queries, log, type_scale, compact)
    if ax_acc is not None:
        _draw_accuracy(ax_acc, bars, colour_of, type_scale, compact)

    bottom = ax_acc or ax_mem or ax_speed
    bottom.set_xticks(range(len(queries)))
    # When a group is a single bar the colour legend is the only thing saying
    # which budget it was measured at, and a legend lookup per bar is a poor
    # trade for two characters under the name.
    one_each = all(len(df[df["query"] == q]) == 1 for q in queries)
    # A figure drawn at a single budget throughout says so once, in its title or
    # caption: neither the tick labels nor the legend should repeat it per bar.
    uniform_err = df["err"].nunique() == 1
    one_each = one_each and not uniform_err
    names = []
    for q in queries:
        name = (labels or {}).get(q, q)
        if one_each:
            name += "\n\u03b5 = {:g}".format(df.loc[df["query"] == q, "err"].iloc[0])
        names.append(name)
    # Six or seven groups in one column leaves ~0.5 in each, less than a query id
    # needs lying flat; three groups have room to spare. The steeper tilt buys
    # back the cosine: a label's horizontal footprint is 0.71 of its length at
    # 45 degrees against 0.87 at 30.
    # Names of a few characters ("Q1") fit flat however many groups there are.
    flat = max(len(n.split("\n")[0]) for n in names) <= 4
    tight = narrow and width / len(queries) < 0.75 and not flat
    tilt = 45 if tight and width / len(queries) < 0.55 else 30
    bottom.set_xticklabels(
        names,
        fontsize=(6.0 if tilt == 45 else 6.5) * type_scale,
        rotation=tilt if tight else 0,
        ha="right" if tight else "center",
        rotation_mode="anchor" if tight else None,
    )
    if narrow:
        for ax in filter(None, (ax_speed, ax_mem, ax_acc)):
            ax.tick_params(labelsize=6 * type_scale, pad=1.5)
    bottom.set_xlim(-0.6, len(queries) - 0.4)
    for ax in filter(None, (ax_speed, ax_mem, ax_acc)):
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
    if title:
        ax_speed.set_title(title)

    # With one bar per group the budget is already under the query name, so the
    # colour swatches would be a second way of saying the same thing.
    handles: list = (
        []
        if one_each or uniform_err
        else [
            Patch(facecolor=palette[i], edgecolor="black", linewidth=0.4, label=lab)
            for i, lab in enumerate(rank_labels)
        ]
        if rank_labels
        else [
            Patch(facecolor=palette[e], edgecolor="black", linewidth=0.4, label=f"{e:g}")
            for e in errs
        ]
    )
    if ax_acc is not None:
        handles += [
            Line2D(
                [],
                [],
                marker="o",
                markersize=3.4,
                color="dimgrey",
                linestyle="none",
                label="precision",
            ),
            Line2D(
                [],
                [],
                marker="^",
                markersize=3.8,
                markerfacecolor="none",
                markeredgecolor="dimgrey",
                linestyle="none",
                label="recall",
            ),
        ]
    if ax_mem is not None:
        # Only DuckDB's rule needs naming: the bars are the engine's, as they are
        # in the panel above, and a swatch for them repeats what the colours and
        # the caption already say.
        handles += [
            Line2D([], [], color="crimson", linewidth=1.6, label="DuckDB", linestyle="--"),
        ]
    legend = (ax_speed if legend_inside else bottom).legend(
        handles=handles,
        title=None if one_each or uniform_err or rank_labels else "max_error_px (stored units)",
        fontsize=(6 if legend_inside else 7) * type_scale,
        title_fontsize=7 * type_scale,
        # Two rows, whatever the suite holds: a column per pair of entries, so
        # adding a budget widens the legend rather than growing it downwards
        # into the caption.
        # One row when the budget swatches are gone and only the key entries
        # remain; two columns of pairs otherwise.
        ncol=(2 if legend_inside else len(handles))
        if (legend_inside or one_each or uniform_err)
        else math.ceil(len(handles) / 2),
        loc="upper right" if legend_inside else "upper center",
        # The tick labels carry a second line when one_each, so the legend has
        # to start below it rather than beside it.
        bbox_to_anchor=None if legend_inside else (
            0.5,
            # Keyed on the tilt, not on how the bars are grouped: rotated
            # labels are what the legend has to clear.
            -0.62
            if tight
            else -0.34
            if one_each
            else -0.30
            if narrow
            else -0.22,
        ),
        frameon=legend_inside,
        framealpha=0.9,
        edgecolor="none",
        # Four short rows over the bars: spacing is what makes it tall.
        **(
            dict(labelspacing=0.15, borderpad=0.25, handlelength=1.3, columnspacing=0.8)
            if legend_inside
            else {}
        ),
    )
    if legend_inside:
        _clear_legend(ax_speed, legend, log)

    # No `tight_layout`: it is incompatible with the explicit `hspace` that keeps
    # the two panels tight against each other. `bbox_inches="tight"` below trims
    # the margins instead, and handles the legend and parity label hanging outside
    # the axes.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out_path


@dataclass(frozen=True)
class SpeedupOptions:
    """The `[[plots]]` keys only this kind understands."""

    log: bool = True
    accuracy: bool = True
    memory: bool = True
    # Query id in the database -> label on the axis. For when a query had to be
    # recorded under a distinct id -- re-measured on a second encoding of the
    # same data, say -- but the figure should still read plainly.
    labels: dict[str, str] | None = None
    # Figure width in inches. Default scales with the number of groups, for a
    # figure that spans both columns; set it to ~3.3 for one column, which also
    # tightens the type so the labels stay legible once TeX scales the file.
    width: float | None = None
    # Figure height in inches. The default suits the default width; set both
    # together when the figure is published at a size TeX does not scale.
    height: float | None = None
    # Names for the bars of each group, left to right, when every query was
    # swept at its own budgets. Colours then mean position, not `err`.
    rank_labels: tuple[str, ...] | None = None
    #: Query id -> its budget for each of `rank_labels`, in that order. The
    #: group is then drawn one bar per label, including where two labels resolve
    #: to the same budget and the bar is drawn twice.
    budgets: dict[str, list[float]] | None = None
    #: Print the budget each bar was measured at on top of the bar, beside its
    #: speedup. For figures whose colours name something indirect.
    err_labels: bool = False
    #: Put the legend inside the speedup panel, over the empty space above the
    #: short bars, instead of under the bottom axis.
    legend_inside: bool = False


def _parse(table: Table) -> SpeedupOptions:
    labels = table.string_map("labels") or None
    return SpeedupOptions(
        log=table.boolean("log", True),
        accuracy=table.boolean("accuracy", True),
        memory=table.boolean("memory", True),
        labels=labels,
        width=table.number("width"),
        height=table.number("height"),
        rank_labels=table.strings("rank_labels"),
        budgets=table.number_list_map("budgets") or None,
        err_labels=table.boolean("err_labels", False),
        legend_inside=table.boolean("legend_inside", False),
    )


def _render(points: pd.DataFrame, plot: Plot) -> Path:
    options: SpeedupOptions = plot.options
    return plot_speedup(
        points,
        plot.out,
        order=plot.order,
        log=options.log,
        accuracy=options.accuracy,
        memory=options.memory,
        labels=options.labels,
        width=options.width,
        height=options.height,
        rank_labels=options.rank_labels,
        budgets=options.budgets,
        err_labels=options.err_labels,
        legend_inside=options.legend_inside,
        title=plot.title,
    )


KIND = PlotKind(
    name="speedup",
    summary="Bar chart of speedup over SQL: one group per query, one bar per error budget.",
    parse=_parse,
    render=_render,
    option_help=(
        ("log", "Log y-axis (default true; false collapses every query but the fastest)."),
        ("accuracy", "Add a precision/recall panel sharing the x-axis (default true)."),
        (
            "memory",
            "Add a peak-memory panel sharing the x-axis (default true; dropped "
            "automatically when the database records no memory).",
        ),
        (
            "err_labels",
            "Print each bar's max_error_px beside its speedup (default false).",
        ),
        (
            "legend_inside",
            "Put the legend inside the speedup panel rather than under the "
            "bottom axis (default false).",
        ),
    ),
)
