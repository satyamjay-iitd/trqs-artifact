"""Linear vs quadratic vs cubic trajectory polynomials.

The degree is a compile-time choice (`--features linear|quadratic|cubic`), and it
moves three things at once: how many segments a trajectory needs to stay inside
the error budget, how much arithmetic a root-find costs, and how wide a segment
is in memory. This draws all three against the budget -- time, segment
count, and what those segments weigh (the count times `_SEGMENT_BYTES`, the
width of one segment at that degree) -- which is what makes the case for degree
2 as the default rather than an assertion. The third panel is the encoding's
own footprint rather than peak RSS, which also carries the process around it
and hides the thing being compared. Accuracy is not among them: the three degrees'
precision against the reference answer differs by at most 1.5 points at any
budget drawn (0.956 to 0.971 at the widest), so a panel of it separated
nothing.

The three databases come from `data/ablation/run.sh`, which builds one binary and
one set of artifacts per degree — the `.trql` filename does not record the
degree, so each needs its own tree. `data/ablation-instrumented/` holds the same
matrix re-run from `--features instrument` builds, which is where the time
panel's stage split comes from.

The split is drawn as a share of the *clean* total, not of the instrumented one:
the timing scopes cost a little on every predicate, so an instrumented run is
1.1-1.8x slower here and only its proportions are meaningful. That overhead
lands on whichever scope is running, and the most-entered scope is root-finding,
so the polynomial share is an upper bound.

    uv run python scripts/degree_ablation.py
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import pandas as pd

_DEGREES = ("linear", "quadratic", "cubic")
# One colour per degree, for the two panels whose bars are a degree rather than
# a stage. The marker and dash that used to ride along went with the lines.
_STYLE = {"linear": "#c1440e", "quadratic": "#1b6ca8", "cubic": "#2e7d32"}
_LABEL = {"linear": "$d{=}1$", "quadratic": "$d{=}2$", "cubic": "$d{=}3$"}

# The stage columns, in the order they stack, with the colour each carries.
# Ordered by when the work happens: pick the candidate tuples, solve the
# polynomials, lift the solved frames into interval sets, compose them.
_STAGES = (
    ("sel", "selection", "#4c72b0"),
    ("poly", "root-finding", "#dd8452"),
    ("temp", "temporal", "#55a868"),
)

# What the engine reports and what the figure draws. The interval lift (solved
# frames into interval sets), the interval-set algebra itself, and the residual
# (planning, tuple materialisation) are each a few per cent of a run and move
# together, so they are one bar: everything after the polynomials are solved.
_RAW_STAGES = ("sel", "poly", "lift", "temp", "other")

# `size_of::<PolySegment<4>>()` per degree, from the layout of
# `trql-poly/src/{linear,quadratic,cubic}.rs`: two u32 frame bounds and four
# `ScalarSegmentData2`, each an enum over `Poly([f64; d+1])` and
# `Direct { values: [u32; 4] }` — so 24, 32 and 40 bytes per coordinate, tag
# included, and 8 + 4 x that per segment. This is what a segment costs wherever
# it is held; peak RSS also carries the process around it, which is the same for
# all three builds and swamps the difference at tight budgets.
_SEGMENT_BYTES = {"linear": 104, "quadratic": 136, "cubic": 168}


def _tint(colour: str, amount: float = 0.58):
    """`colour` blended toward white, for the verbatim part of a bar."""
    import matplotlib.colors as mcolors

    r, g, b = mcolors.to_rgb(colour)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def load(data_dir: Path) -> pd.DataFrame:
    frames = []
    for deg in _DEGREES:
        db = data_dir / f"{deg}.db"
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
            df = pd.read_sql_query(
                """SELECT q.name AS query, r.trql_error_px AS err, r.trql_time AS ms,
                          r.trql_segments AS segments,
                          r.trql_poly_segments AS poly_segments,
                          r.trql_direct_segments AS direct_segments,
                          r.trql_peak_mem_mb AS mem_mb,
                          r.trql_oid_precision AS precision
                   FROM Run r JOIN Query q ON q.id = r.query_id""",
                conn,
            )
        df["degree"] = deg
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_stages(data_dir: Path, query: str) -> pd.DataFrame:
    """Per-stage self-time from the instrumented builds, as a share of the run.

    Returned as fractions rather than milliseconds, because the run they were
    measured on is slower than the one the figure draws — see the module docs.
    """
    frames = []
    for deg in _DEGREES:
        db = data_dir / f"{deg}-inst.db"
        if not db.is_file():
            raise SystemExit(
                f"no instrumented database at {db} — see data/ablation/run.sh, "
                "rebuilt with `--features instrument`"
            )
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
            df = pd.read_sql_query(
                """SELECT r.trql_error_px AS err,
                          r.trql_selection_ms AS sel, r.trql_poly_ms AS poly,
                          r.trql_lift_ms AS lift, r.trql_temporal_ms AS temp,
                          r.trql_other_ms AS other
                   FROM Run r JOIN Query q ON q.id = r.query_id
                   WHERE q.name = ?""",
                conn,
                params=(query,),
            )
        if df.isna().any().any():
            raise SystemExit(f"{db} has no stage breakdown — was it built with `instrument`?")
        df["temp"] = df["lift"] + df["temp"] + df["other"]
        cols = [c for c, _, _ in _STAGES]
        df[cols] = df[cols].div(df[cols].sum(axis=1), axis=0)
        df["degree"] = deg
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    here = Path(__file__).parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=here / "data" / "ablation")
    ap.add_argument(
        "--stage-dir", type=Path, default=here / "data" / "ablation-instrumented"
    )
    ap.add_argument("--query", default="I24-convoy")
    ap.add_argument("--out", type=Path, default=here / "plots" / "comparison-cubic-quadratic-plot.pdf")
    args = ap.parse_args()

    df = load(args.data_dir)
    df = df[df["query"] == args.query].sort_values("err")
    if df.empty:
        raise SystemExit(f"no runs for {args.query} in {args.data_dir}")
    stages = load_stages(args.stage_dir, args.query)

    # Three panels in one column, stacked, sharing the budget axis along the
    # bottom so it is drawn once. Kept shallow: the figure sits in a single
    # column of the paper, where height is what it costs the page.
    fig, axes = plt.subplots(
        3, 1, figsize=(3.4, 2.3), sharex=True, constrained_layout=True
    )
    # The budgets are a percentile ladder of the dataset's own motion, not an
    # arithmetic one; plotted at equal spacing so the tight end, where the
    # degrees differ most, is not squeezed into the left margin.
    budgets = sorted(df["err"].unique())
    xs = list(range(len(budgets)))

    # Three bars per budget: d = 1, 2, 3 left to right, in the order of the
    # degree legend below.
    span = 0.8
    bar_w = span / len(_DEGREES)
    for j, deg in enumerate(_DEGREES):
        d = df[df["degree"] == deg].set_index("err").reindex(budgets)
        share = stages[stages["degree"] == deg].set_index("err").reindex(budgets)
        colour = _STYLE[deg]

        # Proportions from the instrumented build, heights from the clean one:
        # each bar is as tall as the query really took, split as the
        # instrumented run measured it.
        offset = (j - (len(_DEGREES) - 1) / 2) * bar_w
        pos = [x + offset for x in xs]
        bottom = [0.0] * len(budgets)
        for col, label, scolour in _STAGES:
            part = (share[col] * d["ms"]).to_numpy()
            axes[0].bar(
                pos,
                part,
                width=bar_w * 0.88,
                bottom=bottom,
                color=scolour,
                label=label if j == 0 else None,
                edgecolor="black",
                linewidth=0.2,
                zorder=3,
            )
            bottom = [b + p for b, p in zip(bottom, part)]

        # Which bar is which degree, without a second legend: the numeral sits
        # on top of each bar. The bars are stage-coloured, so nothing else in
        # the panel says it.
        for x, total in zip(pos, bottom):
            axes[0].annotate(
                _LABEL[deg].replace("$d{=}", "").replace("$", ""),
                (x, total),
                xytext=(0, 1),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=5.5,
                color="0.35",
                zorder=4,
            )

        # The lower two panels are the same groups again, one bar per degree at
        # the same offset as above, so a column reads as one configuration all
        # the way down. Here the bar is the degree's own colour -- there is
        # nothing to stack -- which is also what the legend names.
        bars = dict(
            width=bar_w * 0.88, color=colour, edgecolor="black", linewidth=0.2, zorder=3
        )
        # Split by what a segment holds: a fitted polynomial, or the raw boxes
        # of a run too short to fit (fewer than five detections). Only the first
        # kind costs a root-find to evaluate, and at eps=0 it is 5 % of them --
        # which is why the budget buys speed faster than it buys compression.
        axes[1].bar(pos, d["poly_segments"] / 1e6, label=_LABEL[deg], **bars)
        axes[1].bar(
            pos,
            d["direct_segments"] / 1e6,
            bottom=d["poly_segments"] / 1e6,
            width=bar_w * 0.88,
            color=_tint(colour),
            edgecolor="black",
            linewidth=0.2,
            zorder=3,
        )
        axes[2].bar(pos, d["segments"] * _SEGMENT_BYTES[deg] / 2**20, **bars)

    # Headroom for the degree numerals standing on the bars.
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.1)
    for ax, ylabel in zip(axes, ("Time (ms)", "Segments (M)", "Segment MB")):
        ax.set_ylabel(ylabel, fontsize=6.5, labelpad=2)
        ax.tick_params(labelsize=5.5, pad=1.5)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    axes[-1].set_xticks(xs)
    axes[-1].set_xticklabels([f"{b:g}" for b in budgets])
    axes[-1].set_xlabel(r"Error budget $\varepsilon$", fontsize=6.5, labelpad=2)
    axes[-1].set_xlim(-0.6, len(budgets) - 0.4)

    # Three entries fit inside the panel they describe, in the top right, which
    # the bars leave clear once the budget opens. The panel is grown to make the
    # room rather than taking it from the bars.
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.28)
    axes[0].legend(
        fontsize=7,
        ncol=len(_STAGES),
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        borderaxespad=0.2,
        frameon=True,
        framealpha=0.85,
        edgecolor="none",
        facecolor="white",
        handlelength=0.9,
        handletextpad=0.3,
        columnspacing=0.8,
        borderpad=0.05,
    )
    # The degrees are named once, in the panel whose curves leave the most room.
    handles, labels = axes[1].get_legend_handles_labels()
    # Only name the verbatim share where there is one to see. A dataset of long
    # tracks (I24-Motion) fits almost every run, so the pale part of a bar is
    # invisible and the swatch would point at nothing.
    verbatim = (df["direct_segments"] / df["segments"]).max()
    if verbatim > 0.02:
        handles.append(Patch(facecolor=_tint("0.45"), edgecolor="black", linewidth=0.2))
        labels.append("verbatim")
    axes[1].legend(
        handles,
        labels,
        fontsize=8,
        ncol=len(labels),
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        borderaxespad=0.2,
        frameon=True,
        framealpha=0.85,
        edgecolor="none",
        handlelength=1.2,
        handletextpad=0.35,
        columnspacing=0.8,
        borderpad=0.05,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0)

    wide = df[df["err"] == max(budgets)].set_index("degree")
    print(f"{args.out}")
    for deg in _DEGREES:
        r = wide.loc[deg]
        held = r["segments"] * _SEGMENT_BYTES[deg] / 2**20
        print(f"  {deg:<9} {r['ms']:7.1f} ms  {r['segments']:>9,.0f} segs  "
              f"{held:6.1f} MB of segments ({_SEGMENT_BYTES[deg]} B each)  "
              f"{r['mem_mb']:5.0f} MB RSS  precision {r['precision']:.4f}")


if __name__ == "__main__":
    main()
