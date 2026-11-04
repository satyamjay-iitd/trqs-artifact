"""The registry of plot kinds.

A new figure type for the paper is a module in this package exporting a `KIND`,
plus a line in `KINDS` below. Nothing else in `benchplot` names a kind: the
manifest reader, the CLI and `benchplot kinds` all work off this dict, so a kind
added here is immediately writable in `manifest.toml`.
"""

from __future__ import annotations

from benchplot.plots import (
    compression,
    cooccurrence,
    epsilon,
    p15,
    recall,
    rekall,
    speedup,
    static_pdf,
    vlm,
    vlmbars,
)
from benchplot.spec import PlotKind

KINDS: dict[str, PlotKind] = {
    kind.name: kind
    for kind in (
        compression.KIND,
        cooccurrence.KIND,
        epsilon.KIND,
        p15.KIND,
        recall.KIND,
        rekall.KIND,
        speedup.KIND,
        static_pdf.KIND,
        vlm.KIND,
        vlmbars.KIND,
    )
}

__all__ = ["KINDS"]
