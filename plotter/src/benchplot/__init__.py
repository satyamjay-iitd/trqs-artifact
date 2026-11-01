"""Plots for the VLDB'27 evaluation, generated from a `trql-cli bench.db`."""

from benchplot.db import collapse_repeats, load_runs
from benchplot.manifest import read_defaults, read_manifest
from benchplot.plots import KINDS
from benchplot.render import Renderer
from benchplot.spec import ManifestError, Plot, PlotKind, Selection

__all__ = [
    "KINDS",
    "ManifestError",
    "Plot",
    "PlotKind",
    "Renderer",
    "Selection",
    "collapse_repeats",
    "load_runs",
    "read_defaults",
    "read_manifest",
]
