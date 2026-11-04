"""Copy a preserved PDF whose original experiment and generator are external."""
from pathlib import Path
import shutil

import pandas as pd

from benchplot.spec import Plot, PlotKind, Table


def _parse(table: Table) -> Path:
    source = table.path("source")
    if source is None:
        raise ValueError("static_pdf requires source")
    return source


def _render(points: pd.DataFrame, plot: Plot) -> Path:
    source = plot.options
    if not source.is_file():
        raise FileNotFoundError(source)
    if plot.out.suffix.lower() != ".pdf":
        raise ValueError("static_pdf output must be a PDF")
    plot.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, plot.out)
    return plot.out


KIND = PlotKind(
    name="static_pdf",
    summary="Copy a preserved PDF; does not rerun or redraw its experiment.",
    parse=_parse,
    render=_render,
    needs_runs=False,
    option_help=(("source", "Path to the preserved PDF, relative to the manifest."),),
)
