"""Repository paths shared by evaluation drivers.

TRQS_DATA_DIR and TRQS_RAW_DIR select processed and raw dataset roots.
TRQS_OUTPUT_DIR keeps generated reports separate from checked-in inputs.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITE = Path(__file__).resolve().parent
CLI = ROOT / "engine" / "target" / "release" / "trql-cli"
PROCESSED = Path(os.environ.get("TRQS_DATA_DIR", ROOT / "datasets" / "processed")).resolve()
RAW = Path(os.environ.get("TRQS_RAW_DIR", ROOT / "datasets" / "raw")).resolve()
OUTPUT = Path(os.environ.get("TRQS_OUTPUT_DIR", ROOT / "results" / "suite")).resolve()
SCRATCH = OUTPUT / "tmp"
VLM = Path(os.environ.get("TRQS_VLM_DIR", SUITE / "vlm")).resolve()
