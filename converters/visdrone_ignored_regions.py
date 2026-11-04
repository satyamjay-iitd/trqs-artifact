#!/usr/bin/env python
"""Record VisDrone's ignored regions in the converted databases.

``visdrone_convert.py`` drops rows with ``score = 0`` or ``object_category = 0`` — the
latter are VisDrone's ignored regions, areas the annotators declined to label.
Ground truth holds nothing there, so a detector firing in one is not making a
mistake, but scoring counts it a false positive: on the densest sequence that is
23% of all predictions, and excusing them moves precision 0.698 -> 0.908.

Unlike UA-DETRAC, whose regions are fixed for a whole sequence, VisDrone marks
them **per frame** — so the table carries a ``fid`` column.
``trql-storage::eval`` treats a NULL ``fid`` as "every frame", which is what the
DETRAC tables (written without the column) mean.

Run with the project venv::

    .trqs-pyenv/bin/python converters/visdrone_ignored_regions.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "datasets" / "raw" / "VisDrone"
PROCESSED = ROOT / "datasets" / "processed" / "visdrone"

Region = tuple[int, float, float, float, float]


def regions_for(sub: str, raw: Path = RAW) -> list[Region] | None:
    """``(fid, x1, y1, x2, y2)`` per ignored region, or None if no annotation.

    A sequence we cannot speak for must not be recorded as having none — that is
    indistinguishable from a correct empty answer and would quietly restore the
    bug this fixes.
    """
    # Processed names are `<split>_<sequence>`; the annotation is per sequence.
    seq = sub.split("_", 1)[1] if "_" in sub else sub
    hits = list(raw.glob(f"VisDrone2019-MOT-*/annotations/{seq}.txt"))
    if not hits:
        return None
    out: list[Region] = []
    for line in hits[0].read_text().splitlines():
        p = line.strip().split(",")
        if len(p) < 8 or int(p[7]) != 0:
            continue
        # `convert.py` uses 0-based fid (frame_index - 1); x,y are top-left.
        fid = int(p[0]) - 1
        x, y, w, h = (float(p[i]) for i in (2, 3, 4, 5))
        out.append((fid, x, y, x + w, y + h))
    return out


def write_ignored_regions(db_path: Path, regions: list[Region]) -> None:
    con = duckdb.connect(str(db_path))
    try:
        con.execute("DROP TABLE IF EXISTS ignored_regions")
        con.execute(
            "CREATE TABLE ignored_regions "
            "(fid BIGINT, x1 DOUBLE, y1 DOUBLE, x2 DOUBLE, y2 DOUBLE)"
        )
        if regions:
            con.executemany(
                "INSERT INTO ignored_regions VALUES (?, ?, ?, ?, ?)", regions
            )
    finally:
        con.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    ap.add_argument("--sub", action="append", help="Only these sub-datasets.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    subs = sorted(d for d in args.processed.iterdir() if (d / "duckdb.db").is_file())
    if args.sub:
        wanted = set(args.sub)
        subs = [d for d in subs if d.name in wanted]

    written = skipped = 0
    for d in subs:
        regions = regions_for(d.name, args.raw)
        if regions is None:
            print(f"  {d.name}: no annotation file found — left alone")
            skipped += 1
            continue
        if args.dry_run:
            print(f"  {d.name}: {len(regions)} regions")
            continue
        write_ignored_regions(d / "duckdb.db", regions)
        written += 1

    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb} ignored_regions for {written} sub-datasets, skipped {skipped}")


if __name__ == "__main__":
    main()
