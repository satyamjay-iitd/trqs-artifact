#!/usr/bin/env python
"""Record MOT17's non-evaluated ground-truth rows as ignored regions.

``mot17_convert.py`` keeps only ``flag == 1`` rows — the pedestrian evaluation targets —
and drops the rest. On MOT17-04 that is 60,448 of 108,005 rows (56%): occluders,
static persons, and the vehicle classes. Ground truth says nothing about those
areas, so a detector firing there is not making a mistake, but scoring counts it a
false positive.

Every dropped row is recorded, not a chosen subset. The reasoning is the same one
the converter used to drop them: `flag == 0` means "outside the evaluation", and a
detection outside the evaluation is neither right nor wrong. Picking which
dropped classes to forgive would be re-deciding that, less carefully.

Rows are per frame, so the table carries a ``fid`` column;
``trql-storage::eval`` reads a NULL ``fid`` as "every frame", which is what the
UA-DETRAC tables mean.

Run with the project venv::

    .trqs-pyenv/bin/python converters/mot17_ignored_regions.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "datasets" / "raw" / "mot17"
PROCESSED = ROOT / "datasets" / "processed" / "mot17"

Region = tuple[int, float, float, float, float]


def regions_for(sub: str, raw: Path = RAW) -> list[Region] | None:
    """``(fid, x1, y1, x2, y2)`` per non-evaluated row, or None if no gt found.

    The ``-DPM``/``-FRCNN``/``-SDP`` variants share identical gt — the converter
    deduplicates them to one subset — so any one of them answers for the sequence.
    """
    hits = sorted(raw.glob(f"*/{sub}-*/gt/gt.txt")) or sorted(raw.glob(f"*/{sub}/gt/gt.txt"))
    if not hits:
        return None
    out: list[Region] = []
    for line in hits[0].read_text().splitlines():
        p = line.strip().split(",")
        if len(p) < 8 or int(p[6]) != 0:
            continue
        # `convert.py` uses 0-based fid (gt frame - 1); x,y are top-left.
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
            con.executemany("INSERT INTO ignored_regions VALUES (?, ?, ?, ?, ?)", regions)
    finally:
        con.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    written = skipped = 0
    for d in sorted(x for x in args.processed.iterdir() if (x / "duckdb.db").is_file()):
        regions = regions_for(d.name, args.raw)
        if regions is None:
            print(f"  {d.name}: no gt.txt found — left alone")
            skipped += 1
            continue
        if args.dry_run:
            print(f"  {d.name}: {len(regions):,} regions")
            continue
        write_ignored_regions(d / "duckdb.db", regions)
        written += 1
    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb} ignored_regions for {written} sub-datasets, skipped {skipped}")


if __name__ == "__main__":
    main()
