#!/usr/bin/env python
"""Convert the UA-DETRAC dataset into the TrQL common format.

UA-DETRAC is vehicle multi-object tracking. Each sequence ships frames (not a
video) and an XML annotation file; both train and test splits have ground truth.

Raw layout expected (see README.md)::

    raw/detrac/DETRAC-Images/DETRAC-Images/<SEQ>/img00001.jpg ...
    raw/detrac/DETRAC-<Split>-Annotations-XML/DETRAC-<Split>-Annotations-XML/<SEQ>.xml

Each sequence becomes one subset::

    <out>/detrac/<SEQ>/video.mp4   (encoded from the frames)
                      /duckdb.db
                      /metadata.json

XML structure: <sequence> has <ignored_region> and many <frame num=..>, each with
a <target_list> of <target id=..> entries holding a <box left top width height/>
and an <attribute vehicle_type=../>.

  * objects.oid        = target id; class = vehicle_type (car/bus/van/others).
  * bbox               = one row per target box; aid is always NULL.
  * activities         = empty (pure tracking dataset).
  * ignored_regions    = the <ignored_region> rectangles, as (x1, y1, x2, y2).

The ignored regions are not objects, but they are not noise either: DETRAC marks
roughly 15% of a typical frame as ignored during scoring, and a detector that
fires there is not making a mistake. Recording them lets `trql-storage::eval`
excuse those predictions — without the table it counts them as false positives,
about 40% of all predictions on DETRAC, worth ~0.16 F1 on MVI_20011.

They do not mean "un-annotated": 23.5% of DETRAC's ground-truth boxes sit inside
one, which is why `eval` excuses only predictions that matched no ground truth.

Run with the project venv::

    .trqs-pyenv/bin/python converters/detrac_convert.py
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_ROOT = SCRIPT_DIR.parent / "datasets"
sys.path.insert(0, str(SCRIPT_DIR))

from common import Obj, Bbox, build_subset  # noqa: E402

DATASET = "detrac"
DEFAULT_FPS = 25.0  # UA-DETRAC is recorded at 25 fps (no seqinfo ships with it)

SPLITS = {
    "Train": "DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML",
    "Test": "DETRAC-Test-Annotations-XML/DETRAC-Test-Annotations-XML",
}
IMAGES_SUBDIR = "DETRAC-Images/DETRAC-Images"


Region = tuple[float, float, float, float]


def parse_ignored_regions(path: Path) -> list[Region]:
    """The sequence's ``<ignored_region>`` rectangles as ``(x1, y1, x2, y2)``."""
    root = ET.parse(path).getroot()
    out: list[Region] = []
    for ir in root.iter("ignored_region"):
        for b in ir.findall("box"):
            left, top = float(b.get("left")), float(b.get("top"))
            out.append((left, top, left + float(b.get("width")), top + float(b.get("height"))))
    return out


def write_ignored_regions(db_path: Path, regions: Sequence[Region]) -> None:
    """(Re)create the ``ignored_regions`` table in an already-built duckdb.

    Written after :func:`build_subset` rather than through it: only this dataset
    has such regions, and the common builder stays the common format.

    The table is created even when ``regions`` is empty — "this sequence declares
    none" and "nobody recorded them" are different facts, and only the table's
    presence tells them apart.
    """
    import duckdb

    con = duckdb.connect(str(db_path))
    try:
        con.execute("DROP TABLE IF EXISTS ignored_regions")
        con.execute(
            "CREATE TABLE ignored_regions (x1 DOUBLE, y1 DOUBLE, x2 DOUBLE, y2 DOUBLE)"
        )
        if regions:
            con.executemany("INSERT INTO ignored_regions VALUES (?, ?, ?, ?)", list(regions))
    finally:
        con.close()


def parse_xml(path: Path) -> tuple[list[Obj], list[Bbox]]:
    """Parse a DETRAC sequence XML into objects and bboxes.

    ``ignored_region`` boxes are not objects and are not returned here; see
    :func:`parse_ignored_regions`. ``fid`` is 0-based (frame num - 1). An
    object's class is the vehicle_type of its first seen box.
    """
    root = ET.parse(path).getroot()
    objects: dict[int, Obj] = {}
    bboxes: list[Bbox] = []
    for frame in root.findall("frame"):
        fid = int(frame.get("num")) - 1
        for target in frame.findall("./target_list/target"):
            oid = int(target.get("id"))
            box = target.find("box")
            if box is None:
                continue
            left = float(box.get("left"))
            top = float(box.get("top"))
            w = float(box.get("width"))
            h = float(box.get("height"))
            if oid not in objects:
                attr = target.find("attribute")
                cls = attr.get("vehicle_type") if attr is not None else None
                objects[oid] = Obj(oid=oid, cls=(cls or "vehicle"))
            bboxes.append(
                Bbox(fid=fid, oid=oid, x1=left, y1=top, x2=left + w, y2=top + h, aid=None)
            )
    return list(objects.values()), bboxes


def iter_sequences(raw_root: Path, splits):
    for split, anno_rel in SPLITS.items():
        if split.lower() not in splits:
            continue
        anno_dir = raw_root / anno_rel
        if not anno_dir.is_dir():
            print(f"!! split '{split}' annotations not found ({anno_dir}); skipping")
            continue
        for xml in sorted(anno_dir.glob("*.xml")):
            seq = xml.stem
            frames_dir = raw_root / IMAGES_SUBDIR / seq
            yield split, seq, xml, frames_dir


def convert(raw_root: Path, out_root: Path, splits, fps: float) -> None:
    out_dataset = out_root / DATASET
    n = 0
    for split, seq, xml, frames_dir in iter_sequences(raw_root, splits):
        if not frames_dir.is_dir():
            print(f"!! {seq}: image folder missing ({frames_dir}); skipping")
            continue
        objects, bboxes = parse_xml(xml)
        regions = parse_ignored_regions(xml)
        obj_classes = sorted({o.cls for o in objects})

        out_dir = out_dataset / seq
        metadata = {
            "dataset": DATASET,
            "source": "UA-DETRAC",
            "split": split.lower(),
            "sequence": seq,
            "obj_classes": obj_classes,
            "activity_classes": [],
            "assumed_fps": fps,
            "notes": (
                "Frames encoded to H.264 at an ASSUMED 25 fps (UA-DETRAC ships no "
                "explicit frame rate). fid is 0-based (XML frame num - 1). Class "
                "is vehicle_type; ignored_region rectangles are kept in the "
                "ignored_regions table (scoring excuses predictions there); "
                "activities are absent (aid=NULL)."
            ),
        }
        meta = build_subset(
            out_dir,
            frames_dir=frames_dir,
            fps=fps,
            frames_pattern="*.jpg",
            objects=objects,
            bboxes=bboxes,
            activities=(),
            metadata=metadata,
        )
        write_ignored_regions(out_dir / "duckdb.db", regions)
        rc = meta["row_counts"]
        print(f"[ok] {seq} ({split}): {rc['objects']} objects, {rc['bbox']} bboxes, "
              f"{len(regions)} ignored regions, "
              f"classes={obj_classes}, frames={meta['num_frames']} -> {out_dir}")
        n += 1
    print(f"\nDone: wrote {n} subset(s) under {out_dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw", type=Path, default=DATASETS_ROOT / "raw" / "detrac")
    parser.add_argument("--out", type=Path, default=DATASETS_ROOT / "processed")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--splits", nargs="+", default=["train", "test"],
                        choices=["train", "test"])
    args = parser.parse_args()
    convert(args.raw.resolve(), args.out.resolve(), args.splits, args.fps)


if __name__ == "__main__":
    main()
