# /// script
# requires-python = ">=3.13"
# dependencies = ["duckdb>=1.5.6", "requests"]
# ///
"""Rekall's cyclist-detection tutorial data, as a TrQS dataset.

Rekall (Fu et al., Stanford TR 2019) publishes the Mask R-CNN detections behind
tutorial 01 ("Cyclist Detection") alongside the notebook, so its cyclist query
can be run on exactly the input its authors ran it on. This script downloads
those pickles and writes them into the processed-dataset schema the engine
ingests, so Rekall, a frame-expanded SQL baseline and TrQS can all be measured
on the same boxes.

    uv run suite/build_rekall_cydet.py            # raw
    uv run suite/build_rekall_cydet.py --track    # with identity

Each pickle is a list over frames of `[x1, y1, x2, y2, class, score, png]` in
**pixel** coordinates (`tutorials/cyclist_tutorial_helpers.py` divides by
width/height to get the frame-relative bounds Rekall stores; the overlap
predicates are invariant under that positive scaling, so the question asked is
the same either way).

THREE THINGS THIS SCRIPT DECIDES, AND WHY.

Integer coordinates. `trql-storage`'s ingest casts every coordinate with `as
u32` (`convert.rs:197`), a truncation. A reference query reading the source
doubles would therefore be asking a question of data the engine never sees --
on DETRAC that was one answer pair in fifty (see `build_detrac_px.py`).
Truncating once here puts all three encodings on the same numbers by
construction.

Every class is kept, not just the two the query names. The tutorial loads all
detections and filters by payload class inside the query
(`bboxes.filter(lambda i: i['payload']['class'] == 'bicycle')`), so the class
filter is part of what each system is being timed on: a `filter` for Rekall, a
join against `objects` for SQL, a selection for TrQS. Pre-filtering the dataset
would quietly remove that work from all three and shrink the input by 9x.

Identity is optional and is ours. The release is per-frame detections with no
track ids, which is all Rekall needs -- it never uses identity. TrQS fits
polynomial segments per object, so on ids-as-detections every trajectory is one
frame long and the segment model has nothing to compress; that is TrQS's floor,
and `--track` adds the identity it is designed for. The tracker below is this
script's addition, not Rekall's data, and its cost and output are reported.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import duckdb
import requests

from paths import PROCESSED, RAW, ROOT

_REPO, _RAW, _PROCESSED = ROOT, RAW / "rekall_cydet", PROCESSED

_BASEURL = "https://storage.googleapis.com/esper/dan_olimar/rekall_tutorials/cydet"
# The two videos `cyclist_tutorial_helpers.py` names, in its order (sorted by
# filename), so a video's index here is the index its `id` has in the tutorial.
_PICKLES = ["maskrcnn_bboxes_0001.pkl", "maskrcnn_bboxes_0004.pkl"]

# Matched to a track only above this; below it the detection starts a new track.
# At 10 fps on dashcam footage a box moves a good fraction of its own width
# between frames, so a threshold tuned for 25-30 fps would fragment everything.
# The sweep on video 0001, in boxes per track:
#
#            misses=2   misses=5   misses=10
#   IoU 0.1      7.97       9.15        9.93
#   IoU 0.2      6.58       7.34        7.83
#   IoU 0.3      5.28       5.74        6.04
#   IoU 0.5      3.15       3.32        3.44
#
# 0.3 with two misses is the conservative end of the defensible range: looser
# settings give TrQS longer trajectories to fit and so a better number, which is
# exactly why they are not the ones chosen here.
_IOU_MATCH = 0.3
# A track survives this many frames without a match before it is closed, which
# bridges a single missed detection without stitching across an occlusion.
_MAX_MISSES = 2

# `trql-storage` links libduckdb 1.1.x (`duckdb = "~1.10505"`), which reads
# storage version 64; a current Python duckdb writes 67 and the engine then
# refuses the file. Every other processed dataset in the tree is at 64, so pin
# the writer rather than leave the version to whatever uv resolves.
_STORAGE_COMPAT = "v1.0.0"

_SCHEMA = """
CREATE TABLE objects (oid BIGINT PRIMARY KEY, class VARCHAR);
CREATE TABLE activities (aid BIGINT PRIMARY KEY, name VARCHAR);
CREATE TABLE bbox (
    fid BIGINT,
    oid BIGINT REFERENCES objects(oid),
    x1 DOUBLE, y1 DOUBLE, x2 DOUBLE, y2 DOUBLE,
    aid BIGINT REFERENCES activities(aid)
);
"""


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def fetch() -> tuple[list[dict], list[list]]:
    """The tutorial's `metadata.json` and pickles, cached under datasets/raw."""
    _RAW.mkdir(parents=True, exist_ok=True)
    for name in ["metadata.json", *_PICKLES]:
        dst = _RAW / name
        if dst.is_file():
            continue
        print(f"fetching {name} ...", file=sys.stderr)
        resp = requests.get(f"{_BASEURL}/{name}", timeout=120)
        resp.raise_for_status()
        dst.write_bytes(resp.content)

    meta = json.loads((_RAW / "metadata.json").read_text())
    meta = sorted(meta, key=lambda vm: vm["filename"])
    frames = [pickle.loads((_RAW / name).read_bytes()) for name in _PICKLES]
    return meta, frames


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #
def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def track(detections: list[tuple[int, str, tuple[int, int, int, int]]]) -> list[int]:
    """Greedy per-class IoU association; returns a track id per detection.

    Detections arrive in frame order. Within a frame, every (open track,
    detection) pair of the same class is scored by IoU and the pairs are
    consumed highest-first, each track and detection at most once -- the
    standard greedy assignment, which on this footage differs from a Hungarian
    solve only where two boxes of one class overlap each other, and is enough
    for an input whose purpose is to carry identity rather than to be correct
    about it.
    """
    # class -> list of [track id, last box, frames since last match]
    open_tracks: dict[str, list[list]] = {}
    out: list[int] = [-1] * len(detections)
    next_id = 0

    i = 0
    n = len(detections)
    while i < n:
        fid = detections[i][0]
        j = i
        while j < n and detections[j][0] == fid:
            j += 1
        frame = detections[i:j]

        by_class: dict[str, list[int]] = {}
        for k in range(i, j):
            by_class.setdefault(detections[k][1], []).append(k)

        for cls, idxs in by_class.items():
            tracks = open_tracks.setdefault(cls, [])
            cands = sorted(
                (
                    (_iou(tr[1], detections[k][2]), t, k)
                    for t, tr in enumerate(tracks)
                    for k in idxs
                ),
                key=lambda c: (-c[0], c[1], c[2]),
            )
            used_t: set[int] = set()
            used_k: set[int] = set()
            for iou, t, k in cands:
                if iou < _IOU_MATCH or t in used_t or k in used_k:
                    continue
                used_t.add(t)
                used_k.add(k)
                out[k] = tracks[t][0]
                tracks[t][1] = detections[k][2]
                tracks[t][2] = 0
            for k in idxs:
                if k not in used_k:
                    tracks.append([next_id, detections[k][2], 0])
                    out[k] = next_id
                    next_id += 1

        # Age every track that went unmatched this frame, and close the stale.
        for cls, tracks in open_tracks.items():
            survivors = []
            for tr in tracks:
                if tr[2] == 0 and any(out[k] == tr[0] for k in by_class.get(cls, ())):
                    tr[2] = 0
                else:
                    tr[2] += 1
                if tr[2] <= _MAX_MISSES:
                    survivors.append(tr)
            open_tracks[cls] = survivors

        i = j
    return out


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build(vm: dict, frame_list: list[list], with_identity: bool) -> dict:
    name = Path(vm["filename"]).stem
    variant = "rekall_cydet_tracked" if with_identity else "rekall_cydet_raw"
    out_dir = _PROCESSED / variant / name
    out_dir.mkdir(parents=True, exist_ok=True)
    db = out_dir / "duckdb.db"
    for stale in (db, db.with_suffix(".db.wal")):
        stale.unlink(missing_ok=True)

    # (fid, class, (x1, y1, x2, y2)) in frame order, coordinates truncated to
    # the integers the engine will store.
    dets: list[tuple[int, str, tuple[int, int, int, int]]] = [
        (
            fid,
            box[4],
            (
                max(int(box[0]), 0),
                max(int(box[1]), 0),
                max(int(box[2]), 0),
                max(int(box[3]), 0),
            ),
        )
        for fid, in_frame in enumerate(frame_list)
        for box in in_frame
    ]

    t0 = time.perf_counter()
    oids = track(dets) if with_identity else list(range(len(dets)))
    track_ms = (time.perf_counter() - t0) * 1000.0

    # A track id is one object; its class is the class every detection in it
    # carries, since matching never crosses classes.
    obj_class: dict[int, str] = {}
    for oid, (_, cls, _) in zip(oids, dets):
        obj_class[oid] = cls

    conn = duckdb.connect(
        str(db), config={"storage_compatibility_version": _STORAGE_COMPAT}
    )
    conn.execute(_SCHEMA)
    conn.executemany(
        "INSERT INTO objects VALUES (?, ?)", sorted(obj_class.items())
    )
    conn.executemany(
        "INSERT INTO bbox VALUES (?, ?, ?, ?, ?, ?, NULL)",
        [
            (fid, oid, bx[0], bx[1], bx[2], bx[3])
            for oid, (fid, _, bx) in sorted(
                zip(oids, dets), key=lambda r: (r[0], r[1][0])
            )
        ],
    )
    n_bbox = conn.execute("SELECT count(*) FROM bbox").fetchone()[0]
    # The engine drops a repeated fid within one object (`convert.rs:191`), so a
    # duplicate here would make the stored trajectory shorter than the rows SQL
    # reads. The tracker cannot produce one -- a track takes at most one
    # detection per frame -- but on `raw` each oid has exactly one row anyway.
    dup = conn.execute(
        "SELECT count(*) FROM (SELECT oid, fid FROM bbox GROUP BY oid, fid HAVING count(*) > 1)"
    ).fetchone()[0]
    # Rekall's `overlaps()` is a four-way disjunction (`predicates.py:147`) that
    # reduces to strict interval overlap -- which is what the TrQL port and the
    # reference SQL write -- only while both boxes are non-degenerate. Truncation
    # can collapse a one-pixel-wide detection, so check the data rather than the
    # claim.
    degen = conn.execute(
        "SELECT count(*) FROM bbox WHERE x2 <= x1 OR y2 <= y1"
    ).fetchone()[0]
    conn.close()
    if degen:
        raise SystemExit(
            f"{variant}/{name}: {degen} degenerate boxes after truncation; "
            "the strict-overlap reduction of Rekall's `overlaps()` would not hold"
        )

    lens = Counter(oids)
    classes = Counter(obj_class.values())
    meta = {
        "fps": float(vm["fps"]),
        "width": int(vm["width"]),
        "height": int(vm["height"]),
        "num_frames": int(vm["num_frames"]),
        "duration_s": int(vm["num_frames"]) / float(vm["fps"]),
        "row_counts": {"objects": len(obj_class), "activities": 0, "bbox": n_bbox},
        "dataset": variant,
        "source": "Rekall tutorial 01 (Cyclist Detection), Mask R-CNN detections",
        "sequence": name,
        "obj_classes": sorted(classes),
        "activity_classes": [],
        "notes": (
            "Mask R-CNN detections published with Rekall's cyclist tutorial "
            f"({_BASEURL}). Pixel coordinates truncated to integers to match "
            "`trql-storage`'s `as u32` ingest. All detected classes are kept: "
            "the tutorial filters by class inside the query, so the filter is "
            "part of what each system is timed on."
        )
        + (
            " Object ids come from a greedy per-class IoU tracker added by "
            f"suite/build_rekall_cydet.py (IoU >= {_IOU_MATCH}, "
            f"closed after {_MAX_MISSES} missed frames); Rekall's release has "
            "no identity."
            if with_identity
            else " Each detection is its own object: the release has no track "
            "ids, and this is that data unchanged."
        ),
        "identity": (
            {
                "source": "greedy per-class IoU tracker (ours)",
                "iou_match": _IOU_MATCH,
                "max_misses": _MAX_MISSES,
                "tracks": len(lens),
                "mean_track_len": n_bbox / len(lens) if lens else 0.0,
                "max_track_len": max(lens.values()) if lens else 0,
                "tracker_ms": track_ms,
            }
            if with_identity
            else {"source": "none (one object per detection)"}
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    per_class = ", ".join(f"{c}={n}" for c, n in classes.most_common(4))
    print(
        f"{variant}/{name}: {n_bbox:,} boxes, {len(obj_class):,} objects "
        f"({n_bbox / len(obj_class):.2f} boxes/object), {dup} duplicate (oid,fid)"
        + (f", tracker {track_ms:.0f} ms" if with_identity else "")
        + f"\n  objects by class: {per_class}"
    )
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--track",
        action="store_true",
        help="add identity with a greedy IoU tracker (writes rekall_cydet_tracked)",
    )
    args = ap.parse_args()

    meta, frames = fetch()
    for vm, frame_list in zip(meta, frames):
        build(vm, frame_list, args.track)


if __name__ == "__main__":
    main()
