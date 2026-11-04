"""Shared helpers for converting raw datasets into the TrQS common format.

Target layout produced for every dataset::

    <out>/<DatasetName>/<subset>/video.mp4      # symlink to the source video
                               /duckdb.db        # annotations (schema below)
                               /metadata.json    # fps, classes, extra info

DuckDB schema (identical for every dataset)::

    objects(oid PRIMARY KEY, class)
    activities(aid PRIMARY KEY, name)
    bbox(fid, oid -> objects.oid, x1, y1, x2, y2, aid -> activities.aid NULLABLE)

Each per-dataset converter is responsible only for parsing the raw annotations
into ``Obj`` / ``Activity`` / ``Bbox`` records; this module owns everything that
is common: probing the video, building the DuckDB file, symlinking the video and
writing ``metadata.json``.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Optional, Sequence

import duckdb


# --------------------------------------------------------------------------- #
# Record types
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Obj:
    """A tracked object. ``oid`` is unique within a subset."""

    oid: int
    cls: str


@dataclass(slots=True)
class Activity:
    """A named activity/event. ``aid`` is unique within a subset."""

    aid: int
    name: str


@dataclass(slots=True)
class Bbox:
    """One bounding box of object ``oid`` in frame ``fid``.

    Coordinates are pixel values: (x1, y1) top-left, (x2, y2) bottom-right.
    ``aid`` links the box to an activity, or is ``None`` when unlabeled.
    """

    fid: int
    oid: int
    x1: float
    y1: float
    x2: float
    y2: float
    aid: Optional[int] = None


# --------------------------------------------------------------------------- #
# Video probing
# --------------------------------------------------------------------------- #
def probe_video(path: os.PathLike | str) -> dict:
    """Return basic video metadata using ``ffprobe``.

    Keys: ``fps``, ``width``, ``height``, ``num_frames``, ``duration_s``.
    Fields that ffprobe cannot determine are returned as ``None``.
    """
    path = os.fspath(path)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_streams",
        "-show_format",
        "-print_format",
        "json",
        path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(out.stdout)
    if not info.get("streams"):
        raise ValueError(f"no video stream found in {path}")
    stream = info["streams"][0]
    fmt = info.get("format", {})

    def _fps() -> Optional[float]:
        for key in ("avg_frame_rate", "r_frame_rate"):
            val = stream.get(key)
            if val and val != "0/0":
                f = float(Fraction(val))
                if f > 0:
                    return f
        return None

    def _int(value) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _float(value) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    num_frames = _int(stream.get("nb_frames"))
    duration = _float(stream.get("duration")) or _float(fmt.get("duration"))
    fps = _fps()
    if num_frames is None and duration is not None and fps:
        num_frames = round(duration * fps)

    return {
        "fps": fps,
        "width": _int(stream.get("width")),
        "height": _int(stream.get("height")),
        "num_frames": num_frames,
        "duration_s": duration,
    }


# --------------------------------------------------------------------------- #
# DuckDB construction
# --------------------------------------------------------------------------- #
SCHEMA = """
CREATE TABLE objects (
    oid   BIGINT PRIMARY KEY,
    class VARCHAR
);

CREATE TABLE activities (
    aid  BIGINT PRIMARY KEY,
    name VARCHAR
);

CREATE TABLE bbox (
    fid BIGINT,
    oid BIGINT REFERENCES objects(oid),
    x1  DOUBLE,
    y1  DOUBLE,
    x2  DOUBLE,
    y2  DOUBLE,
    aid BIGINT REFERENCES activities(aid)
);
"""


_BBOX_COLUMNS = {
    "fid": "BIGINT",
    "oid": "BIGINT",
    "x1": "DOUBLE",
    "y1": "DOUBLE",
    "x2": "DOUBLE",
    "y2": "DOUBLE",
    "aid": "BIGINT",
}


def build_duckdb(
    db_path: os.PathLike | str,
    objects: Sequence[Obj],
    bboxes: Iterable[Bbox],
    activities: Sequence[Activity] = (),
) -> dict:
    """Create ``db_path`` (overwriting) and populate the three tables.

    Object ids are remapped to a **continuous** 0-based range (sorted by the
    original id) so every dataset's ``oid``s are gap-free; the same remap is
    applied to ``bbox.oid``. Bboxes (potentially millions of rows) are streamed
    to a temporary CSV and bulk-loaded with DuckDB's vectorized reader --
    row-by-row ``executemany`` is orders of magnitude too slow for this volume.
    Returns row counts.
    """
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()

    # Remap the (possibly sparse) object ids to a continuous 0-based range.
    oid_remap = {oid: i for i, oid in enumerate(sorted({o.oid for o in objects}))}

    # Stream bboxes to a temp CSV alongside the db (same filesystem). An empty
    # field maps to SQL NULL, which is exactly what we want for a missing aid.
    n_bbox = 0
    fd, csv_path = tempfile.mkstemp(suffix=".bbox.csv", dir=str(db_path.parent))
    try:
        with os.fdopen(fd, "w", newline="") as fh:
            writer = csv.writer(fh)
            for b in bboxes:
                if b.oid not in oid_remap:
                    raise ValueError(f"bbox references unknown oid {b.oid} (not in objects)")
                writer.writerow((b.fid, oid_remap[b.oid], b.x1, b.y1, b.x2, b.y2, b.aid))
                n_bbox += 1

        con = duckdb.connect(str(db_path))
        try:
            con.execute(SCHEMA)
            if objects:
                con.executemany(
                    "INSERT INTO objects VALUES (?, ?)",
                    [(oid_remap[o.oid], o.cls) for o in objects],
                )
            if activities:
                con.executemany(
                    "INSERT INTO activities VALUES (?, ?)",
                    [(a.aid, a.name) for a in activities],
                )
            if n_bbox:
                con.execute(
                    "INSERT INTO bbox SELECT * FROM "
                    "read_csv(?, header=false, columns=?)",
                    [csv_path, _BBOX_COLUMNS],
                )
        finally:
            con.close()
    finally:
        os.unlink(csv_path)

    return {
        "objects": len(objects),
        "activities": len(activities),
        "bbox": n_bbox,
    }


# --------------------------------------------------------------------------- #
# Subset assembly
# --------------------------------------------------------------------------- #
def symlink_video(video_src: os.PathLike | str, dst: os.PathLike | str) -> None:
    """Create/replace ``dst`` as a symlink to ``video_src`` (absolute target)."""
    video_src = Path(video_src).resolve()
    dst = Path(dst)
    if not video_src.exists():
        raise FileNotFoundError(f"source video missing: {video_src}")
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(video_src)


def encode_frames_to_mp4(
    frames_dir: os.PathLike | str,
    dst: os.PathLike | str,
    *,
    fps: float,
    pattern: str = "*.jpg",
    crf: int = 18,
) -> None:
    """Encode a directory of ordered frame images into an H.264 ``dst`` mp4.

    For datasets that ship frames instead of a video. Frames are taken in
    lexicographic order (``pattern``, glob), so zero-padded names sort
    correctly. Odd-sized frames are padded to even dimensions (required by
    ``yuv420p``); padding is added on the right/bottom so top-left-origin
    bounding-box coordinates remain valid.
    """
    frames_dir = Path(frames_dir)
    dst = Path(dst)
    if not any(frames_dir.glob(pattern)):
        raise FileNotFoundError(f"no frames matching {pattern!r} in {frames_dir}")
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        str(frames_dir / pattern),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-an",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def build_subset(
    out_dir: os.PathLike | str,
    *,
    video_src: os.PathLike | str | None = None,
    frames_dir: os.PathLike | str | None = None,
    fps: float | None = None,
    frames_pattern: str = "*.jpg",
    objects: Sequence[Obj],
    bboxes: Iterable[Bbox],
    activities: Sequence[Activity] = (),
    metadata: Optional[dict] = None,
    reuse_video: bool = False,
) -> dict:
    """Assemble one subset directory in the common format.

    Provide exactly one video source:
      * ``video_src`` -- an existing video file, symlinked as ``video.mp4``; or
      * ``frames_dir`` (+ ``fps``) -- a directory of frames, encoded to
        ``video.mp4`` via :func:`encode_frames_to_mp4`.

    Writes ``video.mp4``, ``duckdb.db`` and ``metadata.json``. Video-derived
    fields (fps/width/height/num_frames) are added to metadata automatically;
    values supplied by ``metadata`` win. Returns the metadata that was written.

    When ``reuse_video`` is set and ``video.mp4`` already exists, the video
    source is left untouched (no symlink/re-encode) -- useful for cheaply
    rebuilding just ``duckdb.db``/``metadata.json``.
    """
    if (video_src is None) == (frames_dir is None):
        raise ValueError("pass exactly one of video_src or frames_dir")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "video.mp4"

    if reuse_video and video_path.exists():
        pass  # keep the existing video; only annotations are rebuilt
    elif video_src is not None:
        symlink_video(video_src, video_path)
    else:
        if fps is None:
            raise ValueError("fps is required when encoding from frames_dir")
        encode_frames_to_mp4(frames_dir, video_path, fps=fps, pattern=frames_pattern)

    counts = build_duckdb(out_dir / "duckdb.db", objects, bboxes, activities)

    video_info = probe_video(video_path)
    meta = {
        "fps": video_info["fps"],
        "width": video_info["width"],
        "height": video_info["height"],
        "num_frames": video_info["num_frames"],
        "duration_s": video_info["duration_s"],
        "row_counts": counts,
    }
    meta.update(metadata or {})
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta
