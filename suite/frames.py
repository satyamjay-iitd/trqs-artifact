#!/usr/bin/env python3
"""Coordinate frames: write a region in the units a reader understands, run it
in the units the database stores.

Four of the processed datasets keep their boxes as scaled integers rather than
pixels, and all four record the same affine frame in their metadata::

    stored = natural * coordinate_scale + delta

i24_motion is the one that matters here: its boxes are roadway feet x 100,
shifted non-negative, so a lane band at 11..22 ft is stored as 21100..22200.
Writing 21100 in a query is unreadable and unreviewable; writing ``ft(11)`` and
letting this module apply the dataset's own recorded frame is neither.

The conversion is done at query-authoring time, so `trql-cli` still receives the
plain stored numbers it has always taken and nothing in the engine changes. What
the paper prints is the natural form, with the frame stated once.

Note this deliberately does *not* mirror `trql-server`'s ``QueryFrame``. That one
also flips Y, because a dashboard map's Y runs up the screen while the stored
raster's runs down. i24 has no map raster and no "up" — its Y is lateral offset
across the carriageway — so a flip here would silently mirror the two directions
of travel into each other.
"""
from __future__ import annotations

import json
from pathlib import Path

from paths import PROCESSED, ROOT


class Frame:
    """The affine map from a dataset's natural units into its stored units.

    Two conventions exist in the tree, and they differ only in Y:

    * **unflipped** — `stored = natural * scale + delta` on both axes. i24 uses
      this: its Y is lateral offset across a carriageway, there is no map raster
      and no notion of "up".
    * **Y-up** — X as above, but `stored_y = y_delta - natural_y * scale`, where
      `y_delta` is the map raster's height in stored units. A dataset drawn on a
      map stores Y running *down* the image while the map's own axis runs up.
      nuScenes uses this.

    Getting it wrong mirrors a region about the map's centre line, which produces
    a perfectly plausible-looking rectangle over the wrong piece of ground — so
    the orientation is inferred from the metadata and can be checked against the
    data with :meth:`verify`.
    """

    def __init__(
        self,
        scale: float = 1.0,
        x_delta: float = 0.0,
        y_delta: float = 0.0,
        y_up: bool = False,
        height: float | None = None,
        width: float | None = None,
    ):
        self.scale, self.x_delta, self.y_delta = scale, x_delta, y_delta
        self.y_up, self.height, self.width = y_up, height, width

    @classmethod
    def of(cls, dataset: str, sub: str, y_up: bool | None = None) -> "Frame":
        """Read the frame a sub-dataset recorded, or the identity if it has none.

        A dataset whose boxes are plain pixels (soccernet, clevrer) records no
        `coordinate_scale`, and gets the identity — so the same authoring code
        works whether or not a dataset is scaled.

        `y_up` is inferred as `y_delta == height`: a Y-up frame is anchored at the
        map raster's top edge, so its shift *is* the raster height. Verified
        against the data — on boston-seaport the unflipped reading puts the
        objects at −1921..−231 m, outside a 2118 m map, while the flipped one puts
        them at 231..1922 m, inside it. i24 (no `height`) and SoccerTrackv2
        (`y_delta` 3571 vs `height` 7210) both come out unflipped, correctly.
        Pass `y_up` explicitly to override.
        """
        for name in ("metadata.json", "world-metadata.json"):
            path = PROCESSED / dataset / sub / name
            if path.is_file():
                m = json.loads(path.read_text())
                scale = m.get("coordinate_scale") or 1.0
                yd = m.get("y_delta") or 0.0
                h, w = m.get("height"), m.get("width")
                # Compared with a tolerance, not `==`: singapore-queenstown records
                # y_delta 368710.00000000006 against height 368710, and exact
                # equality silently inferred an unflipped frame that put its
                # objects 3.4 km off the map. One stored unit is 1 cm here, so a
                # sub-unit difference is float noise and never a real distinction.
                flip = (
                    (h is not None and abs(yd - h) < 1.0) if y_up is None else y_up
                )
                return cls(scale, m.get("x_delta") or 0.0, yd, flip, h, w)
        return cls()

    def x(self, natural: float) -> int:
        return round(natural * self.scale + self.x_delta)

    def y(self, natural: float) -> int:
        if self.y_up:
            return round(self.y_delta - natural * self.scale)
        return round(natural * self.scale + self.y_delta)

    def rect(self, x1: float, y1: float, x2: float, y2: float) -> str:
        """A `REGIONS` rectangle, natural units in, stored units out.

        The corners are re-ordered after conversion: a Y-up frame reverses the
        axis, so the natural-lower edge becomes the stored-*upper* one and a
        rectangle written `y1 < y2` would come out inverted. `INSIDE` on an
        inverted rectangle matches nothing, silently.
        """
        sx1, sx2 = sorted((self.x(x1), self.x(x2)))
        sy1, sy2 = sorted((self.y(y1), self.y(y2)))
        return f"[{sx1}, {sy1}, {sx2}, {sy2}]"

    def is_identity(self) -> bool:
        return (
            self.scale == 1.0
            and self.x_delta == 0.0
            and self.y_delta == 0.0
            and not self.y_up
        )

    def to_natural(self, sx: float, sy: float) -> tuple[float, float]:
        """Stored units back to natural — the inverse of :meth:`x` / :meth:`y`."""
        nx = (sx - self.x_delta) / self.scale
        ny = (
            (self.y_delta - sy) / self.scale
            if self.y_up
            else (sy - self.y_delta) / self.scale
        )
        return nx, ny

    def verify(self, dataset: str, sub: str) -> tuple[bool, str]:
        """Check the frame against the data: does it put the boxes on the map?

        Reads the sub-dataset's coordinate extent and converts it back to natural
        units. A correct frame lands inside `[0, width] x [0, height]`; a wrong Y
        orientation lands outside, usually negative. Returns `(ok, message)`
        rather than raising, so a caller can report on several datasets at once.
        """
        import duckdb

        if self.width is None or self.height is None:
            return True, "no map extent recorded — nothing to check against"
        con = duckdb.connect(str(PROCESSED / dataset / sub / "duckdb.db"), read_only=True)
        try:
            x0, x1, y0, y1 = con.execute(
                "SELECT min(x1), max(x2), min(y1), max(y2) FROM bbox"
            ).fetchone()
        finally:
            con.close()
        nx0, ny0 = self.to_natural(x0, y0)
        nx1, ny1 = self.to_natural(x1, y1)
        lo_x, hi_x = sorted((nx0, nx1))
        lo_y, hi_y = sorted((ny0, ny1))
        w, h = self.width / self.scale, self.height / self.scale
        ok = 0 <= lo_x and hi_x <= w and 0 <= lo_y and hi_y <= h
        return ok, (
            f"x {lo_x:.0f}..{hi_x:.0f} of {w:.0f} m, "
            f"y {lo_y:.0f}..{hi_y:.0f} of {h:.0f} m, y_up={self.y_up}"
        )
