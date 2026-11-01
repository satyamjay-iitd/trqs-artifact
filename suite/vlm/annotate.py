# /// script
# requires-python = ">=3.12"
# dependencies = ["duckdb>=1.5.6", "pillow"]
# ///
"""Burn the scoring into the clip so the misses are visible.

One video per Gemini mode. Every car the specification counts as turning right
is boxed for the duration of its turn, green if that mode found it and red if it
did not, with a trail showing the path it took. The model's own answers are drawn
where and when it said they were: blue when the box matched a ground-truth car,
orange when it matched nothing or the wrong car.

The point is the red boxes -- cars making the same manoeuvre, in the same place,
that the model did not report.

    uv run papers/vldb27/suite/vlm/annotate.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import duckdb
from PIL import Image, ImageDraw, ImageFont

_REPO = Path("/home/satyam/dev/frameql")
_HERE = _REPO / "papers/vldb27/suite/vlm"
_DB = _REPO / "datasets/processed/ytstream/uav_0006/duckdb.db"
W, H, FPS = 960, 540, 5

FOUND = (64, 160, 43)        # ctp green
MISSED = (210, 15, 57)       # ctp red
ANSWER_OK = (30, 102, 245)   # ctp blue
ANSWER_BAD = (254, 100, 11)  # ctp peach
INK = (17, 17, 27)


def font(sz: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def label(d, xy, text, colour, f):
    x, y = xy
    box = d.textbbox((x, y), text, font=f)
    d.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=colour)
    d.text((x, y), text, fill=(255, 255, 255), font=f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["static", "agentic"])
    ap.add_argument("--clip", type=Path, default=Path("clip/uav0006_180_210.mp4"),
                    help="the clip that was sent, relative to the vlm/ directory")
    ap.add_argument("--gt", type=Path, default=Path("gt.json"))
    ap.add_argument("--results", type=Path, default=Path("results-vlm.json"))
    ap.add_argument("--suffix", default="",
                    help="appended to annotated_<mode><suffix>.mp4")
    args = ap.parse_args()

    def here(q: Path) -> Path:
        return q if q.is_absolute() else _HERE / q

    clip = here(args.clip)
    gt = json.loads(here(args.gt).read_text())
    res = json.loads(here(args.results).read_text())
    f0 = gt["window_frames"][0]
    turns = {t["oid"]: t for t in gt["turns"]}

    conn = duckdb.connect(str(_DB), read_only=True)
    boxes: dict[int, dict[int, tuple]] = {}
    for oid in turns:
        boxes[oid] = {f: (x1, y1, x2, y2) for f, x1, y1, x2, y2 in conn.execute(
            "SELECT fid, x1, y1, x2, y2 FROM bbox WHERE oid = ? ORDER BY fid",
            [oid]).fetchall()}
    conn.close()

    f_big, f_sm = font(17), font(13)
    work = _HERE / "clip" / "_frames"

    for run in res["runs"]:
        mode = run["mode"]
        if mode not in args.modes or "score" not in run:
            continue
        detail = run["score"]["detail"]
        hits = {d["matched_oid"] for d in detail if d["outcome"] == "hit"}

        # Model answers, placed at the frames it named.
        answers = []
        for d in detail:
            a = d["answer"]
            b = a.get("box_2d")
            if not (isinstance(b, list) and len(b) == 4):
                continue
            ymin, xmin, ymax, xmax = b
            px = (xmin / 1000 * W, ymin / 1000 * H, xmax / 1000 * W, ymax / 1000 * H)
            def sec(v):
                s = str(v)
                return int(s.split(":")[0]) * 60 + float(s.split(":")[1]) if ":" in s else float(s)
            answers.append({"px": px, "s": sec(a["start"]), "e": sec(a.get("end", a["start"])),
                            "ok": d["outcome"] == "hit", "oid": d.get("matched_oid"),
                            "txt": f'{a["start"]}-{a.get("end","?")}'})

        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        subprocess.run(["ffmpeg", "-y", "-i", str(clip),
                        str(work / "f%04d.png")], check=True, capture_output=True)

        for i, fp in enumerate(sorted(work.glob("f*.png"))):
            fid = f0 + i
            t = i / FPS
            img = Image.open(fp).convert("RGB")
            d = ImageDraw.Draw(img)

            for oid, tn in turns.items():
                if not (tn["turn_start_frame"] <= fid <= tn["turn_end_frame"]):
                    continue
                col = FOUND if oid in hits else MISSED
                trail = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                         for f, b in boxes[oid].items()
                         if tn["turn_start_frame"] <= f <= fid]
                if len(trail) > 1:
                    d.line(trail, fill=col, width=3)
                b = boxes[oid].get(fid)
                if b:
                    d.rectangle(b, outline=col, width=3)
                    label(d, (b[0], max(0, b[1] - 19)),
                          f"{oid} {'FOUND' if oid in hits else 'MISSED'}", col, f_sm)

            for a in answers:
                if not (a["s"] - 0.4 <= t <= a["e"] + 0.4):
                    continue
                col = ANSWER_OK if a["ok"] else ANSWER_BAD
                d.rectangle(a["px"], outline=col, width=2)
                label(d, (a["px"][0], min(H - 16, a["px"][3] + 3)),
                      f"gemini {a['txt']}", col, f_sm)

            n_found, n_tot = len(hits), len(turns)
            bar = f"{mode}   t={t:04.1f}s  frame {fid}   found {n_found}/{n_tot} right turns"
            d.rectangle((0, 0, W, 26), fill=INK)
            d.text((8, 5), bar, fill=(255, 255, 255), font=f_big)
            img.save(fp)

        out = _HERE / f"annotated_{mode}{args.suffix}.mp4"
        subprocess.run(["ffmpeg", "-y", "-framerate", str(FPS), "-i",
                        str(work / "f%04d.png"), "-c:v", "libx264", "-crf", "18",
                        "-pix_fmt", "yuv420p", str(out)], check=True, capture_output=True)
        shutil.rmtree(work)
        miss = sorted(set(turns) - hits)
        print(f"{out.relative_to(_REPO)}  found {sorted(hits)}  missed {miss}")


if __name__ == "__main__":
    main()
