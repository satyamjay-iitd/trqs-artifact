# /// script
# requires-python = ">=3.12"
# dependencies = ["duckdb>=1.5.6", "requests"]
# ///
"""Ask Gemini which cars turn right, and score it against the specification.

Two runs of one model over one clip: Gemini's `static` video processing
(fixed sampling, here at the clip's native 5 fps) and its `agentic` processing
(the model picks its own rate). Same file for both -- agentic mode supports
neither clipping nor an fps knob, so feeding static a `start_offset` and agentic
a pre-cut file would vary the input as well as the mode.

`--clip`/`--gt`/`--out` select the window, so the same harness runs the 30 s cell
and the 10 min scaling cell. The clip's length is read from the ground-truth
window and substituted into the prompt, since telling the model "30-second" over
a ten-minute video would be a confound of its own.

Scoring is per object, which is the point of the experiment: a model answer is a
box plus an interval, the box is matched to a tracker id by IoU, and only then is
the interval compared. That is what lets the result say "it found 2 of 5 and
invented 3" rather than just "the timings were off".

`GEMINI_API_KEY` comes from the environment and is never written anywhere.

    uv run suite/vlm/bench_gemini.py --modes static agentic
    uv run suite/vlm/bench_gemini.py --report

    uv run suite/vlm/bench_gemini.py \
        --clip clip/uav0006_180_780.mp4 --gt gt_180_780.json \
        --out results-vlm-600s.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import duckdb
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import PROCESSED, ROOT, VLM

_REPO = ROOT
_HERE = VLM
_DB = PROCESSED / "ytstream/uav_0006/duckdb.db"
_API = "https://generativelanguage.googleapis.com"
MODEL = "gemini-4-argon"
W, H, FPS = 960, 540, 5
IOU_MIN = 0.3
SEARCH = 10          # +/- frames to look for the box, so a wrong clock is not a miss

DEFAULT_EVENT = "every car that makes a right turn"

PROMPT_TMPL = """This is a {length} aerial video of road traffic recorded at 5 frames per second.

Identify {event} during the video.

Return ONLY a JSON array and nothing else. Each element must be:
{{"box_2d": [ymin, xmin, ymax, xmax], "start": "MM:SS", "end": "MM:SS"}}

- box_2d locates that car at the moment its turn begins, in normalised 0-1000 coordinates.
- start and end bound the turn itself, not the car's whole time on screen.
- If no car does this, return [].
"""


def prompt_for(seconds: int, event: str = DEFAULT_EVENT) -> str:
    """The prompt with the clip's real length and the question spelled out.

    The length is substituted rather than hardcoded because telling the model
    "30-second" over a ten-minute video would be a confound of its own; the
    event is substituted so a second query is a new argument, not a new file.
    """
    length = (f"{seconds}-second" if seconds < 120
              else f"{seconds // 60}-minute" if seconds % 60 == 0
              else f"{seconds // 60}-minute {seconds % 60}-second")
    return PROMPT_TMPL.format(length=length, event=event)


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #
def upload(key: str, path: Path) -> str:
    """Files API resumable upload; returns the file uri once ACTIVE."""
    n = path.stat().st_size
    start = requests.post(
        f"{_API}/upload/v1beta/files?key={key}",
        headers={"X-Goog-Upload-Protocol": "resumable",
                 "X-Goog-Upload-Command": "start",
                 "X-Goog-Upload-Header-Content-Length": str(n),
                 "X-Goog-Upload-Header-Content-Type": "video/mp4",
                 "Content-Type": "application/json"},
        json={"file": {"display_name": path.name}}, timeout=60)
    start.raise_for_status()
    url = start.headers["X-Goog-Upload-URL"]
    up = requests.post(
        url, headers={"Content-Length": str(n), "X-Goog-Upload-Offset": "0",
                      "X-Goog-Upload-Command": "upload, finalize"},
        data=path.read_bytes(), timeout=600)
    up.raise_for_status()
    info = up.json()["file"]
    name, uri = info["name"], info["uri"]
    for _ in range(120):                      # wait for server-side processing
        st = requests.get(f"{_API}/v1beta/{name}?key={key}", timeout=30).json()
        if st.get("state") == "ACTIVE":
            return uri
        if st.get("state") == "FAILED":
            sys.exit(f"upload failed: {st}")
        time.sleep(2)
    sys.exit("file never became ACTIVE")


def ask(key: str, uri: str, mode: str, prompt: str) -> dict:
    """One generateContent call. Returns timing, usage and the raw response."""
    # `media_processing` is a sibling of `video_metadata` on the part, carrying
    # the STATIC/AGENTIC enum; the docs' `processing: {type, fps}` object is not
    # a field this API accepts anywhere. `fps` belongs to `video_metadata` and
    # applies to STATIC only -- agentic picks its own rate.
    part = {"file_data": {"mime_type": "video/mp4", "file_uri": uri},
            "media_processing": "STATIC" if mode == "static" else "AGENTIC"}
    if mode == "static":
        part["video_metadata"] = {"fps": FPS}
    body = {"contents": [{"parts": [part, {"text": prompt}]}]}
    t0 = time.perf_counter()
    r = requests.post(f"{_API}/v1beta/models/{MODEL}:generateContent?key={key}",
                      json=body, timeout=1800)
    ms = (time.perf_counter() - t0) * 1000.0
    out = {"mode": mode, "latency_ms": ms, "http_status": r.status_code,
           "request_part": {k: v for k, v in part.items() if k != "file_data"}}
    try:
        j = r.json()
    except ValueError:
        out["error"] = r.text[:2000]
        return out
    if r.status_code != 200:
        out["error"] = json.dumps(j)[:2000]
        return out
    out["usage"] = j.get("usageMetadata", {})
    out["model_version"] = j.get("modelVersion")
    out["finish_reason"] = j.get("candidates", [{}])[0].get("finishReason")
    # Agentic replies lead with a thoughtSignature part, so take the first part
    # that actually carries text rather than assuming it is the first one.
    parts = j.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    text = next((p["text"] for p in parts if isinstance(p, dict) and "text" in p), None)
    if text is None:
        out["error"] = json.dumps(j)[:2000]
    else:
        out["text"] = text
    return out


def parse(text: str) -> tuple[list[dict] | None, str]:
    """First balanced JSON array in the reply; a failure is itself a result."""
    s = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(), flags=re.M)
    i = s.find("[")
    if i < 0:
        return None, "no JSON array in reply"
    depth = 0
    for j, ch in enumerate(s[i:], i):
        depth += (ch == "[") - (ch == "]")
        if depth == 0:
            try:
                v = json.loads(s[i:j + 1])
            except json.JSONDecodeError as e:
                return None, f"array did not parse: {e}"
            return (v, "") if isinstance(v, list) else (None, "not a list")
    return None, "unbalanced array"


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _iou(a, b) -> float:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _mmss(v) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    m = re.match(r"^\s*(\d+):(\d+(?:\.\d+)?)\s*$", str(v))
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    try:
        return float(str(v).rstrip("s"))
    except ValueError:
        return None


def score(answers: list[dict], gt: dict) -> dict:
    """Match each answer to a tracker id by box IoU, then compare intervals."""
    f0 = gt["window_frames"][0]
    conn = duckdb.connect(str(_DB), read_only=True)
    by_frame: dict[int, list] = {}
    for fid, oid, x1, y1, x2, y2 in conn.execute(
            "SELECT fid, oid, x1, y1, x2, y2 FROM bbox WHERE fid BETWEEN ? AND ?",
            gt["window_frames"]).fetchall():
        by_frame.setdefault(fid, []).append((oid, (x1, y1, x2, y2)))
    conn.close()

    gt_ivl = {t["oid"]: (t["turn_start_frame"], t["turn_end_frame"]) for t in gt["turns"]}
    matched: dict[int, dict] = {}
    scored = []
    for a in answers:
        rec = {"answer": a}
        box, st, en = a.get("box_2d"), _mmss(a.get("start")), _mmss(a.get("end"))
        if not (isinstance(box, list) and len(box) == 4) or st is None:
            rec["outcome"] = "unusable"
            scored.append(rec)
            continue
        ymin, xmin, ymax, xmax = box            # Gemini order, 0-1000 normalised
        px = (xmin / 1000 * W, ymin / 1000 * H, xmax / 1000 * W, ymax / 1000 * H)
        want = f0 + round(st * FPS)
        best = (0.0, None, None)
        for d in range(-SEARCH, SEARCH + 1):    # a wrong clock must not look like a lie
            for oid, b in by_frame.get(want + d, []):
                v = _iou(px, b)
                if v > best[0]:
                    best = (v, oid, d)
        iou, oid, off = best
        rec.update(box_iou=round(iou, 3), matched_oid=oid, frame_offset=off)
        if oid is None or iou < IOU_MIN:
            rec["outcome"] = "no object matched"
        elif oid in gt_ivl:
            gs, ge = gt_ivl[oid]
            ae = f0 + round((en if en is not None else st) * FPS)
            inter = max(0, min(ge, ae) - max(gs, want) + 1)
            union = max(ge, ae) - min(gs, want) + 1
            rec.update(outcome="hit", interval_iou=round(inter / union, 3) if union else 0.0,
                       clock_error_s=round((want - gs) / FPS, 1))
            prev = matched.get(oid)
            if prev is None or rec["interval_iou"] > prev.get("interval_iou", 0):
                matched[oid] = rec
        else:
            rec["outcome"] = "wrong car"
        scored.append(rec)

    tp = len(matched)
    fp = sum(1 for r in scored if r["outcome"] != "hit") + max(0, len(
        [r for r in scored if r["outcome"] == "hit"]) - tp)
    missed = sorted(set(gt_ivl) - set(matched))
    return {"answers": len(answers), "true_positives": tp, "false_positives": fp,
            "missed_oids": missed,
            "precision": round(tp / len(answers), 3) if answers else 0.0,
            "recall": round(tp / len(gt_ivl), 3) if gt_ivl else 0.0,
            "mean_interval_iou": round(
                sum(r["interval_iou"] for r in matched.values()) / tp, 3) if tp else 0.0,
            "detail": scored}


# --------------------------------------------------------------------------- #
def report(path: Path) -> None:
    res = json.loads(path.read_text())
    gt = res["ground_truth"]
    w = gt["window_s"]
    print(f"clip {res['clip']}   {w[1] - w[0]} s   ground truth "
          f"{len(gt['turns'])} right turns among {res['objects_in_window']} cars")
    print(f"{'mode':10} {'answers':>8} {'TP':>4} {'FP':>4} {'prec':>6} {'rec':>6} "
          f"{'ivl IoU':>8} {'latency':>9} {'tokens':>8}")
    for r in res["runs"]:
        s = r.get("score")
        if s is None:
            print(f"{r['mode']:10} {'-- ' + r.get('error', 'failed')[:60]}")
            continue
        tok = r.get("usage", {}).get("totalTokenCount", "-")
        print(f"{r['mode']:10} {s['answers']:>8} {s['true_positives']:>4} "
              f"{s['false_positives']:>4} {s['precision']:>6.3f} {s['recall']:>6.3f} "
              f"{s['mean_interval_iou']:>8.3f} {r['latency_ms'] / 1000:>8.1f}s {tok:>8}")
        if s["missed_oids"]:
            print(f"{'':10} missed: {', '.join(map(str, s['missed_oids']))}")
    print(f"\nTrQS on the same window: {res['trqs_ms']:.1f} ms, exact by construction "
          f"(the query defines the event)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["static", "agentic"],
                    choices=["static", "agentic"])
    ap.add_argument("--clip", type=Path, default=Path("clip/uav0006_180_210.mp4"),
                    help="clip to send, relative to the vlm/ directory")
    ap.add_argument("--gt", type=Path, default=Path("gt.json"),
                    help="ground truth JSON, relative to the vlm/ directory")
    ap.add_argument("--event", default=DEFAULT_EVENT,
                    help="the event to ask for, as a noun phrase")
    ap.add_argument("--out", type=Path, default=_HERE / "results-vlm.json")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--trqs_ms", type=float, default=3.2,
                    help="measured TrQS latency for this window, for the table")
    args = ap.parse_args()
    if args.report:
        report(args.out)
        return

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("set GEMINI_API_KEY in the environment (it is never stored)")
    clip = args.clip if args.clip.is_absolute() else _HERE / args.clip
    gt = json.loads((args.gt if args.gt.is_absolute() else _HERE / args.gt).read_text())
    secs = gt["window_s"][1] - gt["window_s"][0]
    prompt = prompt_for(secs, args.event)
    conn = duckdb.connect(str(_DB), read_only=True)
    n_obj = conn.execute(
        "SELECT count(DISTINCT oid) FROM bbox WHERE fid BETWEEN ? AND ?",
        gt["window_frames"]).fetchone()[0]
    conn.close()

    print(f"uploading {clip.name} ({clip.stat().st_size / 2**20:.1f} MB) ...")
    uri = upload(key, clip)
    runs = []
    for mode in args.modes:
        print(f"  {mode} ...", flush=True)
        r = ask(key, uri, mode, prompt)
        if "text" in r:
            ans, err = parse(r["text"])
            if ans is None:
                r["parse_error"] = err
            else:
                r["parsed"] = ans
                r["score"] = score(ans, gt)
        runs.append(r)

    out = {"model": MODEL, "clip": str(clip.relative_to(_REPO)),
           "prompt": prompt, "objects_in_window": n_obj,
           "event": args.event,
           "ground_truth": gt, "trqs_ms": args.trqs_ms, "runs": runs}
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    rel = args.out.resolve()
    rel = rel.relative_to(_REPO) if rel.is_relative_to(_REPO) else rel
    print(f"\nwrote {rel}")
    report(args.out)


if __name__ == "__main__":
    main()
