# Can a video LM answer "which cars turned right, and when?"

One query, one aerial clip, one frontier model in its two video-processing
modes, scored against a stated specification. The query asks for **(object,
interval) pairs**, which is where a VLM has no mechanism: it carries no
persistent object identity across frames.

## Setup

* **Video** `ytstream/uav_0006` — 960x540, 5 fps, 19.6 min, every annotated
  object is class `car`. `frame = second x 5` exactly, so model timestamps map
  to annotation frames with no resampling.
* **Windows** 3:00-3:30 (30 s) and 3:00-13:00 (10 min), cut with an ffmpeg
  re-encode and verified frame-exact (150 and 3000 frames). A stream copy would
  snap to a keyframe and silently shift the window.
* **Model** `gemini-4-argon`, confirmed from the `modelVersion` echo in each
  response, not assumed from the request.
* **Modes** `STATIC` at the clip's native 5 fps (not the 1 fps default, which
  would let any miss be blamed on sampling) and `AGENTIC`, which picks its own
  rate.
* **Ground truth** is `queries/UAV-rightturn.trql`, which *defines* the event:

  ```
  REGIONS (approach = [230, 280, 340, 540], exit_right = [800, 40, 960, 130])
  SELECT (v: class=car)
  WHERE DURATION(v.center INSIDE approach, 3) THEN DURATION(v.center INSIDE exit_right, 3)
  ```

  A turn counts only if the whole manoeuvre -- **last** frame in `approach` to
  **first** frame in `exit_right` -- completes inside the window. Not first
  entry: cars queue in the approach lane for up to a minute, and the dwell is
  not the turn.

## Results

| window | mode | turns | cars | answers | TP | FP | precision | recall | interval IoU | latency | tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 s | static | 5 | 125 | 7 | 2 | 5 | 0.286 | 0.400 | 0.125 | 13.0 s | 12,057 |
| 30 s | agentic | 5 | 125 | 2 | 2 | 0 | 1.000 | 0.400 | 0.366 | 23.7 s | 15,218 |
| 30 s | **TrQS** | 5 | 125 | 5 | 5 | 0 | 1.000 | 1.000 | 1.000 | **5.3 ms** | -- |
| 10 min | static | 50 | 1529 | 45 | 10 | 35 | 0.222 | 0.200 | 0.453 | 56.1 s | 233,304 |
| 10 min | agentic | 50 | 1529 | 46 | 29 | 17 | 0.630 | 0.580 | 0.388 | 146.1 s | 233,446 |
| 10 min | **TrQS** | 50 | 1529 | 50 | 50 | 0 | 1.000 | 1.000 | 1.000 | **5.3 ms** | -- |

TrQS's accuracy is 1.0 **by construction** and is not a result: the TrQL query
is the specification. Only its latency and the fact that the event is
expressible at all are results. The 5.3 ms is five warm runs over the whole
19.6 min video (78 turns, 23 MB peak), so one query already answers both
windows -- it is an upper bound for either, not a per-window cost.

## What the numbers say

**The failure is object identity, not temporal coverage.** Answers are spread
evenly across the ten minutes and their per-minute counts track ground truth
(gt `[9,0,8,2,4,9,2,8,6,2]`, agentic `[6,3,7,5,2,8,1,6,5,3]`): there is no
attention collapse onto the opening. The errors are elsewhere -- 30 of 45
static answers and 14 of 46 agentic answers matched a *real car that did not
turn*. The model finds the place where turning happens and then names the wrong
vehicle.

**Those false positives are real, not ground-truth gaps.** Every one of the 44
wrongly-named cars has a net heading change under 60 degrees over its track:
they drove straight. None is a turn the regions missed.

**Agentic mode is the more accurate and much the more expensive.** At ten
minutes it more than doubles static's recall (0.580 vs 0.200) and nearly
triples its precision, and its timestamps are far tighter (clock error within
+/-1.6 s, against up to -6.2 s for static). It pays 146 s against 56 s.

**Agentic's token saving does not appear at this length.** Both modes cost
~233 k tokens; agentic merely moves the video out of `promptTokenCount` (534)
into `toolUsePromptTokenCount` (220,871, of which 206,700 is video) and adds
9,017 thinking tokens.

**Honest note on recall direction.** Agentic recall *rises* from 0.400 at 30 s
to 0.580 at 10 min. The 30 s cell is near-uninformative: five ground-truth
turns quantise recall into steps of 0.2, and the model answered with only two
boxes. The 10 min cell -- 50 turns among 1529 cars -- is the one to quote. What
scales badly is cost and precision, not recall.

**The gap.** Against agentic at ten minutes, TrQS is ~27,600x faster and exact,
and the one query covers the entire 19.6 min video rather than a 10 min window.

## Ingest: what TrQS pays to see the video at all

The VLM reads pixels, so charging TrQS only its 5.3 ms query time would put two
different things on one axis. `ingest_cost.py` measures the other side --
detection and tracking over the same clips, with the largest YOLO11 checkpoint
in the tree (`yolo11x.pt`), through `scripts/detect_track.py`, the worker the
ingest pipeline actually uses.

| window | frames | pipeline | association | **ingest** | per frame | vs real time |
|---|---:|---:|---:|---:|---:|---:|
| 30 s | 151 | 5.51 s | 0.95 ms/frame | **5.65 s** | 37.4 ms | 5.3x |
| 10 min | 3001 | 46.04 s | 1.13 ms/frame | **49.43 s** | 16.5 ms | 12.1x |
| 15 min | 4501 | 67.67 s | 1.19 ms/frame | **73.01 s** | 16.2 ms | 12.3x |

Two terms, because one cannot be measured end-to-end here. **COCO-pretrained
YOLO detects essentially nothing in a top-down aerial view** -- it has no
overhead-vehicle imagery, and a sweep over `imgsz` 640-1920 and `conf`
0.25/0.10 returns zero cars on a frame that visibly holds twenty. So the
end-to-end run does the detector's work but none of the tracker's association
work. Replaying the dataset's own boxes through the same BoT-SORT recovers that
term at the true density (20-23 boxes/frame), rather than silently omitting it.
It is small -- about 1 ms against the detector's ~15 -- but measured rather than
assumed.

The per-frame cost falls from 36.7 ms to 16.6 ms between the two cells because
model load and warm-up are fixed and amortise; the steady-state rate is
~15 ms/frame.

This makes the ingest number an **estimate**, and the script says so: it is the
right figure for a cost axis, not a measured pipeline that produced these
particular tracks. The tracks themselves are MIRIS ground-truth annotation --
`bbox` holds fractional coordinates, which is why they are not detector output.

## The figure

`plotter/plots/vlm.pdf`, drawn by the `vlm` plot kind from
`results-vlm.json`, `results-vlm-600s.json` and `ingest.json`:

```bash
cd plotter && uv run --locked benchplot render --only vlm
```

F1 against total time, log x, six points, each system's two windows joined.
F1 is computed from counts as `2TP / (2TP + FP + FN)`, so it does not inherit
the rounding in the stored rates.

| window | system | total time | F1 |
|---|---|---:|---:|
| 30 s | static | 12.95 s | 0.333 |
| 30 s | agentic | 23.67 s | 0.571 |
| 30 s | **TrQS** | **5.55 s** | **1.000** |
| 10 min | static | 56.11 s | 0.211 |
| 10 min | agentic | 146.08 s | 0.604 |
| 10 min | **TrQS** | **49.92 s** | **1.000** |

TrQS's x is ingest plus query; ingest dominates so completely that the 5.3 ms
query is invisible on the axis, which is the honest shape of the comparison --
**paid once per video, not once per question.** A second query over the same
clip moves the engine's point to 5.3 ms and leaves both model points where they
are. The figure's `with_ingest = false` draws that variant.

At ten minutes the engine is 2.9x faster than the better of the two modes and
exact, where the model reaches F1 0.604. Again: the 1.0 is **by construction**
-- the caption must say so.

## A second query: does ingestion actually amortise?

Figure 6's caption claims ingestion is "paid once per video, not once per
question." **With one query that claim cannot be tested** — both systems have run
exactly once. So there is a second query on the same video and the same window.

`queries/UAV-rightturn-top.trql` — every car that enters from the top of the
frame and turns right (the car's own right, i.e. westward):

```
REGIONS (from_top = [25, 0, 80, 60], exit_west = [0, 155, 40, 215])
SELECT (v: class=car)
WHERE DURATION(v.center INSIDE from_top, 3) THEN DURATION(v.center INSIDE exit_west, 3)
```

Only the two regions differ from Q1. The flow was found in the data before any
code was written: clustering every track in frames 900-3900 by (entry edge, exit
edge, net heading change) gives a **T -> L, +90 deg** group of **17 tracks**,
entering at median (48, 30), leaving at median (6, 186), median net heading
**+82 deg**. It is distinct from the 75-track T -> L group at net heading ~0,
which is westbound through-traffic along the top road. The region query returns
**exactly those 17**, which is the check that the regions describe the flow
rather than merely selecting some cars, and its cars are **disjoint from Q1's
50**.

### Amortization

| k queries | TrQS | Gemini static | Gemini agentic | vs static | vs agentic |
|---:|---:|---:|---:|---:|---:|
| 1 | 49.92 s | 56.1 s | 146.1 s | 1.1x | 2.9x |
| 2 | **49.92 s** | 87.6 s | 173.1 s | 1.8x | 3.5x |
| 5* | 49.94 s | 219.0 s | 432.7 s | 4.4x | 8.7x |
| 10* | 49.97 s | 438.1 s | 865.3 s | 8.8x | 17.3x |

\* extrapolated from the mean measured per-query cost. Only k = 1 and k = 2 are
measured, and the figure draws the rest dashed.

Two queries cost TrQS 49.92 s against 49.91 s for one: the ingestion, and
essentially nothing else (5.3 ms + 6.3 ms). The model re-reads ten minutes of
video per question and pays again each time.

### Accuracy on Q2: both modes score zero

| mode | answers | TP | FP | precision | recall | F1 | latency | tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| static | 6 | 0 | 6 | 0.000 | 0.000 | **0.000** | 31.5 s | 231,469 |
| agentic | 0 | 0 | 0 | 0.000 | 0.000 | **0.000** | 27.0 s | 68,537 |
| **TrQS** | 17 | 17 | 0 | 1.000 | 1.000 | **1.000** | **6.3 ms** | — |

Both replies are well-formed; this is not a parse failure. The agentic mode
returned a literal `[]` after 27 s and 68 k tokens — it stopped early rather than
answering. The static mode returned six boxes, and they are **not random**: all
six sit in the top-left corner of the frame, which is where the manoeuvre
happens. But every one lands on **empty pavement**, several metres from any
vehicle the tracker holds at that instant — see `q2-miss.png`, which draws the
model's box (orange) against the tracked cars (green) at two of the six
timestamps. Best box IoU over all six answers and a +/-10 frame search is
**0.083**, so no threshold choice rescues them.

This is the same failure as Q1, one step further along: on Q1 the model found
the event and named the wrong car; on Q2 it found the event and named a piece of
road.

**Q2 is the harder question and its F1 is not comparable with Q1's.** Its
manoeuvre lasts a median 3.6 s against Q1's 11.4 s and happens at the edge of
the frame. The amortization claim is about *time* and does not depend on it.

## 15 minutes: the amortization result

The headline cell is the **15 min window, 3:00-18:00**, which contains the 30 s
and 10 min windows so the three are nested rather than independent samples. Q1
has 73 turns and Q2 has 21, among 2170 cars. All four runs were served by
`gemini-4-argon` (confirmed from each reply's `modelVersion`), and the static
mode ran at the clip's native 5 fps throughout -- 351 k tokens, comfortably
inside the context window, so nothing here is a truncation artifact.

| k queries | TrQS | Gemini static | Gemini agentic | vs static | vs agentic |
|---:|---:|---:|---:|---:|---:|
| 1 | 73.02 s | 70.5 s | 73.8 s | 1.0x | 1.0x |
| 2 | **73.02 s** | 137.6 s | 106.4 s | 1.9x | 1.5x |
| 5* | 73.04 s | 344.1 s | 266.1 s | 4.7x | 3.6x |
| 10* | 73.07 s | 688.2 s | 532.2 s | 9.4x | 7.3x |

\* extrapolated from the mean measured per-query cost; only k = 1 and k = 2 are
measured, and the figure draws the rest dashed.

**At one query the three systems are within 5% of each other.** That is the
cleanest form of the result: at this length TrQS has no head start, and the
whole of its advantage is that the 73 s is ingestion, paid once. Two queries
cost it 73.02 s against 73.01 s for one.

### Accuracy at 15 min

| query | static | agentic | TrQS |
|---|---:|---:|---:|
| Q1 (73 turns) | 0.480 | 0.357 | 1.000 |
| Q2 (21 turns) | 0.000 | 0.278 | 1.000 |

The failure mechanism reproduces on the larger sample: on Q1, 38 of 98 static
answers and 18 of 39 agentic answers matched a **real car that did not turn**,
and all 55 of those cars have net heading change below 60 degrees -- they drove
straight. A further 15 static answers matched no tracked object at all. Answers
are spread across all fifteen minutes (gt per minute
`[9,0,8,2,4,9,2,8,6,2,7,1,9,6,0]`, static `[6,6,7,5,2,9,6,11,7,4,12,5,10,7,1]`),
so there is no attention collapse onto the opening.

### Agentic mode is not stable run to run

This matters for how much the length sweep can claim:

| window | mode | tokens | latency | Q1 F1 |
|---|---|---:|---:|---:|
| 10 min | static | 233 k | 56.1 s | 0.211 |
| 10 min | agentic | 233 k | 146.1 s | 0.604 |
| 15 min | static | 351 k | 70.5 s | 0.480 |
| 15 min | agentic | 76 k | 73.8 s | 0.357 |

Between the two windows the agentic mode chose to spend a third as many tokens
on *more* video, its latency halved, and its F1 fell by half while static's
doubled. With one run per cell, **no trend in accuracy or per-query cost across
window length is supportable for the agentic mode**, and the write-up says so
rather than fitting a rate. The amortization claim is unaffected: it compares
systems within a single window, where every number is measured once under the
same conditions.

A `right_panel = "length"` variant of the figure plots cost against video
length; it is not the published default precisely because the agentic line dips
at 15 min, which is this variance and not a scaling effect.

## The per-query bar figure

`plotter/plots/vlmbars.pdf` (`benchplot render --only vlmbars`) is the
whole matrix in one figure: one panel per window, one group per system, two bars
per group for the first and second query, with precision, recall and the time
printed over each bar.

| window | system | 1st query | 2nd query |
|---|---|---|---|
| 30 s | TrQS | 5.7 s, P=1.00 R=1.00 | **6 ms**, P=1.00 R=1.00 |
| 30 s | static | 13.0 s, P=0.29 R=0.40 | 5.4 s, P=0.00 R=0.00 |
| 30 s | agentic | 23.7 s, P=1.00 R=0.40 | 21.8 s, P=1.00 R=0.33 |
| 10 min | TrQS | 49.4 s, P=1.00 R=1.00 | **6 ms**, P=1.00 R=1.00 |
| 10 min | static | 56.1 s, P=0.22 R=0.20 | 31.5 s, P=0.00 R=0.00 |
| 10 min | agentic | 146.1 s, P=0.63 R=0.58 | 27.0 s, P=0.00 R=0.00 |
| 15 min | TrQS | 73.0 s, P=1.00 R=1.00 | **6 ms**, P=1.00 R=1.00 |
| 15 min | static | 70.5 s, P=0.42 R=0.56 | 67.2 s, P=0.00 R=0.00 |
| 15 min | agentic | 73.8 s, P=0.51 R=0.27 | 32.7 s, P=0.33 R=0.24 |

The y axis is logarithmic and has to be: the engine's second query is 6 ms
against the models' tens of seconds, and on a linear axis that bar would be
invisible -- which is the one thing the figure exists to show. The engine's two
bars differ by four orders of magnitude; each model's two bars are the same
order, because a prompt has nothing to re-use.

The engine's P and R are 1.00 **by construction**, so they report
expressibility, not an accuracy win.

Note the 30 s Q2 cell is thin -- only 3 ground-truth turns -- so its P and R
move in steps of 0.33 and should not be read as a measurement of quality.

## Caveats a reviewer will raise

* **Single run per cell.** This is an existence proof, not a distribution. No
  repeats, no fps sweep.
* **The VLM got pixels, TrQS got a tracker.** True; ingestion is a one-time
  cost TrQS amortises across queries, and it is reported separately.
* **The ground truth is our own system's output.** Stated plainly above: the
  query defines the event, the regions were verified by eye, and TrQS's
  accuracy is not claimed as a result.
* **Camera motion.** Ruled out twice: median heading change over frames
  900-1050 is +3.3 deg (44 of 57 tracks within +/-20 deg), and the junction,
  bridge and footpaths sit in the same pixels at 180 s, 480 s and 779 s, so the
  fixed regions stay valid across the whole window.

## Running it

Run from the repository root, with Python 3.13+, uv, FFmpeg, and the CLI built.
Prepare `ytstream/uav_0006` from the raw video and annotations, then build its
`10_0.trql`. Its dataset-specific converter still needs migration into this
repository; the command below assumes that processed input already exists.
Set up a separate working directory so reruns preserve the checked-in labels
and the paper's reference measurements:

```bash
export TRQS_DATA_DIR="$PWD/datasets/processed"
export TRQS_OUTPUT_DIR="$PWD/results/suite"
export TRQS_VLM_DIR="$TRQS_OUTPUT_DIR/vlm"
mkdir -p "$TRQS_VLM_DIR"
cp -r suite/vlm/queries "$TRQS_VLM_DIR/"
cp suite/vlm/gt*.json plotter/data/vlm/*.json "$TRQS_VLM_DIR/"
```

The copied `results-vlm*.json` files can be reported or scored offline. This is
the verification scope for the artifact: live Gemini reruns are excluded.
The model-call commands below document the original experiment and are optional.
Live
model calls require `GEMINI_API_KEY` and access to the configured model. The
`uv run` commands install their declared Python dependencies. The ingestion
command uses the separate `.venv-detect` environment from the root README.
The plotter uses the preserved measurements under `plotter/data/vlm/` by
default; set a plot's `source` to `../results/suite/vlm` to draw new measurements.

```bash
export GEMINI_API_KEY=...        # never written to disk

uv run suite/vlm/clip.py --start 180 --dur 30
uv run suite/vlm/groundtruth.py --start 180 --end 210
uv run suite/vlm/bench_gemini.py --modes static agentic

uv run suite/vlm/clip.py --start 180 --dur 600
uv run suite/vlm/groundtruth.py --start 180 --end 780
uv run suite/vlm/bench_gemini.py --modes static agentic \
    --clip clip/uav0006_180_780.mp4 --gt gt_180_780.json \
    --out results/suite/vlm/results-vlm-600s.json

# the second query, same video and window
uv run suite/vlm/groundtruth.py --start 180 --end 780 \
    --query queries/UAV-rightturn-top.trql       # writes gt_top_180_780.json
uv run suite/vlm/bench_gemini.py --modes static agentic \
    --clip clip/uav0006_180_780.mp4 --gt gt_top_180_780.json --trqs_ms 6.3 \
    --event "every car that enters from the top of the frame and then turns right (the car's own right)" \
    --out results/suite/vlm/results-vlm-top-600s.json

# the 15 min window (both queries), which is what the figure draws
uv run suite/vlm/clip.py --start 180 --dur 900
for q in UAV-rightturn UAV-rightturn-top; do
  uv run suite/vlm/groundtruth.py --start 180 --end 1080 \
      --query queries/$q.trql
done
uv run suite/vlm/bench_gemini.py --modes static agentic \
    --clip clip/uav0006_180_1080.mp4 --gt gt_180_1080.json --trqs_ms 5.3 \
    --out results/suite/vlm/results-vlm-900s.json
uv run suite/vlm/bench_gemini.py --modes static agentic \
    --clip clip/uav0006_180_1080.mp4 --gt gt_top_180_1080.json --trqs_ms 6.3 \
    --event "every car that enters from the top of the frame and then turns right (the car's own right)" \
    --out results/suite/vlm/results-vlm-top-900s.json

uv run suite/vlm/scaling.py        # all cells + amortization

.venv-detect/bin/python suite/vlm/ingest_cost.py   # needs the GPU env
cd plotter && uv run --locked benchplot render --only vlm          # plots/vlm.pdf
cd -
uv run suite/vlm/annotate.py \
    --clip clip/uav0006_180_780.mp4 --gt gt_180_780.json \
    --results results-vlm-600s.json --suffix _600s
```

Prompts, raw responses, parse outcomes and per-answer scoring are all stored in
`results-vlm.json` and `results-vlm-600s.json`.
