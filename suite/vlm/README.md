# Video-language-model comparison

Two right-turn queries on `ytstream/uav_0006`, compared with recorded Gemini
STATIC and AGENTIC responses. The windows start at 180 seconds and last 30,
600, or 900 seconds. The video is 960×540 at 5 FPS. STATIC uses native 5 FPS;
AGENTIC chooses its sampling rate. Recorded responses identify `gemini-4-argon`.

## Queries and scoring

| Query | Definition |
|---|---|
| Q1: `queries/UAV-rightturn.trql` | Approach region followed by the right exit |
| Q2: `queries/UAV-rightturn-top.trql` | Top entrance followed by the west exit |

Each region requires three frames of presence. A turn's reported interval runs
from the last approach frame to the first exit frame and must complete inside
the selected window. `groundtruth.py` derives labels from the annotated tracks
and writes a region overlay for inspection. `bench_gemini.py` matches model
answers to tracks and reports precision, recall, F1, interval IoU, latency, and tokens.

The TrQL queries define the events, so TrQS's perfect scores are by construction;
they are not an independent accuracy result. Each model cell has one recorded
run. The nested windows are not independent samples, and AGENTIC's sampling
varies between runs.

## View recorded results

No API key or video is needed to report the saved results. From the repository root:

```bash
export TRQS_VLM_DIR="$PWD/plotter/data/vlm"
uv run suite/vlm/scaling.py
```

For Figure 6 and other plots, follow the
[plotter README](../../plotter/README.md). It reads `plotter/data/vlm/` by default.

## Prepare inputs and measure ingestion

Follow the root README for [setup](../../README.md#building-and-preparing-inputs)
and the [inference environment](../../README.md#running-detection-and-tracking).
Use `converters/ytstream_convert.py` to prepare `ytstream/uav_0006`, then build
its `10_0.trql` artifact. Run from the repository root:

```bash
export TRQS_DATA_DIR="$PWD/datasets/processed"
export TRQS_OUTPUT_DIR="$PWD/results/suite"
export TRQS_VLM_DIR="$TRQS_OUTPUT_DIR/vlm"
mkdir -p "$TRQS_VLM_DIR"
cp -r suite/vlm/queries "$TRQS_VLM_DIR/"
cp suite/vlm/gt*.json plotter/data/vlm/*.json "$TRQS_VLM_DIR/"

for duration in 30 600 900; do
  uv run suite/vlm/clip.py --start 180 --dur "$duration"
done
uv run suite/vlm/groundtruth.py --start 180 --end 210
.venv-detect/bin/python suite/vlm/ingest_cost.py
```

`clip.py` re-encodes with FFmpeg and verifies the frame count: 150, 3,000, or
4,500 frames. `groundtruth.py --query queries/UAV-rightturn-top.trql` selects Q2;
`--start` and `--end` select another window.

`ingest_cost.py` calls `scripts/detect_track.py` with YOLO11x and BoT-SORT.
The pretrained detector finds few or no cars in this overhead view, so the
script adds tracker association time measured by replaying annotated boxes.
The resulting `ingest.json` is an **estimated cost**, not evidence that the
pipeline produced the annotated tracks. Ingestion is charged once per video;
subsequent queries reuse the tracks. Amortization beyond the two measured
queries is extrapolated.

## Optional model calls

Live Gemini calls require `GEMINI_API_KEY` and access to the configured model.
They are unnecessary for viewing the recorded experiment. With the working
directory and inputs above:

```bash
uv run suite/vlm/bench_gemini.py --modes static agentic
uv run suite/vlm/bench_gemini.py --report
```

For a longer window, pass its clip, labels, and output explicitly:

```bash
uv run suite/vlm/groundtruth.py --start 180 --end 780
uv run suite/vlm/bench_gemini.py --modes static agentic \
  --clip clip/uav0006_180_780.mp4 --gt gt_180_780.json \
  --out "$TRQS_VLM_DIR/results-vlm-600s.json"
```

For Q2, generate labels with `--query queries/UAV-rightturn-top.trql` and pass
`--gt gt_top_180_780.json --trqs_ms 6.3`, an output named
`results-vlm-top-600s.json`, and
`--event "every car that enters from the top of the frame and then turns right (the car's own right)"`.
For 900-second windows, use end time 1080 and the corresponding filenames.
To plot new outputs, set the manifest's `source` directory as described in the
[plotter README](../../plotter/README.md#recorded-inputs-and-new-measurements).
