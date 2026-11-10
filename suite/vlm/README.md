# Video-language-model comparison

Two right-turn queries on `ytstream/uav_0006`, compared with recorded Gemini
STATIC and AGENTIC responses. The windows start at 180 seconds and last 30,
600, or 900 seconds. The video is 960×540 at 5 FPS. STATIC uses native 5 FPS;
AGENTIC chooses its sampling rate. Recorded responses identify `gemini-4-argon`.

## Data used

The experiment uses one video: scene `uav`, video `0006` of the YTStream
dataset from [MIRIS](https://favyen.com/miris/) (Bastani et al., SIGMOD 2020),
downloadable as <https://favyen.com/miris/ytstream-dataset.zip>. Nothing else
from the dataset is used. Two files are taken from the archive:

| File in the archive | What it is | Place it at |
|---|---|---|
| `uav/videos/0006.mp4` | the video, 960×540 at 5 FPS, 5,880 frames (19 min 36 s) | `datasets/raw/ytstream/uav/videos/0006.mp4` |
| `uav/json/0006-baseline.json` | the dataset's own car tracks for that video | `datasets/raw/ytstream/uav/json/0006-baseline.json` |

The queries run over the whole video. The model comparison uses windows of it
that start at 180 s (3:00) and last 30 s, 10 min or 15 min.

The tracks are the dataset's own and are ingested as they are; the artifact
runs no detector or tracker for this experiment. They become 3,014 cars and
128,495 boxes, rescaled from the 1920×1080 annotation space to the video's
960×540. The included artifact in [`data/`](data/README.md) is built from
`0006-baseline.json` alone, so the queries can be run without downloading
anything. The video is needed only for the model calls and the region overlay.
Use the data under the dataset's terms; see `README.txt` in the archive.

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

## Ingestion cost

The engine's first query on a video is charged its ingestion: detection,
tracking and fitting. The artifact reports the recorded values and does not
re-measure them:

| Window | Frames | Ingestion |
|---|---|---|
| 30 s | 151 | 5.65 s |
| 10 min | 3,001 | 49.43 s |
| 15 min | 4,501 | 73.01 s |

They were measured once with YOLO11x and BoT-SORT over each clip. The pretrained
detector finds few cars in this overhead view, so tracker association time was
measured by replaying the annotated boxes and added. The values are an
**estimated cost**; they are not evidence that this pipeline produced the
annotated tracks. Ingestion is charged once per video, and later queries reuse
the tracks. Amortization beyond the two measured queries is extrapolated. The
plots (`INGEST_S` in `plotter/src/benchplot/plots/vlm.py`) and `scaling.py`
carry the same values.

## View recorded results

No API key or video is needed to report the saved results. From the repository root:

```bash
export TRQS_VLM_DIR="$PWD/plotter/data/vlm"
uv run suite/vlm/scaling.py
```

For Figure 6 and other plots, follow the
[plotter README](../../plotter/README.md). It reads `plotter/data/vlm/` by default.

## Run the queries

The built artifact the experiment queried is included:
`data/ytstream/uav_0006/` holds `10_0.trql` (`--max-gap 10 --max-error-px 0`),
the converted tracks in `duckdb.db`, and `metadata.json`. The engine needs
`duckdb.db` beside the `.trql` to recognise the sub-dataset. No download or
conversion is needed. After [building the engine](../../README.md#building-and-preparing-inputs),
from the repository root:

```bash
CLI=engine/target/release/trql-cli
ARGS="--data-dir suite/vlm/data --dataset ytstream --sub-dataset uav_0006 \
  --max-gap 10 --max-error-px 0"
mkdir -p results/vlm
for q in suite/vlm/queries/*.trql; do
  $CLI $ARGS run "$(cat "$q")" "results/vlm/$(basename "$q" .trql).json"
done
```

Each run prints its latency (the recorded `trqs_ms` is 3.2 ms for Q1 and 6.3 ms
for Q2) and writes the matching (car, interval) tuples: 78 cars for Q1 and 22
for Q2 over the whole video. `run` only reads the directory.

To rebuild the artifact from the source instead, place the two files above,
then convert the video and build `10_0.trql`. The converter converts every
video under `--raw`, so only `uav_0006` is produced. It reproduces the included
`duckdb.db` row for row. A `.trql` built by the current engine selects the same
cars; some intervals' recorded bounds differ from the included file, which an
earlier engine version built.

```bash
.trqs-pyenv/bin/python converters/ytstream_convert.py \
  --raw datasets/raw/ytstream --out datasets/processed
engine/target/release/trql-cli --data-dir datasets/processed --dataset ytstream \
  --sub-dataset uav_0006 --max-gap 10 --max-error-px 0 dataset build
```

Then run the loop above with `--data-dir datasets/processed`.

## Optional: model calls

Live Gemini calls require `GEMINI_API_KEY` and access to the configured model.
They are unnecessary for viewing the recorded experiment. They read clipped
windows and frozen labels. These are prepared from the source video, so the
calls need the converted `datasets/processed/ytstream/uav_0006` (with its
`video.mp4`) rather than the included artifact:

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

uv run suite/vlm/bench_gemini.py --modes static agentic
uv run suite/vlm/bench_gemini.py --report
```

`clip.py` re-encodes with FFmpeg and verifies the frame count: 150, 3,000, or
4,500 frames. `groundtruth.py --query queries/UAV-rightturn-top.trql` selects Q2;
`--start` and `--end` select another window.

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
