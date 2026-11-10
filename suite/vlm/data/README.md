# Included artifact: `ytstream/uav_0006`

The built sub-dataset the VLM comparison's TrQL queries run on, included so they
can be run without downloading the source.

| File | Contents |
|---|---|
| `ytstream/uav_0006/10_0.trql` | the fitted trajectories the experiment queried (`--max-gap 10 --max-error-px 0`) |
| `ytstream/uav_0006/duckdb.db` | the converted tracks: `objects` (3,014 cars) and `bbox` (128,495 boxes) |
| `ytstream/uav_0006/metadata.json` | frame rate, size, frame count and conversion notes |

**Source.** Scene `uav`, video `0006` of the YTStream dataset from
[MIRIS](https://favyen.com/miris/) (Bastani et al., *MIRIS: Fast Object Track
Queries in Video*, SIGMOD 2020). Archive:
<https://favyen.com/miris/ytstream-dataset.zip>. Only
`uav/json/0006-baseline.json`, the dataset's own tracks for that video, went into
these files. The video itself (`uav/videos/0006.mp4`) is not included.

**How it was made.** `converters/ytstream_convert.py` with the default
`--ann-size 1920x1080` (boxes rescaled by 0.5 to the 960×540 video, class `car`
for the whole scene), then `trql-cli ... --max-gap 10 --max-error-px 0 dataset
build`. Re-running the converter on the archive's files reproduces `duckdb.db`
row for row; see the [VLM README](../README.md#run-the-queries).

Use the data under the dataset's terms; see `README.txt` in the archive.
