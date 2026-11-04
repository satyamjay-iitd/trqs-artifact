# Raw dataset conversion

Run commands from the repository root. Obtain the original datasets separately
and retain their published directory layouts; the converters' module docstrings
describe the required files and their coordinate/time conventions. Dataset
licences and access requirements still apply.

```sh
python3 -m venv .trqs-pyenv
.trqs-pyenv/bin/python -m pip install -r converters/requirements.txt
.trqs-pyenv/bin/python converters/detrac_convert.py \
  --raw datasets/raw/detrac --out datasets/processed
```

The converters need `ffmpeg` and `ffprobe` on `PATH`. Substitute another converter
and its raw directory below. All converters accept `--raw` and `--out`, and
`--help` lists the dataset-specific options. Outputs generally contain
`<family>/<subset>/duckdb.db`, `metadata.json`, and a `video.mp4` link or encoded
video. Keep the raw inputs: video links refer back to them. World-coordinate
datasets may have no video or may expose one as a separate viewpoint.

| Converter | Default directory under `datasets/raw/` | Required raw inputs |
|---|---|---|
| `aicity22_convert.py` | `AICity22_Track1_MTMC_Tracking` | Scenario/camera videos and MOT GT; calibration and camera timestamps when available |
| `alfheim_convert.py` | `alfheim_Totten` | Panorama H264 segments or `pano.tgz`, ball tracks, ZXY CSV |
| `argoverse2_convert.py` | `argoverse2` | City-merged trajectories and maps, prepared as below |
| `cityflow_convert.py` | `cityflow` | Track JSON and camera videos under `data/` |
| `clevrer_convert.py` | `clevrer` | `obj_clevrer.csv` and `video_train/` |
| `detrac_convert.py` | `detrac` | UA-DETRAC XML and image sequences |
| `i24_3d_convert.py` | `I24-3D` | Per-scene annotation CSV, homography JSON and camera videos |
| `i24_motion_convert.py` | `I24_Motion` | Original trajectory JSON files |
| `i24v_convert.py` | `I24V_WACV2024` | `final_gps.csv`, `WACV2024_hg_save.cpkl` and camera videos |
| `meva_convert.py` | `meva` | KPF geometry/types/activities YAML and videos |
| `mot17_convert.py` | `mot17` | Training sequences: `gt/gt.txt`, `seqinfo.ini`, `img1/` |
| `opendd_convert.py` | `openDD` | Original roundabout SQLite trajectory databases |
| `soccernet_convert.py` | `soccernet` | Tracking sequences: GT, game/sequence INI files and frames |
| `stanford_drone_convert.py` | `StanfordDroneDataset` | `annotations/<scene>/<video>/annotations.txt` and corresponding videos |
| `trackid3x3_convert.py` | `TrackID3x3` | Indoor/Outdoor/Drone MOT annotations and corresponding videos |
| `virat_convert.py` | `virat` | Object and activity mapping annotations, videos |
| `visdrone_convert.py` | `VisDrone` | MOT annotations and sequence JPEGs |
| `ytstream_convert.py` | `ytstream` | Scene baseline JSON annotations and videos |

The converters support datasets beyond the paper's main workload. SoccerNet tracking
is different from the paper's SoccerTrack v2 panorama input. The SoccerTrack
and nuScenes converters are not included.

## Dataset-specific steps

openDD's `--out` is the family directory itself:

```sh
.trqs-pyenv/bin/python converters/opendd_convert.py \
  --raw datasets/raw/openDD --out datasets/processed/opendd --rdb rdb1 --limit 1
```

Omit `--limit 1` and `--rdb rdb1` for all recordings. The converter computes the
roundabout origin across all recording tables even when limiting output.

For MOT17 and VisDrone detector scoring, run ignored-region postprocessing
after conversion:

```sh
.trqs-pyenv/bin/python converters/mot17_ignored_regions.py \
  --raw datasets/raw/mot17 --processed datasets/processed/mot17
.trqs-pyenv/bin/python converters/visdrone_ignored_regions.py \
  --raw datasets/raw/VisDrone --processed datasets/processed/visdrone
```

DETRAC writes its ignored regions during conversion. These helpers filter
ignored detections without changing ground-truth object identities.

I24-3D and I24V require projection dependencies. Use a separate environment (or
the inference environment already created for detection):

```sh
python3 -m venv .venv-detect
.venv-detect/bin/python -m pip install -r converters/requirements-projection.txt
```

I24V also requires `i24_rcs` 1.0.1. Download its source distribution from the
[upstream project](https://github.com/I24-MOTION/i24_rcs) and install the
downloaded wheel or extracted source directory into the same environment:

```sh
.venv-detect/bin/python -m pip install /path/to/i24_rcs-1.0.1-py3-none-any.whl
.venv-detect/bin/python -c 'from i24_rcs import I24_RCS; from importlib.metadata import version; print(version("i24_rcs"))'
```

Then run the converters:

```sh
.venv-detect/bin/python converters/i24_3d_convert.py \
  --raw datasets/raw/I24-3D --out datasets/processed
.venv-detect/bin/python converters/i24v_convert.py \
  --raw datasets/raw/I24V_WACV2024 --out datasets/processed
```

The I24-3D homography utility is included; I24V imports the installed `i24_rcs` package. I24V
retains the original assumption that GPS timestamps are relative to video frame
zero; its metadata states that assumption. Do not independently trim the video
and retain absolute annotation times when preparing a subset.

Argoverse 2 needs a transform from ego to city coordinates and a city merge
before the common-schema conversion. These stages read raw annotations and
poses; they do not require the AV2 SDK or imagery.

```sh
.trqs-pyenv/bin/python -m pip install -r converters/requirements-argoverse2.txt
.trqs-pyenv/bin/python converters/argoverse2/to_world_trajectories.py --split val
.trqs-pyenv/bin/python converters/argoverse2/merge_by_city.py
.trqs-pyenv/bin/python converters/argoverse2_convert.py \
  --raw datasets/raw/argoverse2 --out datasets/processed
```

Run the transform for `--split train` as well when that split is present. Set
`TRQS_RAW_DIR` to the directory containing `argoverse2/` if inputs are elsewhere;
set the converter's `--raw` to that same `argoverse2/` directory. The optional
`fetch_logs.py` downloads annotations/poses/maps; it is unnecessary when these
raw files are already available. `validate.py` checks city transforms and
`plot_map.py` visualizes a log; neither is a required conversion stage.
