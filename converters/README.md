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
| `nuscenes_convert.py` | `nuscenes` | `v1.0-trainval/*.json` annotation tables (the metadata download; no images or maps) |
| `opendd_convert.py` | `openDD` | Original roundabout SQLite trajectory databases |
| `soccernet_convert.py` | `soccernet` | Tracking sequences: GT, game/sequence INI files and frames |
| `stanford_drone_convert.py` | `StanfordDroneDataset` | `annotations/<scene>/<video>/annotations.txt` and corresponding videos |
| `trackid3x3_convert.py` | `TrackID3x3` | Indoor/Outdoor/Drone MOT annotations and corresponding videos |
| `virat_convert.py` | `virat` | Object and activity mapping annotations, videos |
| `visdrone_convert.py` | `VisDrone` | MOT annotations and sequence JPEGs |
| `ytstream_convert.py` | `ytstream` | Scene baseline JSON annotations and videos |

The converters support datasets beyond the paper's main workload. SoccerNet tracking
is different from the paper's SoccerTrack v2 panorama input. The SoccerTrack v2
converter is not included.

`nuscenes_convert.py` writes one map-space sub-dataset per scene from the
annotation tables alone, with no video. It re-implements the rules recorded with
the original processed data, whose converter is not available: on the 467
boston-seaport scenes it gives the same frames and objects, 93.5% of boxes
identical, 96.5% within 2 cm, and 0.7% more boxes. Convert only the paper's map
with `--location boston-seaport`.

## Getting the datasets

No dataset is redistributed here. Download each one from its publisher, accept
its terms, and unpack it under `datasets/raw/<directory>` in the layout the
converter's docstring describes. Several require registration or a signed
agreement, so request access early: approval can take days.

### Datasets behind the paper's figures

| Dataset | Query (figure) | Source | Download | `datasets/raw/` |
|---|---|---|---|---|
| CLEVRER | CL-overtake (5, 8) | <http://clevrer.csail.mit.edu/> | `obj_clevrer.csv` and `video_train.zip` (train split only; extract into `video_train/`) | `clevrer` |
| UA-DETRAC | DETRAC-meet (4a) | Authors' page, <https://sites.google.com/view/daweidu/projects/ua-detrac> (Google Drive links) | `DETRAC-Images` and the train/test `DETRAC-*-Annotations-XML`. After conversion, `suite/build_detrac_px.py` builds the `detrac_px` copy the figures read | `detrac` |
| I-24 MOTION (INCEPTION v1.0.0) | I24-convoy, I24-journey4 (5, 8) | <https://i24motion.org/data> | The trajectory JSON exports (`<hash>__post1.json`, ...; ~19 GB each) | `I24_Motion` |
| VIRAT Video Dataset, Ground release 2.0 | VIRAT-meet (4a) | <https://viratdata.org/>; zip files on <https://data.kitware.com/> | Videos and `annotations/*.viratdata.objects.txt` (and `mapping.txt`) | `virat` |
| Stanford Drone Dataset | SD-carpass (5) | <https://cvgl.stanford.edu/projects/uav_data/> | `video/` and `annotations/` for all 60 clips | `StanfordDroneDataset` |
| nuScenes v1.0 | NS-crossing (5, 8) | <https://www.nuscenes.org/nuscenes#download> | Only the metadata, `v1.0-trainval_meta.tgz`; no images, lidar or maps | `nuscenes` |
| SoccerTrack v2 | ST-swap-img (5) | <https://huggingface.co/datasets/atomscott/soccertrack-v2> | `gsr/`, `bas/` and `raw/<match>/` for all 10 matches (videos not needed) | no converter yet |

### Other supported datasets

| Dataset | Source | `datasets/raw/` |
|---|---|---|
| AI City Challenge 2022, Track 1 (CityFlowV2 MTMC) | Request form via <https://www.aicitychallenge.org/> | `AICity22_Track1_MTMC_Tracking` |
| Alfheim (Tromsø IL vs Tottenham) | <https://datasets.simula.no/alfheim/> | `alfheim_Totten` |
| Argoverse 2 Sensor | <https://www.argoverse.org/av2.html>; `converters/argoverse2/fetch_logs.py` downloads from the public S3 bucket | `argoverse2` |
| CityFlow-NL | <https://github.com/fredfung007/cityflow-nl>; videos from the AI City form above | `cityflow` |
| I24-3D | <https://github.com/I24-MOTION/I24-3D-dataset> | `I24-3D` |
| I-24V (WACV 2024) | I-24 MOTION project, <https://i24motion.org/data>; utilities at <https://github.com/I24-MOTION/i24-video-dataset-utils> | `I24V_WACV2024` |
| MEVA | <https://mevadata.org/> | `meva` |
| MOT17 | <https://motchallenge.net/data/MOT17/> | `mot17` |
| openDD | <https://l3pilot.eu/data/opendd.html> | `openDD` |
| SoccerNet-Tracking | <https://www.soccer-net.org/data> (`SoccerNet` pip package) | `soccernet` |
| TrackID3x3 | <https://github.com/open-starlab/TrackID3x3> (Google Drive link in its README) | `TrackID3x3` |
| VisDrone2019-MOT | <https://github.com/VisDrone/VisDrone-Dataset> | `VisDrone` |
| YTStream (MIRIS) | <https://favyen.com/miris/ytstream-dataset.zip> | `ytstream` |

## Merged sub-datasets used by the paper

Several experiments run on merges of converted sub-datasets. After converting
their sources, build them with:

```sh
converters/paper_merges.sh --list      # members of each merge
converters/paper_merges.sh             # build every merge whose sources exist
```

| Merge | Output | Members |
|---|---|---|
| `clevrer-10k` | `clevrer_merged/video_00000+9999more` | `clevrer/video_00000` to `video_09999` |
| `clevrer-100` | `clevrer_merged/video_00000+99more` | `clevrer/video_00000` to `video_00099` |
| `sdd-60` | `merged/bookstore_video0+59more` | all 60 `stanford_drone` clips |
| `nuscenes-boston` | `nuscenes_maps/boston-seaport` | all 467 boston-seaport scenes |
| `soccertrack-20` | `merged/SoccerTrackPano-117092-1st+19more` | 20 SoccerTrack v2 panorama halves (no converter) |

The memberships match those recorded in the original merges. The CLEVRER and
Stanford Drone merges rebuild byte-identical tables from the same sources.

The converters must use the pinned `duckdb==1.5.5` from `requirements.txt`. A
newer DuckDB writes database files the engine's DuckDB 1.5.5 cannot read.

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
