#!/usr/bin/env bash
# Build the merged sub-datasets the paper's experiments run on, with `trql-cli merge`.
#
#   converters/paper_merges.sh --list                  # what each merge is made of
#   converters/paper_merges.sh                         # build every merge whose sources exist
#   converters/paper_merges.sh --only nuscenes-boston  # one of them
#
# Each merge folds converted sub-datasets of one family, in sorted order, into one
# sub-dataset; `merge` offsets `oid` and `fid` per source so ids stay disjoint. The
# memberships below are the ones recorded in the original merges' `merged_from`.
#
#   name               output (under --data-dir)                    members                       used by
#   clevrer-10k        clevrer_merged/video_00000+9999more          clevrer/video_00000..09999    CL-overtake (Figs 5, 8)
#   clevrer-100        clevrer_merged/video_00000+99more            clevrer/video_00000..00099    interval ablation (Fig 9b)
#   sdd-60             merged/bookstore_video0+59more               all 60 stanford_drone clips   SD-carpass (Fig 5)
#   nuscenes-boston    nuscenes_maps/boston-seaport                 all 467 boston-seaport scenes NS-crossing (Figs 5, 8)
#   soccertrack-20     merged/SoccerTrackPano-117092-1st+19more     20 SoccerTrackv2pano halves   ST-swap-img (Fig 5)
#
# Sources come from converters/clevrer_convert.py, stanford_drone_convert.py and
# nuscenes_convert.py. SoccerTrack v2 has no converter in this repository, so its
# merge runs only where `SoccerTrackv2pano/` was prepared some other way.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLI="${TRQL_CLI:-$ROOT/engine/target/release/trql-cli}"
DATA_DIR="${TRQS_DATA_DIR:-$ROOT/datasets/processed}"
ONLY=()
FORCE=0
LIST=0
DRY_RUN=0

usage() {
    sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
    cat <<EOF

Options:
  --data-dir DIR   processed datasets root (default: \$TRQS_DATA_DIR or datasets/processed)
  --only NAME      build only this merge; repeatable (names: ${NAMES[*]})
  --force          rebuild a merge whose output already exists
  --list           print each merge's members and exit
  --dry-run        print the merge commands without running them
EOF
}

NAMES=(clevrer-10k clevrer-100 sdd-60 nuscenes-boston soccertrack-20)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --only) ONLY+=("$2"); shift 2 ;;
        --force) FORCE=1; shift ;;
        --list) LIST=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# --- Recipes: family, output (relative to DATA_DIR), member list on stdout ---
recipe() {
    case "$1" in
        clevrer-10k)
            FAMILY=clevrer; OUT=clevrer_merged/video_00000+9999more
            members() { for i in $(seq 0 9999); do printf 'video_%05d\n' "$i"; done; } ;;
        clevrer-100)
            FAMILY=clevrer; OUT=clevrer_merged/video_00000+99more
            members() { for i in $(seq 0 99); do printf 'video_%05d\n' "$i"; done; } ;;
        sdd-60)
            # Every Stanford Drone clip, i.e. the 60 that stanford_drone_convert.py writes.
            FAMILY=stanford_drone; OUT=merged/bookstore_video0+59more
            members() {
                local s v
                for s in bookstore:7 coupa:4 deathCircle:5 gates:9 hyang:15 little:4 nexus:12 quad:4; do
                    for ((v = 0; v < ${s#*:}; v++)); do echo "${s%:*}_video$v"; done
                done | LC_ALL=C sort
            } ;;
        nuscenes-boston)
            # Every scene on the boston-seaport map, read from each scene's metadata.
            FAMILY=nuscenes; OUT=nuscenes_maps/boston-seaport
            members() {
                python3 - "$DATA_DIR/nuscenes" <<'EOF'
import json, sys
from pathlib import Path
for d in sorted(Path(sys.argv[1]).iterdir()):
    m = d / "metadata.json"
    if not m.is_file():
        m = d / "world-metadata.json"
    if m.is_file() and json.loads(m.read_text()).get("location") == "boston-seaport":
        print(d.name)
EOF
            } ;;
        soccertrack-20)
            FAMILY=SoccerTrackv2pano; OUT=merged/SoccerTrackPano-117092-1st+19more
            members() {
                local h
                for h in 117092 117093 118575 118576 118577 118578 128057 128058 132831 132877; do
                    echo "SoccerTrackv2-$h-1st"; echo "SoccerTrackv2-$h-2nd"
                done
            } ;;
        *) echo "unknown merge: $1 (names: ${NAMES[*]})" >&2; exit 2 ;;
    esac
}

[[ ${#ONLY[@]} -gt 0 ]] && NAMES=("${ONLY[@]}")
if [[ $LIST -eq 0 && $DRY_RUN -eq 0 && ! -x "$CLI" ]]; then
    echo "trql-cli not found at $CLI; build it (see README.md) or set TRQL_CLI" >&2
    exit 1
fi

tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
built=(); skipped=(); failed=()
for name in "${NAMES[@]}"; do
    recipe "$name"
    list="$tmp/$name.txt"
    members > "$list"
    n=$(wc -l < "$list")
    if [[ $LIST -eq 1 ]]; then
        echo "## $name: $n members of $FAMILY -> $OUT"
        sed 's/^/   /' "$list"
        continue
    fi
    if [[ -e "$DATA_DIR/$OUT/duckdb.db" && $FORCE -eq 0 ]]; then
        skipped+=("$name: $OUT exists (--force rebuilds)"); continue
    fi
    absent=$(while read -r m; do [[ -f "$DATA_DIR/$FAMILY/$m/duckdb.db" ]] || echo "$m"; done < "$list" | wc -l)
    if [[ $absent -gt 0 ]]; then
        skipped+=("$name: $absent of $n sources missing under $DATA_DIR/$FAMILY"); continue
    fi
    cmd=("$CLI" --data-dir "$DATA_DIR" --dataset "$FAMILY" merge run --from-file "$list" --out "$DATA_DIR/$OUT")
    if [[ $DRY_RUN -eq 1 ]]; then
        printf '%q ' "${cmd[@]}"; echo; continue
    fi
    echo "==> $name: merging $n $FAMILY sub-datasets into $OUT"
    if [[ $FORCE -eq 1 ]]; then rm -rf "${DATA_DIR:?}/$OUT"; fi
    if "${cmd[@]}"; then built+=("$name"); else failed+=("$name"); fi
done

[[ $LIST -eq 1 ]] && exit 0
echo
[[ ${#built[@]} -gt 0 ]] && printf 'built:   %s\n' "${built[@]}"
[[ ${#skipped[@]} -gt 0 ]] && printf 'skipped: %s\n' "${skipped[@]}"
[[ ${#failed[@]} -gt 0 ]] && { printf 'FAILED:  %s\n' "${failed[@]}"; exit 1; }
exit 0
