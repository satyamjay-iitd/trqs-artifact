#!/usr/bin/env bash
# Re-run the measurements behind the paper's bench.db figures and redraw them.
#
#   ./reproduce_paper.sh --dry-run          # what would run, nothing executed
#   ./reproduce_paper.sh                    # Figs 4a, 5, 8 (compression, p15, epsilon)
#   ./reproduce_paper.sh --plots "p15"      # one figure
#
# The plan is read from plotter/manifest.toml against the recorded bench.db
# (opened read-only): exactly the (query, sub-dataset, budget, gap, scale)
# points each figure draws. See plotter/scripts/repro_plan.py.
#
# The recorded ./bench.db is never written. Every run goes to --out (default
# results/repro/bench.db) via an explicit --bench-db, the script refuses an
# output that is the recorded file, and the recorded file's SHA-256 is checked
# before and after.
#
# Datasets are not included. Points whose sub-dataset is absent from --data-dir
# are skipped and listed at the end; the figures that need them are not drawn.
# Re-running resumes: points already in the output bench.db are not re-measured.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REF_DB="$ROOT/bench.db"
CLI="${TRQL_CLI:-$ROOT/engine/target/release/trql-cli}"

DATA_DIR="${TRQS_DATA_DIR:-$ROOT/datasets/processed}"
OUT="$ROOT/results/repro"
PLOTS="compression p15 epsilon"
RUNS=3
DRY_RUN=0
RESUME=1
STOP_ON_ERROR=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

  --plots "A B ..."   manifest plots to reproduce (default: "$PLOTS")
  --data-dir DIR      processed datasets root (default: \$TRQS_DATA_DIR or datasets/processed)
  --out DIR           output root; bench.db, plots/, logs/ go here (default: results/repro)
  --runs N            query executions per point (default: $RUNS, as recorded)
  --no-resume         re-measure points already present in the output bench.db
  --stop-on-error     stop at the first failing point (default: keep going)
  --dry-run           print the plan and the commands; run nothing
  -h, --help          this help

Environment: TRQL_CLI overrides the trql-cli binary; DuckDB 1.5.5's libduckdb must
be loadable (see README.md).
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --plots) PLOTS="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --runs) RUNS="$2"; shift 2 ;;
        --no-resume) RESUME=0; shift ;;
        --stop-on-error) STOP_ON_ERROR=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

command -v uv >/dev/null || die "uv is required (https://docs.astral.sh/uv/)"
[[ -f "$REF_DB" ]] || die "recorded bench.db not found at $REF_DB"

mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
NEW_DB="$OUT/bench.db"
PLOT_DIR="$OUT/plots"
LOG_DIR="$OUT/logs"
PLAN="$OUT/plan.tsv"

# --- Guard the recorded database -------------------------------------------
# `-ef` catches the same file through a symlink or hard link, not just the same path.
if [[ "$NEW_DB" -ef "$REF_DB" || "$(realpath -m "$NEW_DB")" == "$(realpath -m "$REF_DB")" ]]; then
    die "--out would write the recorded bench.db ($REF_DB); choose another directory"
fi
REF_SHA="$(sha256sum "$REF_DB" | cut -d' ' -f1)"
check_ref() {
    local now
    now="$(sha256sum "$REF_DB" | cut -d' ' -f1)"
    if [[ "$now" != "$REF_SHA" ]]; then
        printf '\033[31merror:\033[0m the recorded bench.db changed during this run (%s -> %s)\n' \
            "$REF_SHA" "$now" >&2
        exit 3
    fi
}
trap check_ref EXIT

# --- Plan ------------------------------------------------------------------
say "Planning from plotter/manifest.toml for: $PLOTS"
skip_args=()
[[ $RESUME -eq 1 ]] && skip_args=(--skip-done-in "$NEW_DB")
# shellcheck disable=SC2086
(cd "$ROOT/plotter" && uv run --quiet --locked python scripts/repro_plan.py \
    --reference-db "$REF_DB" --plots $PLOTS "${skip_args[@]}") > "$PLAN"

n_total=$(($(wc -l < "$PLAN") - 1))
say "$n_total points -> $PLAN"

# --- Engine ----------------------------------------------------------------
if [[ $DRY_RUN -eq 0 && ! -x "$CLI" ]]; then
    say "Building trql-cli (release)"
    cargo build --locked --release --manifest-path "$ROOT/engine/Cargo.toml" -p trql-cli
fi

# --- Measure ---------------------------------------------------------------
mkdir -p "$LOG_DIR"
ok=0; skipped_done=0; failed=(); missing=(); noquery=()
declare -A missing_seen=() revised_seen=()
i=0
while IFS=$'\t' read -r plots query family sub err gap scale trql sql status sqlnote; do
    [[ "$sqlnote" == revised ]] && revised_seen["$query ($sql)"]=1
    i=$((i + 1))
    tag="$query ${family}/${sub} eps=$err gap=$gap x$scale"
    case "$status" in
        done) skipped_done=$((skipped_done + 1)); continue ;;
        no-query-file) noquery+=("$tag"); warn "[$i/$n_total] $tag: no matching query file"; continue ;;
    esac

    ds="$DATA_DIR/$family/$sub"
    if [[ ! -f "$ds/duckdb.db" ]]; then
        missing+=("$tag  [$plots]")
        missing_seen["$family/$sub"]=1
        continue
    fi

    common=(--data-dir "$DATA_DIR" --dataset "$family" --sub-dataset "$sub"
            --max-gap "$gap" --max-error-px "$err")
    build_cmd=("$CLI" "${common[@]}" dataset build)
    bench_cmd=("$CLI" "${common[@]}" bench "$query" "$(cat "$ROOT/$trql")" "$scale" "$RUNS"
               --sql "$ROOT/$sql" --bench-db "$NEW_DB")

    if [[ $DRY_RUN -eq 1 ]]; then
        printf '[%d/%d] %s\n' "$i" "$n_total" "$tag"
        if [[ ! -f "$ds/${gap}_${err}.trql" ]]; then printf ' %q' "${build_cmd[@]}"; echo; fi
        printf ' %q' "${bench_cmd[@]}"; echo
        continue
    fi

    log="$LOG_DIR/${query}__${family}__${sub//\//_}__g${gap}_e${err}_x${scale}.log"
    say "[$i/$n_total] $tag"
    # `set -e` does not apply inside an `if` condition, so the two steps are
    # chained explicitly: a failed build must not go on to bench.
    if { [[ -f "$ds/${gap}_${err}.trql" ]] || "${build_cmd[@]}"; } >"$log" 2>&1 \
        && "${bench_cmd[@]}" >>"$log" 2>&1; then
        ok=$((ok + 1))
    else
        failed+=("$tag  (log: $log)")
        warn "failed; see $log"
        tail -n 5 "$log" >&2 || true
        [[ $STOP_ON_ERROR -eq 1 ]] && break
    fi
    check_ref
done < <(tail -n +2 "$PLAN")

# --- Render ----------------------------------------------------------------
rendered=(); unrendered=()
if [[ $DRY_RUN -eq 0 ]]; then
    if [[ -f "$NEW_DB" ]]; then
        mkdir -p "$PLOT_DIR"
        for p in $PLOTS; do
            if out=$(cd "$ROOT/plotter" && uv run --quiet --locked benchplot render \
                    --only "$p" --db "$NEW_DB" --out-dir "$PLOT_DIR" 2>&1); then
                rendered+=("$out")
            else
                unrendered+=("$p: $out")
            fi
        done
    else
        for p in $PLOTS; do unrendered+=("$p: no measurements were taken"); done
    fi
fi

# --- Report ----------------------------------------------------------------
echo
say "Summary"
if [[ $DRY_RUN -eq 1 ]]; then
    echo "  dry run: nothing executed; plan in $PLAN"
else
    echo "  measured:        $ok"
    echo "  already present: $skipped_done (in $NEW_DB)"
    echo "  failed:          ${#failed[@]}"
fi
echo "  dataset missing: ${#missing[@]} points over ${#missing_seen[@]} sub-datasets"
for d in "${!missing_seen[@]}"; do echo "      $DATA_DIR/$d"; done | sort
[[ ${#noquery[@]} -gt 0 ]] && printf '  no query file:   %s\n' "${noquery[@]}"
if [[ ${#revised_seen[@]} -gt 0 ]]; then
    echo "  reference SQL revised since the recording (gated against the current file):"
    for q in "${!revised_seen[@]}"; do echo "      $q"; done | sort
fi
[[ ${#failed[@]} -gt 0 ]] && printf '  FAILED: %s\n' "${failed[@]}"
if [[ $DRY_RUN -eq 0 ]]; then
    [[ ${#rendered[@]} -gt 0 ]] && printf '  plot: %s\n' "${rendered[@]}"
    [[ ${#unrendered[@]} -gt 0 ]] && printf '  not drawn: %s\n' "${unrendered[@]}"
fi
echo "  recorded bench.db unchanged: sha256 $REF_SHA"

[[ ${#failed[@]} -eq 0 ]] || exit 1
