#!/bin/sh

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_SCRIPT="$SCRIPT_DIR/run_demo.sh"

ROOT_DIR_INPUT="."
ENABLE_TIMING=0
ENABLE_STEP_TIMING=0
ENABLE_SHUFFLE=0
SHUFFLE_SEED=""
ENABLE_SKIP_RAN=0
TIMING_LOG=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --timing)
            ENABLE_TIMING=1
            shift
            ;;
        --step-timing)
            ENABLE_STEP_TIMING=1
            shift
            ;;
        --shuffle)
            ENABLE_SHUFFLE=1
            shift
            ;;
        --shuffle-seed)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] --shuffle-seed requires a seed value"
                exit 1
            fi
            ENABLE_SHUFFLE=1
            SHUFFLE_SEED="$2"
            shift 2
            ;;
        --skip-if-ran)
            ENABLE_SKIP_RAN=1
            shift
            ;;
        --timing-log)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] --timing-log requires a file path"
                exit 1
            fi
            TIMING_LOG="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: sh run_demo_batch.sh [ROOT_DIR] [--timing] [--step-timing] [--shuffle] [--shuffle-seed SEED] [--skip-if-ran] [--timing-log FILE]"
            echo "  ROOT_DIR           Root folder containing case subfolders (default: .)"
            echo "  --timing           Print per-case elapsed seconds"
            echo "  --step-timing      Print per-python-step elapsed seconds from run_demo.sh"
            echo "  --shuffle          Randomize case execution order"
            echo "  --shuffle-seed N   Randomize with reproducible seed N (implies --shuffle)"
            echo "  --skip-if-ran      Skip case if dated output folder already exists"
            echo "  --timing-log FILE  Save timing lines and batch summary to FILE (implies --timing)"
            exit 0
            ;;
        *)
            if [ "$ROOT_DIR_INPUT" = "." ]; then
                ROOT_DIR_INPUT="$1"
                shift
            else
                echo "[ERROR] Unknown argument: $1"
                echo "Use --help for usage."
                exit 1
            fi
            ;;
    esac
done

if [ -n "$TIMING_LOG" ]; then
    ENABLE_TIMING=1
fi
if [ -n "$SHUFFLE_SEED" ] && ! command -v python >/dev/null 2>&1; then
    echo "[WARN] --shuffle-seed is set but python is unavailable; using non-deterministic shuffle."
fi

if [ ! -f "$PIPELINE_SCRIPT" ]; then
    echo "[ERROR] Missing pipeline script: $PIPELINE_SCRIPT"
    exit 1
fi

if [ ! -d "$ROOT_DIR_INPUT" ]; then
    echo "[ERROR] ROOT_DIR does not exist: $ROOT_DIR_INPUT"
    exit 1
fi

# Normalize to absolute path to avoid caller cwd confusion.
ROOT_DIR="$(cd "$ROOT_DIR_INPUT" && pwd)"

success=0
failed=0
skipped=0
total=0
run_count=0
CASE_TIMINGS=""

if [ -n "$TIMING_LOG" ]; then
    case "$TIMING_LOG" in
        /*) ;;
        *) TIMING_LOG="$(pwd)/$TIMING_LOG" ;;
    esac
    : > "$TIMING_LOG"
    printf "# timing log generated at %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" >> "$TIMING_LOG"
fi

shuffle_file_inplace() {
    src="$1"
    tmp="${src}.shuf"
    if [ "$ENABLE_SHUFFLE" -ne 1 ]; then
        return
    fi
    if [ -n "$SHUFFLE_SEED" ] && command -v python >/dev/null 2>&1; then
        python - "$SHUFFLE_SEED" < "$src" > "$tmp" <<'PY'
import random
import sys

seed = sys.argv[1]
lines = [ln.rstrip("\n") for ln in sys.stdin if ln.rstrip("\n")]
try:
    random.seed(int(seed))
except Exception:
    random.seed(seed)
random.shuffle(lines)
for line in lines:
    print(line)
PY
    elif command -v shuf >/dev/null 2>&1; then
        shuf "$src" > "$tmp"
    elif command -v python >/dev/null 2>&1; then
        python - < "$src" > "$tmp" <<'PY'
import random
import sys

lines = [ln.rstrip("\n") for ln in sys.stdin if ln.rstrip("\n")]
random.shuffle(lines)
for line in lines:
    print(line)
PY
    else
        awk 'BEGIN{srand()} {printf("%.16f\t%s\n", rand(), $0)}' "$src" | sort -k1,1n | cut -f2- > "$tmp"
    fi
    mv "$tmp" "$src"
}

find_existing_run_dir() {
    case_dir="$1"
    case_name="$2"
    for cand in "$case_dir"/*; do
        if [ ! -d "$cand" ]; then
            continue
        fi
        base="$(basename "$cand")"
        case "$base" in
            [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9]-[0-9][0-9]-[0-9][0-9]_"$case_name")
                echo "$cand"
                return 0
                ;;
        esac
    done
    return 1
}

run_case() {
    case_dir="$1"
    case_name="$2"
    ins_path="$case_dir/$case_name.ins"
    hkl_path="$case_dir/$case_name.hkl"
    start_ts=""
    end_ts=""
    elapsed=""

    total=$((total + 1))
    if [ ! -f "$ins_path" ] || [ ! -f "$hkl_path" ]; then
        echo "[SKIP] $case_name (missing $case_name.ins or $case_name.hkl)"
        skipped=$((skipped + 1))
        return
    fi

    if [ "$ENABLE_SKIP_RAN" -eq 1 ]; then
        existing_run_dir="$(find_existing_run_dir "$case_dir" "$case_name" || true)"
        if [ -n "$existing_run_dir" ]; then
            echo "[SKIP] $case_name (existing run dir: $(basename "$existing_run_dir"))"
            skipped=$((skipped + 1))
            # Mark as handled to avoid fallback mode unexpectedly rerunning from ROOT_DIR.
            run_count=$((run_count + 1))
            return
        fi
    fi

    run_count=$((run_count + 1))
    echo "[RUN ] $case_name"
    if [ "$ENABLE_TIMING" -eq 1 ]; then
        start_ts="$(date +%s)"
    fi
    if (
        cd "$case_dir" &&
        if [ "$ENABLE_STEP_TIMING" -eq 1 ]; then
            if [ -n "$TIMING_LOG" ]; then
                PY_STEP_TIMING=1 PY_STEP_TIMING_LOG="$TIMING_LOG" sh "$PIPELINE_SCRIPT" "$case_name"
            else
                PY_STEP_TIMING=1 sh "$PIPELINE_SCRIPT" "$case_name"
            fi
        else
            sh "$PIPELINE_SCRIPT" "$case_name"
        fi
    ); then
        echo "[ OK ] $case_name"
        success=$((success + 1))
    else
        echo "[FAIL] $case_name"
        failed=$((failed + 1))
    fi

    if [ "$ENABLE_TIMING" -eq 1 ]; then
        end_ts="$(date +%s)"
        elapsed=$((end_ts - start_ts))
        timing_line="CASE|$case_name|$case_dir|${elapsed}s"
        echo "[TIME] $case_name ${elapsed}s"
        CASE_TIMINGS="${CASE_TIMINGS:-}${timing_line}
"
        if [ -n "$TIMING_LOG" ]; then
            printf "%s\n" "$timing_line" >> "$TIMING_LOG"
        fi
    fi
}

# Mode A: each sample in its own subdirectory.
mode_a_list="$(mktemp)"
for dir in "$ROOT_DIR"/*; do
    if [ ! -d "$dir" ]; then
        continue
    fi
    name="$(basename "$dir")"
    printf "%s|%s\n" "$dir" "$name" >> "$mode_a_list"
done
if [ -s "$mode_a_list" ]; then
    shuffle_file_inplace "$mode_a_list"
    while IFS='|' read -r dir name; do
        if [ -z "$dir" ] || [ -z "$name" ]; then
            continue
        fi
        run_case "$dir" "$name"
    done < "$mode_a_list"
fi
rm -f "$mode_a_list"

# Mode B fallback: ROOT_DIR directly contains *.ins/*.hkl pairs.
if [ "$run_count" -eq 0 ]; then
    found_ins=0
    mode_b_list="$(mktemp)"
    for ins_path in "$ROOT_DIR"/*.ins; do
        if [ ! -f "$ins_path" ]; then
            continue
        fi
        found_ins=1
        stem="$(basename "$ins_path" .ins)"
        printf "%s\n" "$stem" >> "$mode_b_list"
    done
    if [ "$found_ins" -eq 1 ]; then
        shuffle_file_inplace "$mode_b_list"
        while IFS= read -r stem; do
            if [ -z "$stem" ]; then
                continue
            fi
            run_case "$ROOT_DIR" "$stem"
        done < "$mode_b_list"
    fi
    if [ "$found_ins" -eq 0 ]; then
        echo "[WARN] No subdirectories and no *.ins files found in: $ROOT_DIR"
    fi
    rm -f "$mode_b_list"
fi

echo
echo "Batch summary:"
echo "  total cases  : $total"
echo "  success      : $success"
echo "  failed       : $failed"
echo "  skipped      : $skipped"

if [ "$ENABLE_TIMING" -eq 1 ]; then
    echo "  timing       : enabled"
    if [ -n "$TIMING_LOG" ]; then
        echo "  timing log   : $TIMING_LOG"
    fi
    if [ -n "${CASE_TIMINGS:-}" ]; then
        echo
        echo "Per-case timing:"
        printf "%s" "$CASE_TIMINGS"
    fi
fi

if [ "$ENABLE_STEP_TIMING" -eq 1 ]; then
    echo "  step timing  : enabled"
fi
if [ "$ENABLE_SHUFFLE" -eq 1 ]; then
    if [ -n "$SHUFFLE_SEED" ]; then
        echo "  shuffle      : enabled (seed=$SHUFFLE_SEED)"
    else
        echo "  shuffle      : enabled"
    fi
fi
if [ "$ENABLE_SKIP_RAN" -eq 1 ]; then
    echo "  skip-if-ran  : enabled"
fi

if [ -n "$TIMING_LOG" ]; then
    printf "\n# Batch summary\n" >> "$TIMING_LOG"
    printf "total=%s\n" "$total" >> "$TIMING_LOG"
    printf "success=%s\n" "$success" >> "$TIMING_LOG"
    printf "failed=%s\n" "$failed" >> "$TIMING_LOG"
    printf "skipped=%s\n" "$skipped" >> "$TIMING_LOG"
    printf "step_timing=%s\n" "$ENABLE_STEP_TIMING" >> "$TIMING_LOG"
    printf "shuffle=%s\n" "$ENABLE_SHUFFLE" >> "$TIMING_LOG"
    printf "shuffle_seed=%s\n" "$SHUFFLE_SEED" >> "$TIMING_LOG"
    printf "skip_if_ran=%s\n" "$ENABLE_SKIP_RAN" >> "$TIMING_LOG"
fi

if [ "$failed" -gt 0 ]; then
    exit 1
fi

exit 0
