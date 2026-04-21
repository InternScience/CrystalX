#!/bin/sh

set -eu

# Resolve script directory so this script works from any current directory.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"

resolve_bin() {
    local_path="$SCRIPT_DIR/$1"
    if [ -x "$local_path" ]; then
        echo "$local_path"
        return 0
    fi
    if command -v "$1" >/dev/null 2>&1; then
        command -v "$1"
        return 0
    fi
    echo ""
    return 1
}

now_ts() {
    "$PYTHON_BIN" -c "import time; print(f'{time.time():.6f}')"
}

elapsed_1dp() {
    start_ts="$1"
    end_ts="$2"
    awk -v s="$start_ts" -v e="$end_ts" 'BEGIN { printf "%.1f", (e - s) }'
}

# prepare
fname="${1:-}"
if [ -z "$fname" ]; then
    echo "Usage: sh run_demo.sh <file_prefix>"
    exit 1
fi

PIPELINE_START_TS="$(now_ts)"

PY_STEP_TIMING_ENABLED=0
case "${PY_STEP_TIMING:-0}" in
    1|true|TRUE|yes|YES)
        PY_STEP_TIMING_ENABLED=1
        ;;
esac
PY_STEP_TIMINGS=""
PY_STEP_TIMING_LOG="${PY_STEP_TIMING_LOG:-}"

run_python_step() {
    step_name="$1"
    shift
    step_start=""
    step_end=""
    step_elapsed=""

    if [ "$PY_STEP_TIMING_ENABLED" -eq 1 ]; then
        echo "[PYRUN ] $step_name"
        step_start="$(now_ts)"
    fi

    "$@"

    if [ "$PY_STEP_TIMING_ENABLED" -eq 1 ]; then
        step_end="$(now_ts)"
        step_elapsed="$(elapsed_1dp "$step_start" "$step_end")"
        echo "[PYTIME] $step_name ${step_elapsed}s"
        PY_STEP_TIMINGS="${PY_STEP_TIMINGS}${step_name}|${step_elapsed}s
"
        if [ -n "$PY_STEP_TIMING_LOG" ]; then
            printf "STEP|%s|%s|%ss\n" "$fname" "$step_name" "$step_elapsed" >> "$PY_STEP_TIMING_LOG"
        fi
    fi
}

print_python_step_summary() {
    if [ "$PY_STEP_TIMING_ENABLED" -eq 1 ]; then
        pipeline_end_ts="$(now_ts)"
        pipeline_elapsed="$(elapsed_1dp "$PIPELINE_START_TS" "$pipeline_end_ts")"
        echo
        echo "Python step timing ($fname):"
        if [ -n "${PY_STEP_TIMINGS:-}" ]; then
            printf "%s" "$PY_STEP_TIMINGS"
        fi
        echo "[PYTOTAL] $fname ${pipeline_elapsed}s"
        if [ -n "$PY_STEP_TIMING_LOG" ]; then
            printf "TOTAL|%s|%ss\n" "$fname" "$pipeline_elapsed" >> "$PY_STEP_TIMING_LOG"
        fi
    fi
}

download_weights_if_needed() {
    if [ -f "$main_model_name" ] && [ -f "$hydro_model_name" ]; then
        return 0
    fi

    hf_repo_id="${CRYSTALX_HF_REPO_ID:-}"
    if [ -z "$hf_repo_id" ]; then
        return 0
    fi

    echo "[INFO] Missing local CrystalX checkpoints. Downloading from Hugging Face: $hf_repo_id"
    env PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" -m crystalx_infer.tools.download_weights --repo-id "$hf_repo_id"
}

main_model_name="$REPO_DIR/weights/crystalx-heavy.pth"
hydro_model_name="$REPO_DIR/weights/crystalx-hydro.pth"
legacy_main_model_name="$REPO_DIR/weights/final_main_model_add_no_noise_fold_3.pth"
legacy_hydro_model_name="$REPO_DIR/weights/final_hydro_model_add_no_noise_fold_3.pth"

if [ ! -f "$main_model_name" ] && [ -f "$legacy_main_model_name" ]; then
    main_model_name="$legacy_main_model_name"
fi
if [ ! -f "$hydro_model_name" ] && [ -f "$legacy_hydro_model_name" ]; then
    hydro_model_name="$legacy_hydro_model_name"
fi

shelxt_bin="$(resolve_bin shelxt || true)"
shelxl_bin="$(resolve_bin shelxl || true)"
platon_bin="$(resolve_bin platon || true)"

if [ -z "$shelxt_bin" ]; then
    echo "[ERROR] Cannot find shelxt in $SCRIPT_DIR or PATH"
    exit 1
fi
if [ -z "$shelxl_bin" ]; then
    echo "[ERROR] Cannot find shelxl in $SCRIPT_DIR or PATH"
    exit 1
fi
if [ -z "$platon_bin" ]; then
    echo "[ERROR] Cannot find platon in $SCRIPT_DIR or PATH"
    exit 1
fi
download_weights_if_needed
if [ ! -f "$main_model_name" ]; then
    echo "[ERROR] Missing heavy model checkpoint: $main_model_name"
    echo "[ERROR] Run: python -m crystalx_infer.tools.download_weights --repo-id <hf_repo_id>"
    echo "[ERROR] Or set CRYSTALX_HF_REPO_ID=<hf_repo_id> and rerun."
    exit 1
fi
if [ ! -f "$hydro_model_name" ]; then
    echo "[ERROR] Missing hydro model checkpoint: $hydro_model_name"
    echo "[ERROR] Run: python -m crystalx_infer.tools.download_weights --repo-id <hf_repo_id>"
    echo "[ERROR] Or set CRYSTALX_HF_REPO_ID=<hf_repo_id> and rerun."
    exit 1
fi

date_info=$(date +"%Y-%m-%d_%H-%M-%S")
work_dir="${date_info}_${fname}"


mkdir -p "$work_dir"
cp "${fname}.hkl" "$work_dir/${fname}.hkl"
cp "${fname}.ins" "$work_dir/${fname}.ins"

# Running
"$shelxt_bin" "$work_dir/$fname"

# To do: Check the validility of the phasing results

run_python_step "predict_heavy.py" \
    env PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" -m crystalx_infer.pipelines.predict_heavy --fname="$fname" --work_dir="$work_dir" --model_path="$main_model_name"

err_mes="$work_dir/ERROR.json"
if [ -e "$err_mes" ]; then
    print_python_step_summary
    exit 0  
fi

# Back to normal pipeline: refine heavy first.
"$shelxl_bin" "$work_dir/${fname}_AI"

# Then build AIBond and run SHELXL to generate connectivity .lst for hydro.
run_python_step "prepare_bond_inputs.py" \
    env PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" -m crystalx_infer.postprocess.prepare_bond_inputs --fname="${fname}_AI" --work_dir="$work_dir"
"$shelxl_bin" "$work_dir/${fname}_AIBond"

run_python_step "predict_hydro.py" \
    env PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" -m crystalx_infer.pipelines.predict_hydro --fname="${fname}_AI" --work_dir="$work_dir" --model_path="$hydro_model_name"
"$shelxl_bin" "$work_dir/${fname}_AIhydro"

run_python_step "prepare_weight_refine.py" \
    env PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" -m crystalx_infer.postprocess.prepare_weight_refine --fname="${fname}_AIhydro" --work_dir="$work_dir"
"$shelxl_bin" "$work_dir/${fname}_AIhydroWeight"

# perhaps not needed
# python prepare_weight_refine_2.py --fname="${fname}_AIhydroWeight" --work_dir="$work_dir"
# ./shelxl "$work_dir/${fname}_AIhydroWeight2"

# To do: Check the validility of the final results

"$platon_bin" -u "$work_dir/${fname}_AIhydroWeight.cif"

run_python_step "write_final_outputs.py" \
    env PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" -m crystalx_infer.postprocess.write_final_outputs --fname="${fname}_AIhydroWeight" --work_dir="$work_dir"

print_python_step_summary

exit 0
