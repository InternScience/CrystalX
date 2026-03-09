#!/bin/sh

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

ROOT_DIR="${ROOT_DIR:-all_shelx_file_clean}"
TEST_YEARS="${TEST_YEARS:-2018,2019,2020,2021,2022,2023,2024}"
SPLIT_DIR="${SPLIT_DIR:-splits}"
OUT_DIR="${OUT_DIR:-all_test_shelx}"

YEARS_TAG="$(printf "%s" "$TEST_YEARS" | tr ',' '-')"
TEST_IDS="${TEST_IDS:-$SPLIT_DIR/test_ids_test-$YEARS_TAG.txt}"

cd "$REPO_DIR"

python data_pipeline/collect/copy_dirs.py \
  --root_dir "$ROOT_DIR" \
  --test_ids "$TEST_IDS" \
  --out_dir "$OUT_DIR"

python data_pipeline/collect/get_cifs.py
