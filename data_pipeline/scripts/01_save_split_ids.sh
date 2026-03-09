#!/bin/sh

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

TXT_PATH="${TXT_PATH:-sorted_by_journal_year.txt}"
PT_DIR="${PT_DIR:-all_materials/data/all_anno_density}"
TEST_YEARS="${TEST_YEARS:-2018,2019,2020,2021,2022,2023,2024}"
OUT_DIR="${OUT_DIR:-splits}"

cd "$REPO_DIR"

python data_pipeline/split/save_split_ids.py \
  --txt "$TXT_PATH" \
  --pt_dir "$PT_DIR" \
  --test_years "$TEST_YEARS" \
  --out_dir "$OUT_DIR"
