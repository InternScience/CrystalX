#!/bin/sh

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_DIR"

python data_pipeline/prepare/cif2ins.py
python data_pipeline/prepare/platon_cif2hkl.py
python data_pipeline/prepare/check_empty_sxhkl.py
python data_pipeline/prepare/download_hkl.py
python data_pipeline/prepare/platon_hkl2hkl.py
python data_pipeline/prepare/copy_inshkl.py
