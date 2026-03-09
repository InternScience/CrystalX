#!/bin/sh

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_SHELXL="${RUN_SHELXL:-0}"

cd "$REPO_DIR"

python data_pipeline/run/shelxt_simple.py

if [ "$RUN_SHELXL" = "1" ]; then
  python data_pipeline/run/shelxl_simple.py
fi
