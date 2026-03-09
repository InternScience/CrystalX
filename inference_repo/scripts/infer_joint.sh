#!/bin/sh

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$REPO_DIR"
exec "$PYTHON_BIN" -m crystalx_infer.pipelines.infer_joint_heavy_hydro_temporal "$@"
