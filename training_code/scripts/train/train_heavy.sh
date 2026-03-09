#!/bin/sh

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$TRAINING_ROOT"
exec "$PYTHON_BIN" -m crystalx_train.trainers.trainer_heavy "$@"
