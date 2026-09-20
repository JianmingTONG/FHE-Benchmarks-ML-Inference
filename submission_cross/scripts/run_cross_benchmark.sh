#!/usr/bin/env bash
# Run the ML-inference benchmark against the CROSS/TPU submission.
#
#   scripts/run_cross_benchmark.sh <size> [--seed N] [--num_runs N]
#
# Leaves the resident TPU server running between runs; call
# submission_cross/build/server_stop <size> to release the TPU.
set -euo pipefail

ROOT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/../.." &> /dev/null && pwd )"
SIZE="${1:?usage: run_cross_benchmark.sh <size> [harness args...]}"
shift || true

cd "$ROOT"
export FHE_SUBMISSION_DIR=submission_cross
export CROSS_DEVICE_COUNT="${CROSS_DEVICE_COUNT:-8}"
export CROSS_BSGS_JOBS="${CROSS_BSGS_JOBS:-1}"

echo "[run] submission=$FHE_SUBMISSION_DIR chips=$CROSS_DEVICE_COUNT size=$SIZE"
python3 -u harness/run_submission.py "$SIZE" "$@"

echo
python3 -u submission_cross/profile/report.py --sizes "$SIZE"
