#!/usr/bin/env bash
# TPU v6e profiling sweep for the CROSS submission.
#
#   scripts/run_tpu_profile.sh [out_dir]
#
# Sweeps 1/2/4/8 chips at one ciphertext per chip -- strong scaling -- and
# captures a trace of the 8-chip configuration for the op breakdown. One
# Mapping is built per configuration (~8 min each), because device topology and
# global batch are static compilation inputs in CROSS.
#
# A per-chip batch above 1 is NOT swept: on this JAX/libtpu version it aborts
# the XLA TPU compiler inside the fusion emitter, reproducibly, about eight
# minutes into the Mapping build. See results/tpu_profile/crash/. Set
# CROSS_SWEEP_LOCAL_BATCH to try it anyway on a stack where it is fixed.
set -euo pipefail

ROOT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/../.." &> /dev/null && pwd )"
OUT="${1:-$ROOT/submission_cross/results/tpu_profile}"
cd "$ROOT"

export CROSS_BSGS_JOBS="${CROSS_BSGS_JOBS:-16}"

python3 -u submission_cross/profile/profile_tpu.py \
  --devices "${CROSS_SWEEP_DEVICES:-1,2,4,8}" --local-batch 1 \
  --iterations 20 --warmup 3 \
  --trace-config 8:1 --out "$OUT"

if [[ -n "${CROSS_SWEEP_LOCAL_BATCH:-}" ]]; then
  echo "[profile] per-chip batch sweep (known to abort XLA on this stack)"
  python3 -u submission_cross/profile/profile_tpu.py \
    --devices 8 --local-batch "$CROSS_SWEEP_LOCAL_BATCH" \
    --iterations 20 --warmup 3 \
    --trace-config none --out "$OUT/batched" || \
    echo "[profile] per-chip batch sweep failed (see log)"
fi

echo
echo "[profile] op breakdown for the 8-chip trace"
python3 -u submission_cross/profile/trace_breakdown.py "$OUT/traces/8chip_lb1" --iterations 3
