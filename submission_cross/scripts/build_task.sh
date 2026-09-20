#!/usr/bin/env bash
# ------------------------------------------------------------
# Build the CROSS/TPU submission.
#
# CROSS is a JAX library, so there is no compilation step here: "building"
# this submission means checking the runtime is present and making sure the
# HE-friendly model weights exist. The expensive artifact -- the compiled
# Mapping with its keys, BSGS diagonals and encoded constants -- is built by
# benchmark stage 3 (server_preprocess_model) on the TPU, where it is measured.
# ------------------------------------------------------------
set -euo pipefail

SUBMISSION_DIR="$( cd -- "${1:-$( dirname -- "${BASH_SOURCE[0]}" )/..}" &> /dev/null && pwd )"
CROSS_ROOT="${CROSS_ROOT:-/home/jianming_gatech/CROSS}"

if [[ ! -d "$CROSS_ROOT/jaxite_word" ]]; then
  echo "[build] CROSS not found at $CROSS_ROOT; set CROSS_ROOT" >&2
  exit 1
fi

python3 - "$CROSS_ROOT" <<'PY'
import importlib.util, sys
missing = [name for name in ("jax", "torch", "numpy")
           if importlib.util.find_spec(name) is None]
if missing:
    sys.exit(f"[build] missing python packages: {', '.join(missing)}")
print("[build] runtime dependencies present")
PY

WEIGHTS="$SUBMISSION_DIR/model/he_mlp_weights.pth"
if [[ ! -f "$WEIGHTS" ]]; then
  echo "[build] training the HE-friendly MNIST model (one-off)"
  python3 "$SUBMISSION_DIR/model/train_he_mlp.py"
else
  echo "[build] model weights present: $(basename "$WEIGHTS")"
fi

chmod +x "$SUBMISSION_DIR"/build/*
echo "[build] CROSS submission ready"
