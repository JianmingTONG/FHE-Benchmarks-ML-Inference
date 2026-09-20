#!/usr/bin/env bash
# CPU-only self-test for the CROSS submission. No TPU required.
#
# Checks the parts a reviewer can verify without an accelerator: the model
# folds exactly, the client and server agree on the ring, the CKKS codec
# round-trips, ciphertexts survive the file boundary with their scale intact,
# and the atomic writer does not leave partial files behind.
set -euo pipefail
ROOT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/../.." &> /dev/null && pwd )"
cd "$ROOT"
JAX_PLATFORMS=cpu python3 submission_cross/scripts/selftest.py "$@"
