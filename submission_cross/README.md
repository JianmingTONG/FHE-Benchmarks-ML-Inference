# CROSS / TPU submission — ML Inference (MNIST)

This submission runs the benchmark's encrypted MNIST inference on **Google TPU
v6e** using [CROSS](https://github.com/EfficientPPML/CROSS) (`jaxite_word`), the
JAX CKKS library from the HPCA'26 paper *Leveraging ASIC AI Chips for
Homomorphic Encryption*.

The reference submission in `../submission` is OpenFHE on CPU, compiled with
HEIR. It is left untouched so the two can be compared on the same host; select
one with `FHE_SUBMISSION_DIR` (see *Running* below).

## Model architecture changes

The harness model (`harness/mnist/model.py`) is `784 → 128 → 64 → 10` with
ReLU. **The one change is ReLU → x².** Topology, layer widths, and the
input/output contract are unchanged.

```
784 ──Linear(784,128)──► x² ──Linear(128,64)──► x² ──Linear(64,10)──► 10 logits
```

CKKS evaluates polynomials, not ReLU, so every FHE submission substitutes
something. CROSS's activation registry accepts only a bare `x**(2**k)` squaring
chain — `nn.register_activation` rejects anything else, because reaching a
non-power-of-two exponent needs one operand mod-switched down to meet the
other, and the only level-changing primitive is Rescale, which also divides the
scale. So x² it is.

A bare square is hard to train directly (activations square every layer and the
scale explodes), so `model/he_mlp.py` carries a `BatchNorm1d` before each square
during training and `to_inference_model` folds each one into the preceding
`Linear` at export. BN(Wx+b) is itself affine, so the fold is an exact
algebraic rewrite — `export_weights.py` re-checks it, and refuses to write a
model whose function moved by more than 1e-4. The exported module has no
BatchNorm and no branch, which is what `torch.fx`, and therefore
`jaxite_word.nn.vectorize`, accepts.

Client-side preprocessing applies the same MNIST normalization
`(x - 0.1307) / 0.3081` the harness's own predictor applies in
`harness/mnist/test.py`. It is affine and costs nothing in the clear, so it
does not spend ciphertext depth.

### How this compares to the reference submission's changes

The reference rewrites the network to two `1024 × 1024` fully-connected layers
with a degree-11 polynomial approximate-ReLU, and pads 784 → 1024. This
submission keeps the harness's own layer widths and takes the depth-1 square
instead of the depth-5 approximate-ReLU. Both are documented architecture
changes of the kind the benchmark's README anticipates.

## Cryptographic parameters

**Nothing in this submission picks a security parameter.** `packing.pack`
derives the ring from the program's own slot demand and the depth it actually
emits, and the client samples its key pair at that ring's error width:

| | this submission (CROSS) | reference (`../submission`) |
|---|---|---|
| ring degree | 32768 | 2048 |
| slots | 16384 | 1024 |
| modulus chain | 12 Q towers + 4 P towers, `dnum=3` | depth 9 |
| security | **128-bit classical** | `HEStd_NotSet` |
| multiplicative depth | 5 | 9 |

The reference sets `SetSecurityLevel(HEStd_NotSet)` with `SetRingDim(1 << 11)`,
which is not a secure parameter set — a ring of 2048 at that modulus size does
not meet any standard security target. Runtime comparisons against it should be
read with that in mind: this submission is doing 16× more ring work per
ciphertext, at a real 128-bit security level.

## Stage mapping and the resident server

CROSS is a JAX library and **a TPU admits exactly one process at a time**, so
the benchmark's process-per-stage structure maps as:

| Stage | Where | What it does |
|---|---|---|
| 2 `client_key_generation` | CPU | Packs the architecture, samples the CKKS key pair for that ring |
| 5 `client_preprocess_input` | CPU | MNIST normalization |
| 6 `client_encode_encrypt_input` | CPU | Slot-encode + encrypt, one file per input |
| 3 `server_preprocess_model` | **TPU** | Materializes the `Mapping` and leaves it resident |
| 7 `server_encrypted_compute` | **TPU** | `Mapping.execute` on the uploaded ciphertexts |
| 8 `client_decrypt_decode` | CPU | Decrypt + slot-decode |
| 9 `client_postprocess` | CPU | argmax → label |

Client stages run with `JAX_PLATFORMS=cpu`. That is not a workaround — the
client *is* a CPU-side entity here: it holds the key pair and the slot layout,
never builds a Mapping, and never sees the model's schedule.

Stage 3 starts `src/server_daemon.py` and blocks until the Mapping is
materialized, so the stage-3 measurement is the whole offline server cost: BSGS
plan selection and diagonal encoding per MatVec, evaluation- and rotation-key
materialization, constant encoding, device placement, and the fused XLA
compilation. The daemon then keeps it resident, because **a compiled JAX
executable cannot cross a process boundary** and rebuilding it per request
would report compilation time as inference time. The reference submission
reloads its keys on every stage-7 invocation for the same reason.

`build/server_stop <size>` releases the TPU when you are done.

### The client never loads the server's weights

The ring, the slot layout and the input/output coordinate maps follow from the
program's shapes and emitted depth, not from any weight value. So the client
packs the *same architecture with untrained parameters*
(`cross_task.build_client_packed`) and lands on an identical `RingConfig` and
identical pack/unpack maps, while the weight-dependent artifact — the MatVec
diagonals — stays on the server. `verify_against_manifest` refuses to encrypt
unless both sides agree on degree, slots, tower counts, `dnum` and depth.

### Known deviation: where the evaluation keys are generated

In CROSS a `Mapping` owns its `CKKSContext`, and `mapping.ring_runtime_parameters`
requires the secret key because the evaluation and rotation keys are derived
during Mapping construction. This submission therefore hands the server process
the key pair, rather than shipping client-generated evaluation and rotation keys
as the reference does.

**This is a real deviation from the benchmark's client/server trust model and is
not a security claim.** It reflects CROSS's current API, which offers no way to
build an evaluator context from public key material alone. The key *volume* is
still reported: `Mapping.estimate_live_memory()` gives the evaluation-key and
rotation-key bytes, and stage 3 prints them.

## Running

Step-by-step instructions, expected output, timings and troubleshooting are in
**[EVALUATION.md](EVALUATION.md)**. The short form:

```console
# CROSS on TPU
FHE_SUBMISSION_DIR=submission_cross python3 harness/run_submission.py 0 --seed 3

# the OpenFHE CPU reference, unchanged
python3 harness/run_submission.py 0 --seed 3
```

`CROSS_DEVICE_COUNT` (default 8) sets how many TPU chips the server Mapping is
compiled for; the global ciphertext batch equals that count, and inputs are
processed in chunks of it.

Standalone TPU profiling, independent of the harness:

```console
python3 submission_cross/profile/profile_tpu.py --devices 1,2,4,8 --local-batch 1,2,4
python3 submission_cross/profile/trace_breakdown.py <trace_dir>
```

## Harness changes

Two functions in `harness/utils.py` and four lines in `harness/run_submission.py`:
`FHE_SUBMISSION_DIR` selects the submission subdirectory (default `submission`,
so an unset environment reproduces the original behaviour exactly), a submission
that ships its own `scripts/build_task.sh` owns its whole build, and
`server_preprocess_model` is passed the instance size like every other stage
(the reference's `int main()` ignores it).

## Robustness

The stages are separate processes exchanging files, and the server holds a
multi-gigabyte accelerator context for minutes at a time, so the failure modes
that matter are partial writes, orphaned processes and unbounded memory.

- **Streaming inference.** `server_daemon.infer` loads, executes and writes back
  one chunk at a time. Reading every ciphertext up front would hold ~35 GB live
  at the large instance (10000 x 3.0 MB in, 10000 x 517 KB out) for no gain, as
  the Mapping consumes exactly `global_batch` ciphertexts per call. Peak memory
  is now one chunk at any instance size.
- **Atomic file writes.** Every artifact a later stage reads -- ciphertexts,
  manifest, logits, labels -- is written to a temporary file and `os.replace`d
  into position, so a stage killed midway cannot leave a truncated file that
  the next stage silently accepts.
- **Validated reads.** A ciphertext is checked for rank, for the payload shape
  this deployment was compiled for, and for readability; each error names the
  offending file rather than surfacing as a shape error deep inside the
  evaluator with no clue which of 10000 inputs was at fault.
- **Graceful shutdown.** SIGTERM/SIGINT/SIGHUP release the Mapping's HBM, drop
  the socket and remove the pidfile, so the next run can take the accelerator.
  Killing the daemon without this leaves the TPU pinned.
- **Fail fast, not hang.** Stage 3 watches the daemon process, not just its
  ready marker: a server that dies in XLA or to the OOM killer is reported in
  seconds with the tail of its log, instead of waiting out a 90-minute startup
  timeout. It also clears a stale `libtpu_lockfile` left by a killed process.
- **Bounded, EOF-aware socket I/O.** Requests are newline-framed with a size
  cap and honour EOF, so a peer that dies mid-message cannot hang the server or
  make it buffer without bound. A request-time failure is reported to the
  caller and never takes the server down -- rebuilding the Mapping costs ~8
  minutes.
- **Connect retry.** Stage 7 retries briefly, since stage 3 returns when the
  ready marker appears and the listening socket follows a moment later.
- **Portable CROSS discovery.** `CROSS_ROOT`, then an importable
  `jaxite_word`, then the usual sibling locations, with a named error instead
  of a `ModuleNotFoundError` three imports deep.

### Self-test (no TPU required)

```console
submission_cross/scripts/selftest.sh
```

Nine checks covering the accelerator-independent half of the submission: the
BatchNorm fold is exact, client and server derive the same ring, a mismatched
ring is refused, the CKKS codec round-trips through the file boundary with its
scale intact, corrupt and mis-shaped ciphertexts are rejected by name, a failed
atomic write leaves nothing behind, and reading a path does not create it.

## Layout

```
submission_cross/
├─ model/         he_mlp.py, train_he_mlp.py, export_weights.py, weights
├─ src/           cross_task.py (shared), server_daemon.py (resident TPU server)
├─ build/         the stage executables the harness invokes
├─ EVALUATION.md  how to reproduce every number in RESULTS.md
├─ scripts/       build_task.sh, run_cross_benchmark.sh, run_tpu_profile.sh, selftest.sh
├─ profile/       profile_tpu.py, trace_breakdown.py, report.py
└─ results/       measurements behind RESULTS.md
```
