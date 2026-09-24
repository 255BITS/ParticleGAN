# Port the known CPU winner to GPU

**The active base is `constraints_simple_regularization`, the recorded 22/22 CPU
recipe. Preserve its schedules, noise, losses, and host settings while resolving
the GPU failures.** The continuous-rate optimizer search is secondary.

All six failures from the 16/22 CUDA control pass in fresh local CPU training.
These are new runs of the archived original source, PyTorch 2.13.0+cu126,
ordinary AMD Ryzen 5900X execution, one thread, AVX2, and seed 0. No MKL vendor
override is used. This is a six-host reproduction, not a new full CPU 22/22 run.

| Previously failing toy | Fresh CPU | Native CUDA | CUDA, CPU initial parameters | CUDA, all CPU random draws |
|---|---|---|---|---|
| trajectory | PASS | FAIL | **PASS** | **PASS** |
| img_intensity2 | PASS | FAIL | **PASS** | **PASS** |
| img_bars4 | PASS | FAIL | **PASS** | FAIL |
| img_blobs4 | PASS | FAIL | **PASS** | **PASS** |
| mode_hold | PASS | FAIL | FAIL | FAIL |
| vector_unequal_mass | PASS | FAIL | FAIL | **PASS** |
| Total in this six-host diagnostic | **6/6** | **0/6** | **4/6** | **4/6** |

No diagnostic row has a 22-toy score. Do not add these passes to the previous
sixteen, or combine passes from different columns. Each candidate needs its own
complete qualification before promotion. The ordinary CUDA recipe remains
**16/22**, including the three native 100-mode passes.

## What the controls establish

The native CUDA runs reproduce all six earlier final metric dictionaries
exactly. Their initial trainable parameter hashes differ from CPU despite the
same seed. PyTorch does not guarantee matching CPU/GPU results with the same
seed ([reproducibility documentation](https://docs.pytorch.org/docs/2.14/notes/randomness.html)).

The initialization-only control copies original CPU initial parameters onto
CUDA before training. It retains every random draw from the native CUDA control;
the complete ordered random-draw hashes match. All model computation, gradients,
Adam updates, and optimizer moments stay on CUDA. It recovers trajectory and all
three image gates without changing training samples or losses. Trajectory MSE
falls from .27465 to .0007363. Bars finishes with all four modes and HQ 1.

The full random-stream control uses CPU draws for initialization, data,
particles, noise, and evaluation, transferring those tensors to CUDA. Every
initial trainable tensor and every ordered random draw matches the CPU run
byte-for-byte, including call counts. Model arithmetic and updates still run on
CUDA. This recovers unequal-mass coverage/shape, intensity, blobs, and trajectory.
It is a diagnostic bridge, not a proposed production dependency on CPU sampling.

Ring still ends at seven modes even with matched initialization and random
draws. Bars fails with matched draws but passes with CPU initialization and
native CUDA draws. Matching randomness therefore does not eliminate the
backend sensitivity. These controls isolate a combination of initialization,
sample-stream, and arithmetic sensitivity; they do not establish a broken CUDA
kernel or a single universal fix. The loss functions and frozen scoring gates
are unchanged.

## Next experiments

Start from the original scheduled GAN recipe and use the initialization-only
control as the practical porting reference. Gate first on `mode_hold` and
`vector_unequal_mass`, then the other four regressions, then all 22. The ring
ends at seven modes; unequal mass covers its rare component but fails its
minimum covariance eigenvalue ratio (.03095 versus the .15 gate).

Use small, declared changes with a numerical or training-mechanism rationale;
keep seed 0, budgets, architectures, and thresholds fixed. Do not search seeds
or choose a different random-stream policy for each toy. Preserve the original
recipe's decay and auxiliary AE/token terms. Prioritize quality after
convergence; maintain the frozen acquisition gates as separate results. A full
GPU pass must precede a release claim, followed by an explicit stability check
that preserves this recipe's actual schedule.

## Replay and audit

Use the same PyTorch 2.13.0+cu126 / CUDA 12.6 / RTX A6000 profile as the
[original GPU control](../gpu-known-winner-control/README.md). All GPU runs use
FP32, deterministic algorithms, TF32 disabled, and ordinary Adam without fused
or foreach updates. The recorded controls ran on physical GPU 1.

```bash
python reports/toy100/cpu-recipe-gpu-port/replay.py \
  --gpu 0 --profile cuda_cpu_init --task trajectory \
  --workdir /tmp/cpu-recipe-port-new
python reports/toy100/cpu-recipe-gpu-port/audit.py
```

Profiles: `cpu`, `cuda`, `cuda_cpu_init`, `cuda_cpu_random`. The six names in
the table are supported. `prepare.py` reconstructs the exact sources from
checksum-verified retained archives. `batch.py` runs three workers at a time
and writes one completion per line. Local logs:

```bash
tail -F /ml2/hypergan/cpu-recipe-gpu-port-20260924/{batch,cpu-batch,init-batch}.log
```

[Audit](audit.json) regrades all **24 completed training runs**, checks update
counts/device proofs/config identity, binds worker/source hashes, verifies all
six initialization fixtures, and compares full random-stream hashes. The
initialization fixtures are captured before any optimizer update. Setup errors
(missing archived dependency/declaration files and the diagnostic CPU-stream
guard) are retained separately; all stopped before the first update and were
repaired. No failed training run is discarded or relabeled.

`probe-v1.py` is the exact earlier probe; `probe.py` additionally supports the
initialization-only control. The [portable replay check](replay-check.json) reproduces the initialization-only
trajectory control: all non-timing result fields, initial parameter hashes, and
the full random-draw receipt match. The archived historical training code and
current public package defaults remain unchanged.
