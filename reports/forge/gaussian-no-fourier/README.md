# Scalar Gaussian: remove critic Fourier features

This is the independently declared Fourier-only architecture ablation. It asks
whether a raw-input critic makes Gaussian acquisition and continued learning
easier under the unchanged winning BCAP recipe. The depth ablation retains its
original Fourier features and has its own PR and evidence.

The new Tier 1 question is whether training produces at least one full passing
Gaussian state within 1,000 updates, confirmed by an independent sample draw at
the same state. All 24 paired observations and all 1,000 updates still execute.
Tier 2 tests every remaining stationary check through 4,000, then shifts the
target mean from 2 to 3, requires five terminal passing checks by 5,000, and
requires every subsequent check through 6,000 to pass. Historical five-terminal
acquisition verdicts remain a separately reported comparison.

## Fixed comparison

[Protocol](protocol.json) and explicit [smoke](smoke.json) /
[stability](stability.json) cards bind the architecture change: critic Fourier
count 2 becomes 0. Its first layer receives raw scalar input instead of raw
input plus sine/cosine features at frequencies pi and 2pi. Generator and critic
remain width 32 with two LeakyReLU hidden layers; latent dimension remains 2.
Generator parameter count remains 1,185; critic count drops from 1,281 to 1,153.

The public named deterministic initializer preserves every generator and prior
tensor. The critic first weight, its fan-in dependent initial bias and the
Fourier frequency buffer change as expected. Every deeper critic tensor matches
the archived initial state. Only the critic constructor stream changes; training
streams and initial optimizer state/rates match exactly. This is recorded in
[the zero-update CUDA proof](initial-proof.json), with [reproduction](preflight.py).
The unchanged generator's archived capacity control remains applicable as a
representation diagnostic; it is not a trained pass.

The learned prior remains 256 uniform MoG locations with sigma .1, without
standardization; target N(2,.5²), batch 128 and protocol seed 0 are fixed. Trainer
delta is zero: alternating BCAP dualnorm, constant G .012, D .012×1.5 and prior
.012×2.5, zero momentum, no prior regularizer, EMA, annealing or additive output
noise. Clean live public sampling retains MoG kernel noise.

Every distribution check uses 4,096 samples and the original full bounds: mean
error ≤ .2 target sigma, standard deviation ratio [.8,1.2], KS ≤ .05 and finite
fraction 1. The independent confirmation stream runs at every smoke check, so
the sampling schedule does not depend on outcomes. All 72 stationary hold and
48 shifted checks execute; the last 24 shifted checks form the hold phase. A
no-update copy of the own 4,000 checkpoint uses matched shifted evaluation draws.

The single finite trial reserves 720 seconds: 120 for smoke and 600 for the
5,000-update own-state continuation. It spends at most 6,000 new updates and
768,000 real examples, with zero scientific retries or post-result tuning.
All neural initialization, training, sampling and checkpoint restores use CUDA.
Inherited CPU target generation, saved-output scoring and media rendering are
explicit exceptions. The public shared `GANTrainer` host performs every update;
the architecture adapter validates its frozen inputs and contains no training
loop.

[Archived controls](controls.json) retain the original Fourier-2 alternating
4000/6000 evidence and verdicts under their actual source identities, with zero
new cost. Original sigma-.025 Gaussian acquisition remains another task. This
sigma-.1 architecture diagnostic does not confer ordinary qualification or
retroactively change previous failures. The architecture change is Gaussian-only;
the passing ring recipe and its source-bound evidence require no new ring run.

## Reproduction and logs

Use the frozen scientific commit recorded in the completed provenance. Choose a
fresh ignored raw path for an explicitly authorized reproduction:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 /usr/bin/python -u \
  -m benchmarks.toy_audit.gaussian_no_fourier \
  --device cuda:0 --output runs/api/gaussian-no-fourier-v1 \
  > runs/api/gaussian-no-fourier-v1.log 2>&1
tail -F runs/api/gaussian-no-fourier-v1.log
```

Raw stdout, observation arrays, RNG states, optimizer diagnostics and checkpoints
remain ignored locally or in the immutable archive. Publish saved outputs and
verify exact CUDA restores without training:

```sh
/home/martyn/dev/ParticleGAN/.venv/bin/python \
  -m benchmarks.toy_audit.gaussian_architecture_publish \
  --raw runs/api/gaussian-no-fourier-v1 \
  --output reports/forge/gaussian-no-fourier \
  --protocol reports/forge/gaussian-no-fourier/protocol.json
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 /usr/bin/python \
  -m benchmarks.toy_audit.gaussian_architecture_publish \
  --raw runs/api/gaussian-no-fourier-v1 \
  --output reports/forge/gaussian-no-fourier --verify-state --device cuda:0
```
