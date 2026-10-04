# 1-D Gaussian histogram matching

Can ParticleGAN learn `N(2, 0.5²)` from random initialization within a small
fixed budget? This new question tests acquisition of location, spread and the
whole scalar CDF. Histogram bins illustrate training; they do not grade it.

The [Forge task](../../../../configs/forge/tasks/gaussian1d_acquisition.json)
is the first required Tier 1 task in `discriminator_stability` revision 4.
That view has **6/19/2** required tasks and reserves **2,220 seconds** for the
complete Tier 1. The new task reserves at most **120 seconds**, one CPU thread.
[The v2 campaign](../../../../configs/forge/campaigns/tier1-acquisition-v2.json)
covers the expanded reservation; historical campaigns and evidence keep their
original view. Placement is provisional until independently calibrated.

## Frozen protocol

One standalone public `GANTrainer` run: protocol seed 0, **1,000 updates**,
batch 128, 256 learned MoG rows with two latent coordinates, fixed latent
sigma .025, uniform weights and no standardization. The shared MLP generator
has one scalar output; G/D use width 32, two hidden layers, and D has two
Fourier bands. G/D use deterministic orthogonal seeds 0/1; initial prior
locations are independent Gaussian draws with standard deviation .5.

The public K3P recipe uses LR .00425, D multiplier 1, prior multiplier 2,
betas (0,.999) and prior regularization 0. Training noise follows the resolved
recipe. Evaluation samples the live public MoG/G law with `output_noise=False`;
MoG kernel noise remains. The receipt records the actual recipe and training
noise. This standalone initializer/RNG cohort is separate from Forge's named
streams and supplies **no ordinary whole-view qualification or default credit**.

Before training, freeze 4,096 samples at 24 evenly spaced post-update checks.
All five terminal checks, updates **834, 875, 917, 959, 1,000**, must pass:

| Metric | Bound |
| --- | --- |
| Finite fraction | 1 |
| Sample count | ≥ 4,096 |
| Absolute mean error / target sigma | ≤ .20 |
| Sample standard deviation / target sigma | [.80, 1.20] |
| KS distance from the exact Gaussian CDF | ≤ .05 |

Independent oracle draws and destructive controls test the evaluator without
training: point collapse, shifted location, doubled width, same-moment two-point
and uniform laws, nonfinite samples and undersized evaluation. Controls do not
calibrate the tier. No seed sweep, automatic tuning or continuation is declared.

## Reproduce

Run from the repository root. Each execution needs an unused ignored directory.

```sh
python -m benchmarks.toy_audit.gaussian1d_experiment controls \
  --output /tmp/gaussian1d-controls.json
mkdir -p runs/api/gaussian1d-reproduction
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  timeout 120s python -u -m benchmarks.toy_audit.api_run \
  --case api-gaussian1d-acquisition --device cpu --wall-cap-seconds 120 \
  --output runs/api/gaussian1d-reproduction \
  > runs/api/gaussian1d-reproduction/runner.log 2>&1
tail -F runs/api/gaussian1d-reproduction/runner.log
python -m benchmarks.toy_audit.gaussian1d_experiment publish \
  --raw runs/api/gaussian1d-reproduction/api-gaussian1d-acquisition \
  --output /tmp/gaussian1d-publication
```

The run returns nonzero on FAIL. Raw logs, observation arrays and checkpoints
stay under ignored `runs/api`; publish only controls, endpoint/terminal metrics,
provenance and the actual-training GIF. The existing
[current solution leaderboard](../../../forge/technique-inventory.md) retains
its source-bound results. This task readout creates no competing ranking.
