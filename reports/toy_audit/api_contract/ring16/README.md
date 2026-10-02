# Sixteen Gaussian clusters: acquisition

[Current tier report](../../../forge/EXPERIMENTS_BY_TIER.md#experiment-ring16-acquisition) · [Forge task](../../../../configs/forge/tasks/ring16_acquisition.json) · [Dedicated provisional view](../../../../configs/forge/views/ring16_acquisition.json)

This new retained question asks whether training from scratch acquires all 16 equally weighted 2D Gaussian clusters within a small fixed budget. The means are `3(cos(2πk/16), sin(2πk/16))`, k = 0,…,15; each covariance is `0.01I`. Adjacent centers are about 11.7 Gaussian standard deviations apart. Acquisition is separate from extended retention or target adaptation.

The user chose an initial **Tier 1** placement. This is a provisional research placement, with its own view; it is not a calibrated speed/rejection conclusion. The existing eight-mode `mode_hold` task, frozen screening views, historical qualification and the current release leaderboard keep their original identities and denominators.

## Preregistered protocol

One full standalone public-API run, protocol seed **0**, **400 updates**, batch 128, 256 learned MoG rows in four latent dimensions. Generator/discriminator are the shared vector MLPs, width 64, two hidden layers; the discriminator uses two Fourier bands. The public K3P preset resolves the training mechanism; learning rate .00425, discriminator multiplier 1, prior multiplier 2, betas (0,.999), prior regularization 0. Initial networks use deterministic orthogonal seeds 0/1; prior locations use Gaussian init_std .5 seed 0. Prior sigma is fixed at **.025**, standardization is off. Training noise follows the exact resolved recipe. Scoring uses **live, clean outputs** while retaining the MoG latent kernel; EMA, output-noise and policy-served evidence do not substitute for these draws.

These API initializer/data/trainer RNG streams are a separate cohort from Forge's named `FormulationContext` streams. A completed API demonstration supplies **no ordinary Forge qualification**. The task uses the same target, shared vector host and numerical scorer through the existing public Forge adapter, but a future Forge receipt must bind its own exact candidate/source/runtime.

The budget, geometry and gates are frozen before the sole scientific run. Evaluation draws contain 4,096 samples at 24 fixed post-update observations. The last five checks (updates 334, 350, 367, 384, 400) must jointly pass. These are acquisition checks inside the original budget; there is no post-acquisition continuation. Timeout is 300 seconds, one GPU and one CPU thread. No seed-only repetition, tuning after failure or calibration matrix expansion is included.

| Gate | Numerical bound | Purpose |
|---|---|---|
| Finite output | All generated values finite | Reject execution/NaN failures |
| Sample count | ≥ 4,096 | Preserve declared statistical resolution |
| Meaningful modes | All 16 | Each mode has ≥ 1/4 of its target mass **inside its 3-sigma ball**, divided by all generated samples |
| Mass TV | ≤ .15 | Roughly balanced nearest-component mass |
| HQ | ≥ .85 | Most outputs are inside target clusters |
| Average component covariance error | ≤ .85 | Bound gross local width mismatch |
| Minimum component covariance eigen ratio | ≥ .15 | Reject collapsed width in any component |

These are deliberately broad acquisition bounds, not tight Gaussian density certification. In particular covariance error is averaged across components; the minimum eigen gate applies to every component. Projected CDF and global moments are reported as diagnostics, without adding unregistered gates.

## Scorer controls

The [compact control receipt](controls.json) records an independent full target draw (PASS) and eight destructive draws (all correctly rejected): missing cluster, biased mass, centers only, collapsed width, inflated width, continuous circle, displaced target and nonfinite output. They execute **zero training updates**. Controls establish discrimination against these counterexamples; they do not calibrate the smoke screen's predictive value for trained solutions.

## Result

The sole frozen full-budget demonstration has not run in this preregistration commit. Its endpoint metrics, failed bounds, actual-training GIF and source/provenance receipt will be added without changing this protocol.

## Reproduce and inspect

From the repository root in the project environment, these commands use an unused ignored raw output directory. The scientific command runs the sole declared protocol; its exit status is nonzero on numerical FAIL. The JSON lines are easy to tail.

```sh
python -m benchmarks.toy_audit.ring16_controls --output /tmp/ring16-controls.json
mkdir -p runs/api/ring16-reproduction
python -m benchmarks.toy_audit.api_run --case api-ring16-acquisition \
  --device cuda:0 --output runs/api/ring16-reproduction \
  > runs/api/ring16-reproduction/runner.log 2>&1
tail -f runs/api/ring16-reproduction/runner.log
python -m benchmarks.toy_audit.ring16_publish \
  --raw runs/api/ring16-reproduction/api-ring16-acquisition \
  --output /tmp/ring16-publication
```

The caller selects the declared seed 0; the published metadata records it explicitly. The exporter verifies the complete original receipt, protocol and artifact identities; it copies compact endpoint/terminal evidence and the GIF and launches no training or rescoring. Checkpoints, arrays and the complete observation/log streams stay in ignored local storage.

## How to determine tiers scientifically later

Before a separate bounded calibration, freeze independent positive/negative reference criteria, exact prior/init/recipe/sampling/runtime cohorts and a total compute cap. Compare this acquisition screen against independent later-quality/retention tasks rather than use its own answer as the reference. Measure full execution cost, rejection of failing lineages and false rejection of successful ones. Compare cost savings and incremental predictive value against the other proposed Tier 1 tasks, with unknowns retained.

The existing [frozen Forge calibration criteria](../../../../configs/forge/calibration/criteria-v1.json) provide a useful acceptance template: at least three paired lineages, one reference-positive and two reference-negatives, ≥90% paired coverage, zero false rejection, ≤10% false acceptance, complete cost vectors and smoke/reference cost ratio ≤.1 within each cohort. A new profile must declare how these apply before spending; a single API run or oracle draw cannot satisfy them. A justified future tier or budget change gets a new protocol/view revision and preserves this evidence. No broad calibration campaign follows from this PR.
