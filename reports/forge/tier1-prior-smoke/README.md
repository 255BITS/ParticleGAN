# Tier 1 ring16 and scalar Gaussian: prior diagnosis

This user-requested task-only diagnostic tests particle count, cloud versus MoG,
and MoG width on the two failing acquisition questions. It is based on develop
`7183d65d`. It grants no ordinary Forge qualification, changes no public default,
and preserves every archived result and the provisional calibration status.
The existing [current technique leaderboard](../technique-inventory.md) remains
the only leaderboard for `discriminator_stability`.

## Frozen comparison

[Protocol](protocol.json) declares nine prior configurations: 256, 1,024 and
4,096 learned rows, crossed with public `ParticlePrior` (sigma zero) and
`MoGParticlePrior` at fixed sigma .025 or .1. Locations use the public deterministic
R2 initializer, with the current Forge init_std=1 and uniform mixture weights.
Standardization is false. A cloud is an explicitly different finite-atom cohort.

Every configuration uses the exact [selected BCAP-pure recipe](../dualnorm-pacing-v2/DEFAULT_SELECTION.md):
dualnorm, G step .012, D/G 1.5, prior/G 2.5, momentum zero, relativistic loss,
fixed real/fake BCAP coefficient/cap 1, no spread regularization or additive
training noise, and constant schedules. **Trainer delta: none.** Task-owned
particle count and prior law are the only experimental changes. Network
architectures, latent dimension, target/data law, batch size 128, named seed-0
streams, training budgets and evaluation cadence remain fixed.

Scalar training uses 1,000 updates and at most 120 seconds per attempt; ring16
uses 400 updates and at most 300 seconds. The finite round contains 18 cells,
12,600 updates and 3,780 reserved seconds, with no retries or continuations.
Training and model/prior sampling require CUDA and refuse CPU fallback. Target
batch construction and the frozen scientific scorers retain their explicit CPU
reference law, then batches move to the GPU. Changing those reference draws
would confound the comparison. CPU work here is data generation, grading and
media rendering, not neural training.

Clean live public sampling supplies 4,096 samples at all 24 original checkpoints;
at least five consecutive terminal checks must pass. Initial illustration draws
preserve the evaluation stream. Constructor, target, training-noise and evaluation
streams are isolated and checkpointed. Publication checks identical data-sequence
and initial network hashes across all arms, identical initial locations within
each particle count, and a common trainer recipe after removing the two declared
task-owned resource/prior fields.

## Smoke versus distribution quality

Original full gates stay unchanged. A separately frozen **proposed smoke
projection** tests scalar location and width, or ring coverage, precision and
mass balance, at the same five terminal observations. It does not certify scalar
CDF shape or per-component covariance. Its bounds are existing acquisition
bounds, with quality requirements omitted rather than thresholds fitted to runs.

Independent controls verify both questions before training. Full gates reject
same-moment scalar uniform/two-atom laws and sixteen exact center atoms. The
reduced smoke question intentionally accepts these shape impostors. Both reject
point collapse to one location, a shifted scalar law, doubled scalar width and
an incorrect ring radius. This is explicit scope reduction, not evidence that
the original full-quality failures were false. Oracle controls validate scorers;
they do not calibrate a qualification profile.

## Reproduction and logs

Use the frozen scientific commit recorded in the completed receipt. From its
repository root, in the project Python environment, choose a fresh ignored path:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/test_tier1_prior_smoke.py
mkdir -p runs/api
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 python -u -m benchmarks.toy_audit.tier1_prior_smoke run \
  --device cuda:0 --output runs/api/tier1-prior-smoke-v1 \
  > runs/api/tier1-prior-smoke-v1.runner.log 2>&1
tail -F runs/api/tier1-prior-smoke-v1.runner.log
# Each start event names its per-task log, with live numerical observations.
python -m benchmarks.toy_audit.tier1_prior_smoke publish \
  --raw runs/api/tier1-prior-smoke-v1 \
  --output reports/forge/tier1-prior-smoke
```

Raw stdout, curves, tensors and complete checkpoints remain local. Publication
verifies hashes and renders nine actual saved observations per GIF, with the
target law alongside outputs; it adds no updates or model sampling draws.
Numerical verdicts use all 24 observations, not the nine displayed frames.

## Completed readout

Results, interpretation and recommendations are added after the frozen run.
