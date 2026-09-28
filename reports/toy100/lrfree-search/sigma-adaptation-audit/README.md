# Effective sigma release and shared-clock audit (native grid100)

Both source-isolated sigma follow-ups to the committed [prior-rate coupling](../prior-rate-coupling/README.md) finished the unchanged 7,000-update QR/noisy grid100 gate and independent 100,000-sample holdout. **Neither passes.** The base uses paired birth/death, `batch_feature_zero` initialization, `total_steps: null`, and learnable output noise starting at 0.02. All recipe overrides and BD code are byte-identical across these two follow-ups.

| Candidate | σ at 7k | Noisy precision | Centre / data σ | Covariance eig ratios | Trace bias abs | Radial KS | Frozen result |
|---|---:|---:|---:|---:|---:|---:|---|
| Committed prior coupling | .02000 | .96875 | .12646 | .572–1.330 | .00957 | .03298 | FAIL, precision |
| Effective-scale release | .00954 | .98425 | .12048 | .485–1.205 | .12566 | .10117 | FAIL, trace and radial |
| Shared intrinsic clock | .01905 | .97065 | .12402 | .505–1.302 | .02866 | .04401 | FAIL, radial |

The runner's `passing_checks` field counts an older coverage-style observation rule (12/34 and 2/34 for these arms). The stricter accuracy check passed **0/34** observations for each, including **0/5** at the required terminal checkpoints.

The effective-scale patch changes `_output_sigma()` and `_sigma_intrinsic_scale()` to read **applied** non-sigma G/prior rates rather than the raw prior tester, and excludes sigma's own tester from the floor. This corrects a real state mismatch: the coupled prior's raw tester stayed at 1 while its applied rate had fallen to 1/64. Sigma first released at update 3,625 with its own full rate 0.00425, then shrank rapidly. Its independent holdout also fails trace (.11409) and radial KS (.09406).

The shared-clock patch changes one return value: once the release gate opens, sigma's own tester rate is multiplied by the minimum applied non-sigma scale. At release this is 1/64, giving sigma LR 6.640625e-5, equal to G's applied LR. Its independent holdout passes precision (.97103) but misses radial KS at .04042 versus the .04 limit. The final five live checks do not all pass; the first three miss precision and radial accuracy. Both runs recorded zero stream deviations. The patches reconstruct exact executed training-source hashes in `source-sha256.json`; the full raw result, fixture, per-check metrics and noisy verdict for each arm are included.

## Read-only width and objective probes

Rescaling the saved **prior-coupled** clean/noisy clouds at fixed sigma 0.018–0.020 never passes all five final checks. A fixed post-training sigma adjustment alone does not rescue that generator.

For the **effective-scale** run's saved generator, fixed σ=.017 makes all five terminal live and EMA clouds and independent 100k holdouts pass under the same evaluator. This is a **post-hoc diagnostic selected after seeing outcomes, not an online candidate or valid gate pass**. At σ=.0174, live passes only four of five and EMA three of five; the misses are early precision. The trained σ=.00954 fails. This shows a feasible generator law exists while the sigma update chooses a poor width.

At the saved σ=.00954 checkpoint, 12 independent held-out 8,192-point batches give local Gaussian MMD derivative `dL/dlogσ = −1.5611e−4` (95% CI `[−1.6289e−4,−1.4932e−4]`), favoring inflation, while the saved critic's RpGAN derivative is `+2.9408e−6` (95% CI `[+2.6965e−6,+3.1851e−6]`), favoring shrinkage. The data-derived MMD bandwidth is the median real 10th-neighbor distance, .0218439. MMD's estimated root is about .0178 and forward empirical-mixture likelihood's about .0187; both are wider than the fixed-width interval that passed the saved generator's final five checks. A unit-weight MMD generator auxiliary would have a gradient roughly 46–71 times the GAN G gradient at this checkpoint, so it was **not** launched as a training candidate.

The `probe-*` files hold exact pools, statistics and scripts' machine outputs; `prior-coupling-fixed-sigma-sensitivity.*` records the separate saved-cloud control. No frozen metric, task name or threshold was fed into any online training run in this audit. The post-hoc rescores are explicitly diagnostic and do not change the leaderboard.
