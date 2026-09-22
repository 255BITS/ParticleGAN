# Development stress tasks

These eight tasks were specified before their reference runs. Six contribute to
ranking; two diagnose intentionally weak architecture or ambiguous data. Neither
tier adds eligibility blockers: the existing nine required behavioral toys remain
the blockers. The tiers and thresholds stay fixed even when neither reference
solves a task. Such a result is reported as **reference solvability not
demonstrated**, with both attempts retained.

Every development episode uses seed 0, CPU, 24 fixed live-weight observations and
a passing suffix of at least five observations. EMA is reported separately and
cannot rescue a live failure. Fixed cosine and fixed constant schedules are the
only calibration references; no learned policy is fitted here, and there are no
seed sweeps or post-result threshold changes.

The common target is an equal eight-component ring of radius 3, with Gaussian
standard deviation .12. The base model has 256 particles, latent dimension 4,
two-layer width-64 generator and critic, two Fourier bands, batch 128, generator
LR .001, discriminator multiplier 1.5 and prior multiplier 10. The cap coefficient
is 3, its bound is 1.25, and the prior variance/covariance weight is .05. Other
settings are recorded numerically in [stress_tasks.py](stress_tasks.py).

| Task | Tier | Change | Why it matters | Limitation |
| --- | --- | --- | --- | --- |
| Fast critic | Ranking | D LR multiplier 3 | Ordinary optimizer imbalance | Does not change D update frequency |
| Slow critic | Ranking | D LR multiplier .75 | Generator can outrun density estimation | An adequate network can still need more updates |
| Small batch | Ranking | Batch 64 | Routine memory limits and noisier gradients | Half as many training examples at the same update budget |
| Large critic | Ranking | D width 128 | Transfer across capacity and gradient scale | More work per update; report wall time |
| Long horizon | Ranking | 2,400 updates | Persistence and delayed collapse | Observations are 100 updates apart |
| R1+R2 | Ranking | R1+R2 coefficient .1 | Transfer across supported penalty formulations | Coefficients are not an equal-strength calibration |
| Weak critic | Diagnostic | D width 16, one layer, no Fourier bands | Reveal an architecture bottleneck | Artificially limited resolution must not block selection |
| Overlapping data | Diagnostic | Target standard deviation .6 | Diagnose ambiguous component labels | Score the whole distribution, not unreliable component identities |

The first seven tasks require normalized sliced Wasserstein-1 distance ≤.18,
component mass total variation ≤.15, Mahalanobis-radius-3 HQ ≥.85 and relative
component covariance error ≤.85. These numerical distribution checks prevent
mode-center memorization from receiving full credit. The overlapping-data task
uses only sliced distance ≤.18 because component-conditioned diagnostics are
inappropriate for strongly overlapping labels. All thresholds are declared in
the specs, not inferred from reference results.

The reserved dynamics family uses a previously unseen update cadence: the critic
updates every second outer step, and the generator updates every step. Its static
target permits the same faithful distribution metrics. The definition is
published, but it is **never evaluated during development**. The parent can unlock
it only after freezing the selected method. Report actual D/G update counts and
wall time because outer-step budgets alone do not represent equal work.

## Frozen reference results

All 16 predeclared attempts completed. The overlapping-data diagnostic has a
demonstrated sustained reference solution. **Reference solvability is not
demonstrated for any of the six ranking stresses or the weak-critic diagnostic.**
This is not evidence that those tasks are impossible. Their tiers remain as
declared, and none becomes an eligibility blocker.

| Task | Fixed schedule | Final numerical gate | Passing suffix / 24 | Live HQ | Component covariance error | Seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Fast critic | Cosine | FAIL | 0 | 98.05% | 3.369 | 6.49 |
| Fast critic | Constant | FAIL | 0 | 99.17% | 1.622 | 5.56 |
| Slow critic | Cosine | FAIL | 0 | 82.15% | 26.870 | 5.67 |
| Slow critic | Constant | FAIL | 0 | 86.43% | 18.451 | 5.51 |
| Small batch | Cosine | FAIL | 0 | 76.61% | 38.125 | 5.10 |
| Small batch | Constant | FAIL | 0 | 82.45% | 19.666 | 5.08 |
| Large critic | Cosine | FAIL | 0 | 94.38% | 7.297 | 6.39 |
| Large critic | Constant | FAIL | 0 | 95.73% | 5.540 | 6.27 |
| Long horizon | Cosine | FAIL | 0 | 99.29% | .973 | 9.96 |
| Long horizon | Constant | FAIL | 0 | 98.63% | 1.015 | 10.03 |
| R1+R2 | Cosine | FAIL | 0 | 83.98% | 23.117 | 5.21 |
| R1+R2 | Constant | FAIL | 0 | 88.79% | 12.606 | 5.21 |
| Weak critic — diagnostic | Cosine | FAIL | 0 | 12.28% | 32.130 | 3.73 |
| Weak critic — diagnostic | Constant | FAIL | 0 | 20.65% | 32.580 | 3.69 |
| Overlapping data — diagnostic | Cosine | PASS | 23 | Not scored | Not scored | 5.33 |
| Overlapping data — diagnostic | Constant | PASS | 23 | Not scored | Not scored | 5.21 |

Both overlapping-data references confirm a sustained suffix at update 300 of
1,200; final normalized sliced distances are .0535 and .0357. All sixteen
references pass the final sliced-distance threshold, so the separated-target
failures reveal behavior that global distance alone misses. In particular,
98–99% HQ can coexist with an incorrect component covariance. Covariance uses
all generated points assigned to a component, including tails; a small number of
outliers can therefore matter even when HQ is high. These observations do not
justify weakening that gate or pretending the tasks have a demonstrated training
solution.

All attempts, numerical cards, full live/EMA curves, timings, errors, source
hashes and the never-evaluated reserved spec are in
`/tmp/pr36-transfer-stress-v1/results.json`. The manifest was written before the
first run at `/tmp/pr36-transfer-stress-v1/frozen_specs.json`; the serial runner
and tail-able log are beside it. Total episode time was 94.44 CPU seconds. These
are single runtime observations on a shared host, not statistically precise
performance estimates.

Recorded hashes:

- Stress specification: `e885a085c4bf27a17091cb3c4d1de4cdf691aa6561a657f841fa67871417668d`.
- Common vector runner: `433b0566ca07a0118ea0fcae188975658df48e8b84503998af163aa55664bd3d`.
- Complete reference results: `df1cc9cc06c1dcef55ef398453d748a0f7eb6687b94728bb87c756539fd909d2`.

The stress source stayed unchanged throughout calibration. No learned-policy
fitting, seed experiments, threshold changes, tier changes or reserved-family
evaluation occurred. The metric tests additionally demonstrate that target
samples can satisfy the numerical gate and memorized centers fail despite perfect
HQ and occupancy; the former is a scoring sanity check, **not** evidence of a
trainable reference solution.
