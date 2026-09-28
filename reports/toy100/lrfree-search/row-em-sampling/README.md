# Sampling-only row-mass and width calibration for PR 155

The preceding online row EM candidate changed the fake distribution used by
the adversarial game and failed the frozen grid100 five-check streak. This
isolated successor keeps the critic-floor GAN update and birth/death draw law
uniform, including the original training output sigma. It learns a separate
sampling law from real batches seen by `GANTrainer.step`.

At the first non-sigma G stationarity scale at or below 1/64, the existing
row-associated EM update fits particle-row weights and a shared Gaussian
sampling width from a 16,384-real-example FIFO reservoir. It uses five
independently jittered generator outputs per row, a private random stream,
and no task names or frozen evaluator values. Public `sample()` and the native
harness draw from the weighted rows with the fitted width. The G and D updates
and birth/death continue to draw uniformly with the original training width.
The sampler resets on a detected data-law reopen and can refit after a later
optimizer reopen and new settle transition.

`candidate.patch` is the complete difference against the critic-floor source
package named in `manifest.json`; it applies after the committed
`critic-floor/package.patch` and was verified byte for byte against all
candidate Python sources. `smoke.py` compares an activated calibration
with an inactive copy across the same update: G, D, prior locations, optimizer,
controller, birth/death, all training streams, and losses match exactly. It
also checks private fitting RNG, separate sampling width and checkpoint replay.
The complete local frozen runs are at
`/ml2/hypergan/gan-attempts/row-em-sampling-20260928/runs/`. This report
includes each task's `result.json`, header, fixture, 34 scored rows, 7,000
rate rows, diagnostics, and hashes of its raw final and holdout noisy sample
files. Formal verdicts include the final five noisy checks and independent
holdout.

## Frozen grid100 result

The exact package hash in the runner header is
`db67fb7ff12df9f9352b388e8a1a29ba1bc544c18431a1438c178c84756e14e3`.
Grid100 **PASSES** all five terminal noisy checks and its independent 100,000
sample holdout; 21/34 observations pass, final streak 21, zero data-stream
deviations. Final precision is .98360, centre RMS .18424σ, covariance
eigenvalue range .54249–1.30734, absolute trace bias .02017, and radial KS
.02045. The training width remains .02000; the fitted sampling width is
.020724. The sole EM fit was triggered by G settling at update 2,088.

All 7,000 applied-rate rows match the earlier passing critic-floor run. At the
final checkpoint, G, D, EMA G, live and EMA prior positions, optimizers,
controller, birth/death state, training streams and training output width match
that baseline exactly. The stationarity state also matches numerically,
including identical NaN positions. The sampler calibration changed only the
public sampling distribution. Unchanged rotated100 and staggered100 frozen
transfers were started only after this grid verdict.

## Frozen three-task result

The same package hash and QR/noisy fixture were used for all three runs.
Each trained G, D, EMA G, live and EMA prior position table, optimizer,
controller, stationarity state, birth/death state, training stream, and all
7,000 rate rows matches its uncalibrated critic-floor counterpart. All three
runs have zero data-stream deviations. The public sampler was calibrated once
per task from training real batches; the adversarial learner was unchanged.

| Task | Formal verdict | Full accuracy observations | Last five | Holdout | Fitted sampling σ |
|---|---|---:|---:|---|---:|
| grid100 | PASS | 21/34 | 5/5 | PASS | .020724 |
| rotated100 | FAIL | 17/34 | 1/5 | PASS | .024214 |
| staggered100 | PASS | 21/34 | 5/5 | PASS | .022259 |

Rotated100 misses the five-check streak solely through centre error: at
updates 6,000, 6,250, 6,500 and 6,750 the centre RMS is respectively
.2096σ, .2158σ, .2057σ and .2168σ against a .20σ limit. Its final check
passes at .1964σ; final precision is .97835, covariance eigenvalues
.53249–1.68414, absolute trace bias .05977 and radial KS .02439. Its
independent holdout also passes. The sole fit occurred at update 2,664;
subsequent birth/death teleports and gradual support movement did not refresh
the sampling weights. A fresh-data refit is the next isolated test.

This is a sampling calibration attached to a GAN checkpoint. Its calibrated
public samples are not the distribution presented to the discriminator during
training; that distinction must accompany any result claimed for this package.
