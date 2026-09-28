# Fresh-reservoir sampling calibration: native100 3/3

The sampling-only row EM predecessor at `12e55ede` passed grid100 and
staggered100 but missed rotated100's five-check centre streak. Its single
fit at update 2,664 was not refreshed as the support moved. This isolated
successor refits on every
fully renewed 16,384-real-example reservoir while the generator stationarity
scale is at or below 1/64. Each fit starts with uniform particle-row mass and
the current **training** output width. It estimates public sampling weights
and a separate Gaussian sampling width using five latent-jittered outputs per
row. No task label, target geometry, frozen score, or update deadline enters
the rule. `candidate.patch` changes only `particlegan/row_em.py` relative to
the preceding [sampling-only package](../row-em-sampling/README.md).

The package identified by runner SHA-256
`1fc12d29e2c5964f91c9523006e569cba743eba2e3e9851a68afa063c08cd324`
was frozen before the grid run, then transferred unchanged to rotated100 and
staggered100. All used the same QR/noisy fixture and 7,000-update protocol.
The formal gate requires **all five** full-accuracy observations at updates
6,000–7,000 and an independent 100,000-sample holdout. The console `ok` field
checks coverage only; the verdicts here use `acc_accuracy_pass` and each
runner's `native` result. All three have zero data-stream deviations.

| Task | Verdict | Full-accuracy observations | Final five | Holdout | Final precision | Centre RMS / σ | Covariance eigenvalues | Trace bias | Radial KS |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| grid100 | **PASS** | 21/34 | 5/5 | PASS | .98160 | .13017 | .62450–1.26521 | .00209 | .01307 |
| rotated100 | **PASS** | 16/34 | 5/5 | PASS | .98320 | .15228 | .52700–1.59410 | .04611 | .02806 |
| staggered100 | **PASS** | 16/34 | 5/5 | PASS | .98795 | .12412 | .51159–1.44523 | .01628 | .00920 |

Rotated100's terminal centre errors are .14756, .15639, .15147, .15743,
and .15228σ, each below the .20σ limit. Its previous single-fit package
missed four of these five checks with centre errors .2057–.2168σ, despite
passing its final check and holdout. The fresh-data fits resolve that
specific failure in this frozen protocol. Independent holdout precision is
.98398/.98290/.98737 for grid/rotated/staggered respectively; all holdout
accuracy fields pass.

## Audit and source reconstruction

`audit.py` produced `audit.json` from the raw local run directories and the
earlier critic-floor checkpoints. For every task, all 7,000 applied-rate rows
are byte-identical to the critic-floor baseline. Final G, D, EMA G, live and
EMA prior positions, optimizers, controller, birth/death state, training
streams, and training output sigma are identical, treating matching NaNs as
equal. The public sampler buffers, sampling width, and row EM state are the only
intended differences. The training output width remains .02000. The
sampling widths are .020667, .024322, and .022229 respectively. Public
samples therefore follow a calibrated law distinct from the one presented
to D during training; that distinction is part of this result.

`manifest.json` hashes all 26 Python source files. Applying `candidate.patch`
to the previous sampling-only package reconstructs those 26 files byte for
byte. `smoke.py` and `smoke.json` check a repeated fresh-reservoir fit,
private fitting RNG, unchanged training update, separate sampling width, and
checkpoint replay. This report archives each task's result, job header,
fixture, all 34 scored rows, all 7,000 rate rows, diagnostics, and SHA-256
hashes of final and holdout noisy sample files in `audit.json`. Full raw
samples and final checkpoints remain in the local
`/ml2/hypergan/gan-attempts/row-em-renew-20260928/runs/` directory.

## Native100 leaderboard and cost

| Frozen package | grid100 | rotated100 | staggered100 | Native total |
|---|---|---|---|---:|
| Critic-floor GAN | PASS | FAIL | FAIL | 1/3 |
| Single-fit sampling-only row EM | PASS | FAIL | PASS | 2/3 |
| Fresh-reservoir sampling-only row EM | PASS | PASS | PASS | **3/3** |

The frequent fit is expensive: 615/543/486 EM fits and 1,692/1,539/1,363
seconds for grid/rotated/staggered, versus 354/348/283 seconds for the
uncalibrated critic-floor runs. This is roughly 4.4–4.8 times the wall time.
A follow-up should gate refits on measured change in the public sampling
support, while retaining a fully renewed real reservoir, then rerun all
three frozen tasks. The current 3/3 receipt establishes accuracy for this
package; the full 22-check suite has not been scored for it, and no default
change follows from this native-only result.
