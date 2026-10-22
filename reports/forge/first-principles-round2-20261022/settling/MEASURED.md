# Temporal settling readout: running

Both fresh controls exactly reproduce their first-round Gaussian and overlap
runs. Their retained actual update returns show **thermal scale 1 on all
14,400 updates** across those six cells. Mechanics and the Bregman game remain
pending; there is no completed successor comparison yet.

This is a saved-data diagnostic of the four admitted recipes. The
[local exchangeability thermostat](README.md) failed its fixed cheap screen,
was never implemented and incurred zero scientific GPU spend. None of the
four paid arms replaces the thermal observer. The parent
[comparison](../README.md) owns the complete fifteen-task results, certificate,
stream and paid-cost audits; this report does not add a leaderboard or confer
qualification.

## Frozen scope and temporal criteria

All observations and scalar update returns come from executed source
`a03373da3421303b89c99a39898b091e22910134`, source digest
`668914920afda7bb6fc53d9c918aac92a9c9cd5ae582884fb9f068ff1b8249e8`.
The [protocol](../protocol.json) fixes seed 0, public initialization, task
architecture/data/prior/sampling, consumed streams, budgets and evaluations.
Mechanics is compared with the energy control; the Bregman game is compared
with original BISPF. They are distinct global recipes.

Gaussian smoke passes when at least one of 24 scheduled states passes every
bound with an independent same-state confirmation. Its last sample can fail
while smoke passes. Stability resumes that recipe's own passing smoke
checkpoint, without resetting history. It requires all 72 stationary checks
after step 1000, every last-five recovery check by step 5000, and all 24
shifted hold checks. Each check requires 4096 finite samples, mean error <=.2
target sigma, width ratio in [.8,1.2], and CDF KS <=.05. Target mean changes
from 2 to 3 at update 4001; sigma remains .5. The deadline checks are at
4834, 4875, 4917, 4959 and 5000. Passing an endpoint or an earlier run of
checks cannot substitute for these unchanged temporal requirements.

Overlap retains 24 observations and five required terminal passing checks,
with mean error <=.15, covariance error <=.45 and normalized SW1 <=.18.
Its endpoint and whole-curve diagnostics remain distinct.

## Verified control repeats

For Gaussian smoke, Gaussian stability and overlap, both fresh controls have
exactly equal final metrics/gates, all scored metric dictionaries, all 340
retained sample sets, final G/D/prior/EMA tensors, and both complete optimizer
states relative to their own first-round controls. Sample
comparisons include retained independent smoke confirmation draws and frozen
shift controls. Source-specific state metadata hashes are excluded from
sample equality; artifact bytes and model/optimizer tensor identities are
verified separately. Both Gaussian data-sequence digests match as well.

| Diagnostic | Original control | Energy control |
| --- | --- | --- |
| Smoke | PASS; 6/24 primary checks pass, 4 confirmations | PASS; 8/24 primary checks pass, 6 confirmations |
| First confirmed smoke pass | 500 | 209 |
| Stability: stationary | 6/72 pass | 12/72 pass |
| Stability: recovery | 2/24 pass | 4/24 pass |
| Stability: last five by deadline | 1/5 pass: FAIL | 1/5 pass: FAIL |
| Stability: shifted hold | 2/24 pass | 0/24 pass |
| Frozen pre-shift control | 0/48 pass after shift | 0/48 pass after shift |
| Stability endpoint mean error, sigma units | .054736 | .028908 |
| Stability endpoint width ratio | .876510 | .925797 |
| Stability endpoint KS | .070319: FAIL | .065143: FAIL |
| Overlap passing observations | 10/24 | 20/24 |
| Overlap longest passing run | 3 | 14 |
| Overlap terminal suffix | 0: FAIL | 1: FAIL |

Energy reaches Gaussian quality earlier and improves the final mean/width,
but does not hold that quality. Its continuation fails KS at 104/120 checks,
mean error at 65/120 and width at 41/120; the corresponding original counts
are 110, 55 and 16. Failures can overlap within one observation. Energy also
improves overlap shape: its final covariance error .150126, normalized SW1
.052610 and mean error .040615 all pass. Nevertheless four observations fail
translation, including one near the end, leaving only one terminal pass.
This isolates sustained centering as its remaining overlap failure.

A fixed descriptive Gaussian check removes each saved panel's own mean and
population standard deviation, then measures empirical KS to the standard
normal CDF. This same-panel moment-fit KS is invariant to positive location
and scale changes, so it describes shape separately from those two moments.
Across continuation its median is .03459 for original BISPF and .05943 for
energy; in shifted hold the medians are .03678 and .08351. The final values
are .03336 and .06720. Energy's better final mean/width therefore coexist
with greater non-Gaussian shape departure. This is a descriptive empirical
fit residual, not a calibrated normality p-value, an additive decomposition of
target KS, or a replacement task gate. It uses the same 4096 saved samples
per observation; no reference sample is drawn.

The new trace closes the first round's sparse-telemetry limitation for this
fresh, source-bound controls. Each records earned thermal scale exactly 1 for
all 1000 smoke, 5000 continuation and 1200 overlap updates. Original minimum
EMA residuals are .080342, .073971 and .172530 respectively; energy minima
are .080687, .098649 and .167616. Every minimum is above the unchanged .020
cooler scale. This directly shows that existing cooling never activates on
these runs. It does not retroactively create missing first-round per-update
telemetry.

## Candidate measurements pending

Secant mechanics and Bregman game measurements will be filled only
after their own smoke/continuation/overlap jobs complete or their own checkpoint
prerequisites block continuation. An unexecuted blocked continuation will be
reported as such, without substituting another recipe's checkpoint.

The reducer retains phase-wise actual thermal/residual ranges, mechanics
acceptance and dyadic scale histograms, trial counts, raw normalized proposal
norms, final direction-protection changes and after-policy applied movement.
Its secant certificate refers to the pre-protection normalized field; final
anchor protection is a separate operation. A successful numerical certificate
or shorter move alone is not a quality or settling pass.

Bregman diagnostics retain selected-target routing, prices and canonical
mean-loss force before and after thermal scaling. These forces are in output
coordinates for the identity link and pre-sigmoid coordinates for a sigmoid
link; they include batch/dimension averaging. They are not parameter-space
gradients or final finite optimizer movement. A nonzero force alone does not
establish useful settling or support acquisition.

## Reproduce without training

[analyze_temporal.py](analyze_temporal.py) verifies saved artifact SHA256s,
recomputes unchanged phase bounds from declared task thresholds, checks exact
control sample/model/optimizer identities and reduces retained JSONL returns.
It constructs no model, performs no forward or optimizer update, and makes
no sampling draw. A hard 120 CPU-second process limit bounds the reduction.
Bulk traces and intermediate reduction outputs remain outside Git.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -u \
  reports/forge/first-principles-round2-20261022/settling/analyze_temporal.py \
  --output /mnt/ml7tb/ParticleGAN-forge/first-principles-round2-20261022/settling/temporal-reproduction.json \
  > /mnt/ml7tb/ParticleGAN-forge/first-principles-round2-20261022/settling/temporal-reproduction.log 2>&1
tail -F /mnt/ml7tb/ParticleGAN-forge/first-principles-round2-20261022/settling/temporal-reproduction.log
```

The default reducer refuses pending cells. `--allow-incomplete` produces an
explicitly incomplete intermediate result while the campaign runs; it does
not grade missing work as a pass.
