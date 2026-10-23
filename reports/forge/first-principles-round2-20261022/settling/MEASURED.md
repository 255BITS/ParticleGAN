# Temporal settling readout: completed twelve-cell diagnostic

Both fresh controls exactly reproduce their first-round Gaussian and overlap
runs. Mechanics and the Bregman game both repair overlap's declared terminal
window, while every recipe still fails Gaussian sustained settling. Both new
mechanisms follow the Gaussian mean shift quickly and improve its shape;
mechanics also improves pass frequency substantially. None of these gains
establishes sustained Gaussian quality. The existing thermal cooler remains
inactive on every retained update across all twelve cells.

This is a saved-data diagnostic of the four admitted recipes. The
[local exchangeability thermostat](README.md) failed its fixed cheap screen,
was never implemented and incurred zero scientific GPU spend. None of the
four paid arms replaces the thermal observer. The parent
[comparison](../README.md) owns the complete fifteen-task results, certificate,
stream and paid-cost audits; this report does not add a leaderboard or confer
qualification.

The [compact measured result](temporal-results.json) binds all twelve completed
cells and exact immutable queue inputs. It does not claim completion of the
parent campaign's remaining native/image/control cells.

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
while smoke passes. Stability resumes that recipe's own terminal 1,000-update
smoke checkpoint after the acquisition gate passes, without resetting history.
That terminal state itself can fail output bounds. Stability requires all 72 stationary checks
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

The new trace closes the first round's sparse-telemetry limitation for these
fresh, source-bound controls. Each records earned thermal scale exactly 1 for
all 1000 smoke, 5000 continuation and 1200 overlap updates. Original minimum
EMA residuals are .080342, .073971 and .172530 respectively; energy minima
are .080687, .098649 and .167616. Every minimum is above the unchanged .020
cooler scale. This directly shows that existing cooling never activates on
these runs. It does not retroactively create missing first-round per-update
telemetry.

## Mechanics: reaches quality sooner, still fails sustained settling

The same-batch numerical correction is active and materially changes the
trajectory. It passes smoke at the first independent confirmation at 125,
with 10/24 primary passes and a passing final state. Continuing its own state
raises stationary passes from energy's 12/72 to **40/72**, recovery passes
from 4/24 to **11/24**, and shifted hold from 0/24 to **12/24**. Its first
post-shift full pass is at **4042**, only 42 updates after the target changes;
energy first passes at 4334 and original BISPF at 4709. The method follows
the changed law quickly, then loses quality intermittently.

The original stability gate remains **FAIL**: only 1/5 deadline checks passes,
the longest stationary passing run is six, and shifted hold ends with only
three consecutive passes. Its final KS .038687, mean error .044355 sigma and
width .918820 all pass; an endpoint-only verdict would hide the temporal
failure. KS fails at 57/120 checks, mean error at 26 and width at 18; these
failures overlap. The descriptive moment-fit shape KS median falls from
energy's .05943 to .02124, and from .08351 to .01656 during shifted hold.
The final shape KS is .01087. Shape improves as well as centering, while
stable quality remains unsolved.

All 5000 continuation steps accept a bounded numerical trial, with no zero
actual moves and no recorded secant-bound violations. Alpha is below 1 on
84.8% of updates; its median is .25. The median **actual applied norm/base
proposal norm** is .22877, a different measurement from alpha. Final anchor
direction protection changes 1372/5000 proposals. These are Euclidean norms
over combined G/prior parameters, not output-space displacement or separate
role kinetics. The trace cannot establish which role dominates the norm.

Thermal scale remains exactly 1 at every smoke and continuation update; the
minimum continuation residual is .070802. The observed improvement therefore
accompanies an active numerical update while the existing thermal cooler
remains inactive. The local, pre-protection consistency check has no joint
game or cross-batch stability certificate. Its successful trials and smaller
combined moves do not override the failed temporal gate.

Mechanics also repairs the declared overlap gate: **22/24 observations pass,
with exactly five consecutive terminal passes at 1000–1200**. Excursions at
550 and 950 remain visible. Its longest passing run is ten, versus energy's
fourteen; energy nevertheless fails because its terminal suffix is only one.
The mechanics endpoint passes with covariance error .124908, normalized SW1
.049873 and mean error .059381. This is a repair of the fixed terminal-window
requirement, not uniform stability over every observation.

On overlap all 1200 numerical steps accept, alpha is below 1 on 979 steps,
median alpha is .5 and median actual/base norm is .47279. Final protection
changes 55 proposals. No zero actual move or secant violation is observed.
Thermal scale is always 1, with minimum residual .164169. The overlap repair
therefore occurs with active finite-step correction while the original
thermal observer remains hot.

## Bregman game: shape and fast response without stable centering

Game smoke passes with five independently confirmed passing states, first at
167, while its endpoint fails KS and mean error. Its own continuation passes
**10/72 stationary, 4/24 recovery and 2/24 shifted hold** checks. It first
recovers all bounds at 4042, matching mechanics' 42-update response, but
the deadline still passes only 1/5 checks. No passing run exceeds two checks;
the final state fails KS (.110321) and mean error (.231946 sigma), while
width .913882 passes. KS fails at 104/120 observations, mean error at 55 and
width at nine. The original stability gate remains **FAIL**.

The same-panel moment-fit shape KS median is .02340 across continuation,
.01723 in shifted hold and .01299 at the endpoint. These are much closer
to mechanics' .02124, .01656 and .01087 than to energy's larger shape
residuals. A nearly Gaussian shape can still move away from the target mean.
The shape diagnostic does not turn a failed target-law gate into a pass or
identify the cause of that motion.

All 6000 game smoke and continuation updates have thermal scale exactly 1
and nonzero canonical and thermally scaled force, with no recorded scaling
mismatch. During continuation the canonical force RMS has median .00079480
and range [.00023809, .00959807]; the selected minibatch cost median is
.0012881, target-routing fraction median .53906 and price standard deviation
median .010140. Thus the game is active and its small selected cost coexists
with missed global distributional bounds. A selected-target minibatch cost
does not enforce the full target marginal or certify temporal stability.

Game overlap passes with **21/24 observations and nine consecutive terminal
passes at 800–1200**. Translation excursions at 200, 700 and 750 remain
visible; every covariance and SW1 check passes. Its endpoint covariance error
.259825, mean error .059326 and normalized SW1 .080468 pass. This is a
scoped terminal-window repair relative to original BISPF's zero terminal
passes, distinct from its failed Gaussian centering and retention.

All 1200 overlap updates have nonzero canonical force, thermal scale 1 and
no force-scaling mismatch. The median force RMS is .0012036, selected cost
.017110 and target-routing fraction .52344. These batch/output-coordinate
quantities do not measure final parameter movement or prove uniform quality
across the trajectory. The game has no observed cold update in its combined
7200 smoke, continuation and overlap steps; minimum EMA residuals are
.080029, .073429 and .162494 respectively.

## What the return diagnostics can establish

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

The fixed temporal diagnostics compare acquisition, retention and shape:

| Fixed diagnostic | Original | Energy | Mechanics | Bregman game |
| --- | --- | --- | --- | --- |
| First confirmed smoke pass | 500 | 209 | 125 | 167 |
| Stationary full passes | 6/72 | 12/72 | 40/72 | 10/72 |
| Recovery full passes | 2/24 | 4/24 | 11/24 | 4/24 |
| First post-shift full pass | 4709 | 4334 | 4042 | 4042 |
| Deadline last-five passes | 1/5 | 1/5 | 1/5 | 1/5 |
| Shifted-hold full passes | 2/24 | 0/24 | 12/24 | 2/24 |
| Stability endpoint passes all bounds | No | No | Yes | No |
| Sustained stability gate | FAIL | FAIL | FAIL | FAIL |
| Hold moment-fit shape KS median | .03678 | .08351 | .01656 | .01723 |
| Overlap full passes | 10/24 | 20/24 | 22/24 | 21/24 |
| Overlap longest passing run | 3 | 14 | 10 | 9 |
| Overlap terminal suffix | 0 | 1 | 5 | 9 |
| Overlap declared gate | FAIL | FAIL | PASS | PASS |

All four recipes have exactly 7200 retained updates across these three tasks,
and all **28,800** earned thermal scales are exactly 1. This is direct
source-bound telemetry, rather than an inference from sparse checkpoints.
No inactive-cooler result shows that a different observer would be safe:
the local exchangeability proposal failed before training, and global
energy-U controls previously cooled missing support prematurely.

Keep mechanics' overlap repair and Gaussian progress as measured positives,
while retaining every temporal failure. A future joint-game or noise-aware
settling mechanism is a hypothesis requiring its own support and finite-sample
controls, not a conclusion from the successful frozen-critic secant trials.
No extra training or successor follows from this readout. The rejected local
exchangeability observer and the earlier raw-NN/energy-U negative controls
remain constraints on that future question; changing a cooling threshold
does not resolve their practical-equivalence and missing-support failures.

## Reproduce without training

[analyze_temporal.py](analyze_temporal.py) verifies saved artifact SHA256s,
recomputes unchanged phase bounds from declared task thresholds, checks exact
control sample/model/optimizer identities and reduces retained JSONL returns.
It constructs no model, performs no forward or optimizer update, and makes
no sampling draw. A hard 120 CPU-second process limit bounds the reduction.
Bulk traces and intermediate reduction outputs remain outside Git.

Every invocation archives the exact two queue-state inputs under the raw
`settling/inputs/` directory, using content-addressed filenames. The compact
result binds those immutable paths, sizes and SHA256s. A live queue file is
never the final evidence binding. Reproduction selects the retained inputs
with `--state-snapshot` and `--previous-state-snapshot`; it does not depend on
later queue/readout lifecycle changes. The retained round-two input is
21,958,924 bytes with SHA256
`4402bf5f1950a63cf4e0607cfbfbcd8ede3726c2fb4856f77343af3b7310277a`;
the round-one input is 24,634,480 bytes with SHA256
`53d44a53c8239a9570c4948900d50bbad273b99fdfa85419d2e601b253bca139`.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -u \
  reports/forge/first-principles-round2-20261022/settling/analyze_temporal.py \
  --state-snapshot /mnt/ml7tb/ParticleGAN-forge/first-principles-round2-20261022/settling/inputs/round2-state-4402bf5f1950a63cf4e0607cfbfbcd8ede3726c2fb4856f77343af3b7310277a.json \
  --previous-state-snapshot /mnt/ml7tb/ParticleGAN-forge/first-principles-round2-20261022/settling/inputs/round1-state-53d44a53c8239a9570c4948900d50bbad273b99fdfa85419d2e601b253bca139.json \
  --output /mnt/ml7tb/ParticleGAN-forge/first-principles-round2-20261022/settling/temporal-reproduction.json \
  > /mnt/ml7tb/ParticleGAN-forge/first-principles-round2-20261022/settling/temporal-reproduction.log 2>&1
tail -F /mnt/ml7tb/ParticleGAN-forge/first-principles-round2-20261022/settling/temporal-reproduction.log
```

The completed reduction consumed **1.225 process CPU seconds** inside the
reducer, or **2.197** including Python/Torch imports. A second reduction from
the exact archived inputs reproduces every numerical and provenance field,
excluding only the two measured CPU timers. Added optimizer updates, model
forwards, sampling draws and GPU spend are zero. Neither this saved-state
reduction nor the rejected thermostat is another scientific training run.

| Retained artifact under the raw `settling/` directory | SHA256 |
| --- | --- |
| `temporal-final.json`, identical to committed `temporal-results.json` | `87a689335faf626058f52b279a1a6a27dd189a0f5cd1404ad62134b660a65afb` |
| `temporal-final-reproduction.json` | `04885ea66c31d33d5f8cdd4f8a331323eb3561474a64581d63d8be4a4c706e9a` |
| `temporal-final.log` | `feaa674ca6e538e7e7d69dcb1fd43c0cc09affb7428efb84aac27014cad28d54` |
| `temporal-final-reproduction.log` | `270acb023b583a6b3bdba9ee0905c102bddec39fe182dcc1f9476b926b28e7a1` |

The default reducer refuses pending cells. `--allow-incomplete` produces an
explicitly incomplete intermediate result while the campaign runs; it does
not grade missing work as a pass.
