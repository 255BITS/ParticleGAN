# Combined magnitude response at constant rates

**Combining the fixed network cap `.1` and prior cap `.001` does not solve
continuous 1D Gaussian learning.** All nine trial gates fail. Past extrapolation
retains30/72 stationary checks, compared with41/72 for the network-only cap. It
reaches a five-check window sooner, at2,459 instead of3,875, but misses acquisition
at1,000 and keeps losing fit. Shifted retention remains8/24. Keep the adopted
alternating ring recipe.

The [compact results](results.json), [frozen protocol](protocol.json),
[provenance](provenance.json) and [saved-state audit](saved-state-audit.json)
bind the outcome to the actual recipe, prior, initializer, sampling and source.
This is the interaction study authorized after the
[three independent investigations](../gaussian-response-round/README.md).

## Gaussian results

| Timing | Phase | Acquisition | Hold passes | Longest streak | First five-pass window | Final KS |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Alternating | stationary | FAIL | 13/72 | 3 | none | 0.03713 |
| Simultaneous | stationary | FAIL | 0/72 | 0 | none | 0.15660 |
| Past extrapolation | stationary | FAIL | 30/72 | 6 | 2459 | 0.04666 |
| Alternating | shift | FAIL | 10/24 | 2 | none | 0.03193 |
| Simultaneous | shift | FAIL | 0/24 | 0 | none | 0.21905 |
| Past extrapolation | shift | FAIL | 8/24 | 5 | 6000 | 0.04436 |

Acquisition means five terminal full passes by1,000 stationary updates or5,000
after the mean shift. Hold means every later scheduled full pass through4,000
or6,000. Each phase's combined verdict is FAIL; each complete learner also fails.
The late shifted past window ends at6,000, a thousand updates past reacquisition.
Its final mean error.08646 sigma, std ratio1.00471 and KS.04436 pass the endpoint,
but the last five observations cannot replace the missed deadline or failed hold.

Alternating also ends with good samples: stationary mean error.02647 sigma,
std ratio1.01142, KS.03713; shifted mean error.04079 sigma, std ratio.96714,
KS.03193. It never reaches five consecutive full passes in either phase.
Simultaneous passes none of its96 stationary or48 shifted Gaussian checks.

### Original source-bound comparisons

These comparisons reuse archived observations and their original identities;
they add zero training or qualification cost. [controls.json](controls.json)
retains the original publication commits, protocol hashes, recipes, source and
receipt identities. They are diagnostic cohorts, not a new qualification board.

| Past-extrapolation field | Stationary hold /72 | Longest stationary streak | First five-pass window | Shift hold /24 | Longest shift streak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original normalized networks/prior |3|2|none|0|0|
| Prior cap only, .001 |10|2|none|3|3|
| Network cap only, .1 |41|5|3,875|8|3|
| Both caps, this study |30|6|2,459|8|5|

The interaction trades more consecutive late passes for fewer total stationary
passes. It does not improve the full acquisition/retention gate. This single
protocol-seed comparison rejects the exact pair as a smoke-test repair; it does
not establish that every magnitude-sensitive update or scale pair fails.

### What still fails

For stationary past extrapolation, all42 failed hold observations violate
KS≤.05. Four also violate the mean bound, and six exceed the width bound.
Shifted past fails KS at16/24 hold observations; six also fail the mean bound,
while width stays in range. These overlapping counts identify failed bounds;
they are not separate failures to add together.

At the stationary endpoint the fitted-normal KS is.03044, versus.04666 against
the fixed target law; at shift end it is.02145 versus.04436. Fitting mean/width
is a secondary shape diagnostic, never the qualification gate. During stationary
hold the mean error ranges.00400–.32773 sigma and std ratio.87036–1.87483. During
shift hold those ranges are.00375–.27755 and.80582–1.08253. Good final shape
therefore does not demonstrate stable target-law learning.

The saved stationary past cache has97 moving prior rows with mean direction
norm.45004, implying mean row motion.01350 at the constant.03 rate;5.15% saturate.
After the shift,103 moving rows imply.01016 mean motion, with3.88% saturated.
Network-only past implied approximately.02997/.02999 mean prior motion. The
combined rule really reduces prior response while preserving learning; it does
not freeze the prior. The G/D hidden-matrix cap utilization at stationarity is
3.47%/8.52%, and after the shift7.38%/8.98%. These are different parameter units,
so their relative sizes do not establish causal dominance. The audit uses saved
fields only: zero new updates, gradients or model draws.

## Ring regression

| Timing | Acquisition at1,600 | Hold /144 | Longest full streak | Final component covariance error |
| --- | --- | ---: | ---: | ---: |
| Adopted original alternating control |PASS|144|all hold|0.56385|
| Combined alternating |FAIL|0|0|3.45127|
| Combined simultaneous |FAIL|118|118|.54546|
| Combined past extrapolation |FAIL|0|0|3.14061|

The frozen control's exact endpoint value is retained in controls.json; the
original alternating ring acquisition/hold remains the adopted passing evidence.
Combined simultaneous eventually passes its last118 checks, reaching its first
five-check window at2,117, after the1,600 deadline. Alternating and past fail the
component covariance bound at every144 hold check. All three final samples cover
16 modes, which cannot replace cluster fidelity. Ring past's prior field is
nearly saturated:97.94% of active rows are at their cap, with implied mean motion
.02973. The same absolute cap has different effects across these tasks.

## Mechanism and contract

One global trainer configuration per timing composes the two independently
predeclared scales from [PR319](https://github.com/255BITS/ParticleGAN/pull/319)
and [PR320](https://github.com/255BITS/ParticleGAN/pull/320). For a matrix gradient
`U diag(s) Vt`, network motion is
`U diag(min(s/.1,1)) Vt * sqrt(max(1,out/in))`. Vector/scalar gradients use
`g/max(norm(g),.1)`. Sampled prior rows use `g/max(norm(g),.001)`; other rows
receive zero direction. Both shrink small gradients and cap large ones. Scales
are fixed across tasks, roles and time, without calibration from trained states.
The old polar-factor/unit-row rules remain the public defaults.

Past extrapolation uses the previous field for a temporary joint lookahead,
evaluates the fresh field there, restores the base and corrects once. Preview
and correction share both helpers. The first lookahead is zero, optimizer steps
advance once, and checkpoints retain the cache and every consumed stream. This
adapts equations20–21 in [Gidel et al.](https://arxiv.org/pdf/1802.10551) to a capped
neural field. The paper's convergence assumptions are not established for it.

Public `GANTrainer`, protocol seed0, the public deterministic initializer and
batch128 match the original initial tensors and named streams. Both tasks use
256 learned uniform MoG locations, sigma.1, without standardization. Gaussian
is N(2,.5²), z2, G/D width32/depth2, Fourier critic2, with1,185/1,281 network
parameters. Ring keeps z4, width64/depth2 and the original16-cluster target law.
No architecture, particle count, loss, regularizer or sampling change is hidden
in the timing comparison. Ordinary sigma.025 Gaussian remains a separate cohort;
this sigma.1 diagnostic supplies no ordinary qualification.

Rates remain G.012, D.012×1.5, prior.012×2.5. No annealing, EMA, best-checkpoint
substitution or output averaging occurs. Full Gaussian bounds are mean error≤.2
target sigma, std ratio[.8,1.2], KS≤.05,4,096 samples and finite fraction1. Ring
uses every original full-component bound. Acquisition requires five terminal
full passes by1,000 Gaussian/1,600 ring updates; every72 Gaussian/144 ring hold
check through4,000 must pass. Cadence remains24 per original1,000 Gaussian/
400 ring block, with no thinning to select good windows.

At4,000 the Gaussian mean changes2→3, sigma.5 unchanged. Continue2,000 updates
without resetting optimizer or extrapolation history; reacquire by5,000 and pass
all24 later checks. Each own4,000 checkpoint also supplies a no-update control
with matched evaluation component/kernel draws. All frozen controls pass0/48
shift observations. Shift diagnostics execute even after stationary failure;
they cannot turn that learner into a complete pass.

The six stationary trials and three shift continuations were frozen before spend
at scientific commit `d5256284210354bda7a941b6e4121ec82e4d9c23`. The reservation
was30,000 updates and4,500 loop seconds, with600 per stationary cell,300 per shift,
zero retries, no expansion or scale search. Every cell finished once, including
failures. Actual cost is30,000 updates,3,840,000 real examples and362.353342 measured
loop seconds. Setup, serialization, restore and publication are separate; this
is accounting, not a speed comparison.

CUDA performs training, sampling, mechanism fixtures, cached-field reductions
and exact checkpoint restores. Original CPU target generation, saved-output
numerical scoring and media rendering are explicit protocol exceptions. Exact
real-batch digests match the archived stationary laws and each other after the
shift. Constructor, data, training latent/kernel and evaluation streams are
isolated and checkpointed. Unchanged scorer oracle/destructive controls are
reused under their original source identity.

## Recommendation and stopping

Keep the adopted ring recipe and stop this exact fixed-scale pair as a global
smoke-test repair. The useful next experiment is **fresh, two-pass extragradient**
against this cached-past arm, with identical fixed caps/rates and matched data
and noise. Past timing improves Gaussian over alternating13/72 and simultaneous
0/72 under this same combined field, but its cached direction still misses
continuous stability. A fresh predictor/corrector would test the cache hypothesis;
these results do not prove stale history is the cause. Declare its operator
computation cost, RNG law, full acquisition/hold/shift gates and finite budget
before spending. A separate bounded global cap-scale comparison is another open
question; this study tested exactly one pair and cannot choose a winning scale.

There is still no evidence here that larger z, wider networks, more particles or
another seed resolves the issue. The unchanged-network affine capacity control
already demonstrates Gaussian representation capacity. A position spring adds
another force before the response/timing questions are resolved. No fresh
extragradient trial, further scale search, continuation or promotion was run.

## Verification, media and reproduction

All79 software checks pass:35 CUDA mechanism/matched-host/exact-resume controls
and44 metadata-only ownership/technique checks, with zero final skips. Default
packets and three default optimizer updates match the pinned old implementation
exactly. All9 final CUDA contexts restore exactly, including optimizer counts,
constant rates, cached directions and streams. All1,308 saved metric sets and
nine phase grades reproduce; every GIF contains nine actual GPU observation
frames. See [verification](verification.json), [restores](restore-proof.json),
[stationary metric traces](stationary.svg) and [shift traces](shift.svg).

| Timing | Gaussian stationary | Gaussian shift | Ring stationary |
| --- | --- | --- | --- |
| Alternating |[GIF](alternating-gaussian1d_acquisition-stationary.gif)|[GIF](alternating-gaussian1d_acquisition-shift.gif)|[GIF](alternating-ring16_acquisition-stationary.gif)|
| Simultaneous |[GIF](simultaneous-gaussian1d_acquisition-stationary.gif)|[GIF](simultaneous-gaussian1d_acquisition-shift.gif)|[GIF](simultaneous-ring16_acquisition-stationary.gif)|
| Past extrapolation |[GIF](extrapolation_from_past-gaussian1d_acquisition-stationary.gif)|[GIF](extrapolation_from_past-gaussian1d_acquisition-shift.gif)|[GIF](extrapolation_from_past-ring16_acquisition-stationary.gif)|

Restore the inherited source-bound baseline artifacts at the protocol's declared
`runs/api/tier1-*` paths, then use the frozen scientific checkout:

```sh
mkdir -p runs/api
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 /usr/bin/python -u \
  -m benchmarks.toy_audit.gaussian_combined_magnitude run \
  --output runs/api/gaussian-combined-magnitude-v1 --device cuda:0 \
  > runs/api/gaussian-combined-magnitude.run.log 2>&1
tail -f runs/api/gaussian-combined-magnitude.run.log
```

Each per-trial `.log` inside that directory emits scheduled numerical checks and
can also be tailed directly. Output paths must be fresh; these commands reproduce
the completed study rather than authorize another round. Bulk logs, curves,
samples, states and RNG tensors remain ignored locally and in the content-addressed
archive identified in provenance. Git contains compact finals, receipts,
reproduction sources and actual-training media.

Saved-state verification and publication add no training:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 /usr/bin/python \
  reports/forge/gaussian-combined-magnitude/verify.py \
  --raw runs/api/gaussian-combined-magnitude-v1 --device cuda:0
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/martyn/dev/ParticleGAN/.venv/bin/python \
  reports/forge/gaussian-combined-magnitude/publish.py \
  --raw runs/api/gaussian-combined-magnitude-v1
```

The branch targets develop and depends on open
[PR317](https://github.com/255BITS/ParticleGAN/pull/317). It integrates the cap
capabilities from PR319/320, preserving their independent archived science.
Resolve overlapping API additions if those PRs merge separately, then regenerate
Forge catalog/memory summaries. A merge does not require repeating unchanged
experiments. The existing single qualification leaderboard remains unchanged.
