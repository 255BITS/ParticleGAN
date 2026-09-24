# Fixed-target instability: exact replay and critic tracking

**No replacement qualifies yet.** The new penalized critic-refinement rule
passes all44 saved-state continuation checks and warm200/200, minimum HQ
.99707 with eight modes throughout. Its dense hold through2400 is running;
cold acquisition remains untested. Opponent prediction passes warm200 but fails
22 of1,200 later per-update checks, including temporary mode loss. Cold gates
were not run after that failure. This round investigates the delayed loss of quality on the **unchanged**
dataset. The production recipe still uses LR decay. Research tests use live
weights, the original quality thresholds and fixed nominal rates; a final
recovery cannot erase an earlier failed check. Rest without a useful signal
is allowed. Entire target-distribution shifts are outside this round.

## Failure isolated before another candidate

An observer replay of the original PR84 continuation reproduces its complete
final model/Adam/EMA/RNG state, all 420 previously archived diagnostics and
all 1,400 update records. Checking every update from 1300 through 1600 finds
46 failures in 301 checks: 1325–1327, 1389–1390 and 1531–1571. Update 1325 is
the earliest failure found in that dense window, not a proven global first
failure; updates 1201–1299 were not densely checked in that replay.

At 1325 a single generator update drops HQ from .99658 to .82422, retaining
eight modes. The own-field curvature bound accepts the full step. Most
motion comes from the shared generator network rather than the learned
particles. The later episode accumulates drift before losing a mode at1533.
The bound can limit local field variation without ensuring a useful direction.
[Exact replay and phase-by-phase diagnosis](pr84-stationary-failure-diagnosis.md).

At the fixed post-D state for 1325, all 16 held-out generator minibatches push
four vulnerable particles outward. Bias-corrected coherent energy is .943
for raw network gradients, .945 in the **pre-update** Adam metric, and .959
for actual bounded output motion. The raw negative-gradient direction and
the frozen-metric direction both point outward in all 16 batches, while the
median denominator barely changes across1324–1325. This does not support
zero-mean generator minibatch jitter or a sudden metric sign flip as the
explanation for this transition. It is conditional on the saved critic;
it does not establish the population field or exclude discriminator noise.
[Portable held-out diagnostic and raw evidence](pr84-heldout-signal.md).

Target centers and quality grades are used only by these offline diagnostics.
They never enter the proposed training rule.

## One specific dynamics change

The critic keeps its original accepted update from `D0` to `D*`. Generator
queries instead see a temporary predicted critic `2*D* - D0`; actual critic
parameters are restored after each query, including exceptions. The generator
base/proposal queries share that predicted critic and the same detached
spatial-stencil width. Each player still makes three field evaluations and
one Adam moment update. There is no gain sweep, new zero-centered penalty,
elapsed-time factor or extra field evaluation.

The fixed coefficient has a contracting scalar bilinear example, and an
exact zero-field test rests. These facts motivate a test; they do not prove
stability of nonlinear stochastic Adam. Disabled prediction exactly matches
the original PR84 adapter, including its original width recomputation.

The saved-state filter is explicitly **MIXED_NOT_PASS**:

| Resumed updates | Original passing checks | Prediction passing checks |
| --- | ---: | ---: |
|1324–1335|9/12|12/12|
|1380–1395|14/16|15/16|
|1530–1545|1/16|1/16|

Some repaired steps move farther while changing a harmful direction to a
helpful one, so the mechanism is more than scalar damping. The second branch
nevertheless introduces a new failure at1395, and the severe mode-loss branch
remains. All 13 original single-update full-state hashes and all44 original
continuation supports reproduce exactly. [Frozen filter](pr84-prediction-state-filter.md).

This mixed result justified **one separately declared earlier-intervention
diagnostic**, starting from the shared passing state at1000. Applying the
rule earlier can avoid the saved original trajectory; it does not retroactively
turn failed local branches into passes. Warm200 must pass before a longer
hold; that entire hold must pass before cold acquisition tests.

## Completed warm and dense-hold results

| Method | Warm1001–1200 | Every update1201–2400 | Minimum late HQ | Minimum late modes | Final modes / HQ |
| --- | ---: | ---: | ---: | ---: | --- |
|Scheduled control|200/200|1200/1200|.99707|8|8 / .99951|
|Original PR84|200/200|1111/1200 **FAIL**|.71606|7|8 / .99805|
|G opponent prediction|200/200|1178/1200 **FAIL**|.73755|7|8 / 1.0|

The minimum HQ and minimum mode count need not occur at the same update.
Prediction fails at1391,1417,1532–1533,1648–1653,1967–1971,2053–2055,
2057–2059 and2263. Its worst HQ is at1417 with eight modes; the original's
worst HQ is at2122, also with eight modes. Prediction reduces the number of
failed checks from89 to22, but neither method maintains the required quality.
No cold trajectory, ring or further production run followed the rejection.

Dense observation changes the previous original-PR84 result from112/120 to
1111/1200 because it reveals intervening failures, not because training changed.
The original full final-state hash, all420 shared diagnostics, all1,400 update
records,24 host observations excluding timing, noise, rates and optimizer
receipts match the archived run exactly. The predictor's first200 records and
all220 short-run diagnostics match its separate warm run. Identity matches the
uninterrupted scheduled control. All source snapshots and the generated
observer are hash bound. Observer cadence is the only harness transformation.

Both active methods use constant post-fork G/D rates .00425 and prior .0085,
three field evaluations per player per outer update, and one Adam moment
update. The predictor installs/restores its virtual critic2,800 times across
1,400 continuation updates with zero additional field evaluations. Its
stencil width remains .15 throughout this hold. The candidate episode takes
49.77seconds, versus48.86seconds for the original under concurrent CPU load;
these are descriptive timings, not a controlled performance benchmark.

The [machine-readable ledger](continuous-round5-results.json) and
[complete warm/hold evidence](continuous-evidence/pr84-opponent-prediction-run/manifest.json)
retain every branch. An independent Sol audit verified the hashes, accounting,
gates and numeric claims. Eighteen focused integrated tests pass: eight
prediction tests, two held-out diagnostic tests and eight existing PR84/
alternating checks. These implementation checks do not override training failure.

## Second response channel rejected by the short filter

A separate method keeps the original G update and then changes the actual
critic by `-P_D * (F_D(D*, G_accepted) - F_D(D*, G_base))`. Both fields use
the same D batch/noise and accepted D. Its post-base Adam metric is frozen;
there is no extra moment update, gain search or zero-centered pull. Unlike
G-side prediction, the actual critic now responds to the accepted G movement.
The extra response is explicitly outside the original D curvature bound.

The three declared branches give **11/12,16/16,1/16** passing checks, versus
original **9/12,14/16,1/16**. It repairs the1380 window but fails at1325
(HQ .88916) and still loses mode2 at1533. All three branches were required to
pass, so **no warm or cold run followed**. No nonfinite state or critic explosion
occurred; the correction stayed below .526 times the ordinary D step norm.
All44 ordinary support arrays and curvature records match the capture, as do
the first accepted-state hashes; candidate/ordinary RNG endpoints agree.
Each active update costs four D fields and three G fields, with one moment
update per player. Nine additional focused checks pass, independently rerun.
[Source, strict failed filter and complete evidence](pr84-d-cross-response-filter.md).

Both response channels repair one excursion while retaining the severe1530s
episode. The next diagnostic separates learned-critic lag, the actual penalized
critic's local optimum, and pressure from model/objective mismatch. The ideal
unregularized population density ratio is an offline comparator only; it is
not the finite Fourier critic with the host penalty. Failure of these methods
alone is not an impossibility proof.

## Critic fitting supplies a more useful direction

The [structural research note](rp-misspecification-stability.md) proves that
the frozen mixture cannot exactly match the target distribution. That rules
out a realizable constant-critic convergence premise; it does **not** rule out
sustained high HQ, a local equilibrium, or a responding learner. Its small
population counterexample has a restoring best-response coordinate field
despite nonzero discriminator advantage. The host has not been proved impossible.

The [population-field comparison](pr84-population-field.md) freezes the exact
saved generator and evaluates an unrestricted unregularized Rp density-ratio
critic. Both sharp and stencil versions give inward raw network and frozen-Adam
directions in all64 held-out cases across1325,1389,1530,1540. The learned stencil
field points outward in54/64 raw cases. Matching archived learned-field values
and paired batches rules out a comparison caused by altered draws. But the
unscaled analytic critic has enormous sampled cap penalties, so it cannot
stand in for the actual penalized critic optimum.

A [separate copied-critic fit](pr84-critic-relaxation-diagnosis.md) retains the
actual finite architecture and penalized loss. One bounded L-BFGS attempt per
state lowers held-out D loss and yields inward raw and accepted G motion in
24/24 reserved cases, versus2/24 and1/24 before fitting. At1325 its eight G
proposals all pass at HQ1; at1530 all pass with minimum HQ .99878. At1539 the
mode is already missing and none recovers it in one proposal. These are fixed-G
diagnostics, not a new successful continuation. All three fits remain
**NONCONVERGED_RESIDUAL**; lower loss and repaired guidance do not certify
stationarity or a global best response.

This identifies a concrete candidate: refine the actual penalized critic
against the current generator before taking G's step. The declared rule
uses eight cached training batches and one L-BFGS attempt, capped at40
iterations and80 closures. The lowest finite training-loss point supplies
the materialized critic for both G queries. The fit preserves all training
random streams; no diagnostic quality metric selects an update. D's ordinary
Adam moments advance once before refinement, while the additional fit is
explicitly extra optimization outside the original D curvature bound.

The [strict saved-state filter](pr84-critic-refinement-filter.md) passes
**12/12,16/16,16/16**, minimum HQ .999512, against exactly reproduced original
controls9/12,14/16,1/16. Warm1001–1200 subsequently passes **200/200**, minimum
HQ .997070 and eight modes throughout. An independent audit confirms source
identity, unchanged random-stream receipts and constant applied G/D .00425
and prior .0085 rates. Warm cost is10,674 extra1024-pair D fit closures plus200
128-pair parity queries; the median is51.5 fit closures per update. Eleven
fits exhaust the hard closure budget. These results do not certify a best
response or establish cold acquisition. The source-bound every-update hold
through2400 is the next gate; the active adapter currently requires zero
input noise, and a separate faithful cold extension is being prepared.

The [capped one-dimensional toy](capped-critic-tracking-toy.md) gives a
precise mechanism under the same paired Rp losses and one-sided cap. A narrow
generator and wider Gaussian target have a stationary mean despite residual
distribution mismatch. At that point the fixed-critic G curvature is−3.3148,
while re-optimizing the quadratic critic as the mean moves gives a restoring
G-field slope+4.5088. Equal-rate coupled gradient flow is unstable in that toy;
sufficiently fast critic response is stable locally. This is a controlled
counterexample to mismatch alone forbidding stability, not a theorem for the
finite neural critic, Adam, stochastic batches or the full model.

## Research checked after isolation

[Alex-GDA, ICML2024](https://proceedings.mlr.press/v235/lee24e.html) and the
earlier [prediction-method project](https://www.cs.umd.edu/~tomg/projects/stable_gans/)
motivate evaluating one player against an extrapolated opponent. This is not
a novelty claim or an exact reproduction of their algorithms. Their analyzed
games and metrics do not certify this Adam/stencil implementation.

[Adam in zero-sum games, May2026](https://arxiv.org/html/2605.19392v1) analyzes
deterministic simultaneous dynamics through an ODE approximation. Its smooth
model, epsilon convention and approximation limitations at low first-moment
momentum matter here: the host alternates and uses beta1=0. It supports
measuring moments rather than assuming that nominal LR tells the whole story;
it does not justify another epsilon sweep after the held-out result.

[Nonlinear GD/SGD stability, COLT2026](https://proceedings.mlr.press/v336/mulayoff26a.html)
shows why average or local linear behavior can miss nonlinear and minibatch
instability under its assumptions. It supports inspecting complete actual
trajectories; it is not an adversarial-Adam theorem.
[High-probability SGDA, COLT2026](https://proceedings.mlr.press/v336/ha26a.html)
distinguishes noise tails in structured games, with assumptions not established
here. Our captured coherent wrong direction supplies no direct evidence for
a heavy-tail clipping remedy.

[Constant-step two-timescale linear stochastic approximation, AISTATS2025](https://proceedings.mlr.press/v258/kwon25a.html)
can converge to a stationary distribution with residual bias and variance.
That is compatible with persistent motion and does not guarantee that every
live iterate meets a sample-quality threshold. Finite holds remain rejection
tests, not proofs of indefinite stability.

## Reproduction and gate order

Run from the research branch with Python3.12.13/PyTorch2.13.0, CPU AVX2 and
one thread per independent process. No seed sweep or GitHub CI is involved.

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES='' ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 \
  ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 \
  /tmp/pr38-default-env/bin/python -u reports/toy100/pr84_opponent_prediction_probe.py \
  --phase warm --output NEW_WARM > /tmp/pr84-opponent-warm.log 2>&1
```

The same command with `--phase hold --previous NEW_WARM/summary.json
--output NEW_HOLD` requires the source-bound warm pass. Hold checks every
update through2400, preserving the1200-update noise horizon and all model,
optimizer and RNG state. Its original-method control must match archived
final state and shared observations; the prediction's first200 updates must
match the short run exactly. Cold trajectory400 then ring1200 are enabled
only by a complete hold pass. Subsequent survivors would still require
continuation from their own cold-acquired state, longer holds and unchanged
remaining production gates.
