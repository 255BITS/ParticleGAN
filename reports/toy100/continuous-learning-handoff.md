# PR #60: remove LR decay without losing acquisition or stability

**Unresolved.** The passing shared 22-task recipe still uses LR decay. No new
candidate has passed both acquisition and continued quality. Work here is
limited to replacing those schedules with responsive training dynamics;
production defaults have not been changed.

**Scope clarified September 23, 2026:** PR #60 targets initial acquisition and
sustained live quality on a fixed target distribution without LR decay.
Adapting to a changed target distribution is a separate problem; translation
or replacement tests are optional diagnostics, not promotion requirements.
None of the current candidate verdicts changes: their first failures occur
during fixed-target acquisition or stationary quality checks.

Sitting still at a matched target is acceptable. A replacement must acquire
an initially unlearned target and maintain its quality during continued training.
Nonzero parameter movement is not a success criterion.
Favor changes to the game update over R1/R2 or other zero-centered pulls, but
select by measured results. Do not repeat seed sweeps or the rejected grids.

## Reproduction starting point

Research branch: `research/continuous-learning`, based on PR head `983d037`.
The normal PR branch remains `codex/toy100-coverage-gate`. All new tests run
locally; do not wait for GitHub CI. The tested environment is Python 3.12.13,
PyTorch 2.13.0+cu126 on CPU. On the shared machine it is
`/tmp/pr38-default-env/bin/python`; the default Python is a different version.

```bash
git fetch origin refs/heads/research/continuous-learning:refs/remotes/origin/research/continuous-learning
git worktree add ../ParticleGAN-lr-dynamics origin/research/continuous-learning
cd ../ParticleGAN-lr-dynamics
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2
export ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2
# Use the tested Python environment; independent workers each use one CPU thread.
```

Use fresh output paths. There are 24 CPU cores on the shared machine; run
independent candidates concurrently and keep one tailable log per controller.

## Isolated failure

The real, frozen `mode_hold` host takes about 7–12 seconds for 1,200 updates.
It contains the actual 12 learned particles, MLP generator, Fourier critic,
relativistic loss, cap penalty and noise mechanism. Required live quality is
all eight modes and HQ >= .9 at each of updates 1000/1050/1100/1150/1200.

Removing only LR schedules from the simpler 22/22 recipe makes those terminal
HQ values [.2029, .6267, .4441, .2437, .3694]; the scheduled control passes
all five. Over late training, constant LR .00425 moves clean outputs about
.406 per generator update versus .0197 with the schedule. The HQ radius is
.21. Freezing G alone locally protects quality while the prior keeps learning;
freezing the prior alone does not. Freezing is attribution, not a proposed fix.

At a passing state, a repeated-batch diagnostic finds about 95.6% coherent
energy in the Adam-denominator-scaled G gradient. The drift is therefore not
explained solely by zero-mean minibatch jitter. This does not prove that the
mean gradient improves quality. The twelve equally weighted particles and
output noise .029 cannot exactly match eight equal Gaussian modes of width
.07; **a passing state is not a demonstrated mathematical equilibrium**.

The [mechanism report](continuous-mechanism-diagnosis.md),
[research notes](continuous-learning-research.md), and
[result ledger](continuous-learning-results.json) provide measurements,
research references and artifact provenance. Some historical ledger entries
point to local artifacts; the important warm controls and newest game-update
results are retained under [continuous-evidence](continuous-evidence/).

## Cheapest reliable filters

1. Verify the scheduled pass and constant-rate failure on the unchanged host.
2. Use the matched passing-state fork: train one scheduled prefix to update
   1000, then fork live G/D/prior, Adam moments, EMA and every RNG stream.
   Score **every update** from 1001 to 1200. The unchanged child must match a
   separate uninterrupted control's final state hash exactly. A warm pass is
   only conditional local stability, never evidence of cold acquisition.
3. For warm survivors, run full-budget cold trajectory first: 400 updates,
   identity MSE <= .02 and the original sustained gate. It has rejected the
   newest dynamics candidates cheaply. Trajectory is a fixed conditional
   dataset throughout training; it does not change the target distribution.
   Then cold mode-hold, followed by the
   other cheap hosts in the fail-fast screen. Do not run expensive downstream
   tasks after a failed full-budget host.
4. Extend a surviving learner uninterrupted to at least 2400 mode-hold updates,
   checking every ten updates after 1200. Keep noise burn-in tied to the original
   1200-update horizon; do not restart models, optimizer moments or RNG streams.
5. A viable shared replacement still needs fresh older-19, the three strict
   native 100-mode cases, production common-22 replay, and longer continuation.

**Separate distribution-adaptation study:** the existing +.35 translation
protocol and matched frozen controls remain available for optional diagnosis.
Its historical recovery bounds and results are preserved, but a shift failure
does not veto a replacement that passes the fixed-target requirements above.

No architecture, data, frozen budget or quality threshold is relaxed. The
warm prefix uses decay only as a diagnostic starting state; the candidate's
cold acquisition test must remove decay from the beginning.

### Commands and extension API

```bash
python -u -m benchmarks.toy100.continuous_probe --mode scheduled \
  --output /tmp/lr-scheduled.json --archive-sources /tmp/lr-scheduled-source
python -u -m benchmarks.toy100.continuous_probe --mode constant \
  --output /tmp/lr-constant.json --archive-sources /tmp/lr-constant-source
python -u -m benchmarks.toy100.continuous_probe --mode scheduled --steps 2400 \
  --output /tmp/lr-scheduled-hold.json

# Examples of complete candidate drivers, including their controls:
python -u reports/toy100/secant_extra_warm_probe.py --output /tmp/lr-secant-warm
python -u reports/toy100/secant_extra_probe.py --output /tmp/lr-secant-cold
python -u reports/toy100/implicit_extra_warm_probe.py --output /tmp/lr-implicit-warm
python -u reports/toy100/implicit_extra_probe.py --output /tmp/lr-implicit-cold
python -u reports/toy100/cross_competitive_warm_probe.py --output /tmp/lr-cross-warm
```

[`run_warm_variants`](../../benchmarks/toy100/warm_equilibrium_probe.py)
accepts `{name: factory(state, prefix)}` contexts, including an `identity`
control, plus an optional `prefix_context` for a source adapter. The context
starts only in its child after the fork. `state` exposes live models,
optimizers, noise/streams, `set_step_delegate`, and explicit gradient-call /
moment-update accounting. `steps=2400` adds the longer hold. The existing
drivers show how to bind multi-evaluation methods without hiding their cost.

[`continuous_probe`](../../benchmarks/toy100/continuous_probe.py) supplies
uninterrupted hold/shift and matched frozen-control grading.
[`continuous_screen`](continuous_screen.py) demonstrates full-host fail-fast
ordering and saved-evidence regrading. Extra-gradient and implicit scratch
adapters currently support only mode-hold and trajectory; they are not yet
general production trainers. Scratch evidence must remain explicitly
ineligible for the production gate until a real implementation is audited.

## What has already failed

| Method | Local warm checks | Disqualifying evidence |
| --- | ---: | --- |
| Constant Adam .00425 | 6/200 | Cold mode-hold fails |
| Constant Adam .001 | 200/200 | Extended hold fails at update1950, HQ .7903; final HQ .9990 hides it |
| Joint Lookahead .00425, alpha .5, k=2/5/10 | 1/28/8 of200 | All six cold configurations also fail |
| G-proposal confidence scaling, thresholds .25/1 | Both200/200 | Both fail cold mode-hold,0/5 terminal checks |
| Fixed-metric same/independent-sample EG | Both0/200 | Second gradient divided by small first-gradient metric explodes |
| Same-sample secant EG, c=.25/.5/.9 | All200/200 | Cold trajectory MSE .2872/.3540/.1332 > .02 |
| Full-J linearized implicit response | 200/200 | Cold trajectory MSE .2538 > .02; warm cost11.4 gradient evaluations/player/update |
| Cross-player-only competitive response | 196/200 | Warm HQ falls to .8894; unplanned cold diagnostic ends at MSE .020058 > .02, with no passing checkpoints |
| Cross-only + own-curvature step bound (no nonlinear backtracking) | 199/200 | Update 1085, HQ .8013: cross solve lengthened G's step 15.5× its explicit Adam step ([report](cross-curvature-report.md)) |
| Same + per-player step ≤ explicit Adam step | 198/200 | Updates 1134–1135, HQ .8884: slow ten-update walk with every bound inactive |
| Alternating Adam, G own-curvature bound .25, D bound 2 or 3 | 200/200 | Cold trajectory PASS (.00094); cold ring best 3/5 terminal checks (D bound 3: 7 modes at 1000 and 1050, 8 from 1100); non-monotone in D bound ([report](alternating-curvature-report.md)) ; per-particle latent bound (v13) ring 5 modes/.51; averaged critic (v14) trajectory FAIL .251; state-proportional smoothed critic (v15, alpha .25) warm 200/200 + trajectory PASS but ring max 7 modes; peer stencil critic reproduced (warm 196/200, ring PASS); plain-critic curvature (v17) warm 200/200 but ring 7 modes; per-sample slope weighting (v18) warm 192/200; post-bound slope step scale (v19) warm 200/200 but ring max 7 modes; hard slope gate (v20) warm 199/200 (threshold crossing); rest-damping family closed |

Earlier constant Adam, optimistic Adam/AMSGrad, ExtraAdam, epsilon changes,
fixed output-motion bounds and persistent-noise grids also failed. R1+R2
produced one short mode-hold pass, then failed trajectory and 92/120 dense
continuation checks. The cross-player-only variant improves acquisition over
the full-J implicit method but still fails. Its cold run was mistakenly
launched before consuming the failed warm verdict; it is retained as an
unplanned diagnostic, with no promotion credit. No expensive host followed.

The four cross-only warm failures occur at updates 1194–1197, all with step
factor one. Its own cross-residual guard accepts those steps. Applying a new
full-joint residual cutoff of .5 would still miss two of the four failures
while rejecting 133/200 warm and 305/400 cold proposals. The next useful cheap
investigation is exact replay around update 1194, compared with acquisition
updates, to distinguish own-player curvature, functional output motion and
target mismatch before selecting another guard. This is a research direction,
not a tested fix or a reason to relax either gate.

Small accepted step factors are diagnostic, not automatic failures. The
rejections above come from failed acquisition or quality, not an imposed
minimum movement. Initial learning from scratch remains required; recovery
after changing the target is assessed separately.

Local handoff validation: 149 integrated tests and two additional cross-only
audit tests passed. The portable implicit warm driver
also reproduced the 200/200 result, 6/200 constant control, and exact identity
parity using only files in this worktree.

The [second research round](continuous-round2.md) tests eight additional
configurations, including projected skew response, with **170 integrated tests
passing**. No candidate qualifies. Its [ledger](continuous-round2-results.json)
and [exactly matched population / shift diagnostic](matched-population-diagnostic.md)
separate stationary drift, acquisition and optional response to changed data. The
handoff validation count above records the original publication, not the
current total. The follow-up also distinguishes a conservative warm-state
rejection from impossibility of a different cold attractor.

Functional damping's 20/20 matched-population stationary checks are a useful
partial result. That 200-update hold on a constructed target does not establish
long-term stability or original-task acquisition: its fixed-target trajectory
MSE is .037305 against the unchanged .02 limit. Removing shift recovery from
scope therefore does not promote it.

[PR #81](https://github.com/255BITS/ParticleGAN/pull/81) independently replays the
cross-only method on another environment. Its warm-state hash differs even
with the cu126 wheel and AVX2 caps; it fails at update1002 rather than our
1194–1197. Treat those exact step numbers as environment-specific attribution.
Both studies reject the tested method. Its optional center-aware output cap
is a diagnostic using host quality information, not a general training rule.

## Checks for a new attempt

```bash
python -m pytest -q tests/test_continuous_probe.py \
  tests/test_continuous_candidates.py tests/test_continuous_lookahead.py \
  tests/test_confidence_dynamics_scratch.py tests/test_fixed_metric_extra_scratch.py \
  tests/test_secant_extra_scratch.py tests/test_implicit_extra_audit.py \
  tests/test_cross_competitive_audit.py \
  tests/test_loss_budget_scratch.py tests/test_functional_metric_scratch.py \
  tests/test_energy_signal_scratch.py tests/test_adaptive_d_allocation_scratch.py \
  tests/test_matched_population_diagnostic.py tests/test_projected_skew_scratch.py \
  tests/test_toy100_config.py tests/test_toy100_policy.py tests/test_toy_suite.py
```

Bind source/config hashes and record actual G/D/prior rates, state-based step
factors, gradient evaluations and moment updates separately. Check all live
iterates, including between periodic corrections. Preserve exact RNG replay
for same-sample trials and restore rejected trials. A final good checkpoint,
low loss, stable EMA, or passing analytic bilinear game alone is insufficient.
