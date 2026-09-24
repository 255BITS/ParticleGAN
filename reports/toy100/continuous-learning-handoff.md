**Current status: [practical leaderboard](continuous-practical-leaderboard.md), [round9 evidence](continuous-round9.md).** Pre-start anchors lead completed neural acquisition/own-hold/response; a conditioned missing-bank loss recovers after two updates. PR84 remains the reference with fewer objective changes, but its local cold ring has seven modes. Longer retention, fidelity and production cost remain open. No qualified LR-decay replacement.

# PR #60: remove LR decay without losing acquisition or stability

**Unresolved for production.** The passing shared22-task recipe still uses LR
decay. A neural sampled-anchor candidate passes the toy acquisition, own-hold
and response gates, with measured recovery from a conditioned omission. Its
objective changes and production scaling/fidelity prevent promotion. The
current board above supersedes the historical ranking below; original strict
results and source epochs remain unchanged.

**Latest stability investigation:** exact replay isolates a destructive
generator update at1325 and accumulated mode loss beginning1533. All16 held-out
generator minibatches at1325 push vulnerable particles outward; the raw field
already points that way and Adam's denominator barely changes. G opponent
prediction repairs some saved transitions and passes warm200, but fails22 of
1,200 dense later checks through2400. Original PR84 fails89/1200 under the same
dense observer; the scheduled control passes1200/1200. Both failed methods end
with excellent quality. Exact archived original-state and predictor-prefix
parity hold. Read the [fifth-round diagnosis and tests](continuous-round5.md)
and [current status](stationary-stability-status.md) before proposing another
method. Prediction is rejected before cold gates; no production winner is selected.

**Preceding causal lead:** the actual-D cross-response fails its short filter.
Fitting a copied critic on the existing penalized objective reverses harmful
G guidance on24/24 reserved batches; all proposals at1325 and1530 then pass.
[Exact fit and held-out evidence](pr84-critic-relaxation-diagnosis.md) led to
a bounded critic-refinement rule. That rule now passes **all44 saved-state
checks and warm200/200**, with eight modes and minimum warm HQ .99707.
The dense hold through2400 passes all1200 later checks, minimum HQ .939453
with eight modes throughout. Cold trajectory passes at MSE .000910 with22
passing observations; the original cold ring aborts on a nonfinite inner-fit trial at
update472. [Exact failure audit](pr84-critic-refinement-failure-audit.md)
isolates float32 strong-Wolfe interpolation overflow from otherwise finite accepted
state. The [finite-trial repair](pr84-critic-refinement-finite-recovery.md) restores the
saved best critic exactly, preserves all44 finite-path saved-state checks, and completes
updates472–491 with finite state. Those20 checks still have three modes: numerical
recovery alone does not pass acquisition. The [new source-bound full cold run](pr84-critic-refinement-finite-cold.md)
reproduces trajectory exactly but fails ring:3 modes/HQ .451416,0/24 passing checks. Its same three-mode sampled assignments persist from100 through1200. No own-acquired hold or later host follows; the next saved-state tests isolate early acquisition and differentiated critic response.
Each update costs about53 additional1024-pair critic-gradient evaluations in
the warm run, plus a parity check. Fits are nonconverged and D's retained Adam
moments precede the refinement. [Rule, strict filter and frozen evidence](pr84-critic-refinement-filter.md).
The [capped quadratic-critic toy](capped-critic-tracking-toy.md) demonstrates
a locally restoring fast-critic field where a frozen critic is repelling;
its assumptions do not establish full-host stability.
The [structural note](rp-misspecification-stability.md) proves exact distribution
matching impossible for the frozen mixture but explicitly does not prove
good-quality stability impossible. The [operator probe](pr84-cross-reciprocity-research.md)
also rejects an adverse cross-dominance explanation along the captured1325
update. Do not present either finding as a no-go theorem for this task.

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

**September24 follow-up:** [PR81/PR82 review and the alternating-dynamics
round](continuous-round3.md), with a [machine-readable ledger](continuous-round3-results.json),
now provides the current diagnosis. An independent full-state audit confirms
PR82's disabled alternating adapter exactly matches the host. Alternating
constant Adam passes trajectory at MSE .0034514; the matched simultaneous
control fails at .254434. This finding concerns the older ExtraAdam-derived
joint-field adapters, not direct Adam-step wrappers that already alternate.
PR81's target-error cap reads known target geometry and remains diagnostic.

Nine completed new configurations still fail acquisition or warm stability.
The strongest new D-only line-search arm passes warm200/200 and trajectory
at .0010046, then fails ring acquisition at5 modes/HQ .823. Verifying both
players' own-loss descent also fails seven dense warm checks despite passing
every sparse terminal check. The cap penalty can jump across critic activation
boundaries; exact zero-step replay distinguishes this from changed noise.
Keep the dense warm filter and the unchanged five-check trajectory suffix.
The alternating-field implicit-response retest passes warm200 but fails cold
trajectory at .252398,0/24. It is included in the nine completed candidates.
The latest [PR84 review and stencil research](continuous-round4.md) follows
that closed round; its evidence is kept separate.

**PR84 follow-up:** the original generator-only stencil is the strongest
local partial candidate: warm200/200, trajectory PASS, ring7 modes/HQ1.0
FAIL. Its reported eight-mode ring pass does not reproduce locally. The
shared D/G stencil also fails ring acquisition. Read the [fourth-round
report](continuous-round4.md) before running another candidate; it records
the double-smoothing implementation issue and the local gradient barrier.
The first sampled-data coverage projection recovers the missing mode in a
one-state check, then fails dense warm197/200; no cold run followed. Its
[source and failed gate](coverage-projection-report.md) are available for
other agents. The original PR84 adapter was the selected partial candidate
for that round; the current critic-refinement selection is linked above.
Its bidirectional follow-up passes the three saved-state checks and warm200,
then fails cold ring7/HQ .87012. [Completed result and next small tests](chamfer-projection-report.md)
distinguish nonlinear pullback error from inadequate target coverage. Neither
the objective decrease nor early passing checkpoints establish sustained quality.

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
3. For warm survivors, extend the same fixed-target branch to at least2400,
   preserving the original1200-step noise horizon and all optimizer/RNG state.
   Check **every update** after1200 using the round5 observer; ten-step sampling
   missed many excursions. This conditional stability filter precedes further
   acquisition hosts: both original PR84 and opponent prediction pass warm200
   while failing longer continuation.
4. For survivors, run full-budget cold trajectory first: 400 updates,
   identity MSE <= .02 and the original sustained gate. It has rejected the
   newest dynamics candidates cheaply. Trajectory is a fixed conditional
   dataset throughout training; it does not change the target distribution.
   Then cold mode-hold, followed by the
   other cheap hosts in the fail-fast screen. Do not run expensive downstream
   tasks after a failed full-budget host.
5. Extend a surviving cold learner uninterrupted to at least 2400 mode-hold updates,
   checking every update after 1200. Keep noise burn-in tied to the original
   1200-update horizon; do not restart models, optimizer moments or RNG streams.
6. A viable shared replacement still needs fresh older-19, the three strict
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
