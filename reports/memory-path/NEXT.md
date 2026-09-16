# Latest: memory dynamics and G repair scouts completed

User authorized slow/fast D memory, local feedback stability, and a G-side memory
repair/translator, with config-controlled changes and backward compatibility.
Then suggested that repair could use the generated sample to preserve useful
continuation. We acknowledged that as a next hypothesis, not an implemented
sample-conditioned adapter. The current adapter reads M only.

All work finished: 16 fresh 2k scouts and one exact 2k -> 5k continuation.
Queues dynamics_round8 and dynamics_round8_long have 16/1 done, zero failed,
pending, or running. Both GPUs were used; all diagnostic sessions are closed.
User requested commit and compact; this handoff is included in the commit on
feat/sequential-memory-path. Resume with discussion, not an automatic sweep.
Preserve unrelated .claude/, results/motion/, sparse-ucd.log.

Video: ../memory-handoff/dynamics_round8/slow16_r10_rollout.mp4 (24fps, 1024
saved steps, about43s). It shows cold starts and prefix32 continuations from
slow16_r10, lowest long radial error at BOTH prefixes among new scouts. Examples
are the first CW and first CCW reference episodes, selected without looking at
generated quality. Fixed axes, full faint history, bright last32 steps, and a
moving offline reference point. Script: experiments/render_memory_rollout.py.
Rendering is explanatory only; no experiment selection was based on images.

Read ../memory-handoff/dynamics_round8/assessment.md, leaderboard.md,
mechanism-table.md, long/leaderboard.md, and the local/adapter probe JSON files.

Every new checkpoint: 0/128 cold circles at256/1024 and original-orbit continuation
at prefixes8/32 and256/1024. All retain 0% late stopping; the clock milestone
survives. All late-only cold circles also0. No primary-metric winner.

Useful distinctions:
- translate16 improves prefix32 real-context next-point MSE .00664 -> .00541,
  but long radial error2.545; bypass worsens local prediction yet reduces radial
  error to.856, still0 original-orbit passes.
- repair_raw100 reduces synthetic memory reconstruction error~29% at noise.05;
  local prediction MSE.00474. Bypassing repair barely changes noisy G prediction.
- repair_raw10_n15 at2k has radial1.834 (baseline2.597); bypass worsens to1.981.
  It alone met predeclared continuous-metric extension criteria at BOTH prefixes,
  so received a diagnostic5k continuation. At5k radial worsens to2.456, local MSE
  .00561 -> .00787, all full passes still0. No further extensions selected.
- stable_dg10_cap09 reduces mean worst-direction local feedback gain2.415 ->1.894,
  but radial2.344 and full passes0. Random gain is already below1 on average in
  the baseline; this does not constrain its most sensitive directions.
- Baseline prediction after one generated write MSE.02717 vs.00388 after a clean
  real write atprefix32. This is evaluation-only evidence of short feedback error,
  not proof that memory alone is responsible.

Implementation:
- memory_recent.py: SlowFastWriter damps selected GRU coordinates; same parameters
  as baseline. LocalReader has optional identity-initialized residual adapter.
  No private persistent G state; adapter never writes back into D-owned M.
- memory_handoff_scout.py: slow_dim/rate; g_memory_adapter, adapter_width/bottleneck;
  repair_weight/noise/target; stability_g_weight/d_weight/noise/max_gain;
  dynamics_min_prefix. All default off. Repair trains G adapter only. Stability
  compares two parallel one-step D.write(G(M)) branches, same z/time, detached
  prefix/particle anchors, opposite module frozen. No full generated rollouts.
- Dedicated RNG streams preserve exact resume. Actual old clock checkpoint loads
  strictly; legacy checkpoint resume without new inactive fields/streams tested.
- diagnose_memory_dynamics.py probes completed checkpoints, preserves saved eval
  panel, checks baseline first predictions, tests read/clock/adapter use, noisy
  reads, generated-vs-real write prediction, and optional feedback Jacobians and
  adapter-bypassed autonomous rollouts. Largest singular gain is not a global
  instability certificate; meaningful directions may legitimately be sensitive.
- 74 focused tests passed plus combined full-batch GPU smoke and CPU/GPU probes.
  Metadata-only write counts were clarified after all experiments: archived
  fake_writes_in_training counted feedback replacements only, while stability
  branches are described by local_dynamics and G-call counts. Current code adds
  explicit D/G write counts. No training semantics or results changed.

Next: discuss sample-conditioned repair or preservation of predictive usefulness.
Previous single-write feedback already failed; do not merely rename it repair.
Isolate the new adapter objective/conditioning and bound its local G-call cost.
No new sweep is automatically selected. No extrapolation to unseen radii was
established. Keep fixed particles, D-only writer, default B-cap, no clipping/EMA,
no seed sweeps, metrics-based completed-only decisions, and no unsolicited agents.
Central tail: tail -F runs/memory_path/core_round1/train.log

---

# Compact handoff: clock milestone accepted; first principles next

User explicitly considers sustained motion from the clock a success, even though
full circle preservation remains unsolved. Preserve that distinction: the clock
addresses stopping in these scouts, and memory still supplies predictive
information. This is a milestone to build on, not a reason to discard the clock.

User requested committing the accumulated experiment work before compact. Next
session: hear the user's ideas and work through the formulation from first
principles again, aiming for full-circle behavior without generated trajectory
rollout training. Do not automatically launch another sweep or the previously
suggested writer diagnostic. No experiments or diagnostics are running.

Current runtime concept: fixed particle z, external step counter t, D-owned M;
generate x = G(z, M, Fourier(t)), then M = D.write(x, M), then increment t.
No runtime expert. Keep configuration-controlled changes, default exact B-cap,
no gradient clipping/EMA, metrics-based evaluation, and no seed sweeps.

# Latest completed round: external Fourier clock (round7)

User asked to test an external Fourier metronome for G on both GPUs and determine whether successful models still use memory. Eight 2k scouts completed via runs/memory_path/clock_round7; all jobs done, none failed/pending/running. Four completed models also received clock/memory interventions. Nothing running, no longer runs selected. This work is included in the requested compact-preparation commit.

Read [clock assessment](../memory-handoff/clock_round7/assessment.md), [leaderboard](../memory-handoff/clock_round7/leaderboard.md), and raw results/interventions in the same directory.

**All eight: 0/128 full cold circles at256/1024 and 0/128 original-orbit continuations after prefixes8/32 at both horizons.** All seven advancing clocks prevent late stopping (0%), compared with57.0% in the constant-clock architecture control, but do not preserve circles. Random-origin recent/residual model diverges severely. No winner to promote.

Main clock model: freeze clock after32 generated points ->95.3% late stopping, versus0% normally. On real-prefix32 contexts, matched next-point MSE0.00664; shuffled memory1.45982 (~220x); zero memory0.73404; reset clock0.03601 (~5.4x). Both inputs influence G, but beneficial joint use for autonomous circle fidelity remains unestablished. Clock continues driving activity even with zero/frozen memory. Sustained motion is an accepted milestone; full-circle success remains a separate unmet target.

Implementation: configurable clock_bands/frequency/rate/to_d/origin_max in memory_handoff_scout.py. LocalReader appends sin/cos time features to particle input internally; actual learned particle dimension unchanged. Clock is an external integer counter, no private recurrent state. Explicit target times plus per-episode optional random origins; cold t=0, warm t=prefix length. Shared-clock D sees identical time for both candidate scores. Generic rollout/continuation pass time only for clock-enabled G, preserving other trainers. Old non-clock checkpoints supported. No generated training rollouts or new auxiliary losses, unchanged B-cap, no clipping/EMA.

experiments/diagnose_memory_clock.py validates saved baseline reproduction and startup preservation, then runs frozen/half-speed clocks and zero/shuffled/frozen memory reads after32 generated points, plus paired real-context sensitivity probes. Older handoff/latent diagnostics now also forward explicit time. 68 relevant tests passed; GPU training/evaluation smoke and CPU diagnostic smokes passed.

Earlier recommendation: retain optional clock for controlled comparisons and investigate local write-induced loss of predictive information versus G prediction error. The user's latest direction is first-principles discussion with their ideas before choosing another experiment. Nothing is queued. Preserve first-principles discussion below. Central tail remains `tail -F runs/memory_path/core_round1/train.log`.

---

# Completed: 21 valid bounded-feedback experiments

## Latest user direction: first principles after compact

User is compacting now and says we need to approach this from first principles.
This supersedes automatically launching the local prediction-loss diagnostic
recommended below. Do not start another configuration sweep or add auxiliary
losses before reconsidering what the objective actually identifies.

Start the next discussion with the smallest precise formulation:

```python
# Teacher-context training, with real history strictly before x_t:
M_t = D.write_history(x_real[:t])
x_fake = G(z, M_t)
L_D = paired_GAN_D(D.score(x_real[t], M_t), D.score(x_fake, M_t)) + B_cap
L_G = paired_GAN_G(D.score(x_fake, M_t), D.score(x_real[t], M_t)) + prior_reg
# D updates writer + scorer; G updates reader + particles, with D frozen.

# Autonomous runtime, fixed z throughout:
M = zeros()
repeat:
    x = G(z, M)
    M = D.write(x, M)
```

Questions to derive rather than assume:

- What does matching the conditional next-point distribution under real-history
  memory imply about the autonomous transition M -> D.write(G(z,M),M)? What does
  it leave unconstrained, even at an idealized optimum?
- Training samples z independently of each real episode; runtime memory depends
  on that same fixed z. Are the training and runtime JOINT distributions of (z,M)
  compatible? Distinguish this hypothesis from ordinary state distribution shift.
  A weak particle-swap diagnostic did not establish it as the actual cause.
- Which uncertainty remains after zero, one, or several observed points? How
  should a persistent trajectory particle encode choices during cold startup?
- Is there a coherent real target after a generated replacement? The original
  next real point is a recovery target, not automatically a valid continuation of
  the altered history. Do not assume exposure alone fixes that mismatch.
- What learning signal can constrain local autonomous dynamics without full
  generated training rollouts? Distinguish distribution matching, pointwise
  accuracy, stability, and preservation of the original orbit.

Keep evidence separate from hypotheses: D/G demonstrably use motion information
in M on real contexts. The failures do not prove memory is ignored, OOD is the
sole cause, or pointwise GAN training is impossible. Noiseless controls also
failed at the tested budget. Late oscillations were not reliable cold starts and
collapsed to one direction. Identify a discriminating test only after clarifying
the mathematical objective and the missing constraint.

Preserve the concept: G reads D-owned M, D owns writer parameters, fixed particle
per trajectory, no realtime expert required, and no full training rollouts.
No jobs are running. Work remains uncommitted; no commit requested this turn.

## Completed experimental evidence

Latest session DONE:20 fresh2k scouts and one exact5k extension. No jobs remain.
All full cold256/1024 and1024-step original-orbit pass rates are0. One short warm
exception: recent_bound_control has1/128 atprefix32/256, then0/128 at1024. All other
warm256 rates are0. The toy remains unsolved. Read the comprehensive
[session assessment and recommendation](../memory-handoff/feedback-session.md)
and [verified summary metrics](../memory-handoff/feedback-session.json).

Valid queues: feedback_round3 (8), recent_round4 (6), backprop_round5_corrected (5),
clean_round6 (2). All have zero pending/running/failed; sessions3006,72596,61466,
91344 closed. No further extensions selected. recent4_delta's2k late-circle lead
(60.2%, allCCW) vanished at5k with84.4% stopping. Noiseless controls also failed.

IMPORTANT: backprop_round5 WITHOUT_corrected is INVALIDATED. Missing real-score
context-gradient branch affected G's paired loss:2 completed jobs excluded,
2 active jobs intentionally stopped,1 pending cancelled. Both branches now retain
G gradients when M depends on G; common-offset cancellation has a regression test.
Earlier detached-feedback/recent-memory jobs are unaffected. Report/checkpoint
guards reject invalidated sources/legacy backprop checkpoints.60 focused tests
plus1 reporter-exclusion test passed. Corrected full-batch GPU smoke passed.

Implementation: memory_handoff_scout.py supports configured single-write feedback
(probability,strength,min-prefix,ramp,optional G backprop) and recent-point slots
inside one D-owned M. memory_recent.py implements GRU+raw shift register and
bounded absolute/residual readers. G is feedforward with fixed particles and no
private state. No full generated training rollout, no path loss, default B-cap,
no clipping/EMA. Runtime needs no expert. Per-run source snapshots saved.

Next proposal: configured local G prediction-loss diagnostic on informative real
prefixes, retaining GAN branch, to separate adversarial optimization from
representation/feedback limits. Label its changed objective explicitly. NOT
implemented or queued. Do not keep extending failed models or replace full-path
success with burn-in-only metrics. No seed sweeps or unsolicited agents.

Central tail: `tail -F runs/memory_path/core_round1/train.log`.
Feature branch feat/sequential-memory-path; work uncommitted. No commit requested.
Preserve unrelated .claude/, results/motion/, sparse-ucd.log.

# Completed: pointwise local round 2

## Latest discussion / next round after compact

User is compacting now and wants another round afterward. Do not launch more
jobs during compact preparation. Branch: `feat/sequential-memory-path`.
All current jobs finished; work remains uncommitted. No commit requested here.

User agrees the important mismatch is real-history writes during training versus
generated writes at runtime. D sees fake candidates during training, but those
candidates never affect subsequent memory. Small error is not automatically OOD
(real observations already contain noise); accumulated, correlated errors may
move position and memory away from training support. This is a plausible failure
mechanism, not yet directly established by the completed diagnostics.

Latest proposed next experiment, not implemented or queued:

```python
M = D.encode(real_prefix)  # strictly before x_t
x_fake = G(z, M)
M_next = D.write(M, stop_gradient(x_fake))
# Local GAN prediction of x_(t+1), conditioned on this same M_next for real/fake.
```

This introduces one generated write and two sequential G evaluations per example;
describe that cost honestly, rather than calling it the existing one-point-only
trainer. No long generated training trajectory or trajectory critic is intended.
Use the same particle for both predictions. D alone owns/trains writer parameters;
detach the generated replacement, not all of M_next (D still needs writer grads).
Real x_(t+1) is a recovery/denoising target after replacing x_t; it need not be the
natural continuation of an arbitrarily wrong generated point. Make that assumption
explicit and test a bounded/mixed replacement strength if needed. Rebuild context
after updating D as in the current trainer, keeping causal timing and cached
real/fake contexts aligned. This is the leading discussion direction, not an
already agreed implementation specification. Preserve no-clipping/API B-cap,
fixed particles, expert-free runtime, configs, completed-only metrics, both GPUs,
and completion-driven queue notifications. No seed sweeps or unsolicited agents.

Recent-observation slots inside D memory plus residual G output remain a separate
architecture idea. They do not themselves remove the training/runtime mismatch.
The user deferred further experiments until after compact.

## Completed evidence

Latest user constraint: solve without generated training trajectories. User
correctly challenged the claim that D lacks incentive to use M. Baseline probes
confirm temporal use: matched G MSE .00560 versus shuffled1.405; opposite-direction
histories ending at the same point give D correct-continuation preference~89%.

Eight 2k scouts finished successfully in340.7s on both GPUs. Queue
`runs/memory_path/local_round2` is sealed and empty; drain35723 is closed.
Four ordinary-GAN scouts (G SiLU/tanh, G128, D256) all failed full cold/warm metrics.
Four D-only auxiliary scouts also failed warm fidelity; predictive10 alone had
2/128 cold passes, both CCW at256. No convincing winner; no longer runs queued.
50 tests passed; changes remain uncommitted. No fake training writes or G unrolls.

Read [assessment and next recommendation](../memory-handoff/local_round2/assessment.md),
[leaderboard](../memory-handoff/local_round2/leaderboard.md), and
[plan/formulation](../memory-handoff/local_round2/plan.md).
Next proposal: recent-observation slots inside D memory plus residual G output,
keeping ordinary pointwise GAN objective. Not implemented. Feedback instability
is the working hypothesis; current probes do not establish generated-state OOD.
The toy remains unsolved. Same tail: `tail -F runs/memory_path/core_round1/train.log`.

# Completed: handoff-only scouts and 5k checks

Latest user direction: explore without cold loss because generated trajectories
are expensive. This supersedes extending light/heavy cold-feedback winners now.

Ten 2k-update scouts have finished successfully in
`runs/memory_path/handoff_round1`, using `experiments/memory_handoff_scout.py`.
No cold or warm loss, no trajectory critic, and no generated training unroll.
G makes one batched independent point prediction per D/G update. D still encodes
real history; memories are cached for real/fake/B-cap scoring and rebuilt once
for G after updating D. D alone trains the writer. Four points per episode share z.

Read [the current plan](../memory-handoff/README.md),
[leaderboard](../memory-handoff/round1/leaderboard.md), and
[metrics](../memory-handoff/round1/results.json).

Scouts vary real prefix length (63/8/16), observation corruption (.03/.1), memory
snapshot jitter (.05), memory size (32/8/64), D point/context interaction, and
zero-prefix sampling (.125/.25). Same 2k updates / 10k schedule, API B-cap,
no clipping, no EMA, no seed sweeps. Context/augmentation RNG streams are separate.
The old completed handoff_only baseline is included without retraining.

All 48 focused tests passed, including causal context, active exact B-cap,
cached/recomputed gradient equivalence, ownership, resume, and no generated
training unroll. Batch-128 CUDA smoke passed for memory64 + interaction + noise.

The previous tail command still works because the new queue's train.log points
at the existing central log:

```bash
tail -F runs/memory_path/core_round1/train.log
```

Initial drain session 45793 finished. Completion-only notifications:
`runs/memory_path/handoff_round1/notifications.log`. Reports refresh after jobs
complete, before notifications. Initial queue is sealed and complete. See the follow-up below.
Do not restart this queue blindly. Check completed results only, then compare
original-orbit fidelity, cold stability, both directions, diversity, and training
cost before deciding on longer runs. Long rollouts are final evaluation only.

## Current follow-up

The initial round failed cold256 and real-prefix fidelity in every configuration;
only start25 had a single cold1024 pass (1/128). Input noise .10 reduced late
stopping to 4.7% (default dense 68.8%). Initial queue took 6.59 minutes.
[Full assessment](../memory-handoff/round1/assessment.md).

Both input10 and start25 completed exact continuations to 5k (3k additional updates) in
`runs/memory_path/handoff_round1_long`. These are inexpensive diagnostic extensions,
not convincing winners. No new losses or generated training unrolls.
Drain session 56026 finished without failures; both queues are empty. Same central train.log as before.
[Combined completed-only leaderboard](../memory-handoff/long/leaderboard.md).
Both 5k runs have zero cold256/1024 passes and zero original-orbit passes.
Noise input10 late stopping worsened to 75%; start25 late stopping is 30.5%.
[Final assessment and recommendations](../memory-handoff/long/assessment.md).
Do not promote these failed runs merely for more steps. Next suggested local
scout: temporal mismatch negatives for D, still without any trajectory loss.
No further jobs or implementation changes have been started.

## Previous completed core round

Branch: `feat/sequential-memory-path`. Core formulation trainer, configs, metrics,
and tests are implemented. The shared two-GPU queue in
`runs/memory_path/core_round1` finished all nine jobs successfully in 53.2 minutes.
No jobs are pending or running. Read [the assessment](../memory-core/round1/assessment.md)
before choosing extensions. No extensions have been launched.

## Current work

Seven 2,000-update scouts: handoff_only, handoff_cold, handoff_warm, handoff_both,
warm_only, handoff_cold_light, handoff_cold_heavy. All share GRU32 D memory and
feedforward G, learned writer, fixed particles, default exact B-cap, no clipping,
no EMA, same data/initialization/schedule. No seed experiments.

Training and evaluation rules, comparison caveats, and table of loss weights:
[core round README](../memory-core/README.md).

Two existing autonomous controls (2k and 10k) have completed new evaluation without
retraining. The old 10k model passes cold full256/1024 at 53.1%/51.6%, with 31 CW
and 37 CCW full256 passes, but has zero reference-orbit passes after either prefix.
The 2k control gets 14.1%/15.6% cold and zero reference-orbit passes. Neither proves
that an arbitrary expert trajectory is preserved by its generated feedback loop.
All seven new scouts are now complete. Heavy/light handoff plus cold feedback
reach 33.6%/26.6% cold 1024 success; light preserves balanced passing directions.
Every formulation has zero reference-orbit passes. See the completed assessment.

## Monitoring

```bash
tail -F runs/memory_path/core_round1/train.log
```

- One permanent log for all jobs; labels identify run/GPU.
- Drain emits only completion/failure events. These are saved in `notifications.log`.
- Optional completed-job reporter refreshes the numeric leaderboard before sending
  the completion event. No polling of training logs is needed.
- [Leaderboard](../memory-core/round1/leaderboard.md)
- [Detailed metrics](../memory-core/round1/results.json)
- Queue manifests: pending/, running/, done/, failed/.
- Queue is sealed and finished; its final event is queue_complete with no failures.
- Drain ran in tool exec session 81434. Completion is recorded in notifications.log.

All 41 focused tests passed. Tests cover causal memory timing, read-only candidate
scoring, parameter ownership, full feedback gradients, exact B-cap, exact resume,
old/new cold-only equivalence, and completion-triggered queue reporting. A full
batch-128 CUDA smoke passed with all three losses and full 1024-point evaluation.

## Next decisions after completion

Compare H (real-memory handoff), C (cold feedback), W (mixed real-prefix lengths
0/8/32 then feedback), and their weights. All expose runtime:

```python
M = D.write(real_prefix)  # optional; otherwise zeros
for each generated point:
    x = G(fixed_z, M)
    M = D.write(x, M)
```

H trains next-point prediction from real points strictly before the target. Fake
candidate scoring is read-only. C/W explicitly train generated writes. D alone
updates writer weights; G gradients through generated state remain intact.

Cold and warm results are distinct axes. Warm checks original-circle geometry,
speed, direction and startup discontinuity, not merely any valid resulting circle.
Review continuous errors, particle coverage, passing directions, stopping, state
statistics and interventions before selecting winners. Reported auto-ranked
candidates are provisional; the script does not launch longer runs. If no single
formulation wins both axes, retain a candidate on each axis for further testing.

Then resume promising checkpoints for longer training, with exact optimizer/RNG
state. Add private G recurrence only after interpreting the matched feedforward
comparison. DDGAN, FiLM/Fourier and memory-size sweeps remain deferred.

Previous complete history is committed at dfaa1f1. The toy is still unsolved.
Preserve unrelated `.claude/`, `results/motion/`, `sparse-ucd.log`.
