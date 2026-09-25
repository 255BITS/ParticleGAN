# Search direction: continuous learning from K3P

Start from the exact selected K3P bundle in current-research-base.json. Preserve
it as the measured parent: 22/22 declared GPU toys, ring hold 1200/1200 and
extension 300/300, but target-shift recovery only 28/81 deadline checks. Every
modified formulation earns its own scores. This brief configures a search;
the invocation of the launcher supplies its time/proposal/worker budgets.

## The question

Can K3P retain precision and stability while remaining responsive to new data,
without knowing the training horizon or defining a one-way "late training"
phase? The desired direction is a reversible controller driven by training
signals. Constant-rate learning is a useful control, not a requirement. Neither
is established by the parent. Search time limits bound research compute; they
must not become inputs to the learned formulation's schedule.

K3P never deliberately freezes on convergence. Its selected floors leave G/D
rates at 1% of their initial rates and the prior at 5%. The EMA critic and Adam
updates continue. Its handover is nevertheless schedule-dependent: the mixing
weight reads the critic's last applied LR divided by its maximum observed LR.
It stays entirely on the early penalty until that ratio falls below 0.5, then
reaches capped gradients plus the EMA anchor at ratio 0.01. The native network
schedule has a 1600-update horizon cap; prior/noise schedules also use a horizon.

**Do not merely set learning-rate floors to one:** with the existing handover,
constant critic LR keeps the ratio at one, the mixing weight at one, and the
EMA gradient anchor inactive. A constant-rate candidate must explicitly state
how it decouples the critic constraint from that LR clock.

The current recovery failure does not identify a sole cause. Step size, critic
anchoring and their interaction are hypotheses to test, not established facts.
EMA decay and guard-estimator warmup are memory/estimation choices; they are
different from a rule that must know the final training step to become stable.

## Candidate requirements

- Retain the K3P lineage, learned particle prior, direct response and bounded
  sparse-latent rule unless the assigned lane explicitly declares a change.
  Start from copied, hashed candidate files; never edit the pinned parent.
- A continuous-policy candidate must not read the total training budget,
  evaluation scores, convergence detection, target identities, shift times or
  known target centers. No timed restart/reset at the test's target change.
  Rates or damping may respond reversibly to ordinary training/optimizer
  signals. Audit actual applied rates and critic mixing weights.
  Check that changing the declared training horizon leaves an otherwise
  identical training prefix unchanged; report every remaining budget dependency.
- Declare every LR, penalty, anchor, guard and noise rule. Horizon-dependent
  noise left from K3P is still a scheduled component: retain it only as a
  labeled intermediate ablation, not as a horizon-independent final result.
- Keep architecture, data, existing seeds, optimizer-step budgets, auxiliary
  host losses and canonical quality thresholds fixed. No seed sweeps or
  coefficient grids. Compare equal-step and added-compute costs honestly.
- Keep all training and history tensors CUDA FP32 in the qualified deterministic
  environment. Each attempt owns its output directories, source copies and logs.
  Use the existing frozen runtime and fixtures from the gap-fill manifest.

## Gate order and evidence

1. Run the candidate's own standard hold with its 300-update extension **and**
   the target-shift recovery test. Use the existing drivers and fixed verdicts;
   record both outcomes even if one fails. A passing recovery needs its own
   matched frozen control. Audit optimizer updates and actual rates after the
   shift so stationary output cannot masquerade as active adaptation.
2. Screen the sensitive transfer problems: mode_hold, unequal mass, unequal
   width and stripes. Only survivors advance through their own full 22 gates,
   including full 7000-update coverage AND accuracy for all three natives.
   Reuse parent evidence for comparison, never as a modified candidate's pass.
3. For survivors, run a separately declared long-hold/delayed-change stress
   check and a second target change in the same uninterrupted state. Freeze the
   protocol before observing results; keep canonical scores separate. The
   learner cannot read the stress test's change times. Repeated timely recovery
   is stronger evidence than one successful target shift, but is still a finite
   test rather than a claim of indefinite learning.

Use the existing hold/shift drivers first, not a replacement benchmark platform.
Additional stress checks may extend them locally without altering the canonical
tasks or thresholds. Do not cherry-pick cycle troughs or move the recovery
deadline. Keep model, optimizer, critic EMA, response/latent history and RNG
continuous; fresh-process restart equivalence is not yet qualified for K3P.

Rank results by **hold + extension + timely recovery**, then retained toy passes.
Report worst HQ, passing/failing checks, recovery delay, repeated-change results,
rate/mixing traces and added critic evaluations. Keep FAIL/ERROR/NOT_RUN visible
and distinguish diagnostic screens from qualification. A higher LR alone is not
a successful formulation if it trades stability or precision for faster motion.

First round: three Codex and five Grok attempts, no Claude. Each owns a distinct
lane and isolated checkout, with four benchmark workers assigned to each GPU.
First-round defaults: at most three mechanism proposals per lane, 45 minutes
and one benchmark worker per lane; explicit invocation budgets override the
time/proposal defaults. No nested agents, seed repeats,
pushes or comments from attempts. Those are caps, not quotas. Start candidate
training promptly, adapt from measured failures, and preserve a final leaderboard.
The selected parent has already been measured; do not spend the round rerunning
unchanged baselines. A useful negative result should identify the failed gate
and narrow the next mechanism choice.
