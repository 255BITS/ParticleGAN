# Adversarial exploration: completed scouts and extension

Ten fresh2k scouts completed on both A6000s, with zero failures. All training
used particle GAN losses, the default exact B-cap and the standard particle
prior regularizer. No MSE, prediction, temporal or stability auxiliary was used.
No full generated training rollouts, private G state, clipping, EMA or seed sweep.

Every scout has zero full cold-circle and original-orbit warm-continuation passes
at256/1024 steps and both prefixes8/32. Every scout retains zero late stopping.
The toy remains unsolved. Queue wall time449.76s; aggregate training827.81s.

See [plan](plan.md), [full leaderboard](leaderboard.md), [mechanism table](mechanisms.md),
[raw results](results.json), and [extension decision](extension_decision.json).
The original MSE repair proposal is explicitly superseded.

## Main comparisons

Selected prefix32 metrics; radial error covers1024 generated points. MSE columns
are evaluation only. All rows below are2k updates.

| Model | Long radial error | Position error first32 | Next-point MSE | Following-point MSE after generated write |
|---|---:|---:|---:|---:|
| clock_control | 2.597 | 1.466 | .00664 | .02717 |
| shared_s25 | 1.726 | 1.365 | .00579 | .02038 |
| clean_s25 | 2.056 | 1.609 | .00606 | .02323 |
| clean_s25_detach | 2.467 | 1.414 | .00540 | .01944 |
| shared_full | 2.318 | 1.154 | .01005 | .02636 |
| clean_full | 2.063 | 1.211 | .00505 | .01615 |
| proposal_control | 1.675 | 1.456 | .00609 | .02352 |
| proposal_clean_s25 | 1.487 | 1.448 | .00497 | .01860 |
| proposal_clean_full | 1.803 | 1.799 | .00633 | .02247 |
| residual_clean_s25 | 2.547 | 1.577 | .00442 | .01657 |

**Clean judging is mixed, not a demonstrated improvement.** With mild blended
replacement, shared judging is better on radial and early-position errors.
With full replacement, clean judging improves radial and one-write prediction
errors, but direction agreement remains near50% and neither preserves an orbit.
Connecting G's proposal through the frozen writer improves radial error versus
the detached control, while worsening early and one-write prediction errors.
These local metrics do not rank long-run behavior consistently.

**Proposal-conditioned repair has a measurable effect.** At mild replacement it
has the lowest radial errors at both prefixes among this round's scouts. Its
prefix32 radial error rises1.487->4.158 when the adapter is bypassed. Full
replacement rises1.803->3.421 on bypass; the proposal-only control rises
1.675->1.905. All bypass cold/warm full pass rates remain zero. This establishes
a checkpoint-specific benefit for radial error, not successful process repair.
The memory-only residual adapter has the best immediate prediction but worst
radial error among the exploration adapters; bypass reduces its radial error
2.547->.700, still without a full-circle/orbit pass. Memory and clock perturbation
results are recorded in mechanisms.md and probes.json.

## What the clock and history probes show

The old clock baseline and the video model slow16_r10 have their best late-window
position recurrence near201 steps for all128 particles at cold and both warm
prefixes. Slowing external clock advancement10% changes median best recurrence
to223; speeding it10% changes it to183. The predicted clock periods are223.4 and
182.8. This supports strong clock control of repetition in those checkpoints;
it does not establish that every memory direction is corrupted.

The new proposal_clean_s25 differs from the simple clock-driven controls. At
prefix32, only51.6% have best recurrence within199..203 steps, versus100% for
clock_control/shared_s25/clean_s25. Its median normalized lag201 recurrence error
is.828, versus.000057 for clock_control. The statistic searches integer lags8..250
in the last512 points; the minimum need not be the fundamental period.

Aligning cold and warm trajectories by absolute clock time shows that97.7% of
clock-control particles and94.5% of shared_s25 particles have late normalized
cold/warm squared differences below.001. Only7.0% of proposal_clean_s25 particles
meet that threshold. This is evidence that different initial histories continue
to affect its output, but not that the retained differences encode the correct
process: reference-orbit passes are still zero. Chaotic sensitivity or different
wrong attractors could also produce that result.

Saved recurrence/intervention data: recurrence_controls.json,
recurrence_scouts.json, recurrence_proposal_intervention.json. Reproduce using
experiments/diagnose_memory_recurrence.py; all diagnostics require a completed
summary and reuse saved trajectories/prefixes. Clock interventions preserve the
initial prefix32 clock phase and change only subsequent advancement.

## Extension decision

All primary pass rates tie at zero, so no solved-model winner was declared.
shared_s25 meets the diagnostic extension gates against the fresh clock control:
radial error improves33.3%/33.5% at prefixes8/32, early-position and one-write
prediction errors improve, and direction agreement/stopping do not worsen.

proposal_control also meets those gates versus clock_control. With the
predeclared limit of one diagnostic extension, shared_s25 was selected for lower
one-write error at both prefixes and one reader call per runtime point instead
of two, accepting slightly higher radial error. The best proposal+exploration
scout fails the radial/early-position gates against its proposal-only control.

One exact shared_s25 checkpoint continuation2k->5k completed, with the same10k
schedule, objective and saved optimizer/RNG state. The extra3000 updates took
122.95s. **It regressed.** Prefix8/32 radial error rose1.727/1.726->2.277/2.272;
first32 position error rose1.425/1.365->2.082/2.131. Prefix32 direction agreement
dropped50.5%->48.3%, and local next-point MSE rose.00579->.00963. All full cold
and warm passes remain zero; motion persists with zero late stopping. No further
extensions are selected. Both queues are sealed and empty with zero failures.
See [longer-run results](long/leaderboard.md) and long/probes.json.

## Interpretation for the next experiment

Clean judging changes writer training as well as the reference supplied to the
scoring head: D receives writer gradients only through real judging history,
because the generated candidate is detached in D's update. Shared judging also
trains D's writer through explored states. G reads explored memory in both cases.
Thus clean judging provides an undisturbed reference but does not itself teach
the writer how to behave at generated states.

A targeted next comparison is a mixture of clean and explored judging contexts
using the same generated continuation, with ordinary GAN losses in both views.
This would combine reference anchoring with D exposure to explored memory and
need no extra generated write or full rollout. It is a hypothesis, not yet
implemented or queued. Retain a proposal-conditioned arm because its autonomous
adapter effect and reduced clock-period recurrence deserve distinguishing from
mere local prediction improvement. More capacity or longer training alone is
not supported by the present results.

## Validation and implementation

80 focused tests passed, including causal reference construction, G gradients
through a frozen writer, exact resume, measured internal reader-call counts,
proposal identity initialization/bypass, no MSE calls during adversarial-only
training, legacy behavior and known-period/clock-alignment diagnostics. Two
full-batch GPU training/evaluation smokes and a CPU proposal diagnostic passed.
All ten runs have identical archived trainer source hashes. The fresh clock
control exactly reproduces the old clock baseline's G, D and particle tensors.

New config switches default off/backward-compatible: feedback_judge_memory,
adversarial_only, and g_memory_adapter='proposal'. D retains the only persistent
learned memory. The proposal adapter receives raw M and H's proposed sample;
clock information enters through H. Repaired read vectors are never stored.
Complete G-call counts and internal point-reader counts are reported separately.

Central log: `tail -F runs/memory_path/core_round1/train.log`.
