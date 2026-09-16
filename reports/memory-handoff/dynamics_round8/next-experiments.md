# Proposed next round: repair predictive usefulness

SUPERSEDED: the user rejected MSE training objectives. The current authorized
plan is ../exploration_round9/plan.md: adversarial exploration with clean versus
shared judging memory. This document records an abandoned proposal only.

Planning only. No configs queued or training launched.

## Evidence and hypothesis

The clock sustains motion, but every round8 model has zero full cold-circle and
original-orbit warm-continuation passes. Memory-coordinate denoising and reduced
local feedback gain did not solve continuation. In the clock baseline, prediction
MSE after one generated write is .02717 versus .00388 after a clean real write.
This implicates the short feedback loop without establishing whether the writer
loses information, G misreads it, or both.

The observed repeating path is not yet a measured recurrence. Test it before
interpreting it as clock locking. The Fourier features nearly repeat after201
integer steps; a valid orbit's period depends on its own speed.

Desired behavior: preserve process identity and useful progress while correcting
small observation/generation errors. Equality of arbitrary memory coordinates is
neither necessary nor sufficient for this. A stateless repair cannot restore
information that its inputs no longer contain. Sample conditioning adds a useful
computation/inductive bias, not independent evidence about the correct orbit.

## First: completed-checkpoint diagnostics

Use saved trajectories for late-window normalized position/velocity recurrence
over a broad lag range, including near201 and each reference orbit's period.
Report across all particles and both directions; avoid choosing video examples.
Then evaluate small clock-rate changes with identical prefixes/particles. Test
whether dominant generated periods track the external clock or reference speed.
Clock interventions change G inputs and may themselves be out of distribution;
period tracking is supporting evidence, not a causal proof of the entire failure.

Extend paired one-write probes with bounded observation perturbations and genuine
prefixes from distinct processes. Compare next-point errors, direction, and step
size after clean versus altered writes. These probes should distinguish recovery
of small errors from erasing real differences between processes. No circle
parameters or fitted repair vectors enter training.

## Runtime candidate: stateless proposal-conditioned read repair

Let H be the existing point reader and R an identity-initialized residual adapter.
For a sample-conditioned variant:

    proposal = H(z, M, clock(t))
    readable = M + R(M, proposal, clock(t))
    x = H(z, readable, clock(t))
    M = D.write(M, x)

Only the final sample is written. D retains the only persistent learned memory
and sole ownership of writer parameters. R's output is never stored or fed to D.
There are two reader calls per runtime step; memory-only variants need one.
Both use the same fixed particle throughout an episode. This proposal avoids
requiring a second recurrent state or access to the previous raw memory snapshot.

## Local repair objective

Keep ordinary real-prefix adversarial training as the anchor. Add an auxiliary
that trains only R to preserve clean-context predictions under corruption:

    target = stop_gradient(G_complete(z, M_clean, t))
    prediction = G_complete(z, M_altered, t)
    L_repair = mean_squared_error(prediction, target)

In this auxiliary, freeze H/particles/D parameters while preserving derivatives
through H with respect to its repaired memory input. Detach the raw proposal
conditioning R. In the ordinary GAN loss, train the complete G normally. Build
clean targets from the current model; no EMA or persistent teacher. The teacher
can be wrong, so reduced consistency loss alone cannot justify promotion.

Two corruption sources isolate different hypotheses:

1. Gaussian perturbations of the clean memory, matching existing noise .15.
2. A single generated replacement at an adjacent local transition:

       M_clean = D.write(M_before, real_point_t)
       M_altered = D.write(M_before, stop_gradient(G_complete(z, M_before, t)))
       compare next reads at t+1 using the same z and clock

The second source is explicitly a recovery target toward the original process,
not a claim that the replaced observation has the same natural future. Initially
restrict it to short local errors with a configurable distance gate or blend;
report the active fraction and the unblended error distribution. Choose its
threshold from the completed diagnostic error distribution before starting jobs.
Avoid claiming success if the gate simply excludes difficult cases.

This is one generated write followed by a local read, not full rollout training.
Earlier single-write adversarial feedback failed. The new distinction is the
adapter-specific predictive-consistency target and its conditioning.

## Six proposed fresh scouts

All use the same clock, M32, reader widths, training schedule, and adapter hidden
width/bottleneck. Hold auxiliary weighting fixed across the four repair scouts;
select a common initial weight with a small gradient-scale smoke check, then
record it before launching. No seed sweep. Old raw-memory repair and clock
checkpoints remain historical comparisons, not fresh matched controls.

| Scout | Adapter input | Auxiliary corruption | Question |
|---|---|---|---|
| read_control | M, clock | none | Matched one-pass adapter control |
| proposal_control | M, proposal, clock | none | Does two-pass refinement alone help? |
| read_noise | M, clock | Gaussian memory noise | Does predictive repair beat coordinate denoising? |
| proposal_noise | M, proposal, clock | Gaussian memory noise | Does the sample help repair synthetic damage? |
| read_generated | M, clock | One generated write | Is realistic damage the missing ingredient? |
| proposal_generated | M, proposal, clock | One generated write | Do realistic damage and sample conditioning help together? |

Expected complete G-phase reader calls, sharing the base adversarial prediction:
1/2 for the controls; 2/4 for noise; 3/6 for generated-write repair, respectively.
The D phase uses1/2 reader calls without generated memory writes. These counts
exclude any recomputation needed to route auxiliary gradients separately; log
actual calls and measured wall time. Keep other slow-memory/stability/feedback
auxiliaries off to isolate this experiment. All new features default off.

## Evaluation and promotion

Start with2k updates on the unchanged10k schedule, through the existing two-GPU
queue. Inspect completed results only. Central log remains:

    tail -F runs/memory_path/core_round1/train.log

Primary measures remain full cold-circle and original-orbit warm-continuation
fractions at256/1024 steps, prefixes8/32, including startup. Also report direction,
speed error, first32 position error, long radial error, and stopping. One-write
predictive recovery and recurrence are mechanism diagnostics, not replacements
for task success. Include normal versus adapter bypass and shuffled-memory reads
on promising completed models; evaluate useful autonomous dependence as well as
real-context dependence.

Promote at most two models: first by improvements in primary pass rates with
supporting continuous metrics and no return to widespread stopping. If all pass
rates remain zero, allow at most one explicitly diagnostic extension only when
it improves long radial error by at least25% at BOTH prefixes versus its matched
control, improves early position error and one-write recovery, and does not
worsen direction agreement or stopping. Continue exact checkpoints to5k with
unchanged schedules. Reassess before any further extension.

Warm recovery and cold discovery are separate requirements. M=0 contains no
observed process identity; z must select a process there. A warm-only improvement
is useful but does not solve cold start. Once ordinary continuation improves,
test new radii/speeds using genuine observed prefixes, separately from in-range
performance. Do not equate statistical dependence on M with generalization.

## If repair fails

If prediction-space repair fits its local task but closed-loop behavior remains
wrong, inspect accumulated process/phase drift before adding capacity. If damaged
states no longer support accurate prediction even with a separately trained
diagnostic reader, that supports testing a D-side robust predictive auxiliary;
failure of one diagnostic reader alone does not prove information is absent.
Such a writer experiment must include a clean-prediction control because earlier
clean-only D prediction auxiliaries already failed to solve the task.

If recurrence tracks the clock strongly, test clock encoding/rate dependence as
a separate ablation. Changing a clock can change the wrong orbit without fixing
the learned dynamics. Do not force all processes onto its period.
