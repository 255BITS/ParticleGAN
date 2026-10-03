# Optimizer reopen: source and closed-record review

Scope: CPU and standard-library reads only. No checkpoint interpretation, model construction, forward, optimizer step, draw, scorer, or CUDA call. The RA12 package and all existing evidence remain unchanged. This report distinguishes an optimizer shock from evidence that the real target changed.

## What the signal measures

`OptimizerSurprise.group_q` reads the completed Adam step's gradient divided elementwise by the bias-corrected `exp_avg_sq` root, averages each parameter tensor, then averages the parameter tensors in one group. The detector compares a 12-update fast log mean with a slow reference. It takes the geometric mean across the currently nonzero group signals, with equal weight per group. The usual roles are generator, table, scalar output noise, and critic. A sustained ratio above 2 fires after 12 observations; the existing abrupt-rise and rearming rules remain part of this law.

This is a statistic of the changing game and optimizer memory. It has no observation of the real target. A fixed target can therefore cause an abrupt optimizer shock. An external target change can also cause one. The scalar statistic alone cannot identify which cause occurred.

## Concrete internal transitions

1. **Critic loss epoch.** KA2 uses pure A for calls 1–799 and a blended A/B/anchor loss from call 800. `GANTrainer._step` invokes the penalty once per critic update. The detector receives the changed loss's gradients without an epoch marker or rebase. KA2's own surprise ratio is initialized only after 25 blended observations; an optimizer reopen can occur before that ratio exists.
2. **Active group membership.** Exactly zero gradients are omitted from the aggregate, while their old fast and slow levels are retained. A clamped noise scalar can disappear and later return. Equal weighting makes this scalar as influential as a network group in the geometric mean.
3. **Population ownership.** The policy queues generator/table/noise signals in `after_generator_step`. Copy/birth optimizer-row transport and row-evidence rebase occur later in `finish_step`; the queued scalar is folded in at the next `begin_step`. The detector has no population-incarnation marker or rebase. This is a possible table-signal confound when moves actually occur.
4. **Other critic work.** The critic spike guard activates only after its configured prior-step count. It modifies gradients before Adam; KA2 then observes the completed step, can reseed its anchor, and updates that anchor. These are concrete fields to record, rather than infer from timing.

## Existing records delimit the two failures

| Frozen record | Observation | What this establishes |
| --- | --- | --- |
| RA12 toy | First fire is logged at completed step 820, hence the beginning of update 821. `anchor_events` remains zero. | Its timing is consistent with the call-800 loss change and a fire before KA2's ratio initializes. Exact call/phase attribution still needs the authorized short observation trace. |
| RA12 toy | At step 750 the generator scale is .125; at 1000 it is 1. Output sigma remains at the .029 floor. Copy reactions continue. | The reopen changes the generator's applied scale; a sigma increase is not the recorded post-fire mechanism. Copy effects are not excluded. |
| RA12 MNIST | First fire is logged at completed step 202, hence the beginning of update 203. No population moves are recorded through checkpoint 1250. | Its first fire precedes KA2's call-800 loss change and cannot be explained by table-row transport. |
| RA12 MNIST | All tester scales are 1 at steps 100 and 250. | The fire changes optimizer second-moment memory even when there is no settled LR scale to reopen. It is not a harmless restart. |
| Original quarter-rate moving test | Two post-change periods passed, with fires logged at 514 and 1021. | Any selected control must retain this response. This test uses the earlier trainer package; its `OptimizerSurprise` class AST is identical to RA12's. It does not establish behavior after a proposed control. |

On a fire, the policy shrinks each group's second-moment and AMSGrad maximum memory by its observed ratio squared when that ratio exceeds one, restarts all stationarity testers, and may latch KA2 drift evidence. The first two effects already matter when the KA2 release latch is unavailable.

## Small next diagnostic and repair boundary

Use the parent's authorized short original-stream observations around the two fires. Record exact pending `q`, old fast/slow values, kept keys, roles, pre/post fire ratios, and actual KA2/population/noise transitions. No new training or checkpoint reads are needed in this review. A detector-only CPU replay of those closed scalar records can evaluate a narrowly declared role/epoch control if the parent selects it.

The first defensible control is to stop comparing optimizer signals across a known internal loss epoch. A role-scoped confirmation may also keep table and noise transitions from asserting an external shock. Neither control is yet qualified to fix both failures. In particular, an abrupt network-gradient change on a stationary target remains possible. Do not add a threshold fitted to task quality, steps, seeds, or target rotations. Preserve the default detector behavior and its generic jump/ramp/replay tests unless a separately declared policy scope is selected.

## Limitations

Checkpoint metrics do not retain the per-group ratios at the actual fire. Timing is descriptive. This source review does not claim causal attribution, a new quality pass, or that a known phase reset alone fixes MNIST. Confirmation from an immutable real-reference statistic would address external-target attribution more directly, but adds a separate statistical/state law and is not selected here.
