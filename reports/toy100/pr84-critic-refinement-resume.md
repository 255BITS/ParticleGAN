# Exact fixed-target continuation support

This change adds implementation support, not a new training result. Five
bounded tests pass; no long continuation or perturbation was run.

`resume_mode_hold(recorder, generated_source, saved, completed_steps=...,
target_steps=..., fail_fast=True)` restores a complete post-update snapshot
**before** the host's next `noise_policy.set_step`. It inserts one guarded
pre-step call and replaces only the outer loop bounds; a structural inverse
checks that no other host source changes. Noise keeps its original 1200-step
horizon and G/D/prior rates remain .00425/.00425/.0085. The caller keeps the
original constant-recipe host context and must source-bind the saved artifact.

Restoration covers model buffers and parameters, both Adam state dictionaries,
EMA, data/global/input/output RNG, and every fixed-scale NoisePolicy attribute,
including cumulative counters and traces. Snapshot structure, absolute moment
steps, noise configuration and constant rates are checked. Each subsequent
update is audited for exactly three optimizer callbacks but one actual Adam
moment increment per player. The new recorder's diagnostic row numbers restart;
absolute host updates and Adam counts continue from the saved state.

Every live checkpoint is measured with the unchanged eight-mode/.90-HQ rule.
The optional first-failure stop occurs only after the completed G update and
EMA/checkpoint. It saves a before-set-step state and the failed post-checkpoint
state; it does not roll back or alter an update. `FirstHoldFailure` is explicit
and the context still finalizes its counts. Other exceptions remain errors.

The observer also records accepted clean `G(prior.z)` output RMS/maximum motion,
raw used G/prior gradient norms, and post-base Adam denominator/metric ranges.
Clean forwards must preserve all tracked RNG streams and immutable buffers.
These measurements are diagnostic: there is no minimum-movement requirement,
and they do not prove reactivity to a later learning signal.

Tests split a three-update cold run after its ordinary first checkpoint,
restore, and complete the remaining two updates. Full model/Adam/EMA/RNG state,
all noise histories, live metrics and non-timing update records match an
uninterrupted run bitwise, for both global and isolated output RNG. A second
test resumes from an actual prefix **after final evaluation**: learning state,
training RNG and update records remain exact, while the additional prefix
evaluation counters/history are honestly retained. Only those evaluation
history fields differ from a run that had no prefix finalevaluation. Tests
also cover dense fail-fast capture and rejection of an altered noise horizon.

Do not directly pass a captured pre-gradient state whose current `set_step`
already ran: that state has one clock update beyond its completed Adam steps.
This implementation deliberately rejects that stage. The intended inputs are
source-bound final cold snapshots, or complete post-update/checkpoint snapshots
with current EMA. Intermediate pre-EMA captures are for diagnosis rather than
resuming this outer-loop boundary.

```python
with pr84_critic_refinement_cold(task="mode_hold") as (recorder, source):
    try:
        with resume_mode_hold(recorder, source, saved,
                              completed_steps=1200, target_steps=2400) as run:
            # Use original1200-horizon NoisePolicy and constant recipe context.
            mode_hold.train_mode_hold(ModeHoldRecipe(steps=2400, ...), ...)
    except FirstHoldFailure:
        pass
    receipt = run.receipt()
    torch.save(run.final_state, output_state)
    if run.failure is not None:
        torch.save(run.failure, failure_state)
```
