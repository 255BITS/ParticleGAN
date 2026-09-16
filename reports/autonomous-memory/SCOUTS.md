# Next-pass scouts: memory, recurrence, and training feedback

Planning only: nothing below is implemented or running. The user wants a set
of scout experiments next pass, after compaction. Use both GPUs. Main objective:
improve autonomous circle quality from zero memory while preserving diversity.
Learned writer is the primary development model; frozen remains a control.
Neither has won overall. Latest committed experiment baseline: `8ec5dee`.

## Important clarification from the discussion

Current training already runs the same generated-feedback recurrence as runtime:
`x = G(z, M); M = writer(M, x)`, starting at zero and holding z fixed. G gradients
propagate through the rollout; writer parameters are frozen during G updates.
D scores both real-written and fake-written memories. Thus generated writes are
not an inference-only change. The first observation-conditioned experiment had
a different protocol and should not be confused with the current one.

State-distribution drift beyond the 64-step training horizon remains plausible,
as do fixed-point or distorted-loop attractors. Errors inside the trained horizon
show that extrapolation is not the whole explanation. D's update does not
differentiate through G's generation process into the writer; changing that
gradient route is a separate experiment, not a missing inference operation.

The distinctive property is D training the memory representation while G reads
it. A GRU or conditional generator can preserve that property. Giving G private
recurrent state can remove M as the sole carrier of motion, but G may then ignore
M. An NTM adds addressable external memory; defer that complexity until a simpler
experiment establishes a need for selective storage/retrieval.

## Staged scout queue

Start with independent mechanism changes; do not combine the whole list at once.
Use 2,000 updates as the initial scout budget, train horizon 64 and evaluate 256,
unless a row explicitly changes the horizon. This is a proposal, not a mandate
to run every candidate regardless of earlier results.

| Priority | Experiment | Paired comparison / purpose |
|---|---|---|
| 1 | D-owned GRU writer, G reads its state | Learned vs frozen GRU; tests adaptive recurrent updates instead of fixed-decay traces |
| 2 | Short-window and cold-prefix D scores alongside full-path score | Learned vs frozen trace writer; tests adversarial feedback without changing the memory mechanism |
| 3 | Give G its own GRU state while retaining D-owned M | Same recurrent G with and without access to M; tests whether M helps beyond ordinary recurrence |
| 4 | Longer training or a short-to-long curriculum | Use the best interpretable preceding architecture; tests whether late-state coverage helps |

Priority 1 preserves the existing memory interface if practical: a 32-value
GRU state can be reshaped into the existing 8x4 representation at the boundary.
Do not assume shape compatibility alone preserves gradient ownership. G must
differentiate through state while D alone trains writer parameters. Compare
capacity and wall time as well as update count.

For priority 2, define the score aggregation explicitly. Prefer one normalized
combined scalar critic score passed to the existing API loss and default B-cap;
avoid silently changing penalty strength with the number of windows. Decide
whether windows retain full-prefix memory or reset it, and document that choice.
Keep a cold-prefix score so startup is represented in the objective.

For priority 3, the old fixed-z feedforward no-memory control is insufficient:
the matched recurrent no-memory G can move and is a meaningful competitor.
Zero/shuffle M only as labelled inference interventions, with the same z and
private G-state initialization. G's private state is trained by G's objective;
D-owned M retains its original ownership. A win without M would challenge the
need for the shared-memory channel on this toy.

Priority 4 should reuse a critic that supports varying windows/horizons if
possible. The previous flattened critic grows with horizon; its initialization
also changes the RNG state before prior creation. Avoid claiming a pure horizon
effect if architecture, prior initialization, or compute changes with it.

## Diagnostics before adding more machinery

Record memory trajectories on generated rollouts during and beyond the trained
horizon, plus real trajectories used solely as offline evaluation references.
Compare state norms, saturation where applicable, and late-state distances to
states sampled inside the training horizon across multiple particles. These are
drift indicators, not proof of OOD causality. Check whether shrinking trajectories
approach a fixed point or continue circling at the wrong radius. Keep real inputs
out of the generation path.

For promising checkpoints, extend evaluation to 1,024 steps without retraining;
this is a horizon diagnostic, not a new-seed experiment. Keep the original
256-step leaderboard comparable and label the longer evaluation separately.

Track full cold-start circle passes, late-only passes, late stopping, radial
error/drift, clockwise/counterclockwise coverage, and starting-position spread.
Direction counts conditioned on circle success must be labelled as such.
Inspect unselected first-particle plots. Do not promote a model based only on
late-circle fraction or loss curves. No exact success threshold beyond the
existing geometry diagnostic has been agreed with the user.

## Execution rules and deferred ideas

- No seed sweeps; match seeds/schedules across mechanisms. Preserve common
  component initialization where feasible and document unavoidable differences.
- Public ParticleGAN API, default exact B-cap, no gradient clipping or added
  parameter-gradient norm logging. Keep noise at the current .03 unless testing
  a separately labelled change. Fixed particles and no runtime expert remain.
- Run independent jobs on the two GPUs, each with a fresh directory, flushed
  log, resolved config, source provenance, checkpoint, and trajectories. Supply
  a combined `tail -F` command. No delegation requested.
- Reuse completed baseline results when valid; rerun only if the comparison
  protocol or implementation changed. Do not repeat the static no-memory control.
- Summarize each completed wave with a leaderboard, explanation, limitations,
  and next recommendation. Stop adding variants when a result calls for a more
  focused diagnosis. Do not silently change multiple mechanisms in one scout.

Deferred conceptual branches: allow D gradients through the generation loop into
the writer; separate D-owned persistent knowledge memory from G's trajectory
state; addressable NTM memory; fresh per-step particles; new shapes. Persistent
knowledge memory could be fixed during deployment and still avoid an expert,
but that changes the original online-feedback mechanism and needs its own task
and memory-use controls.

Architecture references discussed: [GRU](https://aclanthology.org/D14-1179/) and
[Neural Turing Machines](https://arxiv.org/abs/1410.5401). These motivate candidate
mechanisms; they do not establish performance in this experiment.
