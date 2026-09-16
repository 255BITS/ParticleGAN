# Completed: local dynamics, translation, and repair

Sixteen 2k scouts plus one exact 2k -> 5k continuation completed successfully.
Both queues are sealed and empty; no jobs or diagnostics remain running.
Selection used numerical metrics only. No generated training trajectories,
private G recurrent state, seed sweeps, gradient clipping, or EMA were introduced.
The default public API exact B-cap is unchanged.

**Every new checkpoint: 0/128 full cold circles at256/1024, and 0/128 original-orbit
continuations after prefixes8/32 at both horizons. Every checkpoint also has 0%
late stopping and 0% late-only circles at1024.** The clock's sustained-motion
milestone remains intact; the toy is not solved.

Full [scout leaderboard](leaderboard.md), [longer-run comparison](long/leaderboard.md),
[mechanism table](mechanism-table.md), [plan](plan.md), and config directory
`experiments/configs/memory_handoff/dynamics_round8` preserve the comparisons.
Historical clock models are explicitly comparison-only, not fresh scouts.

## What changed and what was tested

All features default off. Existing configs and checkpoint parameter names remain
compatible. Strict loading of an actual round7 clock checkpoint passed; old
checkpoints without the new inactive RNG streams resume. The tests also verify
exact interrupted/uninterrupted resume for combined slow memory, repair, and
local stability, and verify which parameters each auxiliary can update.

- **Slow/fast D memory:** the same GRU parameters and fully coupled state, with
  selected coordinates receiving a fraction of each proposed write. Four scouts
  varied slow-coordinate count and rate. D alone owns and trains this memory.
- **G translator:** an identity-initialized residual MLP, owned by G, applied to
  the memory read. Bottlenecks16/64 tested capacity without extra supervision.
  It has no persistent state and never writes its translated code back to D.
- **G repair:** four scouts tested raw versus translated detached clean targets,
  strengths10/100, and Gaussian noise.05/.15. The loss updates only the adapter.
  The repair input is memory only; sample-conditioned repair is not implemented.
- **Feedback stability:** G-only or D+G penalties on the gain between two nearby
  one-step `D.write(M,G(z,M,time))` branches. Prefix anchors and particles are
  detached, the opposite module's parameters frozen, and no generated point is
  followed by another generated prediction in this auxiliary. Three isolated
  scouts and three repair/slow-memory combinations tested the interaction.

Scouts used2k updates, unchanged10k schedule, M32, batch128x4 sampled target
positions, G64, D128, and six clock bands. All16 trainers/writer implementations
have identical source hashes. Queue wall time803.94s; aggregate training1487.30s.
The5k extension added3000 updates in120.47s. The main queue used both A6000s;
the extension and completed-model adapter interventions used separate cards.

Metadata clarification: the archived round8 `fake_writes_in_training` field
counts feedback replacements only. Stability branches do perform generated
writes, as recorded in `local_dynamics` and the G-call counts. After completion,
this reporting field was corrected and explicit D/G-phase write counts added;
this changes metadata only, not training, checkpoints, or evaluated results.

## Useful findings

**Translation can improve immediate prediction without improving feedback.**
At prefix32, `translate16` next-point MSE is0.00541 versus0.00664 for the clock
baseline. Bypassing its adapter raises MSE to0.11530. Yet its1024-step radial
error is2.545, and bypassing the adapter reduces that error to0.856 while still
passing zero original orbits. Readout accuracy and autonomous fidelity diverge.
The larger translator worsens both local prediction and long-run radial error.

**Repair learns some denoising, but that is not sufficient for useful recovery.**
For `repair_raw100`, with coordinate-RMS.05 corruption, memory reconstruction MSE
is0.00177 versus0.00250 before repair (about29% reduction). Its real-context
prediction MSE0.00474 is the best prefix32 local prediction among the2k scouts.
However, noisy-input target MSE is0.01015 with repair and0.01018 when bypassed.
For `repair_raw10_n15`, it is0.00875 with repair and0.00859 bypassed. Removing
memory-coordinate noise does not necessarily remove the errors affecting G.
Across-episode translated-state standard deviations remain close to raw-state
values for these models; these probes do not indicate wholesale state collapse.

There is a modest autonomous benefit within the stronger-noise checkpoint:
`repair_raw10_n15` prefix32 radial error is1.834, worsening to1.981 on adapter
bypass. Prefix8 similarly worsens1.837->1.978. These interventions establish a
checkpoint-specific effect, not a solved circle or a general benefit from any
translator. All four models tested with adapter bypass still have zero full cold
and original-orbit warm passes.

**The local stability penalty changes sensitivity without solving continuation.**
At prefix32, the baseline's sampled perturbation gain is0.772 RMS, whereas its
mean largest Jacobian singular value is2.415. Thus random perturbations can shrink
on average while selected directions amplify. `stable_dg10_cap09` reduces those
numbers to0.700 and1.894, but has radial error2.344 and zero circle passes.
Slow memory similarly offers no primary-metric winner; `slow16_r10` has the best
new-scout prefix32 radial error1.679, but early continuation is worse than the
matched clock baseline. The previous shared-clock control has radial error1.619.

These are local Euclidean sensitivity measurements, not certificates of long-run
instability. Sensitive directions may encode legitimate changes. Globally
contracting every direction could erase useful distinctions.

**Feedback can quickly degrade predictive usefulness.** In the clock baseline,
prefix32 next-point MSE is0.00664. After one generated write, prediction of the
following point has MSE0.02717; after a clean real write,0.00388. This paired
evaluation uses the same context, particle, and advancing time. It is evidence
of short-horizon feedback error, not proof that memory alone causes it.

## Extension decision and outcome

All primary pass rates tied at zero. Only `repair_raw10_n15` satisfied the
predeclared diagnostic extension criteria: at least25% less long radial error
at BOTH prefixes than `clock_fourier6`, no worse direction agreement or stopping,
and improved first32 position error. Improvements in early position error were
small: prefix8 1.638->1.581 and prefix32 1.466->1.431. This was explicitly a
diagnostic continuation, not promotion of a solved or primary-metric winner.

At5k, prefix32 radial error worsens1.834->2.456; first32 position error
1.431->1.713; local next-point MSE0.00561->0.00787. Prefix8 radial error similarly
worsens1.837->2.452. All full cold/warm passes remain zero and motion persists.
Its mean worst-direction gain improves further2.152->1.976 despite worse
continuation, reinforcing that this sensitivity proxy is insufficient.
No further extensions were selected.

## Recommendation

Keep the optional translator/repair machinery for targeted experiments. Next,
test repair against errors produced by G's own samples and judge it by preserving
predictive usefulness. The user's proposal to provide the generated sample to
repair is a concrete direction: it supplies information about the decision whose
consequences need correcting. Retain a translator-only control and distinguish
training effects from same-checkpoint adapter bypass effects.

Previous single-write feedback experiments already failed to solve the task.
Simply adding that feedback again is not a new hypothesis. A new experiment
should isolate sample-conditioned read repair or supervision in prediction
space, explicitly state its bounded local G-call cost, and retain D-only writer
ownership. Matching clean continuation after a generated replacement remains a
recovery target; it is not necessarily the natural future of every changed point.

Do not automatically launch another sweep. Discuss the repair objective and
sample conditioning before selecting configs. No results here establish
generalization to radii outside the training range; that remains a separate
evaluation once ordinary continuation improves. Evaluation uses128 learned
particles from the512-particle prior, not held-out particles.

## Artifacts and validation

- `results.json`, `long/results.json`: full metrics and resolved configs.
- `control_probes.json`, `scout_probes.json`: local memory/clock interventions,
  repair sensitivity, feedback write comparisons, and Jacobian singular values.
- `adapter_interventions.json`, `long/probes.json`: adapter-bypassed autonomous
  rollouts and local probes. No images were used to select runs.
- `experiments/diagnose_memory_dynamics.py`: reproducible completed-only probes;
  verifies normal first predictions against saved trajectories before probing.
- 74 focused tests passed in7.77s; full-batch combined GPU smoke and CPU/GPU
  diagnostic smokes passed. No failed or pending jobs remain.

Central tail remains `tail -F runs/memory_path/core_round1/train.log`.
User subsequently requested committing on `feat/sequential-memory-path` for compact.
The code, configs, reports, and video are included in that commit.

## Saved rollout video

[Watch slow16_r10](slow16_r10_rollout.mp4) (about43 seconds,24fps). This model has
the lowest long radial error at both prefix lengths among the new scouts, with
zero full-circle passes. Top: zero-memory cold starts. Bottom:32 real points, then
autonomous generation with an offline reference circle and moving reference dot.
Bright trails show the last32 steps; faint lines show complete generated history.
Axes stay fixed. Examples are the first CW and first CCW reference episodes, not
selected for good or bad generated behavior. The manifest records particle IDs
and the source-array hash. This video explains saved results and does not inform
selection or alter metrics.

Reproduce with:
```bash
.venv/bin/python experiments/render_memory_rollout.py \
  --run runs/memory_path/dynamics_round8/runs/slow16_r10 \
  --out reports/memory-handoff/dynamics_round8/slow16_r10_rollout.mp4
```
