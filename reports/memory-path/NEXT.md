# Continuation: solve autonomous circle quality and coverage

Branch: `feat/sequential-memory-path`. Code, reports, and this handoff are included
in the preparation-for-compaction commit. No experiment process is running.
Preserve unrelated `.claude/`, `results/motion/`, and `sparse-ucd.log` files.

## Current objective and working choice

The user wants to improve the autonomous results next, after compaction.
**Use the learned writer as the primary development model; retain frozen writer
as a comparison.** This is our recommended working choice, not an established
scientific winner or a user decision to discard either mechanism. Frozen leads
on circle passes, but its successful late circles at horizon 64 all rotate
counterclockwise. Learned stops less often and retains both directions. We have
not demonstrated an overall advantage for learning the writer.

Recommended next experiment: add short-window adversarial scores and an explicit
cold-prefix score alongside the full 64-step trajectory score. Keep 64-step
rollouts, fixed particles, zero memory, and 256-step cold evaluation. The
hypothesis is clearer feedback for local motion and startup alongside whole-orbit
consistency. This is not implemented yet. A short-to-long curriculum is another
candidate if optimization remains difficult. Preserve direction coverage and
initial-position spread in the leaderboard; improving circle geometry alone is
not enough. Do not claim success from late-only refitted circles.

## User constraints

- No realtime expert and no real initialization prefix at runtime.
- One trajectory per batch entry, independent memory, shared network weights.
- One particle z per trajectory, held fixed throughout. Memory belongs to the
  rollout, not permanently to a particle-table row.
- Runtime needs G, writer, and prior only. D's scoring head is training-only.
- Use the public ParticleGAN API; **no gradient clipping**. Keep default B-cap:
  exact autograd, L2 cap 1, coefficient 1, every update (`reg_every=1`).
- Parameter-gradient norm logs were experiment-specific and removed along with
  clipping; `penalty` still logs the B-cap input-gradient penalty.
- No seed sweeps. Make logs easy to tail. Explain completed experiments with
  a leaderboard and recommendations. Both RTX A6000 cards are authorized.
- No subagents/delegation requested. Do not touch unrelated files.

## Completed autonomous studies

Read [latest full report](../autonomous-memory/horizon64/README.md) first.
All rows: 2,000 updates, batch 128, 128 evaluation particles, 256-point generation.

| Writer / setup | Full circles | Late-only circles | Stopped late | Full radial RMSE |
|---|---:|---:|---:|---:|
| Learned / clipped 16 | 0% | 17.2% | 20.3% | 0.466 |
| Frozen / clipped 16 | 0% | 14.1% | 8.6% | 0.410 |
| Learned / unclipped 16 | 0.8% | 23.4% | 16.4% | 0.396 |
| Frozen / unclipped 16 | 0% | 11.7% | 12.5% | 0.406 |
| Learned / unclipped 64 | 6.3% | 26.6% | 3.9% | 0.295 |
| Frozen / unclipped 64 | 7.8% | 34.4% | 8.6% | 0.298 |
| Real noisy reference | 99.2% | 99.2% | 0% | 0.032 |

No-memory fixed-z control is structurally static: 100% stopped, zero circle
passes. It was run only in the first study and smoke checks.

Full circles fit the first 32 points and score all 256 against that fixed fit.
Late-only refits points 129–256. Stopped late is mean displacement <0.01 over the
last 64 transitions. Geometry metrics are heuristic, not distribution distances.

At horizon 64, late passing circles are learned 24 CCW/10 CW versus frozen
44 CCW/0 CW; real 62 CCW/65 CW. Initial-position spread narrows to learned
0.622/frozen 0.675 (real 1.180; 16-step baseline 1.094/1.147). Distorted loops,
spirals, startup transients, and partial mode collapse remain.

The horizon comparison changes the full configuration: D grows from 86,761 to
295,657 parameters, points per update quadruple, and initial prior values differ
because the larger D consumes more initialization RNG. G/writer initialization
is unchanged. Training-data RNG consumption also changes with sequence length.
Do not describe this as a pure horizon ablation at equal compute or identical
initial particle values. Both writer mechanisms at a given horizon match their
initialization/data schedules. No seed experiments were performed.

## Outputs and logs

- `runs/memory_path/autonomous_2k/`: original clipped 16-step study.
- `runs/memory_path/autonomous_unclipped_2k/`: unclipped 16-step baseline.
- `runs/memory_path/autonomous_h64_2k/shared_run/`: learned writer on GPU0,
  400.6 seconds training/evaluation.
- `runs/memory_path/autonomous_h64_2k/frozen_writer_run/`: frozen writer on GPU1,
  324.4 seconds, run concurrently with learned.
- Each run has flushed `experiment.log`, per-variant `metrics.jsonl`, config,
  summary, source snapshots, inference checkpoint, and full evaluation arrays.
- Durable reports: `reports/autonomous-memory/`, its `unclipped/` and `horizon64/`
  subdirectories. The latter includes direction counts and learning curves.
- Raw runs are gitignored. Reports include results, source hashes, and a patch
  identifying trainer changes that were uncommitted when the runs started.

Example two-card log tail, after launching future runs in fresh directories:

```bash
tail -F runs/memory_path/autonomous_h64_2k/{shared_run,frozen_writer_run}/experiment.log
```

These completed logs end in `suite_complete`; nothing currently needs monitoring.
Use `.venv/bin/python`; Torch 2.14.0+cu130. Training checkpoints are inference-only
(no optimizer/RNG state for exact resume). Do not overwrite completed run dirs.

## Implementation

`experiments/autonomous_memory.py` reuses `FastMemory`, `Generator`, and `circles`
from `experiments/memory_path.py`. Public API: `get_recipe`, factories for prior,
loss, gradient penalty, prior regularizer and optimizers, plus LR scheduling.
GAN recipe: relativistic logistic, lr 0.0006, D multiplier 1.5, prior multiplier
10, Adam betas (0,.999), particle spread weight 1, decay after 60%, floor .05.
No EMA. Prior has 512 learned four-dimensional particles.

Runtime recurrence, with independent zero memory Bx8x4 and fixed z:

```python
x = G(z, M.flatten(1))
M = writer.write(M, x)
```

Writer maps x through 2->32->8 tanh features; four temporal traces decay at
[0, .5, .8, .95]. It is a feature-trace memory, not general content-addressed
fast weights. G is a width-64 MLP. Sequence D is a width-128 MLP over the flattened
sequence of (point, memory BEFORE that point's write) features.

D trains its writer from both real and detached fake sequence scoring. During
G updates writer weights are frozen, but state stays differentiable through
all generated steps. No teacher forcing, timestamps, or reconstruction loss.
`frozen_writer` freezes the random writer throughout. New configs explicitly
record `gradient_clipping: null`. Training circles retain observation noise .03;
centers uniform [-.75,.75]^2, radii [.6,1.4], speed magnitude [.12,.40], both signs.

## Validation and analysis

- Original implementation/API validation: 38 tests passed.
- Clipping removal: five autonomous tests and all three five-update CUDA smoke
  runs passed. No training code changes occurred in the substantive followups.
- Reloaded unclipped learned G/writer/prior reproduced all 128 saved 256-step
  trajectories without constructing D's scorer or supplying real inputs.
- Both horizon-64 runs completed with finite outputs; real evaluation arrays
  match across cards and against the 16-step baseline.
- Analysis accepts multiple run directories and plots the actual train horizon.
  Reanalysis reproduces the earlier baseline JSON exactly. Plots inspected.

```bash
.venv/bin/python reports/autonomous-memory/analyze.py \
  runs/memory_path/autonomous_h64_2k/shared_run \
  runs/memory_path/autonomous_h64_2k/frozen_writer_run \
  --out reports/autonomous-memory/horizon64
.venv/bin/python reports/autonomous-memory/horizon64/learning_curves.py
```

Historical observation-conditioned study remains at `reports/memory-path/README.md`:
shared next-point RMSE .098, recent-point buffer .088, but real observations
arrived each step. Do not conflate it with autonomous success.
