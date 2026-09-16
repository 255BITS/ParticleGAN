# Continuation: autonomous generation is now the target

Read this first after compaction. Branch: `feat/sequential-memory-path`.
Experiment code, tests, reports and these notes are committed on this branch.
Preserve the user's unrelated `.claude/`,
`results/motion/`, and `sparse-ucd.log` files.

## User decisions

- Focus entirely on dropping the realtime expert from runtime.
- No real prefix at initialization either: start with zero memory and a particle.
- One trajectory per batch entry, with independent memory per entry.
- Sample a particle once and keep it fixed throughout its trajectory initially.
  Memory belongs to the rollout, not permanently to a row of the particle table.
- Shared network weights across trajectories. Feed each generated point back to
  that trajectory's memory. D's scoring head is not required at runtime.
- Use the new public ParticleGAN API and GPUs for the next experiments.
- No seed sweeps. Keep logs easy to tail. Summarize results, leaderboard,
  explanations, and next recommendations after experiments finish.
- The latest request was to make these changes **before compaction**. Code and
  GPU smoke validation are complete; the substantive autonomous study has NOT
  been launched yet.

## Implemented

New entry point: `experiments/autonomous_memory.py` (default `--device cuda:0`).
It reuses `FastMemory`, `Generator`, `circles`, and `mlp` from the earlier
`experiments/memory_path.py`; that earlier trainer remains the historical
observation-conditioned experiment, not the current objective.

Runtime `rollout(generator, writer, z, steps)` has no real-data argument and
needs no sequence critic. Its exact loop is:

```python
M = zeros(batch, 8, 4)
for t in range(steps):
    x = G(z, M.flatten(1))
    M = writer.write(M, x)
```

It returns `[batch, time, 2]` and final memory. `z` is fixed across time. All
states reset at the beginning of a call. Calls do not retain hidden global state.

Training is now a sequence GAN: a temporal MLP reads the ordered 16-step sequence
of `(candidate point, memory before its write)` features. Real and generated
sequences use identical scoring code and separate zero-initialized memories.
Generated trajectories always start cold; there is no teacher forcing, real
prefix, timestamp input, or supervised reconstruction loss.

Important differences from the old experiment:

- Writes now include generated points. During D scoring, the writer receives
  gradients from **both** real sequences and detached generated sequences. This
  is deliberate, symmetric sequence scoring; the old real-only write protocol
  is not the new runtime protocol.
- During the G update, writer **parameters** are frozen, but gradients propagate
  through memory state and earlier generated points across the full rollout.
  This is full backpropagation through time, not detached-state training. G
  cannot update the writer's weights, but it learns how its outputs affect M.
- During D's update, generated trajectories are detached. No D gradient runs
  through G's generation process.

Variants ready to run:

| Variant | Meaning |
|---|---|
| `shared` | D trains writer; G reads the same writer's state |
| `frozen_writer` | Random writer stays frozen; G and sequence scoring head train |
| `no_memory` | Both networks' memory features are zero; fixed-z G is necessarily static |

The last variant is a structural negative control, not a competitive recurrent
baseline. All variants use the same initialization and data/latent schedules.
Recipe GAN loss, gradient penalty, prior regularizer, optimizers and LR schedule
come from the public API. Global gradient norms are clipped to 10 and the
pre-clipping norms are logged. No EMA. Offline real training trajectories retain
the earlier circle distribution and observation noise std 0.03 per coordinate.

## Validation completed

- 38 tests passed: `tests/test_autonomous_memory.py`, `tests/test_memory_path.py`,
  `tests/test_api_primitives.py`, and `tests/test_api_integration.py`.
- New tests cover fixed latent identity, independent batch states, cold starts,
  repeatable resets, G backpropagation across time without writer parameter
  gradients, D writer gradients, preservation of frozen flags, and diagnostics
  rejecting static/spiraling paths while accepting real clean circles.
- All three variants completed five CUDA updates and 256-step autonomous
  evaluation on GPU 0 (RTX A6000).
- Reloaded the CUDA checkpoint and reproduced the saved 256-step trajectories
  using only G, the writer and particle prior, without constructing D's scorer.
- Smoke directory: `runs/memory_path/autonomous_gpu_smoke/`. Its README includes
  a diagnostic leaderboard and trajectory plot. All variants are stationary
  after five steps; this is execution validation, not a learning result.
- No background experiment remains running.

## First substantive study to run after compaction

```bash
.venv/bin/python -u experiments/autonomous_memory.py \
  --out runs/memory_path/autonomous_2k \
  --device cuda:0 --steps 2000 --batch-size 128 \
  --train-length 16 --eval-steps 256 --eval-batch 128 --log-every 100
```

```bash
tail -f runs/memory_path/autonomous_2k/experiment.log
```

The output directory must be fresh. Three variants run sequentially by default;
`--variants shared frozen_writer` selects a subset. Do not launch seed variants.
GPU 1 was also available; use separate output directories if scheduling manual
parallel processes. No delegation is authorized by the local instructions.

Each run saves config, flushed metrics, inference checkpoint, summary and full
evaluation trajectories. The suite saves both source files and provenance, plus
a trajectory figure and diagnostic leaderboard. Checkpoints save separate
`generator`, `writer`, and `prior` states so runtime does not require the scoring
head; full critic state is also included. They do not save optimizer/RNG states
for exact resume.

## What to measure and decide

There is no paired expert future for a generated trajectory, so the old
next-point RMSE leaderboard is inappropriate. The new metrics fit a circle to
the first 32 points, hold that reference fixed, and measure long-horizon radial
error, late radius drift, angular speed, direction consistency, stationary
fraction, initial-position spread and radius diversity. Real clean/noisy
reference trajectories are scored only after generation, never fed to G.

`circle_like_fraction` is a heuristic conjunction: fitted radius in [0.5, 1.6],
relative radial RMSE < 0.1, late radius drift < 0.2, mean absolute angular speed
in [0.08, 0.45], direction consistency > 0.95, and nondegenerate early geometry.
Other geometry metrics average over valid fits only; always inspect
`valid_fit_fraction` and the stationary fraction to avoid selection bias.
These diagnostics are not a full measure of trajectory-distribution matching.

First inspect whether any model sustains motion rather than converging to a
point. Then compare learned vs frozen writers and the gap between 16-step
training and 256-step generation. Change rollout horizon, writer mechanism, or
training objective as separate experiments if needed. A length curriculum and
clean (noise-free) training circles are reasonable follow-ups, not implemented
options yet. Defer fresh per-step particles/noise and figure-eights until the
fixed-particle autonomous circle works. Report both failures and successes.

## Historical result (do not confuse with autonomous generation)

`reports/memory-path/README.md` documents the earlier four 7k-update CPU runs.
Shared memory achieved 0.098 next-point RMSE with real observations arriving
every step; raw recent-point buffer achieved 0.088, linear extrapolation 0.128.
Shuffling M ruined prediction. This established useful memory reading, not
self-sustaining generation. The old checkpoints and logs remain in
`runs/memory_path/circle_7k/`.
