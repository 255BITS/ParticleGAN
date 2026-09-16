# Autonomous particle/memory GAN: completed optimization round

**Current direction:** the subsequent DDGAN/FiLM/size queue was stopped at the
user's request to prioritize the real-data -> D memory -> G training handoff.
See [round 4 outcome](round4/README.md) and [next experiment](../../memory-path/NEXT.md).
The results below describe the completed earlier autonomous formulation.

**Geometry improved substantially; the toy is not solved yet.** The best geometry
model passes 85.2% of full cold-start circles but only generates counterclockwise
passes. The best balanced model passes 52.5% across all 512 learned particles.

Completed **12 scouts and 6 continuations**, totaling **46,000 new updates** and
3.16 summed GPU job-hours (both cards used). No training job failed. No seed
sweeps, image-based selection, analytic circle generator, burn-in exclusion, or
geometry target loss. All jobs have finished; nothing is queued or running.

## Full-table deployment validation

These are CPU evaluations over **all 512 learned particles**, each starting from
zero memory, with its particle fixed throughout. No D scoring head is constructed.
They enumerate the learned table, not a held-out training split.

| Model | Updates | Full 256 circles | Full 1,024 circles | Passing CW / CCW at 256 | Radial RMSE |
|---|---:|---:|---:|---:|---:|
| Learned GRU memory + private G recurrence | 5,000 | **85.2%** | **84.0%** | **0 / 436** | .070 |
| Learned GRU memory, feedforward G | 10,000 | **52.5%** | **50.4%** | **134 / 135** | .122 |
| Learned GRU memory, feedforward G | 5,000 | 35.2% | 33.6% | 101 / 79 | .150 |
| Real noisy reference | — | 99.6% | — | 259 / 251 | .033 |

Both leading models have **zero passing circles when memory is zeroed or shuffled**.
The private-recurrence model stops 95.3% of trajectories when memory is zeroed.
Thus its private state has not made the shared-memory channel irrelevant.

- [All 18 completed training checkpoints](completed/leaderboard.md), evaluated on
  the common first 128 particles, with [configs, metrics and source hashes](completed/results.json).
- [Full-table validation](validation/leaderboard.md) and [numerical details](validation/results.json).
- [Chronological decisions and explanations](decisions.md).
- [Job/update/compute manifest](manifest.json).
- [Original first wave](round1/leaderboard.md) and [historical 64-step baselines](historical/leaderboard.md).

## What worked, and what did not

**Keep learning the GRU writer.** At 2k, learned GRU beats its random frozen
control 14.1% to 6.3%. In a stronger continuation comparison, freezing the learned
writer at 5k reaches 39.8% at 10k, versus 53.1% when writer learning continues.
The frozen branch's writer was verified bitwise unchanged from its 5k checkpoint.
These comparisons favor learned GRU memory in this setup, not every memory family.

**Longer training helped the balanced model.** Its 128-particle full-pass curve
is 14.1% at 2k, 34.4% at 5k, and 53.1% at 10k. The full-table result agrees.
Radius, speed and center coverage are broad: standard deviations .226 and .084
for radius and speed, and center spread .618.

**Private G recurrence greatly improves geometry, but collapses direction.**
It rises from 55.5% at 2k to 85.2% at 5k. However, all passing trajectories are CCW.
A matched recurrent G without memory reading reaches 52.3% at 2k versus 55.5%
with memory. This does **not establish a shared-memory advantage**; dependence
on M and superiority over a separately trained control are different claims.
There is no matched 5k no-reading result in this round.

**Multiscale scoring, difference features and FiLM help geometry early**, but
all their best 2k full passing circles use one direction. The trace/difference
winner regressed from 38.3% to 28.9% at 5k while beginning to recover the other
mode. More training is not automatically better.

**The geometry-head scout's early advantage did not persist clearly.** At 2k it
beats ordinary GRU 27.3% to 14.1%, but at 5k its radial error is similar and its
direction coverage worse. The SiLU scout has zero full and late-only passes and
was not extended. Delay memory did not outperform the multiscale trace model.

## Recommendation for the next round

Retain the 10k ordinary GRU as the balanced reference and the 5k recurrent-G model
as the geometry reference. The highest priority is **recovering both directions
in the recurrent-G model**, not increasing its one-direction circle rate alone.
Two config-only combinations are prepared, **not run**:

1. `gru_private_film.json`: give the recurrent G a direct particle-conditioned
   modulation of its output network, testing more expressive particle-dependent motion.
2. `gru_private_geometry.json`: give its critic the already tested oriented
   pairwise geometry features, testing more direct feedback about trajectory shape
   and direction. This remains a learned adversarial score, not a prescribed circle loss.

For the balanced reference, test a longer-horizon/curriculum continuation with a
scorer whose capacity does not grow with horizon. It passes 70.5% over 64 points
but 52.5% over 256; radial error causes 243/512 failures. Preserve initialization,
optimizer history where compatible, and report compute changes. Do not conceal
startup failures with generated burn-in or late-only refitting.

The working success target remains at least 95% full-256 and 90% full-1,024
circles, negligible stopping, both directions and reasonable distribution coverage.
The existing heuristic circle thresholds were unchanged throughout the round.

## Formulation and reproducibility

One independent trajectory memory per batch row; zero initial memory and zero
private G state where applicable; one fixed particle per trajectory. Runtime is
G plus writer plus particle prior. D alone trains the shared writer using real
and detached generated sequence scoring. G differentiates through state across
time while writer parameters are frozen during its update. Offline expert paths
train D; none initialize or guide generated runtime paths.

The public API supplies recipe, prior, GAN loss, optimizers, regularizer and LR
schedule. Exact B-cap defaults remain unchanged: L2 cap 1, coefficient 1, every
update. No gradient clipping, parameter-gradient norm logging, or EMA. Observation
noise remains .03; 64-step training, batch 128, 512 learned particles. Scouts stop
at 2k on a predeclared 10k LR schedule; promoted runs restore optimizer and random
states. The learned-writer freeze is explicitly a different training rule.

Config files live in `experiments/configs/memory_scout/`. Each run saves its input,
resolved config, source snapshots/hashes, logs, resumable checkpoint, numerical
summary and raw paths. Some prepared configs were not executed; only entries in
[manifest.json](manifest.json) are training results. Changes between source
snapshots added disabled options; all six first-wave configs reproduced their
launch initialization, rollout and critic outputs exactly on CPU in a compatibility check.

**Validation:** 19 focused tests passed, including ownership, reset/independent
states, exact checkpoint continuation, FiLM initialization and learned-writer
freezing. CUDA exact B-cap backward passed for GRU, delay and oriented geometry
heads. The recurrent G/no-reading control has identical initial G, D, writer and
prior tensors. CPU deployment reproduces the 5k CUDA paths within numerical
precision (max coordinate difference .000917 by step 1,024).

```bash
# Training a config: choose a fresh output directory.
.venv/bin/python -u experiments/memory_scout.py \
  --config experiments/configs/memory_scout/gru_private_film.json \
  --out runs/memory_path/NEXT_FRESH_RUN --device cuda:0

# Current best checkpoints (training is complete):
# runs/memory_path/scout_long/gru_flat_10k/model.pt
# runs/memory_path/scout_recurrent_long/gru_private_5k/model.pt

# Completed logs remain tail-able; they end in complete/queue_complete.
tail -F runs/memory_path/scout_long/queue.log \
        runs/memory_path/scout_recurrent_long/queue.log
```
