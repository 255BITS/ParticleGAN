# Core formulation comparison

This round compares real-memory handoff with explicit generated feedback. All seven
new scouts use the same GRU32 D writer and feedforward G, fixed particle per
trajectory, batch 128, 2,000 updates on the common 10k schedule, and the public API's
exact B-cap defaults. No clipping, EMA, analytic geometry losses, or seed sweeps.

## Training rules

- **H (handoff):** choose a target position independently per real episode. D writes
  only real points strictly before that position into M. G reads M. D scores the
  real target and fake candidate with the same M; candidate scoring never writes.
  Target position is zero with probability 1/8, otherwise uniform over 1..63.
- **C (cold):** generate all 64 points from zero M, writing every emitted point back
  through D's writer. D's existing trajectory critic scores real and fake paths.
- **W (prefix mixture):** choose prefix length uniformly from 0, 8, 32 each update.
  D writes that real prefix, G generates the remaining points with generated writes,
  and D scores the full 64-point real and completed paths. Real prefix is fixed for
  both candidates. Length zero explicitly includes cold starts in this branch.

D alone trains the shared writer. G updates freeze writer parameters while retaining
full gradients through generated feedback. Losses and their B-cap terms are averaged
using the weights below; prior regularization is applied once per G update.
H uses a conditional point head; C and W share the existing path head. Both heads
exist with identical initialization in every new scout, even when inactive.
B-cap differentiates the candidate point for H, full path for C, and only the suffix
for W. Thus objective/penalty domains and active discriminator capacity differ;
these are part of the formulations being tested, not isolated architecture claims.

| Config | H | C | W |
|---|---:|---:|---:|
| handoff_only | 1 | 0 | 0 |
| handoff_cold | 1 | 1 | 0 |
| handoff_warm | 1 | 0 | 1 |
| handoff_both | 1 | 1 | 1 |
| warm_only | 0 | 0 | 1 |
| handoff_cold_light | .25 | 1 | 0 |
| handoff_cold_heavy | 4 | 1 | 0 |

The completed autonomous GRU32 controls at 2k and 10k receive the new evaluation
without retraining. The 10k model is an unequal-budget reference. The cold-only
new trainer matches the old trainer's two-update CPU result bit for bit (including
writer, generator, and particles). The extra conditional head is unused in that test.

## Evaluation and decisions

Each run uses the same 128 particles and same real episodes. This is a controlled
scout panel, not a held-out learned-particle test. Cold-start metrics measure all
256/1024 points, including startup. Prefix tests consume 8 or 32 noisy real points,
then disconnect X entirely for 1024 generated points. No target or future observation
enters G or M after the cutoff; clean reference geometry is used offline only.

Report generated-circle quality separately from fidelity to the original orbit.
The new reference-orbit pass requires relative radial RMSE < .1, matching reference
direction on >95% of transitions, signed-speed error < .03 radians/step, and first
prediction error < .2 reference radii. The first transition out of the expert prefix
is included. Also report continuous errors, early/late position error, memory norms
and saturation, zero/shuffle interventions, stopping, CW/CCW passes, and cold diversity.
The new pass threshold is a diagnostic, not a prevalidated statistical test.

Prefer formulations improving both cold stability and prefix fidelity. If these
trade off, retain one candidate on each axis. Do not promote one-direction collapse
solely on circle rate. Review all completed scouts before scheduling longer runs.
DDGAN, FiLM, private G recurrence, and memory-size changes are deferred.

## Live pipeline

```bash
tail -F runs/memory_path/core_round1/train.log
```

Both GPUs drain one durable shared queue. The same log receives every labelled
training line and lifecycle event. `notifications.log` contains only completed,
failed, reporting-failure, and final queue events. No training-result polling is
needed: completion triggers the numeric report before notification. A report failure
is surfaced explicitly without relabelling a successful training run as failed.

Completed-only leaderboard: [round1/leaderboard.md](round1/leaderboard.md).
Detailed numeric results: [round1/results.json](round1/results.json).

Implementation: `experiments/memory_core_scout.py`. Configs:
`experiments/configs/memory_core/`. Reporting:
`experiments/analyze_memory_core.py`. Queue: `experiments/memory_dispatch.py`.
