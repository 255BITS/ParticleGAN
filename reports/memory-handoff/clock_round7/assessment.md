# Fourier clock round: motion sustained, circle preservation unsolved

The user accepts sustained motion as a successful milestone and wants to build
on it. Full-circle preservation is the next unsolved target. After committing
and compacting, return to first-principles discussion and the user's ideas before
selecting another experiment; no automatic diagnostic or sweep is planned.

Completed eight 2,000-update scouts on both GPUs through the shared queue. All eight jobs succeeded; no pending/running/failed jobs remain. No models promoted to longer training.
Queue wall time: 360.7s. Aggregate training time: 667.7s.

All eight models have **0/128 full cold-circle passes at both 256 and 1,024 steps**, and **0/128 original-orbit passes after both 8- and 32-point prefixes at both horizons**. Primary leaderboard is a tie at failure; secondary metrics below do not establish a winner.

## Completed scouts

| Configuration | Cold 256 / 1024 | Warm prefixes8/32, both horizons | Late stopped at1024 | Prefix32 radial RMSE at1024 | Prefix32 startup error |
|---|---:|---:|---:|---:|---:|
| clock_static6 | 0% / 0% | 0% | 57.0% | 1.561 | 0.134 |
| clock_fourier6 | 0% / 0% | 0% | 0.0% | 2.597 | 0.112 |
| clock_fourier6_shared | 0% / 0% | 0% | 0.0% | 1.619 | 0.138 |
| clock_fourier6_offset | 0% / 0% | 0% | 0.0% | 2.633 | 0.103 |
| clock_fourier6_offset_shared | 0% / 0% | 0% | 0.0% | 1.720 | 0.102 |
| clock_fourier3_fast | 0% / 0% | 0% | 0.0% | 2.331 | 0.145 |
| clock_fourier6_recent | 0% / 0% | 0% | 0.0% | 1.811 | 0.088 |
| clock_fourier6_offset_recent | 0% / 0% | 0% | 0.0% | 1163.717 | 0.088 |

Radial and startup errors are relative to the reference radius. Full rates include startup; no burn-in exclusion. All seven advancing-clock scouts have 0% late stopping, versus 57.0% for the architecture-matched constant-clock control. None of the advancing-clock scouts has passing late-only circles at1024 either. The random-origin recent/residual model diverges severely despite good first-point accuracy.

Historical dense4: 68.8% stopped, prefix32 radial1.922. Historical recent4_delta: 14.8% stopped, radial1.085, 60.2% late-only circles, but full success0 and previously collapsed under longer training. These are comparison controls, not new runs.

## Does G use both clock and memory?

Evaluation-only interventions on four completed models preserve the first32 generated points, then freeze/slow the clock or zero/shuffle/freeze the memory supplied to G. D writer continues consuming generated outputs. Unchanged trajectories were checked against saved arrays, and all interventions were checked to preserve startup exactly. Real-prefix panels come from saved arrays to avoid CPU/CUDA RNG differences.

| Model | Normal late stopped | Frozen-clock late stopped | Clean-target MSE, real prefix32 | Clock reset to0 MSE | Shuffled-memory MSE |
|---|---:|---:|---:|---:|---:|
| clock_static6 | 57.0% | 57.0% | 0.00977 | 0.00977 | 1.48443 |
| clock_fourier6 | 0.0% | 95.3% | 0.00664 | 0.03601 | 1.45982 |
| clock_fourier6_offset_shared | 0.0% | 57.0% | 0.00637 | 0.00928 | 1.44228 |
| clock_fourier6_recent | 0.0% | 48.4% | 0.00443 | 0.07415 | 1.44598 |

- Main clock model: freezing time raises late stopping0% ->95.3%. Zeroing or freezing memory while time advances leaves late stopping0%, showing that continued activity can be clock-driven.
- Its real-context next-point MSE rises0.00664 ->1.45982 with shuffled memory (~220x), and ->0.73404 with zero memory. G still strongly depends on M for conditional prediction.
- Resetting time to0 while holding a real-prefix32 memory fixed increases MSE to0.03601 (~5.4x), so clock input is also used locally. Moving the clock back just one step slightly improves this model's error, so this does not establish precise or correct phase tracking.
- The constant-clock control is exactly unchanged by clock freeze/slow/reset, validating the intervention plumbing.
- Freezing the clock can improve radial error and produce some late-only circles in variants, while increasing stopping. In recent-memory prefix8 continuations, clock freeze yields2/128 generated paths passing the generic1024 circle fit, but0/128 preserve the original orbit; these are warm evaluation interventions, not cold successes. This is not evidence that the advancing clock improves orbit preservation. No intervention produces full cold-circle success.

**Conclusion:** both inputs influence the tested models. Memory supports accurate prediction on real histories; the clock sustains activity. Beneficial joint use for stable autonomous circles is not established because the requested full trajectory metrics remain zero.

## Interpretation and recommendation

The tested clock removes a common way for the feedback loop to become stationary, but it does not supply a preserved orbit reference. The result is consistent with our first-principles distinction between progression and coherent state. It does not prove writer corruption is the sole cause, nor rule out better clock architectures/training.

Keep the clock optional as a diagnostic/architectural control. Do not extend these models based only on reduced stopping or low startup error. The next targeted investigation should separate G prediction error from damage to predictive information after one generated write, using local interventions and recovery tests. No further experiment has been queued.

## Implementation and validation

Config fields: clock_bands, clock_frequency, clock_rate, clock_to_d, clock_origin_max. Time is explicit in training, optional single-write feedback, cold evaluation, warm evaluation, and read-only diagnostic tools. An omitted time index raises an error for a clock-enabled reader. Random origins are per episode and shared by all sampled positions. Resume saves the independent clock RNG; older non-clock checkpoints remain supported.

No full generated training rollouts, trajectory losses, geometry supervision, clipping, EMA, or seed sweeps. One fixed particle per episode/trajectory. D alone trains the writer. Shared-clock variants use identical time for paired real/fake scores; exact default B-cap remains over candidate coordinates.

68 relevant tests passed, including clock units/timing, feedback alignment, active B-cap ownership, deterministic resume, and intervention startup preservation. GPU full-batch smoke and CPU diagnostic smoke passed. git diff --check passed. No training jobs were invalidated in this round.

Sources: [plan](plan.md), [full leaderboard](leaderboard.md), [raw results](results.json), [main interventions](interventions_main.json), [variant interventions](interventions_variants.json).

Central log: `tail -F runs/memory_path/core_round1/train.log`.
