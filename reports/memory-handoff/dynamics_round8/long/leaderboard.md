# Single-point handoff scouts (no trajectory loss)

Completed runs only. Cold = zero-memory start. Warm = real prefix then generated writes only.
Reference-orbit pass requires radial RMSE < 0.1, direction consistency > 0.95,
signed-speed error < 0.03 rad/step, and first-point error < 0.2 reference radii.
This new composite is a diagnostic; see continuous errors and particle coverage in results.json.

| Run | Updates | Cold 256 / 1024 | Cold CW / CCW (256) | Stopped | Prefix8 reference 256 / 1024 | Prefix32 reference 256 / 1024 |
|---|---:|---:|---:|---:|---:|---:|
| repair_raw10_n15_5k | 5000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| clock_fourier6 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| clock_fourier6_shared | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| repair_raw10_n15 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |

## Continuation errors (1,024 generated points)

| Run | Prefix | Radial RMSE | Speed MAE | Direction agreement | Startup error | Position error first32 / last128 |
|---|---:|---:|---:|---:|---:|---:|
| repair_raw10_n15_5k | 8 | 2.452 | 0.263 | 52.3% | 0.120 | 1.762 / 3.738 |
| repair_raw10_n15_5k | 32 | 2.456 | 0.264 | 51.6% | 0.118 | 1.713 / 3.683 |
| clock_fourier6 | 8 | 2.588 | 0.271 | 49.9% | 0.106 | 1.638 / 3.654 |
| clock_fourier6 | 32 | 2.597 | 0.270 | 50.0% | 0.112 | 1.466 / 3.879 |
| clock_fourier6_shared | 8 | 1.619 | 0.264 | 50.6% | 0.122 | 1.642 / 2.691 |
| clock_fourier6_shared | 32 | 1.619 | 0.266 | 50.5% | 0.138 | 1.692 / 2.701 |
| repair_raw10_n15 | 8 | 1.837 | 0.271 | 50.4% | 0.098 | 1.581 / 2.936 |
| repair_raw10_n15 | 32 | 1.834 | 0.270 | 50.7% | 0.103 | 1.431 / 2.889 |

## Interpretation and next-run candidates

- No new scout produced a passing cold-start circle at the long horizon; there is no winner on that metric.
- No scout yet passes reference-orbit fidelity for both prefix lengths; do not call a cold oscillator a solved handoff.
- All queued jobs have finished; rankings do not automatically promote a run or establish scientific success.

## Training cost

| Run | Point examples / update | Max real prefix | Memory size | D prediction / temporal weights | G calls D / G phase | Feedback probability / strength | Seconds / update |
|---|---:|---:|---:|---:|---:|---:|---:|
| repair_raw10_n15_5k | 512 | 63 | 32 | 0 / 0 | 1 / 1 | 0 / 1 | 0.0402 |

## Local memory dynamics settings

| Run | Slow coordinates / rate | G adapter bottleneck | Repair target / weight / noise | Stability D / G weight / max gain |
|---|---:|---:|---|---|
| repair_raw10_n15_5k | 0 / 0.1 | 16 | raw / 10 / 0.15 | 0 / 0 / 1.1 |

All new scouts use local next-point GAN losses. There is no full generated training
rollout, trajectory critic, or cold/warm path loss. Configured feedback adds at most one
generated write before each target; configs control G gradients through that write.
Longer rollouts are evaluation only.
Dense scouts use four points per episode; the older handoff_only trainer used one.
Architecture, context, corruption and optional D-only local auxiliary losses are explicit
config changes. Auxiliary heads do not change the GAN negative class. Compare measured cost too.
Optional G repair trains a stateless read adapter. Local stability adds two parallel
one-step feedback branches per enabled phase, with detached prefix anchors and particles.
These local branches do not feed into another generated prediction; writer updates remain D-only.
No seed sweeps. B-cap defaults unchanged. Prior regularization applied once per update.

Continuous fidelity, interventions, state statistics, and diversity: [results.json](results.json).
