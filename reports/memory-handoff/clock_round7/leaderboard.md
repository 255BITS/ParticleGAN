# Single-point handoff scouts (no trajectory loss)

Completed runs only. Cold = zero-memory start. Warm = real prefix then generated writes only.
Reference-orbit pass requires radial RMSE < 0.1, direction consistency > 0.95,
signed-speed error < 0.03 rad/step, and first-point error < 0.2 reference radii.
This new composite is a diagnostic; see continuous errors and particle coverage in results.json.

| Run | Updates | Cold 256 / 1024 | Cold CW / CCW (256) | Stopped | Prefix8 reference 256 / 1024 | Prefix32 reference 256 / 1024 |
|---|---:|---:|---:|---:|---:|---:|
| clock_fourier3_fast | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| clock_fourier6 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| clock_fourier6_offset | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| clock_fourier6_offset_recent | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| clock_fourier6_offset_shared | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| clock_fourier6_recent | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| clock_fourier6_shared | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| clock_static6 | 2000 | 0.0% / 0.0% | 0 / 0 | 57.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| handoff_dense4 | 2000 | 0.0% / 0.0% | 0 / 0 | 68.8% | 0.0% / 0.0% | 0.0% / 0.0% |
| recent4_delta | 2000 | 0.0% / 0.0% | 0 / 0 | 14.8% | 0.0% / 0.0% | 0.0% / 0.0% |

## Continuation errors (1,024 generated points)

| Run | Prefix | Radial RMSE | Speed MAE | Direction agreement | Startup error | Position error first32 / last128 |
|---|---:|---:|---:|---:|---:|---:|
| clock_fourier3_fast | 8 | 2.313 | 0.268 | 50.8% | 0.145 | 2.553 / 3.344 |
| clock_fourier3_fast | 32 | 2.331 | 0.268 | 50.8% | 0.145 | 2.428 / 3.359 |
| clock_fourier6 | 8 | 2.588 | 0.271 | 49.9% | 0.106 | 1.638 / 3.654 |
| clock_fourier6 | 32 | 2.597 | 0.270 | 50.0% | 0.112 | 1.466 / 3.879 |
| clock_fourier6_offset | 8 | 2.632 | 0.265 | 51.9% | 0.114 | 1.804 / 3.728 |
| clock_fourier6_offset | 32 | 2.633 | 0.268 | 51.0% | 0.103 | 1.801 / 3.550 |
| clock_fourier6_offset_recent | 8 | 1129.811 | 0.267 | 53.3% | 0.090 | 1.301 / 2025.386 |
| clock_fourier6_offset_recent | 32 | 1163.717 | 0.257 | 54.9% | 0.088 | 1.715 / 2110.247 |
| clock_fourier6_offset_shared | 8 | 1.718 | 0.269 | 51.3% | 0.105 | 1.582 / 2.756 |
| clock_fourier6_offset_shared | 32 | 1.720 | 0.274 | 50.3% | 0.102 | 1.632 / 2.874 |
| clock_fourier6_recent | 8 | 1.788 | 0.268 | 50.4% | 0.095 | 1.761 / 2.561 |
| clock_fourier6_recent | 32 | 1.811 | 0.264 | 51.2% | 0.088 | 1.269 / 2.631 |
| clock_fourier6_shared | 8 | 1.619 | 0.264 | 50.6% | 0.122 | 1.642 / 2.691 |
| clock_fourier6_shared | 32 | 1.619 | 0.266 | 50.5% | 0.138 | 1.692 / 2.701 |
| clock_static6 | 8 | 1.560 | 0.274 | 42.4% | 0.121 | 2.069 / 2.639 |
| clock_static6 | 32 | 1.561 | 0.273 | 42.6% | 0.134 | 2.027 / 2.627 |
| handoff_dense4 | 8 | 1.976 | 0.228 | 50.7% | 0.098 | 1.589 / 3.117 |
| handoff_dense4 | 32 | 1.922 | 0.235 | 52.6% | 0.098 | 1.619 / 3.010 |
| recent4_delta | 8 | 1.089 | 0.355 | 48.9% | 0.092 | 1.354 / 2.087 |
| recent4_delta | 32 | 1.085 | 0.355 | 48.6% | 0.102 | 1.405 / 2.100 |

## Interpretation and next-run candidates

- No new scout produced a passing cold-start circle at the long horizon; there is no winner on that metric.
- No scout yet passes reference-orbit fidelity for both prefix lengths; do not call a cold oscillator a solved handoff.
- All queued jobs have finished; rankings do not automatically promote a run or establish scientific success.

## Training cost

| Run | Point examples / update | Max real prefix | Memory size | D prediction / temporal weights | G calls per phase | Feedback probability / strength | Seconds / update |
|---|---:|---:|---:|---:|---:|---:|---:|
| clock_fourier3_fast | 512 | 63 | 32 | 0 / 0 | 1 | 0 / 1 | 0.0389 |
| clock_fourier6 | 512 | 63 | 32 | 0 / 0 | 1 | 0 / 1 | 0.0391 |
| clock_fourier6_offset | 512 | 63 | 32 | 0 / 0 | 1 | 0 / 1 | 0.0388 |
| clock_fourier6_offset_recent | 512 | 63 | 32 | 0 / 0 | 1 | 0 / 1 | 0.0498 |
| clock_fourier6_offset_shared | 512 | 63 | 32 | 0 / 0 | 1 | 0 / 1 | 0.0394 |
| clock_fourier6_recent | 512 | 63 | 32 | 0 / 0 | 1 | 0 / 1 | 0.0496 |
| clock_fourier6_shared | 512 | 63 | 32 | 0 / 0 | 1 | 0 / 1 | 0.0398 |
| clock_static6 | 512 | 63 | 32 | 0 / 0 | 1 | 0 / 1 | 0.0386 |

All new scouts use local next-point GAN losses. There is no full generated training
rollout, trajectory critic, or cold/warm path loss. Configured feedback adds at most one
generated write before each target; configs control G gradients through that write.
Longer rollouts are evaluation only.
Dense scouts use four points per episode; the older handoff_only trainer used one.
Architecture, context, corruption and optional D-only local auxiliary losses are explicit
config changes. Auxiliary heads do not change the GAN negative class. Compare measured cost too.
No seed sweeps. B-cap defaults unchanged. Prior regularization applied once per update.

Continuous fidelity, interventions, state statistics, and diversity: [results.json](results.json).
