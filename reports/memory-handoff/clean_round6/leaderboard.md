# Single-point handoff scouts (no trajectory loss)

Completed runs only. Cold = zero-memory start. Warm = real prefix then generated writes only.
Reference-orbit pass requires radial RMSE < 0.1, direction consistency > 0.95,
signed-speed error < 0.03 rad/step, and first-point error < 0.2 reference radii.
This new composite is a diagnostic; see continuous errors and particle coverage in results.json.

| Run | Updates | Cold 256 / 1024 | Cold CW / CCW (256) | Stopped | Prefix8 reference 256 / 1024 | Prefix32 reference 256 / 1024 |
|---|---:|---:|---:|---:|---:|---:|
| clean_gru | 2000 | 0.0% / 0.0% | 0 / 0 | 49.2% | 0.0% / 0.0% | 0.0% / 0.0% |
| clean_recent4_delta | 2000 | 0.0% / 0.0% | 0 / 0 | 58.6% | 0.0% / 0.0% | 0.0% / 0.0% |
| handoff_dense4 | 2000 | 0.0% / 0.0% | 0 / 0 | 68.8% | 0.0% / 0.0% | 0.0% / 0.0% |
| recent4_delta | 2000 | 0.0% / 0.0% | 0 / 0 | 14.8% | 0.0% / 0.0% | 0.0% / 0.0% |

## Continuation errors (1,024 generated points)

| Run | Prefix | Radial RMSE | Speed MAE | Direction agreement | Startup error | Position error first32 / last128 |
|---|---:|---:|---:|---:|---:|---:|
| clean_gru | 8 | 0.965 | 0.230 | 63.6% | 0.042 | 1.263 / 2.028 |
| clean_gru | 32 | 0.939 | 0.229 | 61.0% | 0.042 | 1.259 / 1.965 |
| clean_recent4_delta | 8 | 2.456 | 0.271 | 46.2% | 0.077 | 2.128 / 3.494 |
| clean_recent4_delta | 32 | 2.456 | 0.271 | 46.7% | 0.076 | 2.189 / 3.511 |
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
| clean_gru | 512 | 63 | 32 | 0 / 0 | 1 | 0 / 1 | 0.0387 |
| clean_recent4_delta | 512 | 63 | 32 | 0 / 0 | 1 | 0 / 1 | 0.0492 |

All new scouts use local next-point GAN losses. There is no full generated training
rollout, trajectory critic, or cold/warm path loss. Configured feedback adds at most one
generated write before each target; configs control G gradients through that write.
Longer rollouts are evaluation only.
Dense scouts use four points per episode; the older handoff_only trainer used one.
Architecture, context, corruption and optional D-only local auxiliary losses are explicit
config changes. Auxiliary heads do not change the GAN negative class. Compare measured cost too.
No seed sweeps. B-cap defaults unchanged. Prior regularization applied once per update.

Continuous fidelity, interventions, state statistics, and diversity: [results.json](results.json).
