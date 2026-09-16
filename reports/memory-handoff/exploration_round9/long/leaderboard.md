# Single-point handoff scouts (no trajectory loss)

Completed runs only. Cold = zero-memory start. Warm = real prefix then generated writes only.
Reference-orbit pass requires radial RMSE < 0.1, direction consistency > 0.95,
signed-speed error < 0.03 rad/step, and first-point error < 0.2 reference radii.
This new composite is a diagnostic; see continuous errors and particle coverage in results.json.

| Run | Updates | Cold 256 / 1024 | Cold CW / CCW (256) | Stopped | Prefix8 reference 256 / 1024 | Prefix32 reference 256 / 1024 |
|---|---:|---:|---:|---:|---:|---:|
| shared_s25_5k | 5000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| clock_control | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| proposal_clean_s25 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| shared_s25 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |

## Continuation errors (1,024 generated points)

| Run | Prefix | Radial RMSE | Speed MAE | Direction agreement | Startup error | Position error first32 / last128 |
|---|---:|---:|---:|---:|---:|---:|
| shared_s25_5k | 8 | 2.277 | 0.276 | 48.5% | 0.150 | 2.082 / 3.398 |
| shared_s25_5k | 32 | 2.272 | 0.276 | 48.3% | 0.132 | 2.131 / 3.420 |
| clock_control | 8 | 2.588 | 0.271 | 49.9% | 0.106 | 1.638 / 3.654 |
| clock_control | 32 | 2.597 | 0.270 | 50.0% | 0.112 | 1.466 / 3.879 |
| proposal_clean_s25 | 8 | 1.497 | 0.267 | 50.9% | 0.096 | 1.572 / 2.543 |
| proposal_clean_s25 | 32 | 1.487 | 0.268 | 50.7% | 0.096 | 1.448 / 2.573 |
| shared_s25 | 8 | 1.727 | 0.270 | 50.6% | 0.102 | 1.425 / 2.847 |
| shared_s25 | 32 | 1.726 | 0.270 | 50.5% | 0.104 | 1.365 / 2.702 |

## Interpretation and next-run candidates

- No new scout produced a passing cold-start circle at the long horizon; there is no winner on that metric.
- No scout yet passes reference-orbit fidelity for both prefix lengths; do not call a cold oscillator a solved handoff.
- All queued jobs have finished; rankings do not automatically promote a run or establish scientific success.

## Training cost

| Run | Point examples / update | Max real prefix | Memory size | D prediction / temporal weights | G calls D / G phase | Feedback probability / strength | Seconds / update |
|---|---:|---:|---:|---:|---:|---:|---:|
| shared_s25_5k | 512 | 63 | 32 | 0 / 0 | 2 / 2 | 0.5 / 0.25 | 0.0410 |

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
