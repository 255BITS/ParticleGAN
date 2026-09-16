# Single-point handoff scouts (no trajectory loss)

Completed runs only. Cold = zero-memory start. Warm = real prefix then generated writes only.
Reference-orbit pass requires radial RMSE < 0.1, direction consistency > 0.95,
signed-speed error < 0.03 rad/step, and first-point error < 0.2 reference radii.
This new composite is a diagnostic; see continuous errors and particle coverage in results.json.

| Run | Updates | Cold 256 / 1024 | Cold CW / CCW (256) | Stopped | Prefix8 reference 256 / 1024 | Prefix32 reference 256 / 1024 |
|---|---:|---:|---:|---:|---:|---:|
| local_predict10 | 2000 | 1.6% / 1.6% | 0 / 2 | 43.8% | 0.0% / 0.0% | 0.0% / 0.0% |
| feedback_p100 | 2000 | 0.0% / 0.0% | 0 / 0 | 3.1% | 0.0% / 0.0% | 0.0% / 0.0% |
| feedback_p100_mix50 | 2000 | 0.0% / 0.0% | 0 / 0 | 20.3% | 0.0% / 0.0% | 0.0% / 0.0% |
| feedback_p25 | 2000 | 0.0% / 0.0% | 0 / 0 | 17.2% | 0.0% / 0.0% | 0.0% / 0.0% |
| feedback_p50 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| feedback_p50_ctx16 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| feedback_p50_mature | 2000 | 0.0% / 0.0% | 0 / 0 | 10.2% | 0.0% / 0.0% | 0.0% / 0.0% |
| feedback_p50_mix50 | 2000 | 0.0% / 0.0% | 0 / 0 | 48.4% | 0.0% / 0.0% | 0.0% / 0.0% |
| feedback_p50_ramp | 2000 | 0.0% / 0.0% | 0 / 0 | 3.1% | 0.0% / 0.0% | 0.0% / 0.0% |
| handoff_dense4 | 2000 | 0.0% / 0.0% | 0 / 0 | 68.8% | 0.0% / 0.0% | 0.0% / 0.0% |

## Continuation errors (1,024 generated points)

| Run | Prefix | Radial RMSE | Speed MAE | Direction agreement | Startup error | Position error first32 / last128 |
|---|---:|---:|---:|---:|---:|---:|
| local_predict10 | 8 | 1.453 | 0.252 | 45.1% | 0.101 | 1.645 / 2.511 |
| local_predict10 | 32 | 1.418 | 0.255 | 46.2% | 0.099 | 1.612 / 2.480 |
| feedback_p100 | 8 | 1.755 | 0.271 | 50.0% | 2.434 | 2.453 / 2.756 |
| feedback_p100 | 32 | 1.795 | 0.270 | 50.0% | 2.866 | 2.475 / 2.790 |
| feedback_p100_mix50 | 8 | 1.832 | 0.259 | 49.4% | 0.319 | 2.073 / 2.670 |
| feedback_p100_mix50 | 32 | 1.793 | 0.258 | 50.4% | 0.199 | 1.971 / 2.657 |
| feedback_p25 | 8 | 1.486 | 0.261 | 51.6% | 0.222 | 1.475 / 2.432 |
| feedback_p25 | 32 | 1.378 | 0.263 | 51.3% | 0.139 | 1.328 / 2.312 |
| feedback_p50 | 8 | 1.251 | 0.268 | 50.8% | 0.188 | 1.654 / 2.171 |
| feedback_p50 | 32 | 1.252 | 0.266 | 50.9% | 0.147 | 1.545 / 2.167 |
| feedback_p50_ctx16 | 8 | 1.439 | 0.271 | 50.2% | 0.182 | 1.986 / 2.363 |
| feedback_p50_ctx16 | 32 | 1.435 | 0.271 | 50.2% | 0.182 | 1.919 / 2.357 |
| feedback_p50_mature | 8 | 1.316 | 0.254 | 51.7% | 0.251 | 1.701 / 2.158 |
| feedback_p50_mature | 32 | 1.328 | 0.261 | 50.8% | 0.157 | 1.589 / 2.192 |
| feedback_p50_mix50 | 8 | 2.179 | 0.265 | 51.5% | 0.181 | 1.619 / 3.277 |
| feedback_p50_mix50 | 32 | 2.082 | 0.273 | 52.6% | 0.144 | 1.556 / 3.161 |
| feedback_p50_ramp | 8 | 2.023 | 0.267 | 49.6% | 0.289 | 2.092 / 3.004 |
| feedback_p50_ramp | 32 | 1.999 | 0.266 | 50.0% | 0.290 | 2.078 / 2.973 |
| handoff_dense4 | 8 | 1.976 | 0.228 | 50.7% | 0.098 | 1.589 / 3.117 |
| handoff_dense4 | 32 | 1.922 | 0.235 | 52.6% | 0.098 | 1.619 / 3.010 |

## Interpretation and next-run candidates

- No new scout produced a passing cold-start circle at the long horizon; there is no winner on that metric.
- No scout yet passes reference-orbit fidelity for both prefix lengths; do not call a cold oscillator a solved handoff.
- All queued jobs have finished; rankings do not automatically promote a run or establish scientific success.

## Training cost

| Run | Point examples / update | Max real prefix | Memory size | D prediction / temporal weights | G calls per phase | Feedback probability / strength | Seconds / update |
|---|---:|---:|---:|---:|---:|---:|---:|
| feedback_p100 | 512 | 63 | 32 | 0 / 0 | 2 | 1 / 1 | 0.0402 |
| feedback_p100_mix50 | 512 | 63 | 32 | 0 / 0 | 2 | 1 / 0.5 | 0.0398 |
| feedback_p25 | 512 | 63 | 32 | 0 / 0 | 2 | 0.25 / 1 | 0.0393 |
| feedback_p50 | 512 | 63 | 32 | 0 / 0 | 2 | 0.5 / 1 | 0.0407 |
| feedback_p50_ctx16 | 512 | 16 | 32 | 0 / 0 | 2 | 0.5 / 1 | 0.0156 |
| feedback_p50_mature | 512 | 63 | 32 | 0 / 0 | 2 | 0.5 / 1 | 0.0403 |
| feedback_p50_mix50 | 512 | 63 | 32 | 0 / 0 | 2 | 0.5 / 0.5 | 0.0395 |
| feedback_p50_ramp | 512 | 63 | 32 | 0 / 0 | 2 | 0.5 / 1 | 0.0402 |

All new scouts use local next-point GAN losses. There is no full generated training
rollout, trajectory critic, or cold/warm path loss. Configured feedback adds at most one
detached generated write before each target. Longer rollouts are evaluation only.
Dense scouts use four points per episode; the older handoff_only trainer used one.
Architecture, context, corruption and optional D-only local auxiliary losses are explicit
config changes. Auxiliary heads do not change the GAN negative class. Compare measured cost too.
No seed sweeps. B-cap defaults unchanged. Prior regularization applied once per update.

Continuous fidelity, interventions, state statistics, and diversity: [results.json](results.json).
