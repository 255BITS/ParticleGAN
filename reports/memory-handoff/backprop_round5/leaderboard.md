# INVALIDATED: use backprop_round5_corrected

This queue used an incomplete G gradient: its real score was detached despite
its dependence on the generated memory. Two jobs completed but their results
must not enter model selection or valid leaderboards. Two active jobs were
intentionally terminated, including an unaffected extension that will restart
from its original2k checkpoint. One pending job was cancelled. No conclusions
are drawn from these outputs. The earlier detached-feedback and recent-memory
rounds are unaffected. The corrected implementation retains both branches of
the paired loss and tests cancellation of shared context-only score offsets.

---
Archived invalidated output:

# Single-point handoff scouts (no trajectory loss)

Completed runs only. Cold = zero-memory start. Warm = real prefix then generated writes only.
Reference-orbit pass requires radial RMSE < 0.1, direction consistency > 0.95,
signed-speed error < 0.03 rad/step, and first-point error < 0.2 reference radii.
This new composite is a diagnostic; see continuous errors and particle coverage in results.json.

| Run | Updates | Cold 256 / 1024 | Cold CW / CCW (256) | Stopped | Prefix8 reference 256 / 1024 | Prefix32 reference 256 / 1024 |
|---|---:|---:|---:|---:|---:|---:|
| feedback_p25_bp | 2000 | 0.0% / 0.0% | 0 / 0 | 58.6% | 0.0% / 0.0% | 0.0% / 0.0% |
| feedback_p50_mature_bp | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| feedback_p25 | 2000 | 0.0% / 0.0% | 0 / 0 | 17.2% | 0.0% / 0.0% | 0.0% / 0.0% |
| feedback_p50_mature | 2000 | 0.0% / 0.0% | 0 / 0 | 10.2% | 0.0% / 0.0% | 0.0% / 0.0% |
| handoff_dense4 | 2000 | 0.0% / 0.0% | 0 / 0 | 68.8% | 0.0% / 0.0% | 0.0% / 0.0% |
| recent2_delta | 2000 | 0.0% / 0.0% | 0 / 0 | 25.8% | 0.0% / 0.0% | 0.0% / 0.0% |
| recent4_absolute | 2000 | 0.0% / 0.0% | 0 / 0 | 70.3% | 0.0% / 0.0% | 0.0% / 0.0% |
| recent4_delta | 2000 | 0.0% / 0.0% | 0 / 0 | 14.8% | 0.0% / 0.0% | 0.0% / 0.0% |
| recent4_delta_fb25 | 2000 | 0.0% / 0.0% | 0 / 0 | 35.9% | 0.0% / 0.0% | 0.0% / 0.0% |
| recent4_delta_fb50_mature | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| recent_bound_control | 2000 | 0.0% / 0.0% | 0 / 0 | 56.2% | 0.0% / 0.0% | 0.8% / 0.0% |

## Continuation errors (1,024 generated points)

| Run | Prefix | Radial RMSE | Speed MAE | Direction agreement | Startup error | Position error first32 / last128 |
|---|---:|---:|---:|---:|---:|---:|
| feedback_p25_bp | 8 | 2.164 | 0.271 | 50.7% | 0.199 | 1.619 / 3.312 |
| feedback_p25_bp | 32 | 2.013 | 0.275 | 48.1% | 0.138 | 1.638 / 3.202 |
| feedback_p50_mature_bp | 8 | 1.779 | 0.272 | 50.2% | 0.215 | 2.105 / 2.839 |
| feedback_p50_mature_bp | 32 | 1.787 | 0.272 | 50.2% | 0.193 | 2.000 / 2.846 |
| feedback_p25 | 8 | 1.486 | 0.261 | 51.6% | 0.222 | 1.475 / 2.432 |
| feedback_p25 | 32 | 1.378 | 0.263 | 51.3% | 0.139 | 1.328 / 2.312 |
| feedback_p50_mature | 8 | 1.316 | 0.254 | 51.7% | 0.251 | 1.701 / 2.158 |
| feedback_p50_mature | 32 | 1.328 | 0.261 | 50.8% | 0.157 | 1.589 / 2.192 |
| handoff_dense4 | 8 | 1.976 | 0.228 | 50.7% | 0.098 | 1.589 / 3.117 |
| handoff_dense4 | 32 | 1.922 | 0.235 | 52.6% | 0.098 | 1.619 / 3.010 |
| recent2_delta | 8 | 14.497 | 0.257 | 58.6% | 0.102 | 1.593 / 15.366 |
| recent2_delta | 32 | 13.861 | 0.255 | 61.1% | 0.095 | 1.478 / 14.996 |
| recent4_absolute | 8 | 1.635 | 0.276 | 29.1% | 0.105 | 1.901 / 2.724 |
| recent4_absolute | 32 | 1.648 | 0.271 | 30.4% | 0.120 | 1.940 / 2.737 |
| recent4_delta | 8 | 1.089 | 0.355 | 48.9% | 0.092 | 1.354 / 2.087 |
| recent4_delta | 32 | 1.085 | 0.355 | 48.6% | 0.102 | 1.405 / 2.100 |
| recent4_delta_fb25 | 8 | 585.869 | 0.254 | 55.7% | 0.099 | 1.465 / 1023.301 |
| recent4_delta_fb25 | 32 | 517.095 | 0.251 | 55.2% | 0.088 | 1.509 / 898.114 |
| recent4_delta_fb50_mature | 8 | 1457.829 | 0.269 | 51.8% | 0.122 | 1.834 / 2527.303 |
| recent4_delta_fb50_mature | 32 | 1466.462 | 0.269 | 49.6% | 0.111 | 1.970 / 2530.524 |
| recent_bound_control | 8 | 1.298 | 0.257 | 39.5% | 0.090 | 1.584 / 2.380 |
| recent_bound_control | 32 | 1.375 | 0.252 | 38.4% | 0.096 | 1.484 / 2.432 |

## Interpretation and next-run candidates

- No new scout produced a passing cold-start circle at the long horizon; there is no winner on that metric.
- No scout yet passes reference-orbit fidelity for both prefix lengths; do not call a cold oscillator a solved handoff.
- Candidate ranking is provisional until the queue finishes; no automatic promotion or scientific success claim.

## Training cost

| Run | Point examples / update | Max real prefix | Memory size | D prediction / temporal weights | G calls per phase | Feedback probability / strength | Seconds / update |
|---|---:|---:|---:|---:|---:|---:|---:|
| feedback_p25_bp | 512 | 63 | 32 | 0 / 0 | 2 | 0.25 / 1 | 0.0407 |
| feedback_p50_mature_bp | 512 | 63 | 32 | 0 / 0 | 2 | 0.5 / 1 | 0.0402 |

All new scouts use local next-point GAN losses. There is no full generated training
rollout, trajectory critic, or cold/warm path loss. Configured feedback adds at most one
generated write before each target; configs control G gradients through that write.
Longer rollouts are evaluation only.
Dense scouts use four points per episode; the older handoff_only trainer used one.
Architecture, context, corruption and optional D-only local auxiliary losses are explicit
config changes. Auxiliary heads do not change the GAN negative class. Compare measured cost too.
No seed sweeps. B-cap defaults unchanged. Prior regularization applied once per update.

Continuous fidelity, interventions, state statistics, and diversity: [results.json](results.json).
