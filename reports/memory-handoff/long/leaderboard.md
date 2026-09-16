# Single-point handoff scouts (no trajectory loss)

Completed runs only. Cold = zero-memory start. Warm = real prefix then generated writes only.
Reference-orbit pass requires radial RMSE < 0.1, direction consistency > 0.95,
signed-speed error < 0.03 rad/step, and first-point error < 0.2 reference radii.
This new composite is a diagnostic; see continuous errors and particle coverage in results.json.

| Run | Updates | Cold 256 / 1024 | Cold CW / CCW (256) | Stopped | Prefix8 reference 256 / 1024 | Prefix32 reference 256 / 1024 |
|---|---:|---:|---:|---:|---:|---:|
| handoff_dense4_start25 | 2000 | 0.0% / 0.8% | 0 / 0 | 46.9% | 0.0% / 0.0% | 0.0% / 0.0% |
| handoff_dense4_input10_5k | 5000 | 0.0% / 0.0% | 0 / 0 | 75.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| handoff_dense4_start25_5k | 5000 | 0.0% / 0.0% | 0 / 0 | 30.5% | 0.0% / 0.0% | 0.0% / 0.0% |
| handoff_only | 2000 | 0.0% / 0.0% | 0 / 0 | 60.2% | 0.0% / 0.0% | 0.0% / 0.0% |
| handoff_dense4 | 2000 | 0.0% / 0.0% | 0 / 0 | 68.8% | 0.0% / 0.0% | 0.0% / 0.0% |
| handoff_dense4_ctx16 | 2000 | 0.0% / 0.0% | 0 / 0 | 35.2% | 0.0% / 0.0% | 0.0% / 0.0% |
| handoff_dense4_ctx8 | 2000 | 0.0% / 0.0% | 0 / 0 | 21.9% | 0.0% / 0.0% | 0.0% / 0.0% |
| handoff_dense4_input03 | 2000 | 0.0% / 0.0% | 0 / 0 | 56.2% | 0.0% / 0.0% | 0.0% / 0.0% |
| handoff_dense4_input10 | 2000 | 0.0% / 0.0% | 0 / 0 | 4.7% | 0.0% / 0.0% | 0.0% / 0.0% |
| handoff_dense4_interaction | 2000 | 0.0% / 0.0% | 0 / 0 | 43.8% | 0.0% / 0.0% | 0.0% / 0.0% |
| handoff_dense4_m64 | 2000 | 0.0% / 0.0% | 0 / 0 | 46.9% | 0.0% / 0.0% | 0.0% / 0.0% |
| handoff_dense4_m8 | 2000 | 0.0% / 0.0% | 0 / 0 | 65.6% | 0.0% / 0.0% | 0.0% / 0.0% |
| handoff_dense4_state05 | 2000 | 0.0% / 0.0% | 0 / 0 | 74.2% | 0.0% / 0.0% | 0.0% / 0.0% |

## Continuation errors (1,024 generated points)

| Run | Prefix | Radial RMSE | Speed MAE | Direction agreement | Startup error | Position error first32 / last128 |
|---|---:|---:|---:|---:|---:|---:|
| handoff_dense4_start25 | 8 | 2.227 | 0.271 | 42.3% | 0.121 | 2.209 / 3.270 |
| handoff_dense4_start25 | 32 | 2.204 | 0.268 | 41.9% | 0.126 | 2.231 / 3.228 |
| handoff_dense4_input10_5k | 8 | 2.863 | 0.257 | 49.9% | 0.237 | 1.785 / 4.040 |
| handoff_dense4_input10_5k | 32 | 2.635 | 0.262 | 46.0% | 0.193 | 1.463 / 3.762 |
| handoff_dense4_start25_5k | 8 | 2.280 | 0.282 | 52.7% | 0.117 | 2.188 / 3.281 |
| handoff_dense4_start25_5k | 32 | 2.365 | 0.282 | 53.2% | 0.122 | 2.245 / 3.354 |
| handoff_only | 8 | 1.947 | 0.267 | 42.2% | 0.121 | 1.623 / 3.012 |
| handoff_only | 32 | 1.845 | 0.271 | 43.0% | 0.118 | 1.682 / 2.914 |
| handoff_dense4 | 8 | 1.976 | 0.228 | 50.7% | 0.098 | 1.589 / 3.117 |
| handoff_dense4 | 32 | 1.922 | 0.235 | 52.6% | 0.098 | 1.619 / 3.010 |
| handoff_dense4_ctx16 | 8 | 1.420 | 0.277 | 49.7% | 0.107 | 1.655 / 2.441 |
| handoff_dense4_ctx16 | 32 | 1.369 | 0.272 | 52.0% | 0.103 | 1.650 / 2.375 |
| handoff_dense4_ctx8 | 8 | 1.714 | 0.258 | 52.6% | 0.124 | 1.878 / 2.719 |
| handoff_dense4_ctx8 | 32 | 1.690 | 0.266 | 52.2% | 0.141 | 1.969 / 2.632 |
| handoff_dense4_input03 | 8 | 2.044 | 0.255 | 50.6% | 0.121 | 1.589 / 3.115 |
| handoff_dense4_input03 | 32 | 1.939 | 0.258 | 48.5% | 0.114 | 1.512 / 2.988 |
| handoff_dense4_input10 | 8 | 1.138 | 0.253 | 55.4% | 0.196 | 1.631 / 2.086 |
| handoff_dense4_input10 | 32 | 1.155 | 0.257 | 55.0% | 0.191 | 1.658 / 2.076 |
| handoff_dense4_interaction | 8 | 1.535 | 0.282 | 49.7% | 0.092 | 1.627 / 2.575 |
| handoff_dense4_interaction | 32 | 1.560 | 0.277 | 49.1% | 0.101 | 1.679 / 2.622 |
| handoff_dense4_m64 | 8 | 1.664 | 0.258 | 56.9% | 0.110 | 1.425 / 2.644 |
| handoff_dense4_m64 | 32 | 1.634 | 0.267 | 54.3% | 0.107 | 1.529 / 2.578 |
| handoff_dense4_m8 | 8 | 1.228 | 0.279 | 33.0% | 0.192 | 1.960 / 2.257 |
| handoff_dense4_m8 | 32 | 1.229 | 0.281 | 31.6% | 0.193 | 1.939 / 2.271 |
| handoff_dense4_state05 | 8 | 1.543 | 0.267 | 39.3% | 0.106 | 1.459 / 2.657 |
| handoff_dense4_state05 | 32 | 1.590 | 0.267 | 38.0% | 0.105 | 1.481 / 2.691 |

## Interpretation and next-run candidates

- No new scout produced a passing cold-start circle at the long horizon; there is no winner on that metric.
- No scout yet passes reference-orbit fidelity for both prefix lengths; do not call a cold oscillator a solved handoff.
- All queued jobs have finished; rankings do not automatically promote a run or establish scientific success.

## Training cost

| Run | Point examples / update | Max real prefix | Memory size | Seconds / update |
|---|---:|---:|---:|---:|
| handoff_dense4_input10_5k | 512 | 63 | 32 | 0.0389 |
| handoff_dense4_start25_5k | 512 | 63 | 32 | 0.0390 |

All new scouts train only independent single-point GAN predictions; no generated training
trajectory, trajectory critic, or cold/warm loss. Longer rollouts are evaluation only.
The historical handoff_only baseline used one point per episode; dense scouts use four
points per episode and therefore more point supervision per update. Memory size, context
length, conditioning and corruption are explicit config changes; compare measured cost too.
No seed sweeps. B-cap defaults unchanged. Prior regularization applied once per update.

Continuous fidelity, interventions, state statistics, and diversity: [results.json](results.json).
