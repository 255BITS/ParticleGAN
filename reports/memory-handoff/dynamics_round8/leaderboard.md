# Single-point handoff scouts (no trajectory loss)

Completed runs only. Cold = zero-memory start. Warm = real prefix then generated writes only.
Reference-orbit pass requires radial RMSE < 0.1, direction consistency > 0.95,
signed-speed error < 0.03 rad/step, and first-point error < 0.2 reference radii.
This new composite is a diagnostic; see continuous errors and particle coverage in results.json.

| Run | Updates | Cold 256 / 1024 | Cold CW / CCW (256) | Stopped | Prefix8 reference 256 / 1024 | Prefix32 reference 256 / 1024 |
|---|---:|---:|---:|---:|---:|---:|
| repair_raw10 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| repair_raw100 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| repair_raw10_n15 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| repair_stable | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| repair_translated10 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| slow16_r03 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| slow16_r10 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| slow16_r25 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| slow24_r10 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| slow_repair | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| slow_stable | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| stable_dg10 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| stable_dg10_cap09 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| stable_g10 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| translate16 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| translate64 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| clock_fourier6 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| clock_fourier6_shared | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| clock_static6 | 2000 | 0.0% / 0.0% | 0 / 0 | 57.0% | 0.0% / 0.0% | 0.0% / 0.0% |

## Continuation errors (1,024 generated points)

| Run | Prefix | Radial RMSE | Speed MAE | Direction agreement | Startup error | Position error first32 / last128 |
|---|---:|---:|---:|---:|---:|---:|
| repair_raw10 | 8 | 2.186 | 0.271 | 50.2% | 0.120 | 1.869 / 3.241 |
| repair_raw10 | 32 | 2.183 | 0.270 | 50.4% | 0.119 | 1.925 / 3.349 |
| repair_raw100 | 8 | 1.813 | 0.272 | 49.8% | 0.100 | 1.603 / 2.932 |
| repair_raw100 | 32 | 1.811 | 0.274 | 49.6% | 0.094 | 1.628 / 2.879 |
| repair_raw10_n15 | 8 | 1.837 | 0.271 | 50.4% | 0.098 | 1.581 / 2.936 |
| repair_raw10_n15 | 32 | 1.834 | 0.270 | 50.7% | 0.103 | 1.431 / 2.889 |
| repair_stable | 8 | 2.671 | 0.272 | 50.6% | 0.103 | 1.398 / 3.820 |
| repair_stable | 32 | 2.687 | 0.272 | 50.6% | 0.103 | 1.426 / 3.991 |
| repair_translated10 | 8 | 2.104 | 0.275 | 50.1% | 0.092 | 1.620 / 3.256 |
| repair_translated10 | 32 | 2.106 | 0.275 | 50.1% | 0.109 | 1.676 / 3.302 |
| slow16_r03 | 8 | 2.151 | 0.269 | 50.4% | 0.117 | 1.462 / 3.239 |
| slow16_r03 | 32 | 2.172 | 0.270 | 50.4% | 0.117 | 1.451 / 3.381 |
| slow16_r10 | 8 | 1.677 | 0.267 | 50.1% | 0.117 | 1.699 / 2.917 |
| slow16_r10 | 32 | 1.679 | 0.267 | 50.2% | 0.133 | 1.635 / 2.834 |
| slow16_r25 | 8 | 1.901 | 0.268 | 51.4% | 0.119 | 2.176 / 2.930 |
| slow16_r25 | 32 | 1.901 | 0.266 | 51.4% | 0.119 | 2.022 / 3.068 |
| slow24_r10 | 8 | 1.821 | 0.268 | 50.7% | 0.140 | 1.409 / 2.922 |
| slow24_r10 | 32 | 1.836 | 0.269 | 50.6% | 0.132 | 1.559 / 3.079 |
| slow_repair | 8 | 2.016 | 0.271 | 50.4% | 0.117 | 1.863 / 3.238 |
| slow_repair | 32 | 2.020 | 0.271 | 50.3% | 0.139 | 1.889 / 3.179 |
| slow_stable | 8 | 1.914 | 0.273 | 49.7% | 0.118 | 1.621 / 3.119 |
| slow_stable | 32 | 1.914 | 0.274 | 49.6% | 0.120 | 1.529 / 3.028 |
| stable_dg10 | 8 | 2.126 | 0.272 | 49.4% | 0.108 | 1.552 / 3.250 |
| stable_dg10 | 32 | 2.123 | 0.273 | 49.2% | 0.109 | 1.401 / 3.284 |
| stable_dg10_cap09 | 8 | 2.345 | 0.269 | 50.8% | 0.110 | 1.507 / 3.455 |
| stable_dg10_cap09 | 32 | 2.344 | 0.268 | 50.9% | 0.121 | 1.642 / 3.417 |
| stable_g10 | 8 | 2.036 | 0.271 | 50.3% | 0.100 | 1.561 / 3.065 |
| stable_g10 | 32 | 2.039 | 0.269 | 50.6% | 0.115 | 1.470 / 3.218 |
| translate16 | 8 | 2.546 | 0.270 | 50.5% | 0.097 | 1.534 / 3.689 |
| translate16 | 32 | 2.545 | 0.269 | 51.1% | 0.098 | 1.511 / 3.897 |
| translate64 | 8 | 3.099 | 0.271 | 49.9% | 0.166 | 2.921 / 4.087 |
| translate64 | 32 | 2.990 | 0.269 | 50.4% | 0.147 | 2.650 / 4.155 |
| clock_fourier6 | 8 | 2.588 | 0.271 | 49.9% | 0.106 | 1.638 / 3.654 |
| clock_fourier6 | 32 | 2.597 | 0.270 | 50.0% | 0.112 | 1.466 / 3.879 |
| clock_fourier6_shared | 8 | 1.619 | 0.264 | 50.6% | 0.122 | 1.642 / 2.691 |
| clock_fourier6_shared | 32 | 1.619 | 0.266 | 50.5% | 0.138 | 1.692 / 2.701 |
| clock_static6 | 8 | 1.560 | 0.274 | 42.4% | 0.121 | 2.069 / 2.639 |
| clock_static6 | 32 | 1.561 | 0.273 | 42.6% | 0.134 | 2.027 / 2.627 |

## Interpretation and next-run candidates

- No new scout produced a passing cold-start circle at the long horizon; there is no winner on that metric.
- No scout yet passes reference-orbit fidelity for both prefix lengths; do not call a cold oscillator a solved handoff.
- All queued jobs have finished; rankings do not automatically promote a run or establish scientific success.

## Training cost

| Run | Point examples / update | Max real prefix | Memory size | D prediction / temporal weights | G calls D / G phase | Feedback probability / strength | Seconds / update |
|---|---:|---:|---:|---:|---:|---:|---:|
| repair_raw10 | 512 | 63 | 32 | 0 / 0 | 1 / 1 | 0 / 1 | 0.0400 |
| repair_raw100 | 512 | 63 | 32 | 0 / 0 | 1 / 1 | 0 / 1 | 0.0402 |
| repair_raw10_n15 | 512 | 63 | 32 | 0 / 0 | 1 / 1 | 0 / 1 | 0.0399 |
| repair_stable | 512 | 63 | 32 | 0 / 0 | 3 / 3 | 0 / 1 | 0.0459 |
| repair_translated10 | 512 | 63 | 32 | 0 / 0 | 1 / 1 | 0 / 1 | 0.0402 |
| slow16_r03 | 512 | 63 | 32 | 0 / 0 | 1 / 1 | 0 / 1 | 0.0533 |
| slow16_r10 | 512 | 63 | 32 | 0 / 0 | 1 / 1 | 0 / 1 | 0.0532 |
| slow16_r25 | 512 | 63 | 32 | 0 / 0 | 1 / 1 | 0 / 1 | 0.0546 |
| slow24_r10 | 512 | 63 | 32 | 0 / 0 | 1 / 1 | 0 / 1 | 0.0543 |
| slow_repair | 512 | 63 | 32 | 0 / 0 | 1 / 1 | 0 / 1 | 0.0567 |
| slow_stable | 512 | 63 | 32 | 0 / 0 | 3 / 3 | 0 / 1 | 0.0578 |
| stable_dg10 | 512 | 63 | 32 | 0 / 0 | 3 / 3 | 0 / 1 | 0.0437 |
| stable_dg10_cap09 | 512 | 63 | 32 | 0 / 0 | 3 / 3 | 0 / 1 | 0.0428 |
| stable_g10 | 512 | 63 | 32 | 0 / 0 | 1 / 3 | 0 / 1 | 0.0416 |
| translate16 | 512 | 63 | 32 | 0 / 0 | 1 / 1 | 0 / 1 | 0.0393 |
| translate64 | 512 | 63 | 32 | 0 / 0 | 1 / 1 | 0 / 1 | 0.0402 |

## Local memory dynamics settings

| Run | Slow coordinates / rate | G adapter bottleneck | Repair target / weight / noise | Stability D / G weight / max gain |
|---|---:|---:|---|---|
| repair_raw10 | 0 / 0.1 | 16 | raw / 10 / 0.05 | 0 / 0 / 1.1 |
| repair_raw100 | 0 / 0.1 | 16 | raw / 100 / 0.05 | 0 / 0 / 1.1 |
| repair_raw10_n15 | 0 / 0.1 | 16 | raw / 10 / 0.15 | 0 / 0 / 1.1 |
| repair_stable | 0 / 0.1 | 16 | raw / 10 / 0.05 | 0.1 / 0.1 / 1.1 |
| repair_translated10 | 0 / 0.1 | 16 | translated / 10 / 0.05 | 0 / 0 / 1.1 |
| slow16_r03 | 16 / 0.03 | off | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| slow16_r10 | 16 / 0.1 | off | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| slow16_r25 | 16 / 0.25 | off | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| slow24_r10 | 24 / 0.1 | off | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| slow_repair | 16 / 0.1 | 16 | raw / 10 / 0.05 | 0 / 0 / 1.1 |
| slow_stable | 16 / 0.1 | off | raw / 0 / 0.05 | 0.1 / 0.1 / 1.1 |
| stable_dg10 | 0 / 0.1 | off | raw / 0 / 0.05 | 0.1 / 0.1 / 1.1 |
| stable_dg10_cap09 | 0 / 0.1 | off | raw / 0 / 0.05 | 0.1 / 0.1 / 0.9 |
| stable_g10 | 0 / 0.1 | off | raw / 0 / 0.05 | 0 / 0.1 / 1.1 |
| translate16 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| translate64 | 0 / 0.1 | 64 | raw / 0 / 0.05 | 0 / 0 / 1.1 |

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
