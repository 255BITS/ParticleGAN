# Single-point handoff scouts (no trajectory loss)

Completed runs only. Cold = zero-memory start. Warm = real prefix then generated writes only.
Reference-orbit pass requires radial RMSE < 0.1, direction consistency > 0.95,
signed-speed error < 0.03 rad/step, and first-point error < 0.2 reference radii.
This new composite is a diagnostic; see continuous errors and particle coverage in results.json.

| Run | Updates | Cold 256 / 1024 | Cold CW / CCW (256) | Stopped | Prefix8 reference 256 / 1024 | Prefix32 reference 256 / 1024 |
|---|---:|---:|---:|---:|---:|---:|
| local_predict10 | 2000 | 1.6% / 1.6% | 0 / 2 | 43.8% | 0.0% / 0.0% | 0.0% / 0.0% |
| local_d256 | 2000 | 0.0% / 0.0% | 0 / 0 | 70.3% | 0.0% / 0.0% | 0.0% / 0.0% |
| local_g128 | 2000 | 0.0% / 0.0% | 0 / 0 | 82.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| local_predict1 | 2000 | 0.0% / 0.0% | 0 / 0 | 35.2% | 0.0% / 0.0% | 0.0% / 0.0% |
| local_predict10_temporal1 | 2000 | 0.0% / 0.0% | 0 / 0 | 60.9% | 0.0% / 0.0% | 0.0% / 0.0% |
| local_silu | 2000 | 0.0% / 0.0% | 0 / 0 | 51.6% | 0.0% / 0.0% | 0.0% / 0.0% |
| local_tanh | 2000 | 0.0% / 0.0% | 0 / 0 | 53.9% | 0.0% / 0.0% | 0.0% / 0.0% |
| local_temporal1 | 2000 | 0.0% / 0.0% | 0 / 0 | 56.2% | 0.0% / 0.0% | 0.0% / 0.0% |
| handoff_dense4 | 2000 | 0.0% / 0.0% | 0 / 0 | 68.8% | 0.0% / 0.0% | 0.0% / 0.0% |

## Continuation errors (1,024 generated points)

| Run | Prefix | Radial RMSE | Speed MAE | Direction agreement | Startup error | Position error first32 / last128 |
|---|---:|---:|---:|---:|---:|---:|
| local_predict10 | 8 | 1.453 | 0.252 | 45.1% | 0.101 | 1.645 / 2.511 |
| local_predict10 | 32 | 1.418 | 0.255 | 46.2% | 0.099 | 1.612 / 2.480 |
| local_d256 | 8 | 2.109 | 0.241 | 51.8% | 0.111 | 1.714 / 3.225 |
| local_d256 | 32 | 1.975 | 0.250 | 52.4% | 0.114 | 1.713 / 3.091 |
| local_g128 | 8 | 2.852 | 0.275 | 36.0% | 0.111 | 2.533 / 3.937 |
| local_g128 | 32 | 2.860 | 0.272 | 36.9% | 0.120 | 2.383 / 3.941 |
| local_predict1 | 8 | 1.377 | 0.238 | 56.1% | 0.094 | 1.379 / 2.457 |
| local_predict1 | 32 | 1.321 | 0.243 | 57.3% | 0.097 | 1.344 / 2.380 |
| local_predict10_temporal1 | 8 | 1.871 | 0.267 | 41.1% | 0.096 | 1.744 / 2.985 |
| local_predict10_temporal1 | 32 | 1.869 | 0.250 | 42.3% | 0.104 | 1.757 / 2.961 |
| local_silu | 8 | 1.916 | 0.305 | 44.2% | 0.102 | 1.898 / 2.998 |
| local_silu | 32 | 1.888 | 0.301 | 41.7% | 0.113 | 1.980 / 2.952 |
| local_tanh | 8 | 1.678 | 0.282 | 36.5% | 0.132 | 2.041 / 2.713 |
| local_tanh | 32 | 1.673 | 0.285 | 36.9% | 0.139 | 2.038 / 2.701 |
| local_temporal1 | 8 | 1.358 | 0.279 | 44.7% | 0.110 | 1.582 / 2.431 |
| local_temporal1 | 32 | 1.446 | 0.271 | 45.7% | 0.118 | 1.588 / 2.498 |
| handoff_dense4 | 8 | 1.976 | 0.228 | 50.7% | 0.098 | 1.589 / 3.117 |
| handoff_dense4 | 32 | 1.922 | 0.235 | 52.6% | 0.098 | 1.619 / 3.010 |

## Interpretation and next-run candidates

- Highest cold long-horizon circle rate so far: **local_predict10**. Check both passing directions and coverage before promotion.
- No scout yet passes reference-orbit fidelity for both prefix lengths; do not call a cold oscillator a solved handoff.
- All queued jobs have finished; rankings do not automatically promote a run or establish scientific success.

## Training cost

| Run | Point examples / update | Max real prefix | Memory size | D prediction / temporal weights | Seconds / update |
|---|---:|---:|---:|---:|---:|
| local_predict10 | 512 | 63 | 32 | 10 / 0 | 0.0392 |
| local_d256 | 512 | 63 | 32 | 0 / 0 | 0.0384 |
| local_g128 | 512 | 63 | 32 | 0 / 0 | 0.0391 |
| local_predict1 | 512 | 63 | 32 | 1 / 0 | 0.0387 |
| local_predict10_temporal1 | 512 | 63 | 32 | 10 / 1 | 0.0408 |
| local_silu | 512 | 63 | 32 | 0 / 0 | 0.0395 |
| local_tanh | 512 | 63 | 32 | 0 / 0 | 0.0389 |
| local_temporal1 | 512 | 63 | 32 | 0 / 1 | 0.0409 |

All new scouts train only independent single-point GAN predictions; no generated training
trajectory, trajectory critic, or cold/warm loss. Longer rollouts are evaluation only.
Dense scouts use four points per episode; the older handoff_only trainer used one.
Architecture, context, corruption and optional D-only local auxiliary losses are explicit
config changes. Auxiliary heads do not change the GAN negative class. Compare measured cost too.
No seed sweeps. B-cap defaults unchanged. Prior regularization applied once per update.

Continuous fidelity, interventions, state statistics, and diversity: [results.json](results.json).
