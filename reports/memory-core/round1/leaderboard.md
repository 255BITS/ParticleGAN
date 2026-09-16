# Core memory GAN scouts

Completed runs only. Cold = zero-memory start. Warm = real prefix then generated writes only.
Reference-orbit pass requires radial RMSE < 0.1, direction consistency > 0.95,
signed-speed error < 0.03 rad/step, and first-point error < 0.2 reference radii.
This new composite is a diagnostic; see continuous errors and particle coverage in results.json.

| Run | Updates | Cold 256 / 1024 | Cold CW / CCW (256) | Stopped | Prefix8 reference 256 / 1024 | Prefix32 reference 256 / 1024 |
|---|---:|---:|---:|---:|---:|---:|
| autonomous_control_10000 | 10000 | 53.1% / 51.6% | 31 / 37 | 0.8% | 0.0% / 0.0% | 0.0% / 0.0% |
| handoff_cold_heavy | 2000 | 28.9% / 33.6% | 4 / 33 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| handoff_cold | 2000 | 28.1% / 28.1% | 35 / 1 | 3.9% | 0.0% / 0.0% | 0.0% / 0.0% |
| handoff_cold_light | 2000 | 26.6% / 26.6% | 18 / 16 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| autonomous_control_2000 | 2000 | 14.1% / 15.6% | 5 / 13 | 1.6% | 0.0% / 0.0% | 0.0% / 0.0% |
| handoff_both | 2000 | 10.2% / 10.9% | 13 / 0 | 7.8% | 0.0% / 0.0% | 0.0% / 0.0% |
| warm_only | 2000 | 7.0% / 10.9% | 9 / 0 | 3.1% | 0.0% / 0.0% | 0.0% / 0.0% |
| handoff_warm | 2000 | 0.8% / 7.8% | 1 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| handoff_only | 2000 | 0.0% / 0.0% | 0 / 0 | 60.2% | 0.0% / 0.0% | 0.0% / 0.0% |

## Continuation errors (1,024 generated points)

| Run | Prefix | Radial RMSE | Speed MAE | Direction agreement | Startup error | Position error first32 / last128 |
|---|---:|---:|---:|---:|---:|---:|
| autonomous_control_10000 | 8 | 0.672 | 0.303 | 51.9% | 0.801 | 1.520 / 1.595 |
| autonomous_control_10000 | 32 | 0.672 | 0.304 | 51.6% | 0.841 | 1.481 / 1.600 |
| handoff_cold_heavy | 8 | 0.623 | 0.285 | 51.1% | 0.237 | 1.407 / 1.547 |
| handoff_cold_heavy | 32 | 0.633 | 0.283 | 51.7% | 0.246 | 1.410 / 1.555 |
| handoff_cold | 8 | 0.622 | 0.298 | 49.7% | 0.461 | 1.476 / 1.543 |
| handoff_cold | 32 | 0.624 | 0.295 | 50.2% | 0.463 | 1.477 / 1.536 |
| handoff_cold_light | 8 | 0.644 | 0.283 | 54.3% | 0.619 | 1.530 / 1.596 |
| handoff_cold_light | 32 | 0.644 | 0.283 | 54.2% | 0.655 | 1.512 / 1.585 |
| autonomous_control_2000 | 8 | 0.649 | 0.282 | 54.9% | 0.875 | 1.487 / 1.560 |
| autonomous_control_2000 | 32 | 0.650 | 0.282 | 55.3% | 0.909 | 1.493 / 1.570 |
| handoff_both | 8 | 0.748 | 0.307 | 50.1% | 0.483 | 1.593 / 1.732 |
| handoff_both | 32 | 0.748 | 0.305 | 50.8% | 0.503 | 1.581 / 1.733 |
| warm_only | 8 | 0.618 | 0.288 | 50.4% | 0.856 | 1.473 / 1.544 |
| warm_only | 32 | 0.618 | 0.287 | 50.3% | 0.884 | 1.482 / 1.547 |
| handoff_warm | 8 | 0.692 | 0.211 | 75.8% | 0.365 | 1.475 / 1.635 |
| handoff_warm | 32 | 0.687 | 0.215 | 74.6% | 0.386 | 1.472 / 1.620 |
| handoff_only | 8 | 1.947 | 0.267 | 42.2% | 0.121 | 1.623 / 3.012 |
| handoff_only | 32 | 1.845 | 0.271 | 43.0% | 0.118 | 1.682 / 2.914 |

## Interpretation and next-run candidates

- Highest cold long-horizon circle rate so far: **handoff_cold_heavy**. Check both passing directions and coverage before promotion.
- No scout yet passes reference-orbit fidelity for both prefix lengths; do not call a cold oscillator a solved handoff.
- Candidate ranking is provisional until the queue finishes; no automatic promotion or scientific success claim.

The 10k control has more training than the 2k scouts. Conditional-head capacity and penalty domains
differ from the old path-only control; active-loss weights are normalized. All new scouts share
the same initialized modules, particles, data stream, optimizer defaults, and 10k schedule.
All evaluation episodes are fixed across formulations; no seed sweeps.

Continuous fidelity, interventions, state statistics, and diversity: [results.json](results.json).
