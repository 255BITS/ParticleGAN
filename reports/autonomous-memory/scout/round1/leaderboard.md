# Completed autonomous memory experiments

Selections use metrics only. Full cold-start circle passes use the existing early-fit diagnostic;
coverage summaries include imperfect trajectories and are not calibrated distribution tests.

| Run | Updates | Full 256 | Full 1024 | Late stopped | Radial RMSE | Passing CW / CCW | Initial spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| trace_multi_diff | 2000 | 38.3% | 37.5% | 1.6% | 0.164 | 49 / 0 | 1.131 |
| trace_multi | 2000 | 33.6% | 32.0% | 3.1% | 0.162 | 43 / 0 | 1.228 |
| delay_multi | 2000 | 32.8% | 32.0% | 1.6% | 0.193 | 42 / 0 | 1.099 |
| gru_flat | 2000 | 14.1% | 15.6% | 1.6% | 0.237 | 5 / 13 | 1.135 |
| gru_frozen | 2000 | 6.2% | 7.0% | 7.8% | 0.272 | 8 / 0 | 1.224 |
| trace_flat | 2000 | 3.1% | 3.9% | 6.2% | 0.254 | 0 / 4 | 0.714 |

Per-run numerical coverage, interventions, sources and real-reference results: [results.json](results.json).
