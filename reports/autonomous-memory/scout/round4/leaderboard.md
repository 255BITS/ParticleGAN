# Completed autonomous memory experiments

Selections use metrics only. Full cold-start circle passes use the existing early-fit diagnostic;
coverage summaries include imperfect trajectories and are not calibrated distribution tests.

| Run | Updates | Full 256 | Full 1024 | Late stopped | Radial RMSE | Passing CW / CCW | Initial spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| gru_private | 2000 | 55.5% | 54.7% | 0.0% | 0.126 | 0 / 71 | 1.146 |
| gru_m8 | 2000 | 22.7% | 22.7% | 10.2% | 0.152 | 0 / 29 | 0.937 |
| gru_flat | 2000 | 14.1% | 15.6% | 1.6% | 0.237 | 5 / 13 | 1.135 |

Per-run numerical coverage, interventions, sources and real-reference results: [results.json](results.json).
