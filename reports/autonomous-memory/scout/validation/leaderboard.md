# Completed autonomous memory experiments

Selections use metrics only. Full cold-start circle passes use the existing early-fit diagnostic;
coverage summaries include imperfect trajectories and are not calibrated distribution tests.

| Run | Updates | Full 256 | Full 1024 | Late stopped | Radial RMSE | Passing CW / CCW | Initial spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| gru_private_5k_all512 | 5000 | 85.2% | 84.0% | 0.2% | 0.070 | 0 / 436 | 1.190 |
| gru_flat_10k_all512 | 10000 | 52.5% | 50.4% | 0.4% | 0.122 | 134 / 135 | 1.183 |
| gru_flat_5k_all512 | 5000 | 35.2% | 33.6% | 0.2% | 0.150 | 101 / 79 | 1.129 |

Per-run numerical coverage, interventions, sources and real-reference results: [results.json](results.json).
