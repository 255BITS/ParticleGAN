# Completed autonomous memory experiments

Selections use metrics only. Full cold-start circle passes use the existing early-fit diagnostic;
coverage summaries include imperfect trajectories and are not calibrated distribution tests.

| Run | Updates | Full 256 | Full 1024 | Late stopped | Radial RMSE | Passing CW / CCW | Initial spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen_writer | 2000 | 7.8% | n/a | 8.6% | 0.298 | 0 / 10 | 0.675 |
| shared | 2000 | 6.2% | n/a | 3.9% | 0.295 | 4 / 4 | 0.622 |

Per-run numerical coverage, interventions, sources and real-reference results: [results.json](results.json).
