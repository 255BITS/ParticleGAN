# Completed autonomous memory experiments

Selections use metrics only. Full cold-start circle passes use the existing early-fit diagnostic;
coverage summaries include imperfect trajectories and are not calibrated distribution tests.

| Run | Updates | Full 256 | Full 1024 | Late stopped | Radial RMSE | Passing CW / CCW | Initial spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| gru_private_5k | 5000 | 85.2% | 84.4% | 0.0% | 0.072 | 0 / 109 | 1.172 |
| gru_private | 2000 | 55.5% | 54.7% | 0.0% | 0.126 | 0 / 71 | 1.146 |
| gru_flat_10k | 10000 | 53.1% | 51.6% | 0.8% | 0.127 | 31 / 37 | 1.185 |
| gru_private_no_memory | 2000 | 52.3% | 53.1% | 0.8% | 0.119 | 66 / 1 | 1.168 |
| trace_multi_film | 2000 | 50.8% | 49.2% | 0.8% | 0.178 | 0 / 65 | 1.078 |
| gru_multi | 2000 | 44.5% | 43.0% | 0.0% | 0.146 | 0 / 57 | 1.133 |
| gru_freeze_10k | 10000 | 39.8% | 37.5% | 0.8% | 0.141 | 31 / 20 | 1.131 |
| trace_multi_diff | 2000 | 38.3% | 37.5% | 1.6% | 0.164 | 49 / 0 | 1.131 |
| gru_geometry_5k | 5000 | 36.7% | 34.4% | 2.3% | 0.151 | 41 / 6 | 1.007 |
| gru_flat_5k | 5000 | 34.4% | 31.2% | 0.0% | 0.149 | 25 / 19 | 1.122 |
| trace_multi | 2000 | 33.6% | 32.0% | 3.1% | 0.162 | 43 / 0 | 1.228 |
| delay_multi | 2000 | 32.8% | 32.0% | 1.6% | 0.193 | 42 / 0 | 1.099 |
| trace_multi_diff_5k | 5000 | 28.9% | 28.1% | 0.0% | 0.221 | 33 / 4 | 1.178 |
| gru_geometry | 2000 | 27.3% | 30.5% | 3.1% | 0.150 | 4 / 31 | 0.927 |
| gru_flat | 2000 | 14.1% | 15.6% | 1.6% | 0.237 | 5 / 13 | 1.135 |
| gru_frozen | 2000 | 6.2% | 7.0% | 7.8% | 0.272 | 8 / 0 | 1.224 |
| trace_flat | 2000 | 3.1% | 3.9% | 6.2% | 0.254 | 0 / 4 | 0.714 |
| gru_silu | 2000 | 0.0% | 0.0% | 2.3% | 0.245 | 0 / 0 | 1.259 |

Per-run numerical coverage, interventions, sources and real-reference results: [results.json](results.json).
