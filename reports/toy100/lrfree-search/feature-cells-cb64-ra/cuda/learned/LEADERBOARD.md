# Learned CUDA comparison

All four 2000-update training runs have valid frozen source, initialization, data and checkpoint receipts. All four required replay checks pass exact semantic restoration, every loss tensor and every state section after each of the ten continuation updates.

| Toy at 2000 updates | Precision | Modes /25 | Mass TV | Independent toy gate | Training seconds |
|---|---:|---:|---:|---|---:|
| E22 | 0.7155 | 25 | 0.2845 | FAIL | 72.34 |
| CB64-RA | 0.6233 | 24 | 0.3780 | FAIL | 80.79 |

| MNIST at 2000 updates | Raw embedding FD | Active embedding FD | Active precision | Active recall | Class mass TV | Training seconds |
|---|---:|---:|---:|---:|---:|---:|
| E22 | 6.8522 | 0.5445 | 0.8691 | 0.8472 | 0.0364 | 90.96 |
| CB64-RA | 19.1878 | 1.4888 | 0.8525 | 0.3228 | 0.1244 | 103.26 |

E22 has better measured learned quality and contemporary throughput in both problems. Both MNIST runs cover all ten confident classifier classes, but CB64-RA has substantially lower embedding recall. There is no predeclared MNIST acceptance threshold; embedding distance, precision, recall and class balance remain comparisons. Neither toy run reaches its predeclared precision/mass gate.

The common training-time comparison also favors E22. The 72.34-second toy budget selects E22 update 2000 and CB64-RA update 1750; CB precision is 0.6409 with 23 modes and TV 0.3697. The 90.96-second MNIST budget selects E22 update 2000 and CB64-RA update 1750; CB active FD is 1.0358, recall 0.2446 and class TV 0.0996. CB leaves 1.61 and 2.53 seconds unused due to checkpoint granularity.

CB64-RA's allocated GPU memory peak is lower: 57.2 versus 91.6 MiB on toy, and 155.0 versus 237.5 MiB on MNIST. Its synchronized training is 11.7% and 13.5% slower respectively under the contemporary shared GPU scheduling. These runs establish lower GPU memory use for these two model sizes; they do not establish a CUDA throughput gain or general scaling claim.

The candidate makes 7140 ordinary moves across the learned runs (1958 toy, 5182 MNIST), plus 41 MNIST isolation moves. The ordinary reaction path acts and its saved-state continuation is exact. These correctness/activity findings coexist with the lower learned quality.

Recommendation from this lane: keep E22 as the default and retain CB64-RA as an experimental implementation with traceable CUDA evidence. Its memory improvement does not compensate for the measured quality and speed regressions here. Original frozen portability/native acceptance remains the root report's responsibility; no earlier CPU or archived GPU verdict is inherited by this learned comparison.

Full metrics, normalization/real-control receipts, resource peaks, state fingerprints and artifact identities are in REPORT.md, results.json and artifact-manifest.json.
