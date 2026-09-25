# Sparse-controller action disagreement on measured learner states

Posthoc heuristic commands on saved selected test traces; zero simulator resets/steps and no model fitting. Overall metrics weight transitions equally. Episode means and outcome groups are in the JSON.

| Controller | Expert-state physical MSE | Learner-state physical MSE | Episode-mean physical MSE | Main agreement | Side agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| probes | 0.07524 | 0.40936 | 0.31135 | 79.6% | 36.1% |
| auxiliary | 0.05669 | 0.71665 | 0.55555 | 53.8% | 30.3% |

| Controller / group | Records | Physical MSE | Side missed / expert on | Side extra / expert off | Side wrong direction / expert on |
| --- | ---: | ---: | ---: | ---: | ---: |
| probes / overall | 15027 | 0.40936 | 90.8% | 16.8% | 2.5% |
| probes / first20 | 1000 | 0.23823 | 75.4% | 3.7% | 0.7% |
| probes / later_flight | 3663 | 0.52343 | 91.4% | 11.3% | 3.6% |
| probes / approach | 6451 | 0.57062 | 92.8% | 6.3% | 2.2% |
| probes / contact | 3913 | 0.08047 | — | 20.8% | — |
| auxiliary / overall | 19374 | 0.71665 | 92.9% | 9.6% | 1.6% |
| auxiliary / first20 | 1000 | 0.25158 | 78.3% | 1.7% | 0.0% |
| auxiliary / later_flight | 7637 | 0.95960 | 95.4% | 9.3% | 1.4% |
| auxiliary / approach | 7121 | 0.83307 | 92.4% | 10.8% | 1.9% |
| auxiliary / contact | 3616 | 0.10287 | — | 9.7% | — |
| auxiliary / outcome_out_of_bounds | 5534 | 1.06975 | 95.3% | 16.4% | 1.7% |

Current contacts first; otherwise approach y<0.25, otherwise flight. Later means zero-based step >=20. Outcome groups use eventual terminal outcome, retrospectively.

Controllers visit different states; this is not a matched-state causal comparison.
Heuristic recommendations are posthoc reference actions, not demonstrated recovery or optimal actions.
No labels from this diagnosis were used in training, preprocessing, or checkpoint selection.
