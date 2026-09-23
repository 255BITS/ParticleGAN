# Separate toy: longer training on eight Gaussian clusters

Measure whether the same recipe learns each cluster's mass and spread with more training. Each budget is a separate full run with cosine scheduled over that budget. It is not a resumed checkpoint or proof of stability between measurements. These results do not enter the ten-toy practical count.

| Formulation | Updates | Sustained live | Final good samples | Covariance error (≤0.85) | Final passing checks | Confirmed step |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| rp_logistic_bcap3 | 2,400 | FAIL | 99.3% | 0.973 | 0 | — |
| rp_logistic_bcap3 | 4,800 | FAIL | 97.6% | 0.451 | 4 | — |
| rp_logistic_bcap3 | 7,200 | PASS | 99.2% | 0.458 | 6 | 6900 |
| rp_logistic_bcap10 | 2,400 | FAIL | 93.5% | 7.164 | 0 | — |

Only the budget changes between the b_cap3 rows; the builder verifies identical loss, regularization, optimizer settings, architecture, target, batch and particle count. b_cap10 has no matching extended-budget run in this comparison.

Every PASS still requires all live metrics for the final five of 24 measurements. EMA remains separate. [Exact settings, verdicts and source artifacts](leaderboard.json).
