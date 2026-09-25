# Solvability search

Seed 0, live weights, unchanged thresholds; 24 measurements and five final passing checks. These are inspected development tasks. Different training budgets and capacities are explicit. A per-task witness establishes solvability, not a shared default.

| Candidate | Task | Sustained | Confirmed step | Budget | Seconds | Final failing metrics |
| --- | --- | --- | ---: | ---: | ---: | --- |
| cap10_lrhalf | vector_two_broad | PASS | 750 | 1200 | 11.98 | — |
| cap10_lrhalf | vector_unequal_mass | FAIL | None | 1200 | 9.90 | component_covariance_error=1.776 |
| cap10_lrhalf | vector_unequal_width | FAIL | None | 1200 | 9.84 | component_covariance_error=2.137 |
| cap10_lrhalf | vector_anisotropic | FAIL | None | 1200 | 10.02 | component_covariance_error=1.206 |
| cap10_lrhalf | vector_overlap | PASS | 650 | 1200 | 9.27 | — |
| cap10_lrhalf | vector_spiral | PASS | 534 | 1600 | 11.78 | — |
| cap10_adam999 | vector_two_broad | PASS | 500 | 1200 | 9.52 | — |
| cap10_adam999 | vector_unequal_mass | FAIL | None | 1200 | 9.59 | component_covariance_error=5.434, component_min_eigen_ratio=0.003633 |
| cap10_adam999 | vector_unequal_width | FAIL | None | 1200 | 9.51 | component_covariance_error=9.148 |
| cap10_adam999 | vector_anisotropic | FAIL | None | 1200 | 8.30 | component_covariance_error=1.46 |
| cap10_adam999 | vector_overlap | FAIL | None | 1200 | 8.54 | — |
| cap10_adam999 | vector_spiral | PASS | 467 | 1600 | 9.06 | — |
| cap10_prior30 | vector_two_broad | PASS | 400 | 1200 | 6.92 | — |
| cap10_prior30 | vector_unequal_mass | FAIL | None | 1200 | 7.23 | component_covariance_error=3.111, component_min_eigen_ratio=0.000524 |
| cap10_prior30 | vector_unequal_width | FAIL | None | 1200 | 7.24 | component_covariance_error=2.315 |
| cap10_prior30 | vector_anisotropic | PASS | 1200 | 1200 | 6.74 | — |
| cap10_prior30 | vector_overlap | FAIL | None | 1200 | 6.66 | — |
| cap10_prior30 | vector_spiral | PASS | 467 | 1600 | 8.96 | — |
