# Solvability search

Seed 0, live weights, unchanged thresholds; 24 measurements and five final passing checks. These are inspected development tasks. Different training budgets and capacities are explicit. A per-task witness establishes solvability, not a shared default.

| Candidate | Task | Sustained | Confirmed step | Budget | Seconds | Final failing metrics |
| --- | --- | --- | ---: | ---: | ---: | --- |
| linear_skip_d64_beta6 | vector_unequal_mass | FAIL | None | 1200 | 7.24 | mass_tv=0.1511, component_covariance_error=1.2 |
| linear_skip_d64_beta10 | vector_unequal_mass | FAIL | None | 1200 | 6.25 | component_covariance_error=4.442, component_min_eigen_ratio=0.04748 |
| linear_skip_d96_beta5 | vector_unequal_mass | PASS | 1150 | 1200 | 6.85 | — |
| linear_skip_d96_beta6 | vector_unequal_mass | FAIL | None | 1200 | 6.90 | component_min_eigen_ratio=0.003289 |
