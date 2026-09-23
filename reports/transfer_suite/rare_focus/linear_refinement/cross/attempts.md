# Solvability search

Seed 0, live weights, unchanged thresholds; 24 measurements and five final passing checks. These are inspected development tasks. Different training budgets and capacities are explicit. A per-task witness establishes solvability, not a shared default.

| Candidate | Task | Sustained | Confirmed step | Budget | Seconds | Final failing metrics |
| --- | --- | --- | ---: | ---: | ---: | --- |
| linear_skip_d96_beta5 | vector_two_broad | PASS | 700 | 1200 | 7.83 | — |
| linear_skip_d96_beta5 | vector_unequal_width | FAIL | None | 1200 | 7.08 | component_covariance_error=8.703 |
| linear_skip_d96_beta5 | vector_anisotropic | FAIL | None | 1200 | 6.93 | component_covariance_error=0.8924 |
| linear_skip_d96_beta5 | vector_overlap | FAIL | None | 1200 | 6.84 | covariance_error=0.5003 |
| linear_skip_d96_beta5 | vector_spiral | PASS | 400 | 1600 | 8.68 | — |
