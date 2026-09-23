# Solvability search

Seed 0, live weights, unchanged thresholds; 24 measurements and five final passing checks. These are inspected development tasks. Different training budgets and capacities are explicit. A per-task witness establishes solvability, not a shared default.

| Candidate | Task | Sustained | Confirmed step | Budget | Seconds | Final failing metrics |
| --- | --- | --- | ---: | ---: | ---: | --- |
| axis_softplus5 | vector_two_broad | PASS | 600 | 1200 | 8.33 | — |
| axis_softplus5 | vector_anisotropic | FAIL | None | 1200 | 6.87 | sw1_normalized=0.1844, mass_tv=0.1794, component_covariance_error=1.655 |
| axis_softplus5 | vector_spiral | PASS | 467 | 1600 | 8.03 | — |
| axis_tanh | vector_two_broad | PASS | 600 | 1200 | 5.94 | — |
| axis_tanh | vector_anisotropic | FAIL | None | 1200 | 5.85 | sw1_normalized=0.2191, mass_tv=0.1934, component_covariance_error=1.166 |
| axis_tanh | vector_spiral | PASS | 934 | 1600 | 7.59 | — |
