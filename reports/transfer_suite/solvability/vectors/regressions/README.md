# Solvability search

Seed 0, live weights, unchanged thresholds; 24 measurements and five final passing checks. These are inspected development tasks. Different training budgets and capacities are explicit. A per-task witness establishes solvability, not a shared default.

| Candidate | Task | Sustained | Confirmed step | Budget | Seconds | Final failing metrics |
| --- | --- | --- | ---: | ---: | ---: | --- |
| lr_half | vector_two_broad | FAIL | None | 1200 | 7.57 | component_covariance_error=1.693 |
| lr_half | vector_anisotropic | FAIL | None | 1200 | 6.27 | component_covariance_error=0.8971 |
| lr_half | vector_spiral | PASS | 534 | 1600 | 7.95 | — |
| cap10 | vector_two_broad | PASS | 450 | 1200 | 6.00 | — |
| cap10 | vector_anisotropic | PASS | 850 | 1200 | 6.14 | — |
| cap10 | vector_spiral | PASS | 400 | 1600 | 7.90 | — |
| adam999 | vector_two_broad | PASS | 400 | 1200 | 5.86 | — |
| adam999 | vector_anisotropic | PASS | 700 | 1200 | 7.13 | — |
| adam999 | vector_spiral | PASS | 400 | 1600 | 8.69 | — |
