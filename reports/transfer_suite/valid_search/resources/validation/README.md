# Solvability search

Seed 0, live weights, unchanged thresholds; 24 measurements and five final passing checks. These are inspected development tasks. Different training budgets and capacities are explicit. A per-task witness establishes solvability, not a shared default.

| Candidate | Task | Sustained | Confirmed step | Budget | Seconds | Final failing metrics |
| --- | --- | --- | ---: | ---: | ---: | --- |
| p512_b128 | vector_two_broad | PASS | 400 | 1200 | 9.34 | — |
| p512_b128 | vector_anisotropic | PASS | 800 | 1200 | 7.76 | — |
| p512_b128 | vector_spiral | PASS | 467 | 1600 | 9.15 | — |
| p512_b512 | vector_two_broad | PASS | 450 | 1200 | 9.69 | — |
| p512_b512 | vector_anisotropic | FAIL | None | 1200 | 10.89 | component_covariance_error=1.465 |
| p512_b512 | vector_spiral | PASS | 400 | 1600 | 13.32 | — |
