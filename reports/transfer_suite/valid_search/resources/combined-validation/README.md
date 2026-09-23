# Solvability search

Seed 0, live weights, unchanged thresholds; 24 measurements and five final passing checks. These are inspected development tasks. Different training budgets and capacities are explicit. A per-task witness establishes solvability, not a shared default.

| Candidate | Task | Sustained | Confirmed step | Budget | Seconds | Final failing metrics |
| --- | --- | --- | ---: | ---: | ---: | --- |
| recipe_p512_b128 | vector_two_broad | PASS | 600 | 1200 | 7.13 | — |
| recipe_p512_b128 | vector_anisotropic | FAIL | None | 1200 | 5.82 | component_covariance_error=2.121 |
| recipe_p512_b128 | vector_spiral | PASS | 400 | 1600 | 8.10 | — |
