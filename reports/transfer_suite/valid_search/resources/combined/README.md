# Solvability search

Seed 0, live weights, unchanged thresholds; 24 measurements and five final passing checks. These are inspected development tasks. Different training budgets and capacities are explicit. A per-task witness establishes solvability, not a shared default.

| Candidate | Task | Sustained | Confirmed step | Budget | Seconds | Final failing metrics |
| --- | --- | --- | ---: | ---: | ---: | --- |
| recipe_p512_b128 | vector_unequal_mass | FAIL | None | 1200 | 9.10 | component_covariance_error=3.306 |
| recipe_p512_b128 | vector_unequal_width | PASS | 700 | 1200 | 7.55 | — |
| recipe_p512_b128 | vector_overlap | PASS | 1100 | 1200 | 8.03 | — |
| recipe_p512_b256 | vector_unequal_mass | FAIL | None | 1200 | 9.22 | component_covariance_error=3.671 |
| recipe_p512_b256 | vector_unequal_width | FAIL | None | 1200 | 8.26 | component_covariance_error=1.597 |
| recipe_p512_b256 | vector_overlap | PASS | 400 | 1200 | 7.31 | — |
