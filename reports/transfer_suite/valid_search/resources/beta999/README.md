# Solvability search

Seed 0, live weights, unchanged thresholds; 24 measurements and five final passing checks. These are inspected development tasks. Different training budgets and capacities are explicit. A per-task witness establishes solvability, not a shared default.

| Candidate | Task | Sustained | Confirmed step | Budget | Seconds | Final failing metrics |
| --- | --- | --- | ---: | ---: | ---: | --- |
| b999_p512_b128 | vector_unequal_mass | FAIL | None | 1200 | 6.86 | component_covariance_error=5.15 |
| b999_p512_b128 | vector_unequal_width | FAIL | None | 1200 | 6.11 | component_covariance_error=1.216 |
| b999_p512_b128 | vector_overlap | PASS | 1000 | 1200 | 6.11 | — |
| b999_p512_b256 | vector_unequal_mass | FAIL | None | 1200 | 6.60 | component_covariance_error=5.472 |
| b999_p512_b256 | vector_unequal_width | FAIL | None | 1200 | 7.08 | component_covariance_error=5.515 |
| b999_p512_b256 | vector_overlap | PASS | 400 | 1200 | 8.96 | — |
