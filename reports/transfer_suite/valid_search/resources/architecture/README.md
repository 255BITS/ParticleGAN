# Solvability search

Seed 0, live weights, unchanged thresholds; 24 measurements and five final passing checks. These are inspected development tasks. Different training budgets and capacities are explicit. A per-task witness establishes solvability, not a shared default.

| Candidate | Task | Sustained | Confirmed step | Budget | Seconds | Final failing metrics |
| --- | --- | --- | ---: | ---: | ---: | --- |
| p512_b128_d128_l3_f4 | vector_unequal_mass | FAIL | None | 1200 | 9.53 | component_covariance_error=1.965 |
| p512_b128_d128_l3_f4 | vector_unequal_width | FAIL | None | 1200 | 8.59 | component_covariance_error=33.02 |
| p512_b128_d128_l3_f4 | vector_overlap | PASS | 1100 | 1200 | 8.90 | — |
