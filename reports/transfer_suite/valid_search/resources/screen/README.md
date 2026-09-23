# Solvability search

Seed 0, live weights, unchanged thresholds; 24 measurements and five final passing checks. These are inspected development tasks. Different training budgets and capacities are explicit. A per-task witness establishes solvability, not a shared default.

| Candidate | Task | Sustained | Confirmed step | Budget | Seconds | Final failing metrics |
| --- | --- | --- | ---: | ---: | ---: | --- |
| p512_b128 | vector_unequal_mass | FAIL | None | 1200 | 7.41 | component_covariance_error=5.963 |
| p512_b128 | vector_unequal_width | PASS | 950 | 1200 | 5.67 | — |
| p512_b128 | vector_overlap | FAIL | None | 1200 | 5.60 | — |
| p512_b256 | vector_unequal_mass | FAIL | None | 1200 | 7.17 | component_covariance_error=0.9129 |
| p512_b256 | vector_unequal_width | FAIL | None | 1200 | 6.37 | component_covariance_error=2.552 |
| p512_b256 | vector_overlap | PASS | 1200 | 1200 | 6.40 | — |
| p512_b512 | vector_unequal_mass | FAIL | None | 1200 | 7.82 | component_covariance_error=3.578 |
| p512_b512 | vector_unequal_width | FAIL | None | 1200 | 9.25 | — |
| p512_b512 | vector_overlap | PASS | 900 | 1200 | 8.83 | — |
| p1024_b256 | vector_unequal_mass | FAIL | None | 1200 | 6.64 | component_covariance_error=1.25 |
| p1024_b256 | vector_unequal_width | FAIL | None | 1200 | 6.54 | component_covariance_error=3.155 |
| p1024_b256 | vector_overlap | FAIL | None | 1200 | 6.55 | mean_error=0.1645 |
| p1024_b512 | vector_unequal_mass | FAIL | None | 1200 | 8.46 | component_covariance_error=4.591 |
| p1024_b512 | vector_unequal_width | FAIL | None | 1200 | 8.22 | component_covariance_error=6.086 |
| p1024_b512 | vector_overlap | PASS | 400 | 1200 | 9.34 | — |
| p2048_b128 | vector_unequal_mass | FAIL | None | 1200 | 7.15 | component_covariance_error=11.31 |
| p2048_b128 | vector_unequal_width | FAIL | None | 1200 | 7.27 | component_covariance_error=13.35 |
| p2048_b128 | vector_overlap | FAIL | None | 1200 | 7.86 | sw1_normalized=0.1832, mean_error=0.2202, covariance_error=0.4555 |
| p2048_b256 | vector_unequal_mass | FAIL | None | 1200 | 8.23 | component_covariance_error=13.62 |
| p2048_b256 | vector_unequal_width | FAIL | None | 1200 | 8.62 | component_covariance_error=10.09 |
| p2048_b256 | vector_overlap | FAIL | None | 1200 | 8.06 | — |
| p2048_b512 | vector_unequal_mass | FAIL | None | 1200 | 9.69 | component_covariance_error=3.55 |
| p2048_b512 | vector_unequal_width | FAIL | None | 1200 | 10.48 | component_covariance_error=2.151 |
| p2048_b512 | vector_overlap | PASS | 800 | 1200 | 11.92 | — |
| p4096_b128 | vector_unequal_mass | FAIL | None | 1200 | 7.55 | component_covariance_error=5.026 |
| p4096_b128 | vector_unequal_width | FAIL | None | 1200 | 11.16 | component_covariance_error=6.497 |
| p4096_b128 | vector_overlap | FAIL | None | 1200 | 9.48 | — |
| p4096_b256 | vector_unequal_mass | FAIL | None | 1200 | 11.48 | component_covariance_error=1.169 |
| p4096_b256 | vector_unequal_width | FAIL | None | 1200 | 10.44 | component_covariance_error=10.94 |
| p4096_b256 | vector_overlap | FAIL | None | 1200 | 9.20 | mean_error=0.1513 |
| p4096_b512 | vector_unequal_mass | FAIL | None | 1200 | 11.11 | component_covariance_error=1.688 |
| p4096_b512 | vector_unequal_width | FAIL | None | 1200 | 10.03 | component_covariance_error=4.86 |
| p4096_b512 | vector_overlap | PASS | 400 | 1200 | 12.16 | — |
