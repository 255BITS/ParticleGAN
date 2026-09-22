# Solvability search

Seed 0, live weights, unchanged thresholds; 24 measurements and five final passing checks. These are inspected development tasks. Different training budgets and capacities are explicit. A per-task witness establishes solvability, not a shared default.

| Candidate | Task | Sustained | Confirmed step | Budget | Seconds | Final failing metrics |
| --- | --- | --- | ---: | ---: | ---: | --- |
| b999_lr075_d2_p30__d128_l3_f2 | vector_unequal_mass | FAIL | None | 1200 | 10.51 | component_covariance_error=3.044 |
| b999_lr075_d2_p30__d128_l3_f2 | vector_unequal_width | FAIL | None | 1200 | 10.67 | component_covariance_error=5.163 |
| b999_lr075_d2_p30__d128_l3_f2 | vector_overlap | FAIL | None | 1200 | 8.84 | — |
| b999_lr075_d2_p30__d64_l4_f3 | vector_unequal_mass | FAIL | None | 1200 | 7.46 | component_covariance_error=0.8978, component_min_eigen_ratio=0.05876 |
| b999_lr075_d2_p30__d64_l4_f3 | vector_unequal_width | FAIL | None | 1200 | 7.20 | component_covariance_error=5.104 |
| b999_lr075_d2_p30__d64_l4_f3 | vector_overlap | PASS | 300 | 1200 | 7.21 | — |
| b999_lr075_d2_p30__d64_l2_f3 | vector_unequal_mass | FAIL | None | 1200 | 5.70 | component_covariance_error=11.12 |
| b999_lr075_d2_p30__d64_l2_f3 | vector_unequal_width | FAIL | None | 1200 | 5.98 | component_covariance_error=4.981 |
| b999_lr075_d2_p30__d64_l2_f3 | vector_overlap | PASS | 400 | 1200 | 6.47 | — |
