# Solvability search

Seed 0, live weights, unchanged thresholds; 24 measurements and five final passing checks. These are inspected development tasks. Different training budgets and capacities are explicit. A per-task witness establishes solvability, not a shared default.

| Candidate | Task | Sustained | Confirmed step | Budget | Seconds | Final failing metrics |
| --- | --- | --- | ---: | ---: | ---: | --- |
| axis_silu | vector_unequal_mass | FAIL | None | 1200 | 6.99 | component_covariance_error=1.44, component_min_eigen_ratio=0.000204 |
| axis_silu | vector_unequal_width | FAIL | None | 1200 | 6.72 | component_covariance_error=0.893 |
| axis_silu | vector_overlap | FAIL | None | 1200 | 6.23 | — |
| axis_softplus1 | vector_unequal_mass | FAIL | None | 1200 | 6.37 | component_covariance_error=3.889, component_min_eigen_ratio=0, min_mass_ratio=0.1953 |
| axis_softplus1 | vector_unequal_width | FAIL | None | 1200 | 6.30 | hq=0.5618, component_covariance_error=60.62 |
| axis_softplus1 | vector_overlap | FAIL | None | 1200 | 6.05 | sw1_normalized=0.2034, mean_error=0.2596 |
| axis_softplus5 | vector_unequal_mass | FAIL | None | 1200 | 6.23 | sw1_normalized=0.1819, mass_tv=0.1667 |
| axis_softplus5 | vector_unequal_width | PASS | 1100 | 1200 | 6.45 | — |
| axis_softplus5 | vector_overlap | FAIL | None | 1200 | 6.41 | — |
| axis_tanh | vector_unequal_mass | FAIL | None | 1200 | 5.95 | component_covariance_error=9.801, component_min_eigen_ratio=0.009287 |
| axis_tanh | vector_unequal_width | FAIL | None | 1200 | 6.22 | component_covariance_error=9.523 |
| axis_tanh | vector_overlap | PASS | 1150 | 1200 | 6.20 | — |
| oriented4_silu | vector_unequal_mass | FAIL | None | 1200 | 6.40 | component_covariance_error=5.955, component_min_eigen_ratio=0.0005364 |
| oriented4_silu | vector_unequal_width | FAIL | None | 1200 | 6.06 | component_covariance_error=2.047 |
| oriented4_silu | vector_overlap | FAIL | None | 1200 | 5.88 | mean_error=0.1851 |
| oriented8_silu | vector_unequal_mass | FAIL | None | 1200 | 5.89 | component_min_eigen_ratio=0, min_mass_ratio=0 |
| oriented8_silu | vector_unequal_width | FAIL | None | 1200 | 5.89 | sw1_normalized=0.3153, mass_tv=0.3198, component_min_eigen_ratio=0 |
| oriented8_silu | vector_overlap | FAIL | None | 1200 | 5.95 | mean_error=0.1581 |
| oriented8_softplus1 | vector_unequal_mass | FAIL | None | 1200 | 6.15 | component_min_eigen_ratio=0.007915 |
| oriented8_softplus1 | vector_unequal_width | FAIL | None | 1200 | 6.42 | sw1_normalized=0.2823, mass_tv=0.4265, component_covariance_error=3.277, component_min_eigen_ratio=0 |
| oriented8_softplus1 | vector_overlap | FAIL | None | 1200 | 6.84 | sw1_normalized=0.2096, mean_error=0.3063 |
| oriented8_tanh | vector_unequal_mass | FAIL | None | 1200 | 6.29 | sw1_normalized=0.4838, mass_tv=0.3542, component_covariance_error=9.494 |
| oriented8_tanh | vector_unequal_width | FAIL | None | 1200 | 6.34 | sw1_normalized=0.2743, mass_tv=0.2568, component_covariance_error=38.41 |
| oriented8_tanh | vector_overlap | PASS | 1200 | 1200 | 6.61 | — |
