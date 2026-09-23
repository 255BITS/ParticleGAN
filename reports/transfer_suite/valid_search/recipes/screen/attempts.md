# Solvability search

Seed 0, live weights, unchanged thresholds; 24 measurements and five final passing checks. These are inspected development tasks. Different training budgets and capacities are explicit. A per-task witness establishes solvability, not a shared default.

| Candidate | Task | Sustained | Confirmed step | Budget | Seconds | Final failing metrics |
| --- | --- | --- | ---: | ---: | ---: | --- |
| b999_lr05_p20 | vector_unequal_mass | FAIL | None | 1200 | 10.82 | component_covariance_error=3.524 |
| b999_lr05_p20 | vector_unequal_width | FAIL | None | 1200 | 5.78 | mass_tv=0.1719, component_covariance_error=3.99 |
| b999_lr05_p20 | vector_overlap | PASS | 450 | 1200 | 5.59 | — |
| b999_lr075_p15 | vector_unequal_mass | FAIL | None | 1200 | 5.76 | component_covariance_error=2.642, component_min_eigen_ratio=0.001692 |
| b999_lr075_p15 | vector_unequal_width | FAIL | None | 1200 | 5.61 | mass_tv=0.1729 |
| b999_lr075_p15 | vector_overlap | FAIL | None | 1200 | 6.26 | — |
| b999_lr15_p5 | vector_unequal_mass | FAIL | None | 1200 | 5.80 | component_covariance_error=4.017, component_min_eigen_ratio=0.09447 |
| b999_lr15_p5 | vector_unequal_width | FAIL | None | 1200 | 5.86 | — |
| b999_lr15_p5 | vector_overlap | FAIL | None | 1200 | 6.08 | covariance_error=0.5053 |
| b999_lr075_d3_p20 | vector_unequal_mass | FAIL | None | 1200 | 6.30 | component_covariance_error=11.14 |
| b999_lr075_d3_p20 | vector_unequal_width | FAIL | None | 1200 | 5.84 | component_covariance_error=10.25 |
| b999_lr075_d3_p20 | vector_overlap | PASS | 1200 | 1200 | 5.69 | — |
| b999_d3_p10 | vector_unequal_mass | FAIL | None | 1200 | 5.87 | sw1_normalized=0.229, mass_tv=0.1516, component_covariance_error=11.68 |
| b999_d3_p10 | vector_unequal_width | FAIL | None | 1200 | 6.29 | component_covariance_error=3.972 |
| b999_d3_p10 | vector_overlap | FAIL | None | 1200 | 7.47 | mean_error=0.1732 |
| b999_d075_p20 | vector_unequal_mass | FAIL | None | 1200 | 6.04 | component_covariance_error=5.718, component_min_eigen_ratio=1.192e-06 |
| b999_d075_p20 | vector_unequal_width | FAIL | None | 1200 | 5.98 | component_covariance_error=13.74 |
| b999_d075_p20 | vector_overlap | FAIL | None | 1200 | 5.88 | sw1_normalized=0.2825, mean_error=0.4109, covariance_error=0.4756 |
| b99_d3_p30 | vector_unequal_mass | FAIL | None | 1200 | 5.91 | component_covariance_error=2.743 |
| b99_d3_p30 | vector_unequal_width | FAIL | None | 1200 | 6.25 | component_covariance_error=2.393 |
| b99_d3_p30 | vector_overlap | FAIL | None | 1200 | 5.98 | — |
| b995_lr075_d2_p20 | vector_unequal_mass | FAIL | None | 1200 | 6.22 | component_covariance_error=6.902, component_min_eigen_ratio=0.0001556 |
| b995_lr075_d2_p20 | vector_unequal_width | PASS | 1100 | 1200 | 6.44 | — |
| b995_lr075_d2_p20 | vector_overlap | FAIL | None | 1200 | 7.58 | — |
| b999_lr075_d2_p30 | vector_unequal_mass | PASS | 1150 | 1200 | 7.00 | — |
| b999_lr075_d2_p30 | vector_unequal_width | FAIL | None | 1200 | 7.57 | component_covariance_error=1.766 |
| b999_lr075_d2_p30 | vector_overlap | PASS | 1100 | 1200 | 7.18 | — |
| mom05_lr05_d2_p20 | vector_unequal_mass | FAIL | None | 1200 | 7.03 | component_covariance_error=0.8932, component_min_eigen_ratio=0.001026 |
| mom05_lr05_d2_p20 | vector_unequal_width | FAIL | None | 1200 | 7.36 | mass_tv=0.1594, component_covariance_error=1.28, component_min_eigen_ratio=0.04108 |
| mom05_lr05_d2_p20 | vector_overlap | FAIL | None | 1200 | 6.97 | — |
| mom05_d2_p10 | vector_unequal_mass | FAIL | None | 1200 | 7.03 | component_covariance_error=3.723, component_min_eigen_ratio=0.05356 |
| mom05_d2_p10 | vector_unequal_width | FAIL | None | 1200 | 7.42 | component_min_eigen_ratio=0.004619 |
| mom05_d2_p10 | vector_overlap | FAIL | None | 1200 | 7.29 | — |
| mom09_lr03_d3_p20 | vector_unequal_mass | FAIL | None | 1200 | 8.32 | sw1_normalized=0.2518, hq=0.5913, component_covariance_error=8.901, component_min_eigen_ratio=0, min_mass_ratio=0 |
| mom09_lr03_d3_p20 | vector_unequal_width | FAIL | None | 1200 | 6.95 | sw1_normalized=0.1844, mass_tv=0.1506, component_covariance_error=0.9589, component_min_eigen_ratio=0.01005 |
| mom09_lr03_d3_p20 | vector_overlap | FAIL | None | 1200 | 10.36 | mean_error=0.1943 |
| g2_b999_p20 | vector_unequal_mass | FAIL | None | 1200 | 8.57 | component_covariance_error=4.35 |
| g2_b999_p20 | vector_unequal_width | FAIL | None | 1200 | 7.91 | component_covariance_error=12.15 |
| g2_b999_p20 | vector_overlap | PASS | 400 | 1200 | 7.51 | — |
| g2_b999_lr15_d1 | vector_unequal_mass | FAIL | None | 1200 | 7.41 | component_covariance_error=8.347 |
| g2_b999_lr15_d1 | vector_unequal_width | FAIL | None | 1200 | 6.47 | component_covariance_error=7.591 |
| g2_b999_lr15_d1 | vector_overlap | PASS | 1050 | 1200 | 6.60 | — |
| g2_b999_lr2_d075 | vector_unequal_mass | FAIL | None | 1200 | 6.51 | component_covariance_error=5.226, component_min_eigen_ratio=-5.588e-09 |
| g2_b999_lr2_d075 | vector_unequal_width | FAIL | None | 1200 | 6.09 | component_covariance_error=9.189 |
| g2_b999_lr2_d075 | vector_overlap | PASS | 1150 | 1200 | 6.09 | — |
| g3_b999_lr15_p20 | vector_unequal_mass | FAIL | None | 1200 | 6.28 | component_covariance_error=4.133 |
| g3_b999_lr15_p20 | vector_unequal_width | FAIL | None | 1200 | 6.88 | component_covariance_error=8.561 |
| g3_b999_lr15_p20 | vector_overlap | PASS | 700 | 1200 | 5.76 | — |
| d2_b999_lr075_d3_p20 | vector_unequal_mass | FAIL | None | 1200 | 5.22 | component_covariance_error=3.708, component_min_eigen_ratio=0.001456 |
| d2_b999_lr075_d3_p20 | vector_unequal_width | FAIL | None | 1200 | 5.43 | component_covariance_error=7.845 |
| d2_b999_lr075_d3_p20 | vector_overlap | FAIL | None | 1200 | 5.47 | — |
| b999_lr075_d3_p5 | vector_unequal_mass | FAIL | None | 1200 | 7.24 | component_covariance_error=1.184, component_min_eigen_ratio=0.01699 |
| b999_lr075_d3_p5 | vector_unequal_width | FAIL | None | 1200 | 8.08 | sw1_normalized=0.2192, mass_tv=0.2056, component_covariance_error=29.89 |
| b999_lr075_d3_p5 | vector_overlap | PASS | 750 | 1200 | 7.95 | — |
