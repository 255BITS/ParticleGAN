# Solvability search

Seed 0, live weights, unchanged thresholds; 24 measurements and five final passing checks. These are inspected development tasks. Different training budgets and capacities are explicit. A per-task witness establishes solvability, not a shared default.

| Candidate | Task | Sustained | Confirmed step | Budget | Seconds | Final failing metrics |
| --- | --- | --- | ---: | ---: | ---: | --- |
| long2 | vector_unequal_mass | FAIL | None | 2400 | 14.29 | component_covariance_error=2.437, component_min_eigen_ratio=0.07596 |
| long2 | vector_unequal_width | FAIL | None | 2400 | 13.03 | component_covariance_error=4.307 |
| long2 | vector_overlap | FAIL | None | 2400 | 12.71 | — |
| prior_1 | vector_unequal_mass | FAIL | None | 1200 | 7.30 | component_covariance_error=2.628, component_min_eigen_ratio=0.02123 |
| prior_1 | vector_unequal_width | FAIL | None | 1200 | 7.39 | component_covariance_error=9.43 |
| prior_1 | vector_overlap | FAIL | None | 1200 | 7.59 | sw1_normalized=0.1825, mean_error=0.1937, covariance_error=0.57 |
| prior_0 | vector_unequal_mass | FAIL | None | 1200 | 6.88 | component_covariance_error=5.892, component_min_eigen_ratio=0.04727 |
| prior_0 | vector_unequal_width | FAIL | None | 1200 | 6.19 | sw1_normalized=0.1889, mass_tv=0.1941 |
| prior_0 | vector_overlap | FAIL | None | 1200 | 6.32 | mean_error=0.179 |
| prior_lr1 | vector_unequal_mass | FAIL | None | 1200 | 5.81 | sw1_normalized=0.1954, mass_tv=0.1501, component_covariance_error=9.253 |
| prior_lr1 | vector_unequal_width | FAIL | None | 1200 | 6.25 | component_covariance_error=15.85 |
| prior_lr1 | vector_overlap | FAIL | None | 1200 | 7.29 | mean_error=0.1698 |
| prior_lr30 | vector_unequal_mass | FAIL | None | 1200 | 6.72 | component_min_eigen_ratio=0.0001082 |
| prior_lr30 | vector_unequal_width | FAIL | None | 1200 | 6.91 | component_covariance_error=2.341 |
| prior_lr30 | vector_overlap | FAIL | None | 1200 | 7.39 | — |
| cap1 | vector_unequal_mass | FAIL | None | 1200 | 7.68 | component_covariance_error=4.095 |
| cap1 | vector_unequal_width | FAIL | None | 1200 | 7.59 | mass_tv=0.1597 |
| cap1 | vector_overlap | FAIL | None | 1200 | 6.70 | covariance_error=0.5505 |
| cap10 | vector_unequal_mass | FAIL | None | 1200 | 7.42 | component_covariance_error=4.431 |
| cap10 | vector_unequal_width | PASS | 1200 | 1200 | 8.19 | — |
| cap10 | vector_overlap | FAIL | None | 1200 | 6.65 | — |
| kappa05 | vector_unequal_mass | FAIL | None | 1200 | 6.76 | component_min_eigen_ratio=0.00555 |
| kappa05 | vector_unequal_width | FAIL | None | 1200 | 6.43 | component_covariance_error=7.738 |
| kappa05 | vector_overlap | FAIL | None | 1200 | 6.76 | — |
| lr_half | vector_unequal_mass | FAIL | None | 1200 | 6.75 | component_covariance_error=4.013, component_min_eigen_ratio=0.01807 |
| lr_half | vector_unequal_width | FAIL | None | 1200 | 8.13 | component_covariance_error=2.387 |
| lr_half | vector_overlap | PASS | 1150 | 1200 | 6.48 | — |
| adam999 | vector_unequal_mass | FAIL | None | 1200 | 7.19 | component_covariance_error=3.582, component_min_eigen_ratio=0.004975 |
| adam999 | vector_unequal_width | FAIL | None | 1200 | 8.49 | mass_tv=0.1558, component_covariance_error=1.325 |
| adam999 | vector_overlap | PASS | 1200 | 1200 | 7.12 | — |
| hinge | vector_unequal_mass | FAIL | None | 1200 | 7.44 | component_covariance_error=5.917 |
| hinge | vector_unequal_width | FAIL | None | 1200 | 7.52 | sw1_normalized=0.6477, mass_tv=0.6421, component_min_eigen_ratio=0 |
| hinge | vector_overlap | FAIL | None | 1200 | 7.67 | — |
| vanilla | vector_unequal_mass | FAIL | None | 1200 | 7.61 | component_covariance_error=4.18, component_min_eigen_ratio=0.001886 |
| vanilla | vector_unequal_width | PASS | 1200 | 1200 | 8.99 | — |
| vanilla | vector_overlap | FAIL | None | 1200 | 10.92 | — |
| ra | vector_unequal_mass | FAIL | None | 1200 | 10.92 | component_covariance_error=5.552 |
| ra | vector_unequal_width | FAIL | None | 1200 | 11.12 | component_covariance_error=1.082 |
| ra | vector_overlap | FAIL | None | 1200 | 9.86 | — |
| particles1024 | vector_unequal_mass | FAIL | None | 1200 | 9.41 | component_covariance_error=4.558 |
| particles1024 | vector_unequal_width | FAIL | None | 1200 | 10.12 | component_covariance_error=3.173 |
| particles1024 | vector_overlap | FAIL | None | 1200 | 9.47 | — |
| fourier4 | vector_unequal_mass | FAIL | None | 1200 | 9.14 | component_covariance_error=2.004 |
| fourier4 | vector_unequal_width | FAIL | None | 1200 | 9.41 | hq=0.8323, component_covariance_error=27.22 |
| fourier4 | vector_overlap | FAIL | None | 1200 | 7.80 | — |
