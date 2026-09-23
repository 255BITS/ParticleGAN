# Solvability search

Seed 0, live weights, unchanged thresholds; 24 measurements and five final passing checks. These are inspected development tasks. Different training budgets and capacities are explicit. A per-task witness establishes solvability, not a shared default.

| Candidate | Task | Sustained | Confirmed step | Budget | Seconds | Final failing metrics |
| --- | --- | --- | ---: | ---: | ---: | --- |
| interp_cap3 | vector_unequal_mass | FAIL | None | 1200 | 7.49 | component_covariance_error=8.813 |
| interp_cap3 | vector_unequal_width | FAIL | None | 1200 | 5.99 | component_covariance_error=1.809 |
| interp_cap3 | vector_overlap | PASS | 1150 | 1200 | 5.31 | — |
| interp_cap10 | vector_unequal_mass | FAIL | None | 1200 | 5.00 | component_covariance_error=3.133 |
| interp_cap10 | vector_unequal_width | FAIL | None | 1200 | 5.16 | component_covariance_error=1.18 |
| interp_cap10 | vector_overlap | FAIL | None | 1200 | 5.29 | — |
| wgan_gp10 | vector_unequal_mass | FAIL | None | 1200 | 5.00 | sw1_normalized=0.4763, mass_tv=0.1944, hq=0, component_min_eigen_ratio=0, min_mass_ratio=0 |
| wgan_gp10 | vector_unequal_width | FAIL | None | 1200 | 5.20 | sw1_normalized=1.231, mass_tv=0.5, hq=0, component_covariance_error=11.48, component_min_eigen_ratio=0 |
| wgan_gp10 | vector_overlap | FAIL | None | 1200 | 5.04 | sw1_normalized=0.2587 |
| rp_gp10 | vector_unequal_mass | FAIL | None | 1200 | 5.08 | sw1_normalized=1.315, mass_tv=0.9766, hq=0, component_covariance_error=27.35, component_min_eigen_ratio=0, min_mass_ratio=0 |
| rp_gp10 | vector_unequal_width | FAIL | None | 1200 | 5.05 | sw1_normalized=0.4121, mass_tv=0.5, hq=0, component_covariance_error=2.446, component_min_eigen_ratio=0 |
| rp_gp10 | vector_overlap | FAIL | None | 1200 | 4.97 | sw1_normalized=0.3753, mean_error=0.284, covariance_error=0.9124 |
| rp_eikonal1 | vector_unequal_mass | FAIL | None | 1200 | 6.01 | sw1_normalized=0.837, mass_tv=0.3211, hq=0.52, component_covariance_error=24.3, component_min_eigen_ratio=0.0001355, min_mass_ratio=0.2173 |
| rp_eikonal1 | vector_unequal_width | FAIL | None | 1200 | 5.89 | sw1_normalized=0.1902, mass_tv=0.2297, hq=0.8325, component_covariance_error=20.78, component_min_eigen_ratio=0.04252 |
| rp_eikonal1 | vector_overlap | FAIL | None | 1200 | 6.70 | sw1_normalized=0.3217, mean_error=0.4606, covariance_error=0.6213 |
| rp_r1r2_1 | vector_unequal_mass | FAIL | None | 1200 | 7.14 | component_covariance_error=2.72 |
| rp_r1r2_1 | vector_unequal_width | FAIL | None | 1200 | 7.10 | component_covariance_error=2.176 |
| rp_r1r2_1 | vector_overlap | PASS | 900 | 1200 | 6.65 | — |
