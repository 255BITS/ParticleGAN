# Vector fixed-reference calibration (protocol v2)

Re-scored from retained v1 training; no new training, no reserved evaluation. All rows have 24/24 observations. Minimum eigenvalue >=0.15 applies to every separated mixture. Timing is original CPU1 elapsed time, including evaluation.

| Task | Control | Tier | Final | Sustained | Tail / 24 | Seconds | Failed final bounds |
|---|---|---|---:|---:|---:|---:|---|
| vector_anisotropic | fixed_cosine | ranking | PASS | PASS | 15 | 5.52 | none |
| vector_narrow | fixed_cosine | diagnostic | FAIL | FAIL | 0 | 7.71 | sw1_normalized=0.4988 (<=0.18), mass_tv=0.5 (<=0.15), component_covariance_error=36.84 (<=0.85), component_min_eigen_ratio=0 (>=0.15) |
| vector_overlap | fixed_cosine | ranking | PASS | FAIL | 4 | 5.35 | none |
| vector_scale_drift | fixed_cosine | diagnostic | PASS | PASS | 22 | 6.91 | none |
| vector_spiral | fixed_cosine | ranking | PASS | PASS | 23 | 6.76 | none |
| vector_two_broad | fixed_cosine | ranking | PASS | PASS | 19 | 6.35 | none |
| vector_unequal_mass | fixed_constant | ranking | FAIL | FAIL | 0 | 6.30 | component_covariance_error=2.892 (<=0.85) |
| vector_unequal_mass | fixed_cosine | ranking | FAIL | FAIL | 0 | 5.57 | component_covariance_error=5.8 (<=0.85) |
| vector_unequal_mass | fixed_cosine_r1r2_0p1 | ranking | FAIL | FAIL | 0 | 5.28 | component_covariance_error=2.738 (<=0.85) |
| vector_unequal_width | fixed_constant | ranking | FAIL | FAIL | 0 | 5.39 | component_covariance_error=4.442 (<=0.85) |
| vector_unequal_width | fixed_cosine | ranking | FAIL | FAIL | 0 | 5.38 | component_covariance_error=7.809 (<=0.85) |
| vector_unequal_width | fixed_cosine_r1r2_0p1 | ranking | FAIL | FAIL | 0 | 5.33 | mass_tv=0.1567 (<=0.15), component_covariance_error=16.68 (<=0.85) |
