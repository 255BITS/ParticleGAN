# Formulation × architecture results

One entry per formulation. PASS means at least one listed architecture sustains every live metric; each case counts once. Architecture failures remain visible. Architecture cells cannot mix numerical formulations, optimizer settings, data or training budgets.

## rp_logistic_bcap3

Required: 9/9. Practical: 7/16. Eligible on required tests: True.

| Problem / condition | Supported | Architecture observations |
| --- | --- | --- |
| vector_two_broad | PASS | original architecture: PASS |
| vector_unequal_mass | FAIL | original architecture: FAIL |
| vector_unequal_width | FAIL | original architecture: FAIL |
| vector_anisotropic | PASS | original architecture: PASS |
| vector_overlap | FAIL | original architecture: FAIL |
| vector_spiral | PASS | original architecture: PASS |
| stress_fast_critic | FAIL | original architecture: FAIL |
| stress_slow_critic | FAIL | original architecture: FAIL |
| stress_small_batch | FAIL | original architecture: FAIL |
| stress_large_critic | FAIL | original architecture: FAIL |
| stress_long_horizon | FAIL | original architecture: FAIL |
| stress_nominal_ring | FAIL | bcap3, nominal ring: FAIL |
| img_stripes2 | PASS | baseline: FAIL; residual16: PASS; transpose24: FAIL; residual12: PASS; transpose16: FAIL |
| img_bars4 | PASS | baseline: FAIL; residual16: PASS; transpose24: FAIL; residual12: FAIL; transpose16: FAIL |
| img_blobs4 | PASS | baseline: PASS; residual16: PASS; transpose24: FAIL; residual12: FAIL; transpose16: FAIL |
| img_intensity2 | PASS | baseline: FAIL; residual16: PASS; transpose24: FAIL; residual12: PASS; transpose16: FAIL |

## rp_logistic_bcap10

Required: 8/9. Practical: 7/16. Eligible on required tests: False.

| Problem / condition | Supported | Architecture observations |
| --- | --- | --- |
| vector_two_broad | PASS | cap10 original architecture: PASS |
| vector_unequal_mass | FAIL | cap10 original architecture: FAIL |
| vector_unequal_width | PASS | cap10 original architecture: PASS |
| vector_anisotropic | PASS | cap10 original architecture: PASS |
| vector_overlap | FAIL | cap10 original architecture: FAIL |
| vector_spiral | PASS | cap10 original architecture: PASS |
| stress_fast_critic | FAIL | cap10 original architecture: FAIL |
| stress_slow_critic | FAIL | cap10 original architecture: FAIL |
| stress_small_batch | FAIL | cap10 original architecture: FAIL |
| stress_large_critic | FAIL | cap10 original architecture: FAIL |
| stress_long_horizon | FAIL | cap10 original architecture: FAIL |
| stress_nominal_ring | FAIL | bcap10, nominal ring: FAIL |
| img_stripes2 | PASS | residual16_cap10: PASS |
| img_bars4 | FAIL | residual16_cap10: FAIL |
| img_blobs4 | PASS | residual16_cap10: PASS |
| img_intensity2 | PASS | residual16_cap10: PASS |

The image variants include changes to G as well as D. Discriminator width changes alone, generator changes, and their combination are explicit in the machine-readable architecture records. These are inspected development cases with one initialization; architecture support is not a fresh-transfer result.

[Exact axes, metrics and source artifacts](leaderboard.json).
