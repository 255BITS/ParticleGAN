# Formulation × architecture results

One entry per formulation. PASS means at least one listed architecture sustains every live metric; each case counts once. Architecture failures remain visible. Architecture cells cannot mix numerical formulations, optimizer settings, data or training budgets.

Current PR scope: nine required regressions, six data toys and four image toys. The previous 7/16 is now 7/10 because the scope changed, not because any run improved. [Longer training](LONG_TRAINING.md) is a separate toy; [imposed-setting diagnostics](DIAGNOSTICS.md) do not affect this comparison.

## rp_logistic_bcap3

Required: 9/9. Practical: 7/10. Eligible on required tests: True.

| Problem / condition | Supported | Architecture observations |
| --- | --- | --- |
| vector_two_broad | PASS | original architecture: PASS |
| vector_unequal_mass | FAIL | original architecture: FAIL |
| vector_unequal_width | FAIL | original architecture: FAIL |
| vector_anisotropic | PASS | original architecture: PASS |
| vector_overlap | FAIL | original architecture: FAIL |
| vector_spiral | PASS | original architecture: PASS |
| img_stripes2 | PASS | baseline: FAIL; residual16: PASS; transpose24: FAIL; residual12: PASS; transpose16: FAIL |
| img_bars4 | PASS | baseline: FAIL; residual16: PASS; transpose24: FAIL; residual12: FAIL; transpose16: FAIL |
| img_blobs4 | PASS | baseline: PASS; residual16: PASS; transpose24: FAIL; residual12: FAIL; transpose16: FAIL |
| img_intensity2 | PASS | baseline: FAIL; residual16: PASS; transpose24: FAIL; residual12: PASS; transpose16: FAIL |

## rp_logistic_bcap10

Required: 8/9. Practical: 7/10. Eligible on required tests: False.

| Problem / condition | Supported | Architecture observations |
| --- | --- | --- |
| vector_two_broad | PASS | cap10 original architecture: PASS |
| vector_unequal_mass | FAIL | cap10 original architecture: FAIL |
| vector_unequal_width | PASS | cap10 original architecture: PASS |
| vector_anisotropic | PASS | cap10 original architecture: PASS |
| vector_overlap | FAIL | cap10 original architecture: FAIL |
| vector_spiral | PASS | cap10 original architecture: PASS |
| img_stripes2 | PASS | residual16_cap10: PASS |
| img_bars4 | FAIL | residual16_cap10: FAIL |
| img_blobs4 | PASS | residual16_cap10: PASS |
| img_intensity2 | PASS | residual16_cap10: PASS |

The image variants include changes to G as well as D. Discriminator width changes alone, generator changes, and their combination are explicit in the machine-readable architecture records. These are inspected development cases with one initialization; architecture support is not a fresh-transfer result.

[Exact axes, metrics and source artifacts](leaderboard.json).
