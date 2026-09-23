# Shared gradient-update adaptation

All candidates retain one common recipe and fixed architectures. The same role-blind equation caps every tensor update on every task. EMA is separate; only19/19 live sustained qualifies.

| Candidate | Sustained live | Attempted /19 | Overall |
| --- | ---: | ---: | --- |
| lr00425_prior2 (archived) | 15 |19/19 | FAIL |
| relative_cap_01 | 1 | 6/19 | INCOMPLETE |
| relative_cap_025 | 1 | 6/19 | INCOMPLETE |
| relative_cap_05 | 12 | 19/19 | FAIL |

| Candidate | Task | Live | Final passing checks | EMA |
| --- | --- | --- | ---: | --- |
| relative_cap_01 | ae_gan_hold | PASS | 17 | N/A |
| relative_cap_01 | img_bars4 | FAIL | 0 | PASS |
| relative_cap_01 | mode_hold | FAIL | 0 | N/A |
| relative_cap_01 | vector_unequal_mass | FAIL | 0 | FAIL |
| relative_cap_01 | vector_unequal_width | FAIL | 0 | FAIL |
| relative_cap_01 | vector_overlap | FAIL | 0 | PASS |
| relative_cap_025 | ae_gan_hold | PASS | 21 | N/A |
| relative_cap_025 | img_bars4 | FAIL | 0 | PASS |
| relative_cap_025 | mode_hold | FAIL | 0 | N/A |
| relative_cap_025 | vector_unequal_mass | FAIL | 0 | FAIL |
| relative_cap_025 | vector_unequal_width | FAIL | 0 | FAIL |
| relative_cap_025 | vector_overlap | FAIL | 4 | PASS |
| relative_cap_05 | ae_gan_hold | PASS | 22 | N/A |
| relative_cap_05 | img_bars4 | PASS | 9 | FAIL |
| relative_cap_05 | mode_hold | FAIL | 0 | N/A |
| relative_cap_05 | vector_unequal_mass | FAIL | 0 | FAIL |
| relative_cap_05 | vector_unequal_width | FAIL | 0 | FAIL |
| relative_cap_05 | vector_overlap | FAIL | 2 | PASS |
| relative_cap_05 | two_pole | PASS | 11 | N/A |
| relative_cap_05 | trajectory | FAIL | 0 | N/A |
| relative_cap_05 | residual_student | PASS | 22 | N/A |
| relative_cap_05 | unipolar | PASS | 18 | N/A |
| relative_cap_05 | cover_leftover | PASS | 14 | N/A |
| relative_cap_05 | unused_token_hold | PASS | 13 | N/A |
| relative_cap_05 | mid_scale_identity | PASS | 17 | N/A |
| relative_cap_05 | vector_two_broad | PASS | 17 | PASS |
| relative_cap_05 | vector_anisotropic | FAIL | 1 | FAIL |
| relative_cap_05 | vector_spiral | PASS | 24 | PASS |
| relative_cap_05 | img_stripes2 | FAIL | 4 | PASS |
| relative_cap_05 | img_blobs4 | PASS | 7 | FAIL |
| relative_cap_05 | img_intensity2 | PASS | 8 | PASS |
