# Original 22-toy winner: CUDA control

CPU recorded evidence: **22/22 PASS**, independently regraded.
CUDA preserved-recipe control: **16/22 PASS, 6 FAIL**. All 22 were executed; no unsupported cases.

| Toy | CUDA verdict |
|---|---|
| two_pole | PASS |
| mode_hold | FAIL |
| unipolar | PASS |
| mid_scale_identity | PASS |
| cover_leftover | PASS |
| trajectory | FAIL |
| residual_student | PASS |
| img_stripes2 | PASS |
| img_bars4 | FAIL |
| vector_overlap | PASS |
| img_blobs4 | FAIL |
| img_intensity2 | FAIL |
| vector_unequal_mass | FAIL |
| vector_unequal_width | PASS |
| vector_two_broad | PASS |
| vector_anisotropic | PASS |
| vector_spiral | PASS |
| ae_gan_hold | PASS |
| unused_token_hold | PASS |
| grid100 | PASS |
| rotated100 | PASS |
| staggered100 | PASS |

All three native 100-mode problems pass both coverage and accuracy gates.
The GPU ring ends at seven modes / HQ .997802734375.
This control retains learning-rate decay and the original auxiliary host losses.

[Protocol and differences](README.md) · [GPU audit](audit.json) · [CPU regrade](cpu-regrade.json)
