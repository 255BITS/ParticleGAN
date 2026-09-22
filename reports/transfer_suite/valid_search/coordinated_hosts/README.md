# Shared relative optimizer recipe: required and image verification

Rp logistic,b_cap3/κ1.25,prior regularization.05,no L2. Adam(0,.999); relative to each host baseline: G LR×.75,D LR×1,particle LR×2.25. Architecture is unchanged for required hosts and residual16 for images. Seed0, complete24 live checks, final5 passing. EMA separate.

| Host | Sustained live | Confirmed step | Final failing metrics |
| --- | --- | ---: | --- |
| two_pole | PASS | 57 | — |
| trajectory | FAIL | None | identity_mse |
| residual_student | PASS | 134 | — |
| unipolar | PASS | 200 | — |
| ae_gan_hold | PASS | 125 | — |
| cover_leftover | PASS | 634 | — |
| unused_token_hold | PASS | 150 | — |
| mid_scale_identity | PASS | 667 | — |
| mode_hold | FAIL | None | modes, hq |
| img_stripes2 | PASS | 250 | — |
| img_bars4 | FAIL | None | modes |
| img_blobs4 | FAIL | None | — |
| img_intensity2 | PASS | 575 | — |
