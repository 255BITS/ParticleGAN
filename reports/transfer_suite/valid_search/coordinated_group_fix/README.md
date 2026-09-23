# Shared relative optimizer recipe: required and image verification

Rp logistic,b_cap3/κ1.25,prior regularization.05,no L2. Adam(0,.999); relative to each host baseline: G LR×.75,D LR×1,particle LR×2.25. Architecture is unchanged for required hosts and residual16 for images. Seed0, complete24 live checks, final5 passing. EMA separate.

| Host | Sustained live | Confirmed step | Final failing metrics |
| --- | --- | ---: | --- |
| ae_gan_hold | PASS | 125 | — |
