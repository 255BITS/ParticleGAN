# Declared affine-policy replay

The normal combined CLI completed with **FAIL 18/22** for
`configs/toy100/accuracy_shared_policy.json`: all three native problems pass
coverage and strict accuracy, while the identical recipe passes 15/19 older
toys. The separate installed-wheel public-default control passes 19/19.

| Section | Result |
| --- | --- |
| Native final five checks and 100k holdouts | 3/3 PASS |
| Shared candidate on frozen transfer toys | 15/19 FAIL |
| Public-default control | 19/19 PASS |
| Common recipe and applied-noise identity | Valid |

The four candidate failures are `mode_hold`, `vector_overlap`,
`img_stripes2`, and `img_blobs4`. A control pass does not rescue them.
All arrays in each native final 20k draw and independent 100k holdout are
bit-identical to the corresponding
[archived network-horizon experiment](accuracy-network-horizon/README.md).
Thus the declared production model and schedule reproduce that experiment's
numerical result without a scratch trainer override.

The replay began before full public-source archiving was added. Its native
archive is explicitly classified as `native-policy-limited-v1`; the current
regrader preserves its numerical FAIL 18/22 and reports limited source
coverage. New combined passes require the full-source v2 archive. This
replay is distinct from the later κ=1.0/network-floor=0.01 candidate that
passes 18/19 older toys but fails `residual_student`.

Complete local evidence is at
`artifacts/toy100-accuracy/production-shared-policy22/`, including
`compatibility.json`, `regrade-v2.log`, `scratch-parity.json`, all source
archives, exact configurations, optimizer receipts, and scored samples.
