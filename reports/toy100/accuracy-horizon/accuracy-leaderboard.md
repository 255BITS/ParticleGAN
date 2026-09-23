# 100-Gaussian accuracy gate: FAIL

Original coverage criteria, five final 20k-draw fidelity checks, and a separate 100k-draw holdout are required. EMA is diagnostic.

| Problem | Status | Terminal checks | Holdout mass TV | Center RMS / σ | Covariance trace bias | Radial KS | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| grid100 | FAIL | 0 | 0.2145 | — | — | — | coverage, sustained live accuracy, or independent holdout failed |
| rotated100 | PASS | 5 | 0.0375 | 0.1316 | -0.0290 | 0.0096 | sustained live accuracy and independent holdout pass |
| staggered100 | PASS | 5 | 0.0398 | 0.1141 | -0.0254 | 0.0093 | sustained live accuracy and independent holdout pass |
