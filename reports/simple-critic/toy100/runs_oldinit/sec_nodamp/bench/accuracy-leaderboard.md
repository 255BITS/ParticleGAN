# 100-Gaussian accuracy gate: FAIL

Original coverage criteria, five final 20k-draw fidelity checks, and a separate 100k-draw holdout are required. EMA is diagnostic.

| Problem | Status | Terminal checks | Holdout mass TV | Center RMS / σ | Covariance trace bias | Radial KS | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| grid100 | FAIL | 0 | 0.0247 | 0.9008 | -0.1562 | 0.0919 | coverage, sustained live accuracy, or independent holdout failed |
| rotated100 | FAIL | 0 | 0.0216 | 0.4411 | 0.1179 | 0.0784 | coverage, sustained live accuracy, or independent holdout failed |
| staggered100 | FAIL | 0 | 0.0222 | 0.3108 | -0.2294 | 0.1250 | coverage, sustained live accuracy, or independent holdout failed |
