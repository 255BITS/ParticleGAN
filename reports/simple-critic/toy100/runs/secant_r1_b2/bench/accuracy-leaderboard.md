# 100-Gaussian accuracy gate: FAIL

Original coverage criteria, five final 20k-draw fidelity checks, and a separate 100k-draw holdout are required. EMA is diagnostic.

| Problem | Status | Terminal checks | Holdout mass TV | Center RMS / σ | Covariance trace bias | Radial KS | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| grid100 | FAIL | 0 | 0.0245 | 0.3109 | -0.1579 | 0.0921 | coverage, sustained live accuracy, or independent holdout failed |
| rotated100 | FAIL | 0 | 0.0219 | 0.4281 | 0.0870 | 0.0618 | coverage, sustained live accuracy, or independent holdout failed |
| staggered100 | FAIL | 0 | 0.0217 | 0.3464 | -0.1842 | 0.0829 | coverage, sustained live accuracy, or independent holdout failed |
