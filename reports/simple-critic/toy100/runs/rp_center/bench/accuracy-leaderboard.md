# 100-Gaussian accuracy gate: FAIL

Original coverage criteria, five final 20k-draw fidelity checks, and a separate 100k-draw holdout are required. EMA is diagnostic.

| Problem | Status | Terminal checks | Holdout mass TV | Center RMS / σ | Covariance trace bias | Radial KS | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| grid100 | FAIL | 0 | 0.0568 | 0.2330 | -0.0165 | 0.0125 | coverage, sustained live accuracy, or independent holdout failed |
| rotated100 | FAIL | 0 | 0.0479 | 0.4903 | -0.2094 | 0.0711 | coverage, sustained live accuracy, or independent holdout failed |
| staggered100 | FAIL | 0 | 0.0339 | 0.3113 | -0.1069 | 0.0379 | coverage, sustained live accuracy, or independent holdout failed |
