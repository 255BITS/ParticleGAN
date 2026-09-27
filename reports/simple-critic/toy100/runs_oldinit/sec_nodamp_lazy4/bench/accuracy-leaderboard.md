# 100-Gaussian accuracy gate: FAIL

Original coverage criteria, five final 20k-draw fidelity checks, and a separate 100k-draw holdout are required. EMA is diagnostic.

| Problem | Status | Terminal checks | Holdout mass TV | Center RMS / σ | Covariance trace bias | Radial KS | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| grid100 | FAIL | 0 | 0.0525 | 0.8656 | -0.0538 | 0.1156 | coverage, sustained live accuracy, or independent holdout failed |
| rotated100 | FAIL | 0 | 0.0285 | 0.7707 | 0.2030 | 0.1797 | coverage, sustained live accuracy, or independent holdout failed |
| staggered100 | FAIL | 0 | 0.0305 | 0.4455 | 0.0310 | 0.0467 | coverage, sustained live accuracy, or independent holdout failed |
