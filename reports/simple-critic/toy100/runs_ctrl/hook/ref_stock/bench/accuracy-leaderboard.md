# 100-Gaussian accuracy gate: FAIL

Original coverage criteria, five final 20k-draw fidelity checks, and a separate 100k-draw holdout are required. EMA is diagnostic.

| Problem | Status | Terminal checks | Holdout mass TV | Center RMS / σ | Covariance trace bias | Radial KS | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| grid100 | FAIL | 0 | 0.1367 | — | — | — | coverage, sustained live accuracy, or independent holdout failed |
| rotated100 | PASS | 5 | 0.0377 | 0.0761 | -0.0411 | 0.0165 | sustained live accuracy and independent holdout pass |
| staggered100 | PASS | 5 | 0.0324 | 0.0805 | -0.0420 | 0.0177 | sustained live accuracy and independent holdout pass |
