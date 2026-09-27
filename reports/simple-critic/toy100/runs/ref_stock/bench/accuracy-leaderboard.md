# 100-Gaussian accuracy gate: FAIL

Original coverage criteria, five final 20k-draw fidelity checks, and a separate 100k-draw holdout are required. EMA is diagnostic.

| Problem | Status | Terminal checks | Holdout mass TV | Center RMS / σ | Covariance trace bias | Radial KS | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| grid100 | FAIL | 0 | 0.0752 | 0.3095 | -0.0376 | 0.0080 | coverage, sustained live accuracy, or independent holdout failed |
| rotated100 | FAIL | 0 | 0.1026 | 0.7658 | -0.0383 | 0.0889 | coverage, sustained live accuracy, or independent holdout failed |
| staggered100 | FAIL | 0 | 0.1939 | — | — | — | coverage, sustained live accuracy, or independent holdout failed |
