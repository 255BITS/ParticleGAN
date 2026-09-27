# 100-Gaussian accuracy gate: FAIL

Original coverage criteria, five final 20k-draw fidelity checks, and a separate 100k-draw holdout are required. EMA is diagnostic.

| Problem | Status | Terminal checks | Holdout mass TV | Center RMS / σ | Covariance trace bias | Radial KS | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| grid100 | FAIL | 0 | 0.0195 | 0.2854 | -0.2142 | 0.1440 | coverage, sustained live accuracy, or independent holdout failed |
| rotated100 | FAIL | 0 | 0.0152 | 0.4591 | -0.1965 | 0.0633 | coverage, sustained live accuracy, or independent holdout failed |
| staggered100 | FAIL | 0 | 0.0170 | 0.8772 | -0.0992 | 0.0981 | coverage, sustained live accuracy, or independent holdout failed |
