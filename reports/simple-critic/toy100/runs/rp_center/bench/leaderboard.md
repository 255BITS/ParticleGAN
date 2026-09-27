# 100-Gaussian toy gate: FAIL

Scope: **all declared problems**. Verdicts use complete live-weight curves at the stated training budget. A PASS needs five consecutive passing postinitial checks at the end; all 100 modes, sample quality, mode mass, and per-mode shape are evaluated by the toy metrics.

| Problem | Status | Final modes | Final HQ | Mass TV | Cov eig ratio | Radial ratio | First 100 modes | First full quality | Stable from | Confirmed | Budget / elapsed | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| grid100 | FAIL | 100 | 0.980 | 0.060 | 0.176–1.855 | 0.546–1.458 | 500 (13.9s) | — | — | — | 7,000 (144.1s) | only 0/5 terminal postinitial checks pass |
| rotated100 | FAIL | 98 | 0.948 | 0.053 | 0.139–2.182 | 0.630–1.371 | — | — | — | — | 7,000 (156.0s) | only 0/5 terminal postinitial checks pass |
| staggered100 | FAIL | 100 | 0.972 | 0.038 | 0.188–1.636 | 0.546–1.465 | 750 (20.8s) | — | — | — | 7,000 (175.6s) | only 0/5 terminal postinitial checks pass |

The initial step is displayed in the raw events and animation but does not count toward convergence.
