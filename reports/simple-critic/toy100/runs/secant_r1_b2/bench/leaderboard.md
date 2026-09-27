# 100-Gaussian toy gate: FAIL

Scope: **all declared problems**. Verdicts use complete live-weight curves at the stated training budget. A PASS needs five consecutive passing postinitial checks at the end; all 100 modes, sample quality, mode mass, and per-mode shape are evaluated by the toy metrics.

| Problem | Status | Final modes | Final HQ | Mass TV | Cov eig ratio | Radial ratio | First 100 modes | First full quality | Stable from | Confirmed | Budget / elapsed | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| grid100 | FAIL | 100 | 0.925 | 0.034 | 0.395–1.630 | 0.668–1.138 | 500 (15.4s) | — | — | — | 7,000 (200.0s) | only 0/5 terminal postinitial checks pass |
| rotated100 | FAIL | 100 | 0.900 | 0.037 | 0.355–2.033 | 0.674–1.446 | 1,500 (53.5s) | — | — | — | 7,000 (208.8s) | only 0/5 terminal postinitial checks pass |
| staggered100 | FAIL | 100 | 0.948 | 0.038 | 0.127–2.147 | 0.619–1.428 | 500 (22.6s) | — | — | — | 7,000 (128.8s) | only 0/5 terminal postinitial checks pass |

The initial step is displayed in the raw events and animation but does not count toward convergence.
