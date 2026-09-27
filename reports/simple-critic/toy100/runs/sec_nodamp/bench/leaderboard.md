# 100-Gaussian toy gate: FAIL

Scope: **all declared problems**. Verdicts use complete live-weight curves at the stated training budget. A PASS needs five consecutive passing postinitial checks at the end; all 100 modes, sample quality, mode mass, and per-mode shape are evaluated by the toy metrics.

| Problem | Status | Final modes | Final HQ | Mass TV | Cov eig ratio | Radial ratio | First 100 modes | First full quality | Stable from | Confirmed | Budget / elapsed | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| grid100 | FAIL | 100 | 0.948 | 0.037 | 0.192–1.873 | 0.640–1.457 | 500 (14.0s) | — | — | — | 7,000 (140.2s) | only 0/5 terminal postinitial checks pass |
| rotated100 | FAIL | 100 | 0.907 | 0.035 | 0.314–1.909 | 0.605–1.381 | 1,250 (28.8s) | — | — | — | 7,000 (155.2s) | only 0/5 terminal postinitial checks pass |
| staggered100 | FAIL | 100 | 0.944 | 0.037 | 0.210–1.502 | 0.555–1.200 | 500 (14.4s) | — | — | — | 7,000 (177.3s) | only 0/5 terminal postinitial checks pass |

The initial step is displayed in the raw events and animation but does not count toward convergence.
