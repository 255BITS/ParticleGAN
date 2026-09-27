# 100-Gaussian toy gate: FAIL

Scope: **all declared problems**. Verdicts use complete live-weight curves at the stated training budget. A PASS needs five consecutive passing postinitial checks at the end; all 100 modes, sample quality, mode mass, and per-mode shape are evaluated by the toy metrics.

| Problem | Status | Final modes | Final HQ | Mass TV | Cov eig ratio | Radial ratio | First 100 modes | First full quality | Stable from | Confirmed | Budget / elapsed | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| grid100 | FAIL | 66 | 0.853 | 0.299 | 0.000–2.333 | 0.000–2.380 | 250 (9.4s) | — | — | — | 7,000 (149.5s) | only 0/5 terminal postinitial checks pass |
| rotated100 | FAIL | 88 | 0.841 | 0.139 | 0.005–2.357 | 0.289–1.384 | — | — | — | — | 7,000 (164.0s) | only 0/5 terminal postinitial checks pass |
| staggered100 | FAIL | 9 | 0.148 | 0.277 | 0.000–3.524 | 0.000–2.547 | 500 (17.8s) | — | — | — | 7,000 (176.8s) | only 0/5 terminal postinitial checks pass |

The initial step is displayed in the raw events and animation but does not count toward convergence.
