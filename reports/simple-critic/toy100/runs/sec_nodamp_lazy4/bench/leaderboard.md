# 100-Gaussian toy gate: FAIL

Scope: **all declared problems**. Verdicts use complete live-weight curves at the stated training budget. A PASS needs five consecutive passing postinitial checks at the end; all 100 modes, sample quality, mode mass, and per-mode shape are evaluated by the toy metrics.

| Problem | Status | Final modes | Final HQ | Mass TV | Cov eig ratio | Radial ratio | First 100 modes | First full quality | Stable from | Confirmed | Budget / elapsed | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| grid100 | FAIL | 98 | 0.851 | 0.064 | 0.300–2.668 | 0.776–1.529 | 1,000 (22.9s) | — | — | — | 7,000 (138.5s) | only 0/5 terminal postinitial checks pass |
| rotated100 | FAIL | 100 | 0.849 | 0.042 | 0.305–2.236 | 0.719–1.688 | 2,500 (56.6s) | — | — | — | 7,000 (156.0s) | only 0/5 terminal postinitial checks pass |
| staggered100 | FAIL | 95 | 0.843 | 0.074 | 0.187–2.261 | 0.584–1.705 | 2,000 (51.0s) | — | — | — | 7,000 (173.8s) | only 0/5 terminal postinitial checks pass |

The initial step is displayed in the raw events and animation but does not count toward convergence.
