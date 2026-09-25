# 100-Gaussian toy gate: FAIL

Scope: **individual**. Verdicts use complete live-weight curves at the stated training budget. A PASS needs five consecutive passing postinitial checks at the end; all 100 modes, sample quality, mode mass, and per-mode shape are evaluated by the toy metrics.

| Problem | Status | Final modes | Final HQ | Mass TV | Cov eig ratio | Radial ratio | First 100 modes | First full quality | Stable from | Confirmed | Budget / elapsed | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| grid100 | FAIL | 99 | 0.858 | 0.042 | 0.896–2.448 | 1.044–1.716 | 1,500 (24.5s) | — | — | — | 7,000 (106.0s) | only 0/5 terminal postinitial checks pass |

The initial step is displayed in the raw events and animation but does not count toward convergence.
