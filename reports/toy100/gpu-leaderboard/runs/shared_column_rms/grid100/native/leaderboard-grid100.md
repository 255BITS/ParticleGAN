# 100-Gaussian toy gate: FAIL

Scope: **individual**. Verdicts use complete live-weight curves at the stated training budget. A PASS needs five consecutive passing postinitial checks at the end; all 100 modes, sample quality, mode mass, and per-mode shape are evaluated by the toy metrics.

| Problem | Status | Final modes | Final HQ | Mass TV | Cov eig ratio | Radial ratio | First 100 modes | First full quality | Stable from | Confirmed | Budget / elapsed | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| grid100 | FAIL | 100 | 0.866 | 0.042 | 0.879–2.531 | 1.096–1.580 | 1,750 (26.3s) | — | — | — | 7,000 (103.3s) | only 0/5 terminal postinitial checks pass |

The initial step is displayed in the raw events and animation but does not count toward convergence.
