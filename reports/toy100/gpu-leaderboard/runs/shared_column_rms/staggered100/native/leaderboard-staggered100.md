# 100-Gaussian toy gate: FAIL

Scope: **individual**. Verdicts use complete live-weight curves at the stated training budget. A PASS needs five consecutive passing postinitial checks at the end; all 100 modes, sample quality, mode mass, and per-mode shape are evaluated by the toy metrics.

| Problem | Status | Final modes | Final HQ | Mass TV | Cov eig ratio | Radial ratio | First 100 modes | First full quality | Stable from | Confirmed | Budget / elapsed | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| staggered100 | FAIL | 100 | 0.907 | 0.040 | 0.791–2.076 | 1.045–1.533 | 1,500 (26.7s) | — | — | — | 7,000 (116.9s) | only 0/5 terminal postinitial checks pass |

The initial step is displayed in the raw events and animation but does not count toward convergence.
