# 100-Gaussian toy gate: FAIL

Scope: **all declared problems**. Verdicts use complete live-weight curves at the stated training budget. A PASS needs five consecutive passing postinitial checks at the end; all 100 modes, sample quality, mode mass, and per-mode shape are evaluated by the toy metrics.

| Problem | Status | Final modes | Final HQ | Mass TV | Cov eig ratio | Radial ratio | First 100 modes | First full quality | Stable from | Confirmed | Budget / elapsed | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| grid100 | FAIL | 100 | 0.934 | 0.034 | 0.246–2.916 | 0.612–1.627 | 1,500 (32.6s) | — | — | — | 7,000 (144.1s) | only 0/5 terminal postinitial checks pass |
| rotated100 | FAIL | 97 | 0.811 | 0.035 | 0.293–2.976 | 0.667–1.750 | 1,250 (30.6s) | — | — | — | 7,000 (156.0s) | only 0/5 terminal postinitial checks pass |
| staggered100 | FAIL | 100 | 0.945 | 0.032 | 0.261–2.171 | 0.556–1.445 | 750 (22.3s) | — | — | — | 7,000 (179.5s) | only 0/5 terminal postinitial checks pass |

The initial step is displayed in the raw events and animation but does not count toward convergence.
