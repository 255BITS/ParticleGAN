# 100-Gaussian toy gate: FAIL

Scope: **all declared problems**. Verdicts use complete live-weight curves at the stated training budget. A PASS needs five consecutive passing postinitial checks at the end; all 100 modes, sample quality, mode mass, and per-mode shape are evaluated by the toy metrics.

| Problem | Status | Final modes | Final HQ | Mass TV | Cov eig ratio | Radial ratio | First 100 modes | First full quality | Stable from | Confirmed | Budget / elapsed | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| grid100 | FAIL | 98 | 0.973 | 0.062 | 0.183–2.236 | 0.493–1.453 | — | — | — | — | 7,000 (61.9s) | only 0/5 terminal postinitial checks pass |
| rotated100 | FAIL | 100 | 0.957 | 0.052 | 0.209–1.783 | 0.615–1.325 | 4,750 (41.3s) | — | — | — | 7,000 (60.7s) | only 0/5 terminal postinitial checks pass |
| staggered100 | FAIL | 100 | 0.969 | 0.045 | 0.134–2.021 | 0.519–1.434 | 1,000 (9.6s) | — | — | — | 7,000 (62.9s) | only 0/5 terminal postinitial checks pass |

The initial step is displayed in the raw events and animation but does not count toward convergence.
