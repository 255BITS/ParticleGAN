# 100-Gaussian toy gate: FAIL

Scope: **all declared problems**. Verdicts use complete live-weight curves at the stated training budget. A PASS needs five consecutive passing postinitial checks at the end; all 100 modes, sample quality, mode mass, and per-mode shape are evaluated by the toy metrics.

| Problem | Status | Final modes | Final HQ | Mass TV | Cov eig ratio | Radial ratio | First 100 modes | First full quality | Stable from | Confirmed | Budget / elapsed | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| grid100 | FAIL | 100 | 0.967 | 0.034 | 0.152–1.945 | 0.440–1.409 | 500 (12.5s) | — | — | — | 7,000 (143.5s) | only 0/5 terminal postinitial checks pass |
| rotated100 | FAIL | 100 | 0.922 | 0.028 | 0.164–2.052 | 0.559–1.699 | 1,750 (42.1s) | — | — | — | 7,000 (156.7s) | only 0/5 terminal postinitial checks pass |
| staggered100 | FAIL | 100 | 0.972 | 0.035 | 0.163–2.190 | 0.527–1.304 | 250 (10.1s) | — | — | — | 7,000 (175.6s) | only 0/5 terminal postinitial checks pass |

The initial step is displayed in the raw events and animation but does not count toward convergence.
