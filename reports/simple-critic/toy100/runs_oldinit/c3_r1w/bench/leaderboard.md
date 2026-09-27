# 100-Gaussian toy gate: FAIL

Scope: **all declared problems**. Verdicts use complete live-weight curves at the stated training budget. A PASS needs five consecutive passing postinitial checks at the end; all 100 modes, sample quality, mode mass, and per-mode shape are evaluated by the toy metrics.

| Problem | Status | Final modes | Final HQ | Mass TV | Cov eig ratio | Radial ratio | First 100 modes | First full quality | Stable from | Confirmed | Budget / elapsed | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| grid100 | FAIL | 100 | 0.950 | 0.037 | 0.163–1.870 | 0.535–1.291 | 500 (6.4s) | — | — | — | 7,000 (63.8s) | only 0/5 terminal postinitial checks pass |
| rotated100 | FAIL | 100 | 0.938 | 0.029 | 0.168–1.744 | 0.521–1.413 | 1,500 (14.6s) | — | — | — | 7,000 (64.5s) | only 0/5 terminal postinitial checks pass |
| staggered100 | FAIL | 98 | 0.869 | 0.035 | 0.293–2.998 | 0.581–1.680 | 250 (3.3s) | — | — | — | 7,000 (68.8s) | only 0/5 terminal postinitial checks pass |

The initial step is displayed in the raw events and animation but does not count toward convergence.
