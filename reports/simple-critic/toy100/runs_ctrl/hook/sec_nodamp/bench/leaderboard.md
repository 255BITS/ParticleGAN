# 100-Gaussian toy gate: FAIL

Scope: **all declared problems**. Verdicts use complete live-weight curves at the stated training budget. A PASS needs five consecutive passing postinitial checks at the end; all 100 modes, sample quality, mode mass, and per-mode shape are evaluated by the toy metrics.

| Problem | Status | Final modes | Final HQ | Mass TV | Cov eig ratio | Radial ratio | First 100 modes | First full quality | Stable from | Confirmed | Budget / elapsed | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| grid100 | FAIL | 100 | 0.925 | 0.032 | 0.288–2.014 | 0.607–1.425 | 250 (4.9s) | — | — | — | 7,000 (79.5s) | only 0/5 terminal postinitial checks pass |
| rotated100 | FAIL | 100 | 0.865 | 0.033 | 0.335–2.377 | 0.772–1.529 | 1,750 (24.9s) | — | — | — | 7,000 (95.6s) | only 0/5 terminal postinitial checks pass |
| staggered100 | FAIL | 100 | 0.903 | 0.032 | 0.326–1.902 | 0.638–1.357 | 500 (8.9s) | — | — | — | 7,000 (90.4s) | only 0/5 terminal postinitial checks pass |

The initial step is displayed in the raw events and animation but does not count toward convergence.
