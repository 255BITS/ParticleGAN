# 100-Gaussian toy gate: FAIL

Scope: **all declared problems**. Verdicts use complete live-weight curves at the stated training budget. A PASS needs five consecutive passing postinitial checks at the end; all 100 modes, sample quality, mode mass, and per-mode shape are evaluated by the toy metrics.

| Problem | Status | Final modes | Final HQ | Mass TV | Cov eig ratio | Radial ratio | First 100 modes | First full quality | Stable from | Confirmed | Budget / elapsed | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| grid100 | FAIL | 98 | 0.859 | 0.059 | 0.198–2.052 | 0.721–1.822 | 2,500 (30.5s) | — | — | — | 7,000 (81.3s) | only 0/5 terminal postinitial checks pass |
| rotated100 | FAIL | 99 | 0.813 | 0.044 | 0.415–2.225 | 0.785–1.830 | 2,500 (34.2s) | — | — | — | 7,000 (89.9s) | only 0/5 terminal postinitial checks pass |
| staggered100 | FAIL | 100 | 0.922 | 0.040 | 0.238–2.361 | 0.599–1.396 | 2,500 (40.5s) | — | — | — | 7,000 (90.2s) | only 0/5 terminal postinitial checks pass |

The initial step is displayed in the raw events and animation but does not count toward convergence.
