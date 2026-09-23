# 100-Gaussian toy gate: FAIL

Scope: **all declared problems**. Verdicts use complete live-weight curves at the stated training budget. A PASS needs five consecutive passing postinitial checks at the end; all 100 modes, sample quality, mode mass, and per-mode shape are evaluated by the toy metrics.

| Problem | Status | Final modes | Final HQ | Mass TV | Cov eig ratio | Radial ratio | First 100 modes | First full quality | Stable from | Confirmed | Budget / elapsed | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| grid100 | FAIL | 63 | 0.572 | 0.164 | 0.000–3.469 | 0.000–2.134 | — | — | — | — | 7,000 (61.8s) | only 0/5 terminal postinitial checks pass |
| rotated100 | FAIL | 94 | 0.836 | 0.135 | 0.148–2.307 | 0.427–1.404 | — | — | — | — | 7,000 (62.5s) | only 0/5 terminal postinitial checks pass |
| staggered100 | FAIL | 88 | 0.724 | 0.122 | 0.000–2.267 | 0.000–1.824 | — | — | — | — | 7,000 (64.0s) | only 0/5 terminal postinitial checks pass |

The initial step is displayed in the raw events and animation but does not count toward convergence.
