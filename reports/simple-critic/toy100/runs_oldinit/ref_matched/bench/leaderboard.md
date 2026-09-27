# 100-Gaussian toy gate: FAIL

Scope: **all declared problems**. Verdicts use complete live-weight curves at the stated training budget. A PASS needs five consecutive passing postinitial checks at the end; all 100 modes, sample quality, mode mass, and per-mode shape are evaluated by the toy metrics.

| Problem | Status | Final modes | Final HQ | Mass TV | Cov eig ratio | Radial ratio | First 100 modes | First full quality | Stable from | Confirmed | Budget / elapsed | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| grid100 | FAIL | 22 | 0.285 | 0.300 | 0.000–5.343 | 0.000–2.532 | 250 (4.3s) | — | — | — | 7,000 (73.0s) | only 0/5 terminal postinitial checks pass |
| rotated100 | FAIL | 92 | 0.856 | 0.117 | 0.045–2.196 | 0.357–1.389 | — | — | — | — | 7,000 (112.3s) | only 0/5 terminal postinitial checks pass |
| staggered100 | FAIL | 13 | 0.187 | 0.308 | 0.000–3.173 | 0.000–2.406 | 250 (7.2s) | — | — | — | 7,000 (117.2s) | only 0/5 terminal postinitial checks pass |

The initial step is displayed in the raw events and animation but does not count toward convergence.
