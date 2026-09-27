# 100-Gaussian toy gate: FAIL

Scope: **all declared problems**. Verdicts use complete live-weight curves at the stated training budget. A PASS needs five consecutive passing postinitial checks at the end; all 100 modes, sample quality, mode mass, and per-mode shape are evaluated by the toy metrics.

| Problem | Status | Final modes | Final HQ | Mass TV | Cov eig ratio | Radial ratio | First 100 modes | First full quality | Stable from | Confirmed | Budget / elapsed | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| grid100 | FAIL | 100 | 0.947 | 0.083 | 0.638–1.570 | 0.883–1.287 | 1,000 (23.2s) | — | — | — | 7,000 (150.1s) | only 0/5 terminal postinitial checks pass |
| rotated100 | FAIL | 71 | 0.615 | 0.104 | 0.006–2.810 | 0.794–1.823 | — | — | — | — | 7,000 (165.3s) | only 0/5 terminal postinitial checks pass |
| staggered100 | FAIL | 40 | 0.449 | 0.192 | 0.000–2.961 | 0.000–2.407 | — | — | — | — | 7,000 (181.4s) | only 0/5 terminal postinitial checks pass |

The initial step is displayed in the raw events and animation but does not count toward convergence.
