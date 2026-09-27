# 100-Gaussian toy gate: FAIL

Scope: **all declared problems**. Verdicts use complete live-weight curves at the stated training budget. A PASS needs five consecutive passing postinitial checks at the end; all 100 modes, sample quality, mode mass, and per-mode shape are evaluated by the toy metrics.

| Problem | Status | Final modes | Final HQ | Mass TV | Cov eig ratio | Radial ratio | First 100 modes | First full quality | Stable from | Confirmed | Budget / elapsed | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| grid100 | FAIL | 100 | 0.903 | 0.039 | 0.166–1.831 | 0.695–1.662 | 1,000 (10.8s) | — | — | — | 7,000 (64.3s) | only 0/5 terminal postinitial checks pass |
| rotated100 | FAIL | 100 | 0.886 | 0.035 | 0.360–2.120 | 0.719–1.511 | 1,500 (20.9s) | — | — | — | 7,000 (103.9s) | only 0/5 terminal postinitial checks pass |
| staggered100 | FAIL | 100 | 0.954 | 0.037 | 0.209–1.758 | 0.558–1.278 | 500 (10.6s) | — | — | — | 7,000 (112.6s) | only 0/5 terminal postinitial checks pass |

The initial step is displayed in the raw events and animation but does not count toward convergence.
