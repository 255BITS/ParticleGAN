# 100-Gaussian toy gate: FAIL

Scope: **all declared problems**. Verdicts use complete live-weight curves at the stated training budget. A PASS needs five consecutive passing postinitial checks at the end; all 100 modes, sample quality, mode mass, and per-mode shape are evaluated by the toy metrics.

| Problem | Status | Final modes | Final HQ | Mass TV | Cov eig ratio | Radial ratio | First 100 modes | First full quality | Stable from | Confirmed | Budget / elapsed | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| grid100 | FAIL | 100 | 0.948 | 0.035 | 0.291–1.727 | 0.654–1.227 | 1,250 (23.1s) | — | — | — | 7,000 (112.9s) | only 0/5 terminal postinitial checks pass |
| rotated100 | FAIL | 100 | 0.897 | 0.039 | 0.293–1.886 | 0.735–1.466 | 2,000 (35.0s) | — | — | — | 7,000 (114.8s) | only 0/5 terminal postinitial checks pass |
| staggered100 | FAIL | 100 | 0.951 | 0.042 | 0.271–1.719 | 0.618–1.247 | 500 (9.6s) | — | — | — | 7,000 (82.4s) | only 0/5 terminal postinitial checks pass |

The initial step is displayed in the raw events and animation but does not count toward convergence.
