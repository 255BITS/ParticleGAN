# 100-Gaussian toy gate: FAIL

Scope: **all declared problems**. Verdicts use complete live-weight curves at the stated training budget. A PASS needs five consecutive passing postinitial checks at the end; all 100 modes, sample quality, mode mass, and per-mode shape are evaluated by the toy metrics.

| Problem | Status | Final modes | Final HQ | Mass TV | Cov eig ratio | Radial ratio | First 100 modes | First full quality | Stable from | Confirmed | Budget / elapsed | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| grid100 | FAIL | 100 | 0.909 | 0.033 | 0.138–3.224 | 0.581–1.740 | 1,250 (12.9s) | — | — | — | 7,000 (64.3s) | only 0/5 terminal postinitial checks pass |
| rotated100 | FAIL | 100 | 0.908 | 0.032 | 0.255–1.894 | 0.530–1.566 | 1,500 (15.0s) | — | — | — | 7,000 (66.2s) | only 0/5 terminal postinitial checks pass |
| staggered100 | FAIL | 100 | 0.940 | 0.032 | 0.195–2.438 | 0.453–1.488 | 500 (5.7s) | — | — | — | 7,000 (68.9s) | only 0/5 terminal postinitial checks pass |

The initial step is displayed in the raw events and animation but does not count toward convergence.
