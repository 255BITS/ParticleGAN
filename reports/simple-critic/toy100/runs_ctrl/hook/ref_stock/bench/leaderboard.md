# 100-Gaussian toy gate: FAIL

Scope: **all declared problems**. Verdicts use complete live-weight curves at the stated training budget. A PASS needs five consecutive passing postinitial checks at the end; all 100 modes, sample quality, mode mass, and per-mode shape are evaluated by the toy metrics.

| Problem | Status | Final modes | Final HQ | Mass TV | Cov eig ratio | Radial ratio | First 100 modes | First full quality | Stable from | Confirmed | Budget / elapsed | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| grid100 | FAIL | 44 | 0.479 | 0.135 | 0.000–3.151 | 0.000–2.547 | — | — | — | — | 7,000 (94.5s) | only 0/5 terminal postinitial checks pass |
| rotated100 | PASS | 100 | 0.985 | 0.052 | 0.663–1.192 | 0.867–1.141 | 750 (13.4s) | 5,750 (93.3s) | 5,750 (93.3s) | 6,750 (107.0s) | 7,000 (110.0s) | terminal live metrics sustained |
| staggered100 | PASS | 100 | 0.990 | 0.039 | 0.670–1.178 | 0.854–1.082 | 750 (11.7s) | 5,500 (78.1s) | 5,500 (78.1s) | 6,500 (89.7s) | 7,000 (95.7s) | terminal live metrics sustained |

The initial step is displayed in the raw events and animation but does not count toward convergence.
