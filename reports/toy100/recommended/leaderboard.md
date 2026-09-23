# 100-Gaussian toy gate: PASS

Scope: **all declared problems**. Verdicts use complete live-weight curves at the stated training budget. A PASS needs five consecutive passing postinitial checks at the end; all 100 modes, sample quality, mode mass, and per-mode shape are evaluated by the toy metrics.

| Problem | Status | Final modes | Final HQ | Mass TV | Cov eig ratio | Radial ratio | First 100 modes | First full quality | Stable from | Confirmed | Budget / elapsed | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| grid100 | PASS | 100 | 0.981 | 0.083 | 0.552–1.024 | 0.823–1.355 | 5,750 (81.4s) | 6,000 (84.9s) | 6,000 (84.9s) | 7,000 (98.8s) | 7,000 (98.8s) | terminal live metrics sustained |
| rotated100 | PASS | 100 | 0.989 | 0.055 | 0.579–1.071 | 0.783–1.039 | 5,250 (75.3s) | 6,000 (85.8s) | 6,000 (85.8s) | 7,000 (99.9s) | 7,000 (99.9s) | terminal live metrics sustained |
| staggered100 | PASS | 100 | 0.983 | 0.075 | 0.499–1.038 | 0.777–1.118 | 5,500 (137.7s) | 6,000 (150.8s) | 6,000 (150.8s) | 7,000 (176.1s) | 7,000 (176.1s) | terminal live metrics sustained |

The initial step is displayed in the raw events and animation but does not count toward convergence.
