# 100-Gaussian toy gate: PASS

Scope: **all declared problems**. Verdicts use complete live-weight curves at the stated training budget. A PASS needs five consecutive passing postinitial checks at the end; all 100 modes, sample quality, mode mass, and per-mode shape are evaluated by the toy metrics.

| Problem | Status | Final modes | Final HQ | Mass TV | Cov eig ratio | Radial ratio | First 100 modes | First full quality | Stable from | Confirmed | Budget / elapsed | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| grid100 | PASS | 100 | 0.985 | 0.044 | 0.622–1.175 | 0.840–1.129 | 750 (13.9s) | 750 (13.9s) | 5,750 (85.9s) | 6,750 (100.2s) | 7,000 (103.6s) | terminal live metrics sustained |
| rotated100 | PASS | 100 | 0.986 | 0.040 | 0.680–1.180 | 0.848–1.134 | 750 (14.2s) | 5,750 (91.5s) | 5,750 (91.5s) | 6,750 (106.9s) | 7,000 (110.9s) | terminal live metrics sustained |
| staggered100 | PASS | 100 | 0.989 | 0.045 | 0.587–1.286 | 0.881–1.108 | 750 (14.1s) | 1,000 (17.8s) | 5,500 (73.8s) | 6,500 (84.9s) | 7,000 (90.5s) | terminal live metrics sustained |

The initial step is displayed in the raw events and animation but does not count toward convergence.
