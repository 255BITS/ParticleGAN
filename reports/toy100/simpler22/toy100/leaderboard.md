# 100-Gaussian toy gate: PASS

Scope: **all declared problems**. Verdicts use complete live-weight curves at the stated training budget. A PASS needs five consecutive passing postinitial checks at the end; all 100 modes, sample quality, mode mass, and per-mode shape are evaluated by the toy metrics.

| Problem | Status | Final modes | Final HQ | Mass TV | Cov eig ratio | Radial ratio | First 100 modes | First full quality | Stable from | Confirmed | Budget / elapsed | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| grid100 | PASS | 100 | 0.985 | 0.043 | 0.700–1.265 | 0.886–1.231 | 750 (31.1s) | 750 (31.1s) | 5,750 (235.2s) | 6,750 (276.8s) | 7,000 (287.2s) | terminal live metrics sustained |
| rotated100 | PASS | 100 | 0.986 | 0.042 | 0.657–1.255 | 0.848–1.161 | 750 (30.7s) | 5,750 (255.9s) | 5,750 (255.9s) | 6,750 (310.3s) | 7,000 (323.4s) | terminal live metrics sustained |
| staggered100 | PASS | 100 | 0.989 | 0.043 | 0.675–1.153 | 0.859–1.090 | 750 (41.9s) | 5,500 (348.3s) | 5,500 (348.3s) | 6,500 (400.8s) | 7,000 (426.5s) | terminal live metrics sustained |

The initial step is displayed in the raw events and animation but does not count toward convergence.
