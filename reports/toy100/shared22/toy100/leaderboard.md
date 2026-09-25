# 100-Gaussian toy gate: PASS

Scope: **all declared problems**. Verdicts use complete live-weight curves at the stated training budget. A PASS needs five consecutive passing postinitial checks at the end; all 100 modes, sample quality, mode mass, and per-mode shape are evaluated by the toy metrics.

| Problem | Status | Final modes | Final HQ | Mass TV | Cov eig ratio | Radial ratio | First 100 modes | First full quality | Stable from | Confirmed | Budget / elapsed | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| grid100 | PASS | 100 | 0.983 | 0.045 | 0.665–1.201 | 0.864–1.142 | 750 (32.1s) | 1,000 (41.3s) | 6,000 (223.4s) | 7,000 (259.8s) | 7,000 (259.8s) | terminal live metrics sustained |
| rotated100 | PASS | 100 | 0.979 | 0.048 | 0.647–1.195 | 0.870–1.149 | 750 (28.8s) | 6,000 (224.5s) | 6,000 (224.5s) | 7,000 (261.3s) | 7,000 (261.3s) | terminal live metrics sustained |
| staggered100 | PASS | 100 | 0.985 | 0.051 | 0.645–1.269 | 0.848–1.126 | 750 (28.1s) | 1,000 (37.3s) | 5,750 (232.6s) | 6,750 (275.8s) | 7,000 (285.7s) | terminal live metrics sustained |

The initial step is displayed in the raw events and animation but does not count toward convergence.
