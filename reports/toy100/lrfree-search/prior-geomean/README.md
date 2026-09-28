# Geometric prior-rate coupling: grid100 holdout pass, frozen run fail

This isolated candidate starts from the committed paired birth/death graft and changes only the prior applied stationarity scale to `min(s_prior, sqrt(s_prior * s_G))`, where `s_G` is the slowest non-sigma generator tester scale. It never exceeds the prior's own tester and adds no task name, clock schedule, metric feedback or recipe override. QR `batch_feature_zero` initialization, learnable output noise at .02, the paired BD rule and noisy evaluation are unchanged. `training.patch` applies to the graft training source; `source-sha256.json` records the exact source and unchanged BD/override hashes.

The unchanged native grid100 7,000-update run ended **FAIL**. The runner's `13/34` count is its older coverage-style check; the stricter accuracy check passed **4/34** observations and **0/5** required terminal observations. Its independent 100k holdout **PASSES**. This is a promising but **unqualified** native lead; no rotated100 or staggered100 was run.

| Final live measure | Geometric coupling | Frozen requirement |
|---|---:|---:|
| Modes | 100 | 100 |
| Noisy precision | .97990 | ≥ .97000 |
| Centre RMS / data σ | .20427 | ≤ .20 |
| Covariance eig ratios | .42251–1.68585 | .40–1.70 |
| Mass TV | .0364 | ≤ .06 |
| Trace bias abs | .02337 | ≤ .10 |
| Radial KS | .01168 | ≤ .04 |
| Independent holdout | **PASS** | PASS |
| Frozen verdict | **FAIL**; accuracy 4/34, terminal 0/5 | last five + holdout |

The centre error at terminal updates 6000/6250/6500/6750/7000 was `.2100/.2165/.2172/.2099/.2043σ`—all above .20. Maximum covariance eigenvalue was `2.004/1.994/1.791/1.708/1.686`; it only entered the ≤1.70 band at the final check. Prior LR remained about `.00075` at the end while G LR was `3.32e−5`, and the latent table could keep moving after G settled. The next isolated state-driven test uses this geometric coupling while G is still active, then the committed min-rate cap once G's existing settle test reaches 1/64. That successor is not part of this result.

The raw result, all 34 observation rows, native fixture, noisy verdict and smoke receipt are committed here. The result had zero training-stream deviations. The saved final state remains in the isolated run directory for read-only analysis.
