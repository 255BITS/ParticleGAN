# Paired output-sigma sensitivity of H2 prior-coupling grid100

The completed 7,000-update run used output sigma 0.02 and failed the frozen
native100 grid100 verdict. Its terminal geometry was inside the covariance
limits; live precision was 0.96875 versus the 0.97 minimum.

I rescored saved sample clouds without changing the candidate or the frozen
evaluator. The clean and noisy arrays use the same latent draws, and
`noisy - clean` is the realized Gaussian noise at sigma 0.02. For each value
below, I evaluated `clean + float32(sigma / 0.02) * (noisy - clean)` with the
unchanged frozen `evaluate_samples` and `evaluate_accuracy`. At sigma 0.02 I
used the original noisy array exactly; its metrics reproduce the stored run.
All five required 20,000-sample terminal clouds (updates 6000, 6250, 6500,
6750, 7000) and the independent 100,000-sample holdout were rescored. EMA
was checked as a diagnostic; the published pass condition uses live samples.

| Sigma | Live terminal passes / 5 | Final live precision | Final live radial KS | Final live covariance eig range | Holdout live precision | Holdout live radial KS | Holdout live passes? |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 0.01800 | 0 | 0.97370 | 0.06680 | 0.513–1.274 | 0.97404 | 0.06122 | No |
| 0.01900 | 0 | 0.97160 | 0.04874 | 0.542–1.283 | 0.97187 | 0.04516 | No |
| 0.01950 | 0 | 0.97040 | 0.04074 | 0.557–1.306 | 0.97077 | 0.03715 | Yes |
| 0.01955 | 0 | 0.97010 | 0.04013 | 0.558–1.308 | 0.97062 | 0.03639 | Yes |
| 0.01960 | 0 | 0.96995 | 0.03928 | 0.560–1.310 | 0.97047 | 0.03560 | Yes |
| 0.02000 | 0 | 0.96875 | 0.03298 | 0.572–1.330 | 0.96948 | 0.02942 | No |

Frozen limits relevant here: precision at least 0.97, radial KS at most 0.04,
minimum covariance eigenvalue ratio at least 0.40, and maximum at most 1.70.
At 0.01950 the holdout passes but update 7000 narrowly fails radial KS and
updates 6000–6750 still miss precision. At 0.01960 update 7000 passes radial
KS but misses precision; earlier updates also miss precision. EMA likewise
does not pass the five terminal checks at any evaluated sigma. Covariance
remains within its limits throughout this range; it is not the binding issue.

This finite grid provides no evidence that a small, fixed post-training sigma
change can rescue the required sustained verdict. It is a sensitivity test of
one trained model, not a training experiment: changing sigma during updates
can change the clean distribution and therefore the result. The exact rows,
input hashes, and reproducible script are in [results.json](results.json) and
[evaluate.py](evaluate.py).
