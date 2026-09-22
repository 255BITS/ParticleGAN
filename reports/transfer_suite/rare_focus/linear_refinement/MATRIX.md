# Final raw-linear bypass refinements

Architecture-only rare-mode screen under the original recipe. Four predeclared width/Softplus variants share the same zero-initialized raw linear bypass. No target-derived features, covariance-supervision loss, normalization, optimizer changes or extra training. Original Fourier features are unchanged wherever present; amplitudes are not swept.

| Card | D parameters | Rare sustained | Final suffix | HQ | Mass TV | Covariance error | Min eigen ratio | Min mass ratio | Seconds |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| linear_skip_d96_beta5 | 10467 | PASS | 6/24 | 0.997070 | 0.046826 | 0.441314 | 0.269769 | 0.914862 | 6.846 |
| linear_skip_d64_beta6 | 4931 | FAIL | 0/24 | 0.971436 | 0.151074 | 1.200072 | 0.340556 | 0.725320 | 7.239 |
| linear_skip_d96_beta6 | 10467 | FAIL | 0/24 | 0.990479 | 0.031377 | 0.545536 | 0.003289 | 0.549316 | 6.905 |
| linear_skip_d64_beta10 | 4931 | FAIL | 0/24 | 0.969727 | 0.107354 | 4.442183 | 0.047485 | 0.500488 | 6.254 |

Original D has4,929 parameters; architectural capacity changes above are explicit. G stays64×2/z4. All cards use256 particles,batch128,1,200 rare-task steps,Adam(0,.99),G/D/prior LRs .001/.0015/.01,Rp logistic,b_cap3/kappa1.25,prior regularization .05,no particle L2,cosine and1:1 updates. The original five/six behavioral bounds and final-five-of24 stability rule are unchanged. EMA is separate.

| Card | vector_two_broad | vector_unequal_mass | vector_unequal_width | vector_anisotropic | vector_overlap | vector_spiral |
| --- | --- | --- | --- | --- | --- | --- |
| linear_skip_d96_beta5 | PASS | PASS | FAIL | FAIL | FAIL | PASS |

Actual GAN episodes: **9**; live observations: **216**; summed recorded wall time: **64.604s**.

[Combined index](index.json.gz) retains exact cards, original/effective specs, full-episode artifact paths and hashes, all failures, live/EMA curves, actions and runtime. [Architecture checks](architecture_checks.json.gz) verify pointwise behavior, reproducible initialization and active cap backward for every card.
