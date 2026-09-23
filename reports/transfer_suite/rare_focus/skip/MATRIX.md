# Raw-coordinate and quadratic-skip discriminator study

Architecture-only rare-mode screen under the original recipe. Extra raw-coordinate/linear/quadratic paths bypass shared nonlinear features, with zero-initialized added outputs. No target-derived features, covariance-supervision loss, normalization, optimizer changes or extra training. Original Fourier features are unchanged wherever present; amplitudes are not swept.

| Card | D parameters | Rare sustained | Final suffix | HQ | Mass TV | Covariance error | Min eigen ratio | Min mass ratio | Seconds |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fourier_softplus5_linear_skip | 4931 | FAIL | 4/24 | 0.980225 | 0.110664 | 0.557634 | 0.490872 | 0.831410 | 7.330 |
| fourier_residual2_softplus5_quadratic_skip | 17414 | FAIL | 0/24 | 1.000000 | 0.053174 | 0.400273 | 0.105426 | 0.822754 | 10.625 |
| fourier_softplus5_quad_mlp_skip | 6209 | FAIL | 0/24 | 0.991211 | 0.089404 | 0.385831 | 0.018368 | 0.537109 | 8.438 |
| fourier_softplus5_raw_mlp_skip | 6113 | FAIL | 0/24 | 0.993408 | 0.031855 | 0.466265 | 0.004349 | 0.754958 | 7.595 |
| raw_softplus5_quadratic_skip | 4422 | FAIL | 0/24 | 1.000000 | 0.060986 | 0.775713 | 0.003236 | 0.889116 | 6.248 |
| raw_residual2_silu_quadratic_skip | 16902 | FAIL | 0/24 | 1.000000 | 0.028418 | 0.775991 | 0.002026 | 0.796274 | 9.443 |
| fourier_silu_quadratic_skip | 4934 | FAIL | 0/24 | 0.992676 | 0.057666 | 0.907650 | 0.000360 | 0.785006 | 6.996 |
| fourier_softplus5_quadratic_skip | 4934 | FAIL | 0/24 | 0.979736 | 0.048535 | 9.813329 | 0.585353 | 0.911754 | 7.225 |

Original D has4,929 parameters; architectural capacity changes above are explicit. G stays64×2/z4. All cards use256 particles,batch128,1,200 rare-task steps,Adam(0,.99),G/D/prior LRs .001/.0015/.01,Rp logistic,b_cap3/kappa1.25,prior regularization .05,no particle L2,cosine and1:1 updates. The original five/six behavioral bounds and final-five-of24 stability rule are unchanged. EMA is separate.


Actual GAN episodes: **8**; live observations: **192**; summed recorded wall time: **63.901s**.

[Combined index](index.json.gz) retains exact cards, original/effective specs, full-episode artifact paths and hashes, all failures, live/EMA curves, actions and runtime. [Architecture checks](architecture_checks.json.gz) verify pointwise behavior, reproducible initialization and active cap backward for every card.
