# Pointwise local-feature discriminator study

Architecture-only rare-mode screen under the original recipe. No target-derived centers or features, extra objectives, normalization, optimizer changes or extra training. Gaussian centers use one fixed local seed0 standard-normal initialization; declared octave widths may remain fixed or learn through the existing D loss.

| Card | D parameters | Rare sustained | Final suffix | HQ | Mass TV | Covariance error | Min eigen ratio | Min mass ratio | Seconds |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rbf128_full_softplus5 | 12993 | FAIL | 0/24 | 0.988037 | 0.087207 | 0.479072 | 0.131176 | 0.779622 | 11.488 |
| rbf64_full_softplus5 | 8705 | FAIL | 0/24 | 0.966553 | 0.049385 | 0.603043 | 0.044806 | 0.833834 | 8.829 |
| rbf128_full_silu | 12993 | FAIL | 0/24 | 1.000000 | 0.095020 | 0.737839 | 0.005584 | 0.415039 | 11.361 |
| rbf64_full_silu | 8705 | FAIL | 0/24 | 1.000000 | 0.171338 | 0.655482 | 0.001936 | 0.688477 | 8.567 |
| rbf64_residual2_silu | 21185 | FAIL | 0/24 | 0.994873 | 0.022900 | 1.152409 | 0.009068 | 0.958363 | 12.426 |
| rbf32_fixed_silu | 6465 | FAIL | 0/24 | 0.881104 | 0.009473 | 1.363200 | 0.007018 | 0.982777 | 7.820 |
| rbf64_centers_silu | 8641 | FAIL | 0/24 | 0.935303 | 0.213672 | 0.635741 | 0.000000 | 0.195312 | 7.832 |
| rbf64_fixed_silu | 8513 | FAIL | 0/24 | 0.942383 | 0.136768 | 4.396852 | 0.004778 | 0.595328 | 7.472 |

Original D has4,929 parameters; architectural capacity changes above are explicit. G stays64×2/z4. All cards use256 particles,batch128,1,200 rare-task steps,Adam(0,.99),G/D/prior LRs .001/.0015/.01,Rp logistic,b_cap3/kappa1.25,prior regularization .05,no particle L2,cosine and1:1 updates. The original five/six behavioral bounds and final-five-of24 stability rule are unchanged. EMA is separate.


Actual GAN episodes: **8**; live observations: **192**; summed recorded wall time: **75.794s**.

[Combined index](index.json.gz) retains exact cards, original/effective specs, full-episode artifact paths and hashes, all failures, live/EMA curves, actions and runtime. [Architecture checks](architecture_checks.json.gz) verify pointwise behavior, reproducible initialization and active cap backward for every card.
