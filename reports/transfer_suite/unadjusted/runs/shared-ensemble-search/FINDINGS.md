# Fixed-recipe smooth critic architecture study

**Result: no new live PASS.** Across five frozen stages, 48 complete
discriminator episodes tested the two remaining vector cases under the
unchanged `shared_c6` recipe. This study does not change the prior 17/19
supported score. Every trial and failure is retained with 24 live and EMA
observations, original/effective specs, actual optimizer receipts, and its
stage's source archive.

| Stage | Cards × cases | Unequal mass PASS | Unequal width PASS | Evidence |
| --- | ---: | ---: | ---: | --- |
| Independent smooth ensembles | 12 × 2 | 0 | 0 | [screen](screen/README.md) |
| Pointwise normalized raw critics | 6 × 2 | 0 | 0 | [screen](../shared-pointnorm-search/screen/README.md) |
| LayerNorm Softplus refinement | 6 × 1 | 0 | — | [rare stage](../shared-pointnorm-search/refinement-rare/README.md) |
| Predeclared width cross | 2 × 1 | — | 0 | [width stage](../shared-pointnorm-search/refinement-width/README.md) |
| Final beta interpolation | 4 × 1 | 0 | — | [rare stage](../shared-pointnorm-search/beta-interpolation-rare/README.md) |

The strongest rare-mass result in this study is the LayerNorm-all,
width-96, three-layer Softplus(beta=4) critic: final minimum normalized
component variance **.12525**, still below the **.15** gate. The final five
variance ratios are .0824, .0310, .0981, .1215, and .1253, so it has no
passing suffix. At step 1100, sample quality .8413 also misses .85 and
component covariance error .8936 misses .85. Its final occupancy and sample
quality pass, which shows why those aggregate metrics alone cannot certify
rare-component spread. The earlier beta-5 LayerNorm critic finished at .11905
and likewise oscillated in the final window.

The best ensemble final eigen ratio was .03546 on unequal mass and .11250 on
unequal width. The two refinement cards cross-tested on unequal width both
failed. The parallel raw-Softplus width study supplies a separate width PASS;
this architecture study found no rare-mass witness. These results support a
pointwise normalization lead for rare spread, while showing that the fixed
recipe and budget have not yet delivered sustained rare variance.

The five stages were frozen separately before training. Their exact plans
are in `benchmarks/transfer_suite/plans/`; source snapshots differ by stage
as the bounded follow-ups were added. Architecture cards expose every input,
score, normalization, and activation parameterization change. The generator,
prior, training recipe, seed, batch, particles, budgets, all 24 checkpoints,
and evaluation gates were held fixed. There were no seed sweeps or
target-derived features. EMA was measured separately and never selected a
pass.

[Validation log](VALIDATION.log) checks every source archive hash, episode
checksum, complete curve, recomputed live/EMA verdict, optimizer LR/betas, and
discriminator parameter count. [Focused test log](TESTS.log) checks pointwise
independence and active b_cap double backward for every declared critic.
