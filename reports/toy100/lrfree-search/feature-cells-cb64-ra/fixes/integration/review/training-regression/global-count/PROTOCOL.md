# Final proposed joint mass/local/global support family

Keep frozen RA2, RA3, v4, 2K and joint3K source/evidence intact. This separate proposal adds the global inside/outside aggregation diagnosed as missing finite-bin power. There is one fixed proposal with Q=.05, the existing even-fit .95 score boundary, the original odd support calibration/parent p>Q law, and unchanged quality gates. No seed, cutoff, score, boundary, order or hyperparameter sweep is allowed. All local execution is CPU-only, deterministic, with existing saved streams and no optimizer update.

## Common overlapping family

Evaluate the original K-cell, refined2K cell+inside/outside, and global2 inside/outside exact conditional Hypergeometric tests. Each family uses its own counts summing to the actual real/fake sample size. The global counts are sums of the same frozen2K partition, adding no fit. All K+2K+2 hypotheses, including empty bins, use Q/(3K+2). Recompute every significance mask at that common cutoff. Under the fixed-partition iid null, each conditional p-value is valid given its pooled total; the Bonferroni union bound needs no independence between overlapping families. Repeated adaptive head/FIFO decisions still have no cumulative guarantee.

## Shared actions

Retain original v4 mass transport first and local support transport second, including their unchanged method interfaces and quota allocation blocks. Both receive the new common corrected evidence. A third global support phase uses only the remaining floor(Q*N) ordinary slots. It requires global outside excess and global inside deficit certificates, deletes only flagged outside rows, and copies only unflagged inside p>Q parents. Its physical allocation obeys actual cell vacancies, real-center group vacancies, bounded parent pools and distinct rows.

Reserve all previous mass/local children and parents before global pools or deaths. Begin from the exact post-local supported ledger. Subtract every earlier outside-category child from the global outside death quota and every earlier inside-category parent from the global inside birth quota, clamp at zero, and expose raw/spent/residual scalar quotas. Opposite-direction earlier moves do not restore capacity. Every global allocation is bounded by both residual certificates and the remaining ordinary budget. Local support already reserves mass category quotas as in frozen3K.

Isolation receives all three ordinary plans once, retains its original small-flag guard and group/parent policy, excludes all earlier rows, and subtracts only supported ordinary children. The ordinary budget is shared by all three phases; legacy isolation retains its separate bounded small-flag repair count. All parent/child identities and supported ledgers must agree across phases. Per-cell caps apply to ordinary actions; combined legacy isolation retains its group caps.

## Fixed qualification

- Reuse frozen joint3K inputs and streams. Add deterministic no-signal counts, an opposite aggregate signal, and a broad aggregate-only cell-power case without selecting levels.
- Original no-flag supported imbalance must still actuate via the exact v4 mass phase. Nominal/rare, absent-parent, small51/52 guards, balanced supported table and supported-mass+small-hole ledgers remain valid.
- Inspect each global donor/parent region certificate, gross quota reservations from both prior phases, remaining budget, unique rows, supported cell/group targets, and final isolation ledger.
- An exhaustion case must leave zero global quota despite available parents/vacancies. Opposite-direction earlier actions must never replenish global capacity. Empty/tied/degenerate bins remain in actual3K+2 multiplicity.
- Enumerate exact overlapping K+2K+2 pooled nulls, including unequal real/fake sizes and empty bins. Verify original score/geometry/even boundary/RNG and absence of odd/fake boundary fitting.
- Produce a separate patch/READY and root-only CUDA contract runner with --package-root and --output. Root owns physical GPU0 and canonical matched toy/MNIST/replay quality. More fixed-input actions are not a quality result.
