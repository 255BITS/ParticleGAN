# Pointwise discriminator normalization structure: rare mass

**No sustained live PASS in 34 frozen architecture trials.** All 34 use the
unchanged `shared_c6` recipe on `vector_unequal_mass`; each complete episode
has 24 live and EMA observations, actual optimizer receipts, original and
effective specs, source hashes, and the declared discriminator card. There are
no training errors or seed sweeps. This is one test, so it does not claim a
complete 19-case profile.

| Discriminator | Final minimum eigen ratio | Worst of final five | Final passing streak | Other late bounds |
| --- | ---: | ---: | ---: | --- |
| LayerNorm pre-activation plus .5 raw activation at every hidden layer | .1507 | .0307 | 1/5 | Covariance exceeds by .0003 at step 1000 |
| Fixed center-only normalization before activation | .1296 | .1022 | 0/5 | Pass |
| LayerNorm only at the last hidden layer | .0941 | .0941 | 0/5 | Pass |
| GroupNorm with two groups and affine parameters | .1250 | .0120 | 0/5 | Covariance fails at final checkpoint |
| Fixed centering, width 96, Softplus β6 | .1663 | .1278 | 1/5 | Covariance fails at steps 1050 and 1100 |
| Fixed centering, width 128, Softplus β4 | .2164 | .1277 | 0/5 | Covariance fails at steps 1150 and 1200 |

The eigen-ratio gate is **at least .15 at each of five consecutive final
observations**, along with all other behavioral bounds. The raw bypass meets
all bounds at step 1200 only. Fixed centering has a smaller final shortfall
and more stable late eigen ratios, but remains below .15 at every final-window
observation. The final six-card center-only follow-up found higher final eigen
ratios with β6 and width 128, but their earlier eigen/covariance readings fail.
Power scaling, GroupNorm, selective raw bypasses, post-activation normalization,
RMSNorm, and residual paths did not improve sustained support.

[Frozen first-stage cards](../../../../../benchmarks/transfer_suite/plans/shared_norm_structure_rare.json) ·
[First-stage index](screen/index.json) · [First-stage log](screen/run.log) ·
[Frozen refinement cards](../../../../../benchmarks/transfer_suite/plans/shared_norm_structure_refinement_rare.json) ·
[Refinement index](refinement/index.json) · [Refinement log](refinement/run.log) ·
[Frozen final cards](../../../../../benchmarks/transfer_suite/plans/shared_norm_structure_center_followup_rare.json) ·
[Final index](center-followup/index.json) · [Final log](center-followup/run.log).
Each stage has its own `protocol.json` and exact `source.tar.gz` because the
constructor was extended only after the preceding stage's source freeze.

The [primary importer audit](import_check.json) accepted all 34 episodes,
including exact source archives, original specifications, optimizer receipts,
behavioral verdicts, and distinct architecture identities. It raised the
complete evidence inventory from 515 to 549 episodes while the supported
`shared_c6` score remained 18/19. The audit restored the repository's
leaderboard files afterward; the main branch can import these three indexes
alongside its other candidate evidence.
