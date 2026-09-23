# Current default versus proposed default: measured tests

The [primary unadjusted leaderboard](../unadjusted/README.md) carries the public
preset baselines into the shared-default search. Only its unchanged-recipe rows
compete; the per-host comparisons below remain historical explanatory evidence.

**Both presets were trained on all 19 behavioral tests.** Data, architecture, particle support, batch, initialization and update budget match within every pair. The [current master recipe](current-master-audit.json) matches every `gan_legacy` field exactly. Each arm uses its public recipe's loss, regularization, absolute G/D/particle LRs and Adam betas. Seed 0; no seed search or changed thresholds.

## Direct public-default comparison

| Preset | Required | Data | Images | Total live |
| --- | ---: | ---: | ---: | ---: |
| Current master / `gan_legacy` | 1/9 | 3/6 | 1/4 | **5/19** |
| Proposed / `gan` | 1/9 | 6/6 | 1/4 | **8/19** |

**The public preset does not pass all 19 tests unchanged.** The earlier 19/19 result belongs to the new core formulation with established per-host training recipes. It must not be presented as an all-pass result for one universal numerical optimizer preset.

Current: cap 1/κ1, spread 1, Adam(0,.999), G/D/prior LRs .0006/.0009/.006. Proposed: cap 3/κ1.25, spread .05, Adam(0,.99), LRs .001/.0015/.01. Both use Rp logistic, no particle L2, and the same delayed cosine schedule. Toy-specific reconstruction/identity/cover losses and initialization remain fixed. The toy's support and budget replace the generic 20,000 particles/7,000 updates; this is a matched test-host transfer, not a rerun of the separate 100-Gaussian benchmark.

## Isolating the core formulation

This second comparison retains exactly the published host LRs, Adam settings, EMA, schedules, resources and architectures. It changes only cap coefficient, cap threshold and spread weight. The new formulation column reuses the archived passing run for each architecture; the old core was freshly trained under those same settings. These are not scores for the public optimizer presets.

| Formulation with matched host training | Required | Data | Images | Total live |
| --- | ---: | ---: | ---: | ---: |
| Current core: cap 1/κ1, spread 1 | 7/9 | 6/6 | 4/4 | **17/19** |
| Proposed core: cap 3/κ1.25, spread .05 | 9/9 | 6/6 | 4/4 | **19/19** |

The current core meets every final metric, but two-pole has only **4** final passing observations and mode-hold only **1**, below the required 5. The new core sustains both. This is a stability improvement; it does not mean every individual final metric improves.

## Every test

| Test | Current preset | Proposed preset | Current core + host recipe | Proposed core + host recipe |
| --- | --- | --- | --- | --- |
| two_pole | [FAIL](current/episodes/current__two_pole.json.gz) | [FAIL](proposed/episodes/proposed__two_pole.json.gz) | [FAIL](current_core/episodes/current_core__two_pole.json.gz) | [PASS](../study/episodes/cosine__two_pole.json.gz) |
| trajectory | [FAIL](current/episodes/current__trajectory.json.gz) | [FAIL](proposed/episodes/proposed__trajectory.json.gz) | [PASS](current_core/episodes/current_core__trajectory.json.gz) | [PASS](../study/episodes/cosine__trajectory.json.gz) |
| residual_student | [FAIL](current/episodes/current__residual_student.json.gz) | [FAIL](proposed/episodes/proposed__residual_student.json.gz) | [PASS](current_core/episodes/current_core__residual_student.json.gz) | [PASS](../study/episodes/cosine__residual_student.json.gz) |
| unipolar | [FAIL](current/episodes/current__unipolar.json.gz) | [FAIL](proposed/episodes/proposed__unipolar.json.gz) | [PASS](current_core/episodes/current_core__unipolar.json.gz) | [PASS](../study/episodes/cosine__unipolar.json.gz) |
| ae_gan_hold | [PASS](current/episodes/current__ae_gan_hold.json.gz) | [PASS](proposed/episodes/proposed__ae_gan_hold.json.gz) | [PASS](current_core/episodes/current_core__ae_gan_hold.json.gz) | [PASS](../study/episodes/cosine__ae_gan_hold.json.gz) |
| cover_leftover | [FAIL](current/episodes/current__cover_leftover.json.gz) | [FAIL](proposed/episodes/proposed__cover_leftover.json.gz) | [PASS](current_core/episodes/current_core__cover_leftover.json.gz) | [PASS](../study/episodes/cosine__cover_leftover.json.gz) |
| unused_token_hold | [FAIL](current/episodes/current__unused_token_hold.json.gz) | [FAIL](proposed/episodes/proposed__unused_token_hold.json.gz) | [PASS](current_core/episodes/current_core__unused_token_hold.json.gz) | [PASS](../study/episodes/cosine__unused_token_hold.json.gz) |
| mid_scale_identity | [FAIL](current/episodes/current__mid_scale_identity.json.gz) | [FAIL](proposed/episodes/proposed__mid_scale_identity.json.gz) | [PASS](current_core/episodes/current_core__mid_scale_identity.json.gz) | [PASS](../study/episodes/cosine__mid_scale_identity.json.gz) |
| mode_hold | [FAIL](current/episodes/current__mode_hold.json.gz) | [FAIL](proposed/episodes/proposed__mode_hold.json.gz) | [FAIL](current_core/episodes/current_core__mode_hold.json.gz) | [PASS](../study/episodes/cosine__mode_hold.json.gz) |
| vector_two_broad | [PASS](current/episodes/current__vector_two_broad.json.gz) | [PASS](proposed/episodes/proposed__vector_two_broad.json.gz) | [PASS](current_core/episodes/current_core__vector_two_broad.json.gz) | [PASS](../study/episodes/cosine__vector_two_broad.json.gz) |
| vector_unequal_mass | [FAIL](current/episodes/current__vector_unequal_mass.json.gz) | [PASS](proposed/episodes/proposed__vector_unequal_mass.json.gz) | [PASS](current_core/episodes/current_core__vector_unequal_mass.json.gz) | [PASS](../rare_focus/linear_refinement/screen/episodes/linear_skip_d96_beta5__vector_unequal_mass.json.gz) |
| vector_unequal_width | [FAIL](current/episodes/current__vector_unequal_width.json.gz) | [PASS](proposed/episodes/proposed__vector_unequal_width.json.gz) | [PASS](current_core/episodes/current_core__vector_unequal_width.json.gz) | [PASS](../valid_search/smooth_discriminator/screen/episodes/axis_softplus5__vector_unequal_width.json.gz) |
| vector_anisotropic | [PASS](current/episodes/current__vector_anisotropic.json.gz) | [PASS](proposed/episodes/proposed__vector_anisotropic.json.gz) | [PASS](current_core/episodes/current_core__vector_anisotropic.json.gz) | [PASS](../study/episodes/cosine__vector_anisotropic.json.gz) |
| vector_overlap | [FAIL](current/episodes/current__vector_overlap.json.gz) | [PASS](proposed/episodes/proposed__vector_overlap.json.gz) | [PASS](current_core/episodes/current_core__vector_overlap.json.gz) | [PASS](../valid_search/discriminator/episodes/d128_l3_f3__vector_overlap.json.gz) |
| vector_spiral | [PASS](current/episodes/current__vector_spiral.json.gz) | [PASS](proposed/episodes/proposed__vector_spiral.json.gz) | [PASS](current_core/episodes/current_core__vector_spiral.json.gz) | [PASS](../study/episodes/cosine__vector_spiral.json.gz) |
| img_stripes2 | [PASS](current/episodes/current__img_stripes2.json.gz) | [PASS](proposed/episodes/proposed__img_stripes2.json.gz) | [PASS](current_core/episodes/current_core__img_stripes2.json.gz) | [PASS](../solvability/images/stage1/episodes/residual16__img_stripes2.json.gz) |
| img_bars4 | [FAIL](current/episodes/current__img_bars4.json.gz) | [FAIL](proposed/episodes/proposed__img_bars4.json.gz) | [PASS](current_core/episodes/current_core__img_bars4.json.gz) | [PASS](../solvability/images/stage1/episodes/residual16__img_bars4.json.gz) |
| img_blobs4 | [FAIL](current/episodes/current__img_blobs4.json.gz) | [FAIL](proposed/episodes/proposed__img_blobs4.json.gz) | [PASS](current_core/episodes/current_core__img_blobs4.json.gz) | [PASS](../solvability/images/stage1/episodes/residual16__img_blobs4.json.gz) |
| img_intensity2 | [FAIL](current/episodes/current__img_intensity2.json.gz) | [FAIL](proposed/episodes/proposed__img_intensity2.json.gz) | [PASS](current_core/episodes/current_core__img_intensity2.json.gz) | [PASS](../solvability/images/stage1/episodes/residual16__img_intensity2.json.gz) |

## Failing metrics and convergence

Each row needs all bounds for at least five final observations of the complete 24-point curve. A final passing point alone cannot earn PASS. The artifact links above contain every live/EMA measurement and schedule action; [machine-readable leaderboard](leaderboard.json) includes all thresholds, measured values, margins and convergence results.

| Test / preset | Final failed bounds | Final passing streak |
| --- | --- | ---: |
| two_pole / current | mean_abs=0.14728 (needs >=0.3) | 0 |
| trajectory / current | identity_mse=0.24901 (needs <=0.02) | 0 |
| residual_student / current | identity_mse=0.043406 (needs <=0.02); success_rate=0.58333 (needs >=1.0); wrong_pad_rate=0.41667 (needs <=0.0) | 0 |
| unipolar / current | cover=0.59275 (needs >=0.85) | 0 |
| cover_leftover / current | u_kept=0.34715 (needs >=0.85); content_kept=0.54116 (needs >=0.75); pole_rel_err_plus=0.46129 (needs <=0.2); pole_rel_err_minus=0.47404 (needs <=0.2) | 0 |
| unused_token_hold / current | concept_move=0.25748 (needs >=0.85) | 0 |
| mid_scale_identity / current | concept_cos_minus=0.84036 (needs >=0.85); concept_mag_plus=0.52327 (needs >=0.75); concept_mag_minus=0.48737 (needs >=0.75); identity_at_0=0.52041 (needs >=0.85) | 0 |
| mode_hold / current | modes=7 (needs >=8) | 0 |
| vector_unequal_mass / current | component_covariance_error=1.9269 (needs <=0.85) | 0 |
| vector_unequal_width / current | component_covariance_error=3.0654 (needs <=0.85) | 0 |
| vector_overlap / current | Final bounds pass; insufficient sustained checks | 3 |
| img_bars4 / current | modes=3 (needs >=4) | 0 |
| img_blobs4 / current | modes=2 (needs >=4); hq=0.71875 (needs >=0.9) | 0 |
| img_intensity2 / current | Final bounds pass; insufficient sustained checks | 1 |
| two_pole / proposed | mean_abs=0.11966 (needs >=0.3) | 0 |
| trajectory / proposed | identity_mse=0.12622 (needs <=0.02) | 0 |
| residual_student / proposed | identity_mse=0.04112 (needs <=0.02); success_rate=0.58333 (needs >=1.0); wrong_pad_rate=0.41667 (needs <=0.0) | 0 |
| unipolar / proposed | cover=0.8497 (needs >=0.85); neu_hold=0.83184 (needs >=0.85) | 0 |
| cover_leftover / proposed | u_kept=0.57498 (needs >=0.85); pole_rel_err_plus=0.28271 (needs <=0.2); pole_rel_err_minus=0.28469 (needs <=0.2) | 0 |
| unused_token_hold / proposed | concept_move=0.39859 (needs >=0.85) | 0 |
| mid_scale_identity / proposed | concept_mag_plus=0.69238 (needs >=0.75); concept_mag_minus=0.68561 (needs >=0.75); identity_at_0=0.80205 (needs >=0.85) | 0 |
| mode_hold / proposed | modes=7 (needs >=8); hq=0.73853 (needs >=0.9) | 0 |
| img_bars4 / proposed | modes=3 (needs >=4); hq=0.84375 (needs >=0.9) | 0 |
| img_blobs4 / proposed | modes=3 (needs >=4); hq=0.875 (needs >=0.9) | 0 |
| img_intensity2 / proposed | hq=0.78125 (needs >=0.9) | 0 |

## EMA, reported separately

EMA is scored using the same complete-curve rule where the host records all EMA metrics. N/A means no complete EMA curve is available; it earns no pass. Live selection is unchanged.

| Test | Current preset EMA | Proposed preset EMA | Current core EMA | Proposed core EMA |
| --- | --- | --- | --- | --- |
| two_pole | N/A | N/A | N/A | N/A |
| trajectory | N/A | N/A | N/A | N/A |
| residual_student | N/A | N/A | N/A | N/A |
| unipolar | N/A | N/A | N/A | N/A |
| ae_gan_hold | N/A | N/A | N/A | N/A |
| cover_leftover | N/A | N/A | N/A | N/A |
| unused_token_hold | N/A | N/A | N/A | N/A |
| mid_scale_identity | N/A | N/A | N/A | N/A |
| mode_hold | N/A | N/A | N/A | N/A |
| vector_two_broad | PASS | PASS | PASS | PASS |
| vector_unequal_mass | FAIL | FAIL | FAIL | FAIL |
| vector_unequal_width | FAIL | FAIL | FAIL | FAIL |
| vector_anisotropic | PASS | PASS | PASS | PASS |
| vector_overlap | PASS | PASS | PASS | PASS |
| vector_spiral | PASS | PASS | PASS | PASS |
| img_stripes2 | PASS | PASS | PASS | PASS |
| img_bars4 | FAIL | FAIL | PASS | PASS |
| img_blobs4 | FAIL | FAIL | PASS | PASS |
| img_intensity2 | FAIL | FAIL | FAIL | PASS |

## Verification and reproduction

57 new complete training episodes, 19 archived reference episodes, and all failed attempts are retained. All six proposed vector curves and action traces exactly match their published winners, including the rare-mode linear-skip fix. The verifier checks all 1,368 new observations, source archives, artifact hashes, actual optimizer-group LRs/betas, identical paired test conditions and unchanged core-comparison schedules. Five focused adapter tests catch mixed/AE prior groups and direct particle optimizers.

Architectures were selected before these runs from the already published winning row and shared by both arms. No architecture search was performed for stock, so the comparison does not establish its best achievable score after architecture tuning. Wall times are single CPU observations and do not establish a speedup. Legacy EMA behavior remains host-specific; vector/image preset runs use recipe decay .995.

```bash
python -u -m benchmarks.transfer_suite.compare_defaults --arm current --output /tmp/compare-current > /tmp/compare-current.log 2>&1
python -u -m benchmarks.transfer_suite.compare_defaults --arm proposed --output /tmp/compare-proposed > /tmp/compare-proposed.log 2>&1
python -u -m benchmarks.transfer_suite.compare_formulations --output /tmp/compare-core > /tmp/compare-core.log 2>&1
tail -f /tmp/compare-proposed.log
python -m reports.transfer_suite.default_comparison.build
```

[Adapter tests](tests.log) · [Artifact validation](validation.json) · [Archive manifest](archive_manifest.json).
