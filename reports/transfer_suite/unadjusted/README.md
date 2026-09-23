# Unadjusted ParticleGAN default leaderboard

Reproduce the selected profile with [one command](../../../benchmarks/transfer_suite/UNADJUSTED_SEARCH.md). [Findings and discriminator details](FINDINGS.md). Generated JSON, curves and replay archives linked below are local artifacts; source, plans and this readable leaderboard stay in Git ([artifact policy](../../README.md)).

**This is the primary comparison for selecting a shared default.** Each candidate uses one unchanged loss/regularization/optimizer recipe on every test. No per-example LR, Adam, prior-rate or loss-weight adjustments. The earlier adjusted 19/19 result does not compete on this leaderboard.

**Overall PASS requires 19/19 live behavioral tests**, each passing every metric for at least five final observations of a complete 24-point curve. EMA is separate. Missing cases stay in the denominator; partial rows cannot beat a completed candidate or qualify as winners.

| Candidate | Required | Data | Images | Live total | Reference D profile | Attempted | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| GAN v3 (current default; shared_c6) (`shared_c6`) | 9/9 | 6/6 | 4/4 | **19/19** | 15/19 | 19/19 | **PASS** |
| lr00425 prior2 (`lr00425_prior2`) | 8/9 | 3/6 | 4/4 | **15/19** | 15/19 | 19/19 | **FAIL** |
| Ratio G0.0034 D0.0068 P0.0085 (`ratio_g34_d68_p85`) | 8/9 | 3/6 | 4/4 | **15/19** | 15/19 | 19/19 | **FAIL** |
| equal lr 00425 (`equal_lr_00425`) | 8/9 | 3/6 | 3/4 | **14/19** | 14/19 | 19/19 | **FAIL** |
| equal lr 0025 (`equal_lr_0025`) | 7/9 | 3/6 | 4/4 | **14/19** | 14/19 | 19/19 | **FAIL** |
| shared k075 (`shared_k075`) | 7/9 | 4/6 | 3/4 | **14/19** | 14/19 | 19/19 | **FAIL** |
| shared spread01 (`shared_spread01`) | 9/9 | 1/6 | 3/4 | **13/19** | 13/19 | 19/19 | **FAIL** |
| Ratio G0.00425 D0.006375 P0.0085 (`ratio_g425_d6375_p85`) | 9/9 | 2/6 | 2/4 | **13/19** | 13/19 | 19/19 | **FAIL** |
| equal lr 0034 (`equal_lr_0034`) | 8/9 | 3/6 | 2/4 | **13/19** | 13/19 | 19/19 | **FAIL** |
| lr0034 dprior125 (`lr0034_dprior125`) | 7/9 | 3/6 | 3/4 | **13/19** | 13/19 | 19/19 | **FAIL** |
| relative cap 05 (`relative_cap_05`) | 7/9 | 2/6 | 3/4 | **12/19** | 12/19 | 19/19 | **FAIL** |
| equal lr 0017 (`equal_lr_0017`) | 5/9 | 2/6 | 4/4 | **11/19** | 11/19 | 19/19 | **FAIL** |
| GAN v2 (previous default; archived gan) (`gan`) | 1/9 | 6/6 | 1/4 | **8/19** | 8/19 | 19/19 | **FAIL** |
| GAN v1 (original preset) (`gan_legacy`) | 1/9 | 3/6 | 1/4 | **5/19** | 5/19 | 19/19 | **FAIL** |

Rank complete candidates by live passes, then lower normalized final metric shortfall. An all-pass candidate is the target; a partial improvement is not an all-pass stamp.

Live total counts a test once when the unchanged recipe supports a declared discriminator architecture. The reference column shows passes with the original frozen D profile. Architecture trials never change the optimizer recipe; every failed trial remains archived.

## Screening results — not ranked

| Candidate | Passing measured tests | Attempted | Status |
| --- | ---: | ---: | --- |
| shared_b05_999 | 3 | 6/19 | INCOMPLETE |
| ratio_g25_d15_p85 | 2 | 7/19 | INCOMPLETE |
| ratio_g425_d2125_p85 | 2 | 7/19 | INCOMPLETE |
| ratio_g25_d375_p100 | 2 | 7/19 | INCOMPLETE |
| shared_k10 | 2 | 6/19 | INCOMPLETE |
| shared_k15 | 2 | 6/19 | INCOMPLETE |
| shared_c10 | 2 | 6/19 | INCOMPLETE |
| shared_b0_999 | 2 | 6/19 | INCOMPLETE |
| shared_p3 | 2 | 6/19 | INCOMPLETE |
| ratio_g34_d255_p85 | 1 | 7/19 | INCOMPLETE |
| relative_cap_025 | 1 | 6/19 | INCOMPLETE |
| shared_b05 | 1 | 6/19 | INCOMPLETE |
| relative_cap_01 | 1 | 6/19 | INCOMPLETE |
| shared_c20 | 1 | 6/19 | INCOMPLETE |
| shared_p4 | 0 | 6/19 | INCOMPLETE |
| sched_h50_f01 | 0 | 1/19 | INCOMPLETE |
| sched_h40_f01 | 0 | 1/19 | INCOMPLETE |
| c6_profile_prior4 | 0 | 1/19 | INCOMPLETE |
| sched_h30_f01 | 0 | 1/19 | INCOMPLETE |
| sched_h30_f05 | 0 | 1/19 | INCOMPLETE |
| c6_profile_prior3 | 0 | 1/19 | INCOMPLETE |
| c6_profile_beta2_095 | 0 | 1/19 | INCOMPLETE |
| c6_profile_beta1_01 | 0 | 1/19 | INCOMPLETE |

## One recipe per row

| Candidate | G / D / particle LR | Adam betas | b_cap coefficient / κ | Spread weight | Schedule hold / floor | Update rule |
| --- | --- | --- | --- | ---: | --- | --- |
| shared_c6 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 6 / 1.25 | 0.05 | 60% / 5% | Adam |
| lr00425_prior2 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 3 / 1.25 | 0.05 | 60% / 5% | Adam |
| ratio_g34_d68_p85 | 0.0034 / 0.0068 / 0.0085 | (0.0, 0.99) | 3 / 1.25 | 0.05 | 60% / 5% | Adam |
| equal_lr_00425 | 0.00425 / 0.00425 / 0.00425 | (0.0, 0.99) | 3 / 1.25 | 0.05 | 60% / 5% | Adam |
| equal_lr_0025 | 0.0025 / 0.0025 / 0.0025 | (0.0, 0.99) | 3 / 1.25 | 0.05 | 60% / 5% | Adam |
| shared_k075 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 3 / 0.75 | 0.05 | 60% / 5% | Adam |
| shared_spread01 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 3 / 1.25 | 0.01 | 60% / 5% | Adam |
| ratio_g425_d6375_p85 | 0.00425 / 0.006375 / 0.0085 | (0.0, 0.99) | 3 / 1.25 | 0.05 | 60% / 5% | Adam |
| equal_lr_0034 | 0.0034 / 0.0034 / 0.0034 | (0.0, 0.99) | 3 / 1.25 | 0.05 | 60% / 5% | Adam |
| lr0034_dprior125 | 0.0034 / 0.00425 / 0.00425 | (0.0, 0.99) | 3 / 1.25 | 0.05 | 60% / 5% | Adam |
| relative_cap_05 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 3 / 1.25 | 0.05 | 60% / 5% | Adam + relative step cap 0.05 |
| equal_lr_0017 | 0.0017 / 0.0017 / 0.0017 | (0.0, 0.99) | 3 / 1.25 | 0.05 | 60% / 5% | Adam |
| gan | 0.001 / 0.0015 / 0.01 | (0.0, 0.99) | 3 / 1.25 | 0.05 | 60% / 5% | Adam |
| gan_legacy | 0.0006 / 0.0009 / 0.006 | (0.0, 0.999) | 1 / 1 | 1 | 60% / 5% | Adam |
| shared_b05_999 | 0.00425 / 0.00425 / 0.0085 | (0.5, 0.999) | 3 / 1.25 | 0.05 | 60% / 5% | Adam |
| ratio_g25_d15_p85 | 0.0025 / 0.0015 / 0.0085 | (0.0, 0.99) | 3 / 1.25 | 0.05 | 60% / 5% | Adam |
| ratio_g425_d2125_p85 | 0.00425 / 0.002125 / 0.0085 | (0.0, 0.99) | 3 / 1.25 | 0.05 | 60% / 5% | Adam |
| ratio_g25_d375_p100 | 0.0025 / 0.00375 / 0.01 | (0.0, 0.99) | 3 / 1.25 | 0.05 | 60% / 5% | Adam |
| shared_k10 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 3 / 1 | 0.05 | 60% / 5% | Adam |
| shared_k15 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 3 / 1.5 | 0.05 | 60% / 5% | Adam |
| shared_c10 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 10 / 1.25 | 0.05 | 60% / 5% | Adam |
| shared_b0_999 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.999) | 3 / 1.25 | 0.05 | 60% / 5% | Adam |
| shared_p3 | 0.00425 / 0.00425 / 0.01275 | (0.0, 0.99) | 3 / 1.25 | 0.05 | 60% / 5% | Adam |
| ratio_g34_d255_p85 | 0.0034 / 0.00255 / 0.0085 | (0.0, 0.99) | 3 / 1.25 | 0.05 | 60% / 5% | Adam |
| relative_cap_025 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 3 / 1.25 | 0.05 | 60% / 5% | Adam + relative step cap 0.025 |
| shared_b05 | 0.00425 / 0.00425 / 0.0085 | (0.5, 0.99) | 3 / 1.25 | 0.05 | 60% / 5% | Adam |
| relative_cap_01 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 3 / 1.25 | 0.05 | 60% / 5% | Adam + relative step cap 0.01 |
| shared_c20 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 20 / 1.25 | 0.05 | 60% / 5% | Adam |
| shared_p4 | 0.00425 / 0.00425 / 0.017 | (0.0, 0.99) | 3 / 1.25 | 0.05 | 60% / 5% | Adam |
| sched_h50_f01 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 6 / 1.25 | 0.05 | 50% / 1% | Adam |
| sched_h40_f01 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 6 / 1.25 | 0.05 | 40% / 1% | Adam |
| c6_profile_prior4 | 0.00425 / 0.00425 / 0.017 | (0.0, 0.99) | 6 / 1.25 | 0.05 | 60% / 5% | Adam |
| sched_h30_f01 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 6 / 1.25 | 0.05 | 30% / 1% | Adam |
| sched_h30_f05 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 6 / 1.25 | 0.05 | 30% / 5% | Adam |
| c6_profile_prior3 | 0.00425 / 0.00425 / 0.01275 | (0.0, 0.99) | 6 / 1.25 | 0.05 | 60% / 5% | Adam |
| c6_profile_beta2_095 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.95) | 6 / 1.25 | 0.05 | 60% / 5% | Adam |
| c6_profile_beta1_01 | 0.00425 / 0.00425 / 0.0085 | (0.1, 0.99) | 6 / 1.25 | 0.05 | 60% / 5% | Adam |

All current entries use Rp logistic and no particle L2. The schedule holds rates for the declared fraction of each budget, then follows cosine toward the declared floor. Recorded schedule actions are checked against each recipe. Rates above are absolute and are applied to every optimizer group, including directly optimized particles and AE prior groups.

The update-rule column declares any additional transformation of the Adam proposal. Its complete equation and identical global parameters are retained in leaderboard.json and each episode; reported LRs are the base schedule before that transformation.

## Every test: leading complete candidates and public baselines

All candidate metrics and EMA profiles remain in [leaderboard.json](leaderboard.json).

| Test | shared_c6 | lr00425_prior2 | ratio_g34_d68_p85 | gan | gan_legacy |
| --- | --- | --- | --- | --- | --- |
| two_pole | [PASS](runs/round1-0/episodes/shared_c6__two_pole.json.gz) | [PASS](runs/round0-2/episodes/lr00425_prior2__two_pole.json.gz) | [PASS](runs/shared-ratio-search/screen/episodes/ratio_g34_d68_p85__two_pole.json.gz) | [FAIL](../default_comparison/proposed/episodes/proposed__two_pole.json.gz) | [FAIL](../default_comparison/current/episodes/current__two_pole.json.gz) |
| trajectory | [PASS](runs/completion-1/episodes/shared_c6__trajectory.json.gz) | [PASS](runs/round0-2/episodes/lr00425_prior2__trajectory.json.gz) | [PASS](runs/shared-ratio-search/completion/episodes/ratio_g34_d68_p85__trajectory.json.gz) | [FAIL](../default_comparison/proposed/episodes/proposed__trajectory.json.gz) | [FAIL](../default_comparison/current/episodes/current__trajectory.json.gz) |
| residual_student | [PASS](runs/completion-1/episodes/shared_c6__residual_student.json.gz) | [PASS](runs/round0-2/episodes/lr00425_prior2__residual_student.json.gz) | [PASS](runs/shared-ratio-search/completion/episodes/ratio_g34_d68_p85__residual_student.json.gz) | [FAIL](../default_comparison/proposed/episodes/proposed__residual_student.json.gz) | [FAIL](../default_comparison/current/episodes/current__residual_student.json.gz) |
| unipolar | [PASS](runs/completion-1/episodes/shared_c6__unipolar.json.gz) | [PASS](runs/round0-2/episodes/lr00425_prior2__unipolar.json.gz) | [PASS](runs/shared-ratio-search/completion/episodes/ratio_g34_d68_p85__unipolar.json.gz) | [FAIL](../default_comparison/proposed/episodes/proposed__unipolar.json.gz) | [FAIL](../default_comparison/current/episodes/current__unipolar.json.gz) |
| ae_gan_hold | [PASS](runs/completion-1/episodes/shared_c6__ae_gan_hold.json.gz) | [PASS](runs/round0-2/episodes/lr00425_prior2__ae_gan_hold.json.gz) | [PASS](runs/shared-ratio-search/completion/episodes/ratio_g34_d68_p85__ae_gan_hold.json.gz) | [PASS](../default_comparison/proposed/episodes/proposed__ae_gan_hold.json.gz) | [PASS](../default_comparison/current/episodes/current__ae_gan_hold.json.gz) |
| cover_leftover | [PASS](runs/completion-1/episodes/shared_c6__cover_leftover.json.gz) | [PASS](runs/round0-2/episodes/lr00425_prior2__cover_leftover.json.gz) | [PASS](runs/shared-ratio-search/completion/episodes/ratio_g34_d68_p85__cover_leftover.json.gz) | [FAIL](../default_comparison/proposed/episodes/proposed__cover_leftover.json.gz) | [FAIL](../default_comparison/current/episodes/current__cover_leftover.json.gz) |
| unused_token_hold | [PASS](runs/completion-1/episodes/shared_c6__unused_token_hold.json.gz) | [PASS](runs/round0-2/episodes/lr00425_prior2__unused_token_hold.json.gz) | [PASS](runs/shared-ratio-search/completion/episodes/ratio_g34_d68_p85__unused_token_hold.json.gz) | [FAIL](../default_comparison/proposed/episodes/proposed__unused_token_hold.json.gz) | [FAIL](../default_comparison/current/episodes/current__unused_token_hold.json.gz) |
| mid_scale_identity | [PASS](runs/completion-1/episodes/shared_c6__mid_scale_identity.json.gz) | [PASS](runs/round0-2/episodes/lr00425_prior2__mid_scale_identity.json.gz) | [PASS](runs/shared-ratio-search/completion/episodes/ratio_g34_d68_p85__mid_scale_identity.json.gz) | [FAIL](../default_comparison/proposed/episodes/proposed__mid_scale_identity.json.gz) | [FAIL](../default_comparison/current/episodes/current__mid_scale_identity.json.gz) |
| mode_hold | [PASS](runs/round1-0/episodes/shared_c6__mode_hold.json.gz) | [FAIL](runs/round0-2/episodes/lr00425_prior2__mode_hold.json.gz) | [FAIL](runs/shared-ratio-search/screen/episodes/ratio_g34_d68_p85__mode_hold.json.gz) | [FAIL](../default_comparison/proposed/episodes/proposed__mode_hold.json.gz) | [FAIL](../default_comparison/current/episodes/current__mode_hold.json.gz) |
| vector_two_broad | [PASS](runs/completion-1/episodes/shared_c6__vector_two_broad.json.gz) | [PASS](runs/round0-2/episodes/lr00425_prior2__vector_two_broad.json.gz) | [PASS](runs/shared-ratio-search/completion/episodes/ratio_g34_d68_p85__vector_two_broad.json.gz) | [PASS](../default_comparison/proposed/episodes/proposed__vector_two_broad.json.gz) | [PASS](../default_comparison/current/episodes/current__vector_two_broad.json.gz) |
| vector_unequal_mass | [PASS](runs/shared-batch-feature-search/screen/episodes/shared_c6__batchfeat_center6_distance_head__vector_unequal_mass.json.gz) | [FAIL](runs/round0-2/episodes/lr00425_prior2__vector_unequal_mass.json.gz) | [FAIL](runs/shared-ratio-search/screen/episodes/ratio_g34_d68_p85__vector_unequal_mass.json.gz) | [PASS](../default_comparison/proposed/episodes/proposed__vector_unequal_mass.json.gz) | [FAIL](../default_comparison/current/episodes/current__vector_unequal_mass.json.gz) |
| vector_unequal_width | [PASS](runs/shared-width-search/last_refinement/episodes/shared_c6__width_last_softplus8_128_l3__vector_unequal_width.json.gz) | [FAIL](runs/round0-2/episodes/lr00425_prior2__vector_unequal_width.json.gz) | [FAIL](runs/shared-ratio-search/screen/episodes/ratio_g34_d68_p85__vector_unequal_width.json.gz) | [PASS](../default_comparison/proposed/episodes/proposed__vector_unequal_width.json.gz) | [FAIL](../default_comparison/current/episodes/current__vector_unequal_width.json.gz) |
| vector_anisotropic | [PASS](runs/shared-discriminator-search/cross/episodes/shared_c6__additive_raw_fourier64_l2__vector_anisotropic.json.gz) | [PASS](runs/round0-2/episodes/lr00425_prior2__vector_anisotropic.json.gz) | [FAIL](runs/shared-ratio-search/screen/episodes/ratio_g34_d68_p85__vector_anisotropic.json.gz) | [PASS](../default_comparison/proposed/episodes/proposed__vector_anisotropic.json.gz) | [PASS](../default_comparison/current/episodes/current__vector_anisotropic.json.gz) |
| vector_overlap | [PASS](runs/shared-discriminator-search/cross/episodes/shared_c6__raw_softplus96_l3__vector_overlap.json.gz) | [FAIL](runs/round0-2/episodes/lr00425_prior2__vector_overlap.json.gz) | [PASS](runs/shared-ratio-search/screen/episodes/ratio_g34_d68_p85__vector_overlap.json.gz) | [PASS](../default_comparison/proposed/episodes/proposed__vector_overlap.json.gz) | [FAIL](../default_comparison/current/episodes/current__vector_overlap.json.gz) |
| vector_spiral | [PASS](runs/completion-1/episodes/shared_c6__vector_spiral.json.gz) | [PASS](runs/round0-2/episodes/lr00425_prior2__vector_spiral.json.gz) | [PASS](runs/shared-ratio-search/completion/episodes/ratio_g34_d68_p85__vector_spiral.json.gz) | [PASS](../default_comparison/proposed/episodes/proposed__vector_spiral.json.gz) | [PASS](../default_comparison/current/episodes/current__vector_spiral.json.gz) |
| img_stripes2 | [PASS](runs/completion-1/episodes/shared_c6__img_stripes2.json.gz) | [PASS](runs/round0-2/episodes/lr00425_prior2__img_stripes2.json.gz) | [PASS](runs/shared-ratio-search/completion/episodes/ratio_g34_d68_p85__img_stripes2.json.gz) | [PASS](../default_comparison/proposed/episodes/proposed__img_stripes2.json.gz) | [PASS](../default_comparison/current/episodes/current__img_stripes2.json.gz) |
| img_bars4 | [PASS](runs/round1-0/episodes/shared_c6__img_bars4.json.gz) | [PASS](runs/round0-2/episodes/lr00425_prior2__img_bars4.json.gz) | [PASS](runs/shared-ratio-search/completion/episodes/ratio_g34_d68_p85__img_bars4.json.gz) | [FAIL](../default_comparison/proposed/episodes/proposed__img_bars4.json.gz) | [FAIL](../default_comparison/current/episodes/current__img_bars4.json.gz) |
| img_blobs4 | [PASS](runs/completion-1/episodes/shared_c6__img_blobs4.json.gz) | [PASS](runs/round0-2/episodes/lr00425_prior2__img_blobs4.json.gz) | [PASS](runs/shared-ratio-search/screen/episodes/ratio_g34_d68_p85__img_blobs4.json.gz) | [FAIL](../default_comparison/proposed/episodes/proposed__img_blobs4.json.gz) | [FAIL](../default_comparison/current/episodes/current__img_blobs4.json.gz) |
| img_intensity2 | [PASS](runs/completion-1/episodes/shared_c6__img_intensity2.json.gz) | [PASS](runs/round0-2/episodes/lr00425_prior2__img_intensity2.json.gz) | [PASS](runs/shared-ratio-search/completion/episodes/ratio_g34_d68_p85__img_intensity2.json.gz) | [FAIL](../default_comparison/proposed/episodes/proposed__img_intensity2.json.gz) | [FAIL](../default_comparison/current/episodes/current__img_intensity2.json.gz) |

## Selected data architectures: `shared_c6`

These choices share the exact recipe above. The passing streak counts consecutive observations ending at the final checkpoint; at least five are required. Every alternative and failure remains in the trial table below.

| Test | Discriminator | Live | Final passing streak | EMA |
| --- | --- | --- | ---: | --- |
| vector_two_broad | original architecture | [PASS](runs/completion-1/episodes/shared_c6__vector_two_broad.json.gz) | 20/24 | PASS |
| vector_unequal_mass | batchfeat_center6_distance_head | [PASS](runs/shared-batch-feature-search/screen/episodes/shared_c6__batchfeat_center6_distance_head__vector_unequal_mass.json.gz) | 7/24 | FAIL |
| vector_unequal_width | width_last_softplus8_128_l3 | [PASS](runs/shared-width-search/last_refinement/episodes/shared_c6__width_last_softplus8_128_l3__vector_unequal_width.json.gz) | 5/24 | PASS |
| vector_anisotropic | additive_raw_fourier64_l2 | [PASS](runs/shared-discriminator-search/cross/episodes/shared_c6__additive_raw_fourier64_l2__vector_anisotropic.json.gz) | 8/24 | PASS |
| vector_overlap | raw_softplus96_l3 | [PASS](runs/shared-discriminator-search/cross/episodes/shared_c6__raw_softplus96_l3__vector_overlap.json.gz) | 10/24 | PASS |
| vector_spiral | original architecture | [PASS](runs/completion-1/episodes/shared_c6__vector_spiral.json.gz) | 23/24 | PASS |

## Remaining failures in the leading complete recipes

The final passing streak must reach five observations. A good last checkpoint alone does not pass.

| Recipe | Test | Final failing metrics (value; required bound) | Final passing streak |
| --- | --- | --- | ---: |
| lr00425_prior2 | [mode_hold](runs/round0-2/episodes/lr00425_prior2__mode_hold.json.gz) | modes: 7; needs >= 8 | 0/5 |
| lr00425_prior2 | [vector_unequal_mass](runs/round0-2/episodes/lr00425_prior2__vector_unequal_mass.json.gz) | component_min_eigen_ratio: 5.0555e-05; needs >= 0.15 | 0/5 |
| lr00425_prior2 | [vector_unequal_width](runs/round0-2/episodes/lr00425_prior2__vector_unequal_width.json.gz) | component_min_eigen_ratio: 0.0048636; needs >= 0.15 | 0/5 |
| lr00425_prior2 | [vector_overlap](runs/round0-2/episodes/lr00425_prior2__vector_overlap.json.gz) | Final metrics pass | 3/5 |
| ratio_g34_d68_p85 | [mode_hold](runs/shared-ratio-search/screen/episodes/ratio_g34_d68_p85__mode_hold.json.gz) | modes: 7; needs >= 8 | 0/5 |
| ratio_g34_d68_p85 | [vector_unequal_mass](runs/shared-ratio-search/screen/episodes/ratio_g34_d68_p85__vector_unequal_mass.json.gz) | component_min_eigen_ratio: 9.6136e-07; needs >= 0.15 | 0/5 |
| ratio_g34_d68_p85 | [vector_unequal_width](runs/shared-ratio-search/screen/episodes/ratio_g34_d68_p85__vector_unequal_width.json.gz) | component_covariance_error: 2.0177; needs <= 0.85; component_min_eigen_ratio: 0.058472; needs >= 0.15 | 0/5 |
| ratio_g34_d68_p85 | [vector_anisotropic](runs/shared-ratio-search/screen/episodes/ratio_g34_d68_p85__vector_anisotropic.json.gz) | component_covariance_error: 1.1134; needs <= 0.85 | 0/5 |

## Discriminator architecture trials

Architecture support is within a single unchanged recipe. All trials are shown, including failures; it does not mean one universal discriminator works everywhere.

<details><summary>All 244 architecture trials, including failures</summary>

| Recipe | Test | Discriminator | Live |
| --- | --- | --- | --- |
| shared_c6 | vector_unequal_mass | raw_softplus96_l3 | [FAIL](runs/shared-discriminator-search/screen/episodes/shared_c6__raw_softplus96_l3__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | raw_silu128_l3 | [FAIL](runs/shared-discriminator-search/screen/episodes/shared_c6__raw_silu128_l3__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | quadratic_softplus96_l2 | [FAIL](runs/shared-discriminator-search/screen/episodes/shared_c6__quadratic_softplus96_l2__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | quadratic_tanh96_l3 | [FAIL](runs/shared-discriminator-search/screen/episodes/shared_c6__quadratic_tanh96_l3__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | residual_raw_softplus96_l3 | [FAIL](runs/shared-discriminator-search/screen/episodes/shared_c6__residual_raw_softplus96_l3__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | residual_lowfreq_softplus96_l3 | [FAIL](runs/shared-discriminator-search/screen/episodes/shared_c6__residual_lowfreq_softplus96_l3__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | halfscore_fourier_skip96_l2 | [FAIL](runs/shared-discriminator-search/screen/episodes/shared_c6__halfscore_fourier_skip96_l2__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | additive_raw_fourier64_l2 | [FAIL](runs/shared-discriminator-search/screen/episodes/shared_c6__additive_raw_fourier64_l2__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | local_rbf64_direct | [FAIL](runs/shared-local-density-search/screen/episodes/shared_c6__local_rbf64_direct__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | local_rbf128_direct | [FAIL](runs/shared-local-density-search/screen/episodes/shared_c6__local_rbf128_direct__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | local_rbf256_direct | [FAIL](runs/shared-local-density-search/screen/episodes/shared_c6__local_rbf256_direct__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | local_rbf128_adaptive | [FAIL](runs/shared-local-density-search/screen/episodes/shared_c6__local_rbf128_adaptive__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | local_cauchy128_direct | [FAIL](runs/shared-local-density-search/screen/episodes/shared_c6__local_cauchy128_direct__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | local_rbf128_softplus64 | [FAIL](runs/shared-local-density-search/screen/episodes/shared_c6__local_rbf128_softplus64__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | local_quad16_width1 | [FAIL](runs/shared-local-density-search/screen/episodes/shared_c6__local_quad16_width1__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | local_quad32_width1 | [FAIL](runs/shared-local-density-search/screen/episodes/shared_c6__local_quad32_width1__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | local_quad32_width05 | [FAIL](runs/shared-local-density-search/screen/episodes/shared_c6__local_quad32_width05__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | local_quad32_adaptive | [FAIL](runs/shared-local-density-search/screen/episodes/shared_c6__local_quad32_adaptive__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | local_product_silu64_l2 | [FAIL](runs/shared-local-density-search/screen/episodes/shared_c6__local_product_silu64_l2__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | local_product_silu96_l2 | [FAIL](runs/shared-local-density-search/screen/episodes/shared_c6__local_product_silu96_l2__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | local_product_silu128_l2 | [FAIL](runs/shared-local-density-search/screen/episodes/shared_c6__local_product_silu128_l2__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | local_squared_silu64_l3 | [FAIL](runs/shared-local-density-search/screen/episodes/shared_c6__local_squared_silu64_l3__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | local_squared_silu96_l3 | [FAIL](runs/shared-local-density-search/screen/episodes/shared_c6__local_squared_silu96_l3__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | local_product_softplus96_l2 | [FAIL](runs/shared-local-density-search/screen/episodes/shared_c6__local_product_softplus96_l2__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | curvature_raw_silu128_l3_q32_w1p0 | [FAIL](runs/shared-local-density-search/refinement/episodes/shared_c6__curvature_raw_silu128_l3_q32_w1p0__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | curvature_raw_silu128_l3_q64_w1p0 | [FAIL](runs/shared-local-density-search/refinement/episodes/shared_c6__curvature_raw_silu128_l3_q64_w1p0__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | curvature_raw_silu128_l3_q32_w0p5 | [FAIL](runs/shared-local-density-search/refinement/episodes/shared_c6__curvature_raw_silu128_l3_q32_w0p5__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | curvature_raw_softplus96_l3_q32_w1p0 | [FAIL](runs/shared-local-density-search/refinement/episodes/shared_c6__curvature_raw_softplus96_l3_q32_w1p0__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | curvature_raw_softplus96_l3_q64_w1p0 | [FAIL](runs/shared-local-density-search/refinement/episodes/shared_c6__curvature_raw_softplus96_l3_q64_w1p0__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | curvature_raw_softplus96_l3_q32_w0p5 | [FAIL](runs/shared-local-density-search/refinement/episodes/shared_c6__curvature_raw_softplus96_l3_q32_w0p5__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | width_last_softplus8_128_l3 | [FAIL](runs/shared-width-search/cross/episodes/shared_c6__width_last_softplus8_128_l3__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | ensemble2_equal_softplus64_l2 | [FAIL](runs/shared-ensemble-search/screen/episodes/shared_c6__ensemble2_equal_softplus64_l2__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | ensemble2_multiscale_softplus64_l2 | [FAIL](runs/shared-ensemble-search/screen/episodes/shared_c6__ensemble2_multiscale_softplus64_l2__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | ensemble3_multiscale_softplus64_l2 | [FAIL](runs/shared-ensemble-search/screen/episodes/shared_c6__ensemble3_multiscale_softplus64_l2__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | ensemble3_multiscale_silu64_l2 | [FAIL](runs/shared-ensemble-search/screen/episodes/shared_c6__ensemble3_multiscale_silu64_l2__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | ensemble2_multiscale_silu64_l3 | [FAIL](runs/shared-ensemble-search/screen/episodes/shared_c6__ensemble2_multiscale_silu64_l3__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | ensemble3_broad_softplus48_l3 | [FAIL](runs/shared-ensemble-search/screen/episodes/shared_c6__ensemble3_broad_softplus48_l3__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | ensemble4_broad_silu48_l2 | [FAIL](runs/shared-ensemble-search/screen/episodes/shared_c6__ensemble4_broad_silu48_l2__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | ensemble2_highscale_softplus96_l2 | [FAIL](runs/shared-ensemble-search/screen/episodes/shared_c6__ensemble2_highscale_softplus96_l2__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | ensemble2_mixed64_l2 | [FAIL](runs/shared-ensemble-search/screen/episodes/shared_c6__ensemble2_mixed64_l2__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | ensemble2_multiscale_halfscore64_l2 | [FAIL](runs/shared-ensemble-search/screen/episodes/shared_c6__ensemble2_multiscale_halfscore64_l2__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | ensemble2_multiscale_fullscore64_l2 | [FAIL](runs/shared-ensemble-search/screen/episodes/shared_c6__ensemble2_multiscale_fullscore64_l2__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | ensemble2_multiscale_spectrum64_l2 | [FAIL](runs/shared-ensemble-search/screen/episodes/shared_c6__ensemble2_multiscale_spectrum64_l2__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | pointnorm_layer_first_softplus96_l3 | [FAIL](runs/shared-pointnorm-search/screen/episodes/shared_c6__pointnorm_layer_first_softplus96_l3__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | pointnorm_layer_all_softplus96_l3 | [FAIL](runs/shared-pointnorm-search/screen/episodes/shared_c6__pointnorm_layer_all_softplus96_l3__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | pointnorm_layer_first_silu128_l3 | [FAIL](runs/shared-pointnorm-search/screen/episodes/shared_c6__pointnorm_layer_first_silu128_l3__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | pointnorm_layer_all_silu128_l3 | [FAIL](runs/shared-pointnorm-search/screen/episodes/shared_c6__pointnorm_layer_all_silu128_l3__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | pointnorm_rms_all_silu128_l3 | [FAIL](runs/shared-pointnorm-search/screen/episodes/shared_c6__pointnorm_rms_all_silu128_l3__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | pointnorm_weight_all_silu128_l3 | [FAIL](runs/shared-pointnorm-search/screen/episodes/shared_c6__pointnorm_weight_all_silu128_l3__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | pointnorm_layer_all_softplus64_l3 | [FAIL](runs/shared-pointnorm-search/refinement-rare/episodes/shared_c6__pointnorm_layer_all_softplus64_l3__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | pointnorm_layer_all_softplus128_l3 | [FAIL](runs/shared-pointnorm-search/refinement-rare/episodes/shared_c6__pointnorm_layer_all_softplus128_l3__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | pointnorm_layer_all_softplus160_l3 | [FAIL](runs/shared-pointnorm-search/refinement-rare/episodes/shared_c6__pointnorm_layer_all_softplus160_l3__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | pointnorm_layer_all_softplus96_beta2_l3 | [FAIL](runs/shared-pointnorm-search/refinement-rare/episodes/shared_c6__pointnorm_layer_all_softplus96_beta2_l3__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | pointnorm_layer_all_softplus96_beta10_l3 | [FAIL](runs/shared-pointnorm-search/refinement-rare/episodes/shared_c6__pointnorm_layer_all_softplus96_beta10_l3__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | pointnorm_layer_all_softplus96_skip_l3 | [FAIL](runs/shared-pointnorm-search/refinement-rare/episodes/shared_c6__pointnorm_layer_all_softplus96_skip_l3__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | pointnorm_layer_all_softplus96_beta3_l3 | [FAIL](runs/shared-pointnorm-search/beta-interpolation-rare/episodes/shared_c6__pointnorm_layer_all_softplus96_beta3_l3__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | pointnorm_layer_all_softplus96_beta4_l3 | [FAIL](runs/shared-pointnorm-search/beta-interpolation-rare/episodes/shared_c6__pointnorm_layer_all_softplus96_beta4_l3__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | pointnorm_layer_all_softplus96_beta6_l3 | [FAIL](runs/shared-pointnorm-search/beta-interpolation-rare/episodes/shared_c6__pointnorm_layer_all_softplus96_beta6_l3__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | pointnorm_layer_all_softplus96_beta8_l3 | [FAIL](runs/shared-pointnorm-search/beta-interpolation-rare/episodes/shared_c6__pointnorm_layer_all_softplus96_beta8_l3__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | rare_sn_raw_first | [FAIL](runs/shared-rare-gradient/screen/episodes/shared_c6__rare_sn_raw_first__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | rare_sn_raw_hidden | [FAIL](runs/shared-rare-gradient/screen/episodes/shared_c6__rare_sn_raw_hidden__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | rare_sn_raw_head | [FAIL](runs/shared-rare-gradient/screen/episodes/shared_c6__rare_sn_raw_head__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | rare_sn_raw_all | [FAIL](runs/shared-rare-gradient/screen/episodes/shared_c6__rare_sn_raw_all__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | rare_sn_layer_first | [FAIL](runs/shared-rare-gradient/screen/episodes/shared_c6__rare_sn_layer_first__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | rare_sn_layer_hidden | [FAIL](runs/shared-rare-gradient/screen/episodes/shared_c6__rare_sn_layer_hidden__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | rare_sn_layer_head | [FAIL](runs/shared-rare-gradient/screen/episodes/shared_c6__rare_sn_layer_head__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | rare_sn_layer_all | [FAIL](runs/shared-rare-gradient/screen/episodes/shared_c6__rare_sn_layer_all__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | rare_bound_layer_1 | [FAIL](runs/shared-rare-gradient/screen/episodes/shared_c6__rare_bound_layer_1__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | rare_bound_layer_2 | [FAIL](runs/shared-rare-gradient/screen/episodes/shared_c6__rare_bound_layer_2__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | rare_bound_layer_4 | [FAIL](runs/shared-rare-gradient/screen/episodes/shared_c6__rare_bound_layer_4__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | rare_bound_raw_2 | [FAIL](runs/shared-rare-gradient/screen/episodes/shared_c6__rare_bound_raw_2__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_ln_pre_fixed | [FAIL](runs/shared-norm-structure-search/screen/episodes/shared_c6__normstruct_ln_pre_fixed__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_ln_post_affine | [FAIL](runs/shared-norm-structure-search/screen/episodes/shared_c6__normstruct_ln_post_affine__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_ln_post_fixed | [FAIL](runs/shared-norm-structure-search/screen/episodes/shared_c6__normstruct_ln_post_fixed__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_ln_first_last | [FAIL](runs/shared-norm-structure-search/screen/episodes/shared_c6__normstruct_ln_first_last__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_ln_last | [FAIL](runs/shared-norm-structure-search/screen/episodes/shared_c6__normstruct_ln_last__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_rms_pre_affine | [FAIL](runs/shared-norm-structure-search/screen/episodes/shared_c6__normstruct_rms_pre_affine__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_rms_pre_fixed | [FAIL](runs/shared-norm-structure-search/screen/episodes/shared_c6__normstruct_rms_pre_fixed__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_rms_post_affine | [FAIL](runs/shared-norm-structure-search/screen/episodes/shared_c6__normstruct_rms_post_affine__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_center_pre_affine | [FAIL](runs/shared-norm-structure-search/screen/episodes/shared_c6__normstruct_center_pre_affine__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_center_pre_fixed | [FAIL](runs/shared-norm-structure-search/screen/episodes/shared_c6__normstruct_center_pre_fixed__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_ln_raw_blend025 | [FAIL](runs/shared-norm-structure-search/screen/episodes/shared_c6__normstruct_ln_raw_blend025__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_ln_raw_blend050 | [FAIL](runs/shared-norm-structure-search/screen/episodes/shared_c6__normstruct_ln_raw_blend050__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_ln_post_raw_blend025 | [FAIL](runs/shared-norm-structure-search/screen/episodes/shared_c6__normstruct_ln_post_raw_blend025__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_ln_residual025 | [FAIL](runs/shared-norm-structure-search/screen/episodes/shared_c6__normstruct_ln_residual025__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_ln_input_injection025 | [FAIL](runs/shared-norm-structure-search/screen/episodes/shared_c6__normstruct_ln_input_injection025__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_ln_rms_mix050 | [FAIL](runs/shared-norm-structure-search/screen/episodes/shared_c6__normstruct_ln_rms_mix050__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_power025_fixed | [FAIL](runs/shared-norm-structure-search/refinement/episodes/shared_c6__normstruct_power025_fixed__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_power050_fixed | [FAIL](runs/shared-norm-structure-search/refinement/episodes/shared_c6__normstruct_power050_fixed__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_power075_fixed | [FAIL](runs/shared-norm-structure-search/refinement/episodes/shared_c6__normstruct_power075_fixed__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_power025_affine | [FAIL](runs/shared-norm-structure-search/refinement/episodes/shared_c6__normstruct_power025_affine__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_power050_affine | [FAIL](runs/shared-norm-structure-search/refinement/episodes/shared_c6__normstruct_power050_affine__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_group2_affine | [FAIL](runs/shared-norm-structure-search/refinement/episodes/shared_c6__normstruct_group2_affine__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_group4_affine | [FAIL](runs/shared-norm-structure-search/refinement/episodes/shared_c6__normstruct_group4_affine__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_group2_fixed | [FAIL](runs/shared-norm-structure-search/refinement/episodes/shared_c6__normstruct_group2_fixed__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_ln_blend050_first | [FAIL](runs/shared-norm-structure-search/refinement/episodes/shared_c6__normstruct_ln_blend050_first__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_ln_blend050_last | [FAIL](runs/shared-norm-structure-search/refinement/episodes/shared_c6__normstruct_ln_blend050_last__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_ln_blend050_first_last | [FAIL](runs/shared-norm-structure-search/refinement/episodes/shared_c6__normstruct_ln_blend050_first_last__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_center_fixed_blend025 | [FAIL](runs/shared-norm-structure-search/refinement/episodes/shared_c6__normstruct_center_fixed_blend025__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_center_fixed96_beta3 | [FAIL](runs/shared-norm-structure-search/center-followup/episodes/shared_c6__normstruct_center_fixed96_beta3__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_center_fixed96_beta5 | [FAIL](runs/shared-norm-structure-search/center-followup/episodes/shared_c6__normstruct_center_fixed96_beta5__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_center_fixed96_beta6 | [FAIL](runs/shared-norm-structure-search/center-followup/episodes/shared_c6__normstruct_center_fixed96_beta6__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_center_fixed96_beta8 | [FAIL](runs/shared-norm-structure-search/center-followup/episodes/shared_c6__normstruct_center_fixed96_beta8__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_center_fixed128_beta4 | [FAIL](runs/shared-norm-structure-search/center-followup/episodes/shared_c6__normstruct_center_fixed128_beta4__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | normstruct_center_fixed128_beta8 | [FAIL](runs/shared-norm-structure-search/center-followup/episodes/shared_c6__normstruct_center_fixed128_beta8__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | geometry_poly2_lnsp96 | [FAIL](runs/shared-geometry-search/screen/episodes/shared_c6__geometry_poly2_lnsp96__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | geometry_poly3_lnsp96 | [FAIL](runs/shared-geometry-search/screen/episodes/shared_c6__geometry_poly3_lnsp96__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | geometry_tanhcoords_lnsp96 | [FAIL](runs/shared-geometry-search/screen/episodes/shared_c6__geometry_tanhcoords_lnsp96__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | geometry_asinhcoords_lnsp96 | [FAIL](runs/shared-geometry-search/screen/episodes/shared_c6__geometry_asinhcoords_lnsp96__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | geometry_ridgehinge2_lnsp96 | [FAIL](runs/shared-geometry-search/screen/episodes/shared_c6__geometry_ridgehinge2_lnsp96__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | geometry_ridgehinge8_lnsp96 | [FAIL](runs/shared-geometry-search/screen/episodes/shared_c6__geometry_ridgehinge8_lnsp96__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | geometry_ridgetanh2_lnsp96 | [FAIL](runs/shared-geometry-search/screen/episodes/shared_c6__geometry_ridgetanh2_lnsp96__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | geometry_ridgewindow4_lnsp96 | [FAIL](runs/shared-geometry-search/screen/episodes/shared_c6__geometry_ridgewindow4_lnsp96__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | geometry_raw_lngelu96 | [FAIL](runs/shared-geometry-search/screen/episodes/shared_c6__geometry_raw_lngelu96__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | geometry_raw_lnmish96 | [FAIL](runs/shared-geometry-search/screen/episodes/shared_c6__geometry_raw_lnmish96__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | geometry_raw_lntanh96 | [FAIL](runs/shared-geometry-search/screen/episodes/shared_c6__geometry_raw_lntanh96__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | geometry_raw_lngate96 | [FAIL](runs/shared-geometry-search/screen/episodes/shared_c6__geometry_raw_lngate96__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | geometry_poly2_lngate96 | [FAIL](runs/shared-geometry-search/screen/episodes/shared_c6__geometry_poly2_lngate96__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | geometry_raw_lnquadhead96 | [FAIL](runs/shared-geometry-search/screen/episodes/shared_c6__geometry_raw_lnquadhead96__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | geometry_poly2_lnquadhead96 | [FAIL](runs/shared-geometry-search/screen/episodes/shared_c6__geometry_poly2_lnquadhead96__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | geometry_resid_poly2_first | [FAIL](runs/shared-geometry-search/residual/episodes/shared_c6__geometry_resid_poly2_first__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | geometry_resid_hinge_first | [FAIL](runs/shared-geometry-search/residual/episodes/shared_c6__geometry_resid_hinge_first__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | geometry_resid_hinge_all | [FAIL](runs/shared-geometry-search/residual/episodes/shared_c6__geometry_resid_hinge_all__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | geometry_resid_raw_later | [FAIL](runs/shared-geometry-search/residual/episodes/shared_c6__geometry_resid_raw_later__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | geometry_resid_poly2_later | [FAIL](runs/shared-geometry-search/residual/episodes/shared_c6__geometry_resid_poly2_later__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | geometry_resid_hinge_later | [FAIL](runs/shared-geometry-search/residual/episodes/shared_c6__geometry_resid_hinge_later__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | geometry_resid_poly2_head | [FAIL](runs/shared-geometry-search/residual/episodes/shared_c6__geometry_resid_poly2_head__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | geometry_resid_hinge_head | [FAIL](runs/shared-geometry-search/residual/episodes/shared_c6__geometry_resid_hinge_head__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | geometry_resid_tanh_activation | [FAIL](runs/shared-geometry-search/residual/episodes/shared_c6__geometry_resid_tanh_activation__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | geometry_readout_asinh2 | [FAIL](runs/shared-geometry-search/readout/episodes/shared_c6__geometry_readout_asinh2__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | geometry_readout_rational2 | [FAIL](runs/shared-geometry-search/readout/episodes/shared_c6__geometry_readout_rational2__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | batchfeat_center6_std_scalar | [FAIL](runs/shared-batch-feature-search/screen/episodes/shared_c6__batchfeat_center6_std_scalar__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | batchfeat_center6_std_vector | [FAIL](runs/shared-batch-feature-search/screen/episodes/shared_c6__batchfeat_center6_std_vector__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | batchfeat_layer4_std_scalar | [FAIL](runs/shared-batch-feature-search/screen/episodes/shared_c6__batchfeat_layer4_std_scalar__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | batchfeat_center6_density_head | [FAIL](runs/shared-batch-feature-search/screen/episodes/shared_c6__batchfeat_center6_density_head__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_mass | batchfeat_center6_distance_head | [PASS](runs/shared-batch-feature-search/screen/episodes/shared_c6__batchfeat_center6_distance_head__vector_unequal_mass.json.gz) |
| shared_c6 | vector_unequal_width | raw_softplus96_l3 | [FAIL](runs/shared-discriminator-search/screen/episodes/shared_c6__raw_softplus96_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | raw_silu128_l3 | [FAIL](runs/shared-discriminator-search/screen/episodes/shared_c6__raw_silu128_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | quadratic_softplus96_l2 | [FAIL](runs/shared-discriminator-search/screen/episodes/shared_c6__quadratic_softplus96_l2__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | quadratic_tanh96_l3 | [FAIL](runs/shared-discriminator-search/screen/episodes/shared_c6__quadratic_tanh96_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | residual_raw_softplus96_l3 | [FAIL](runs/shared-discriminator-search/screen/episodes/shared_c6__residual_raw_softplus96_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | residual_lowfreq_softplus96_l3 | [FAIL](runs/shared-discriminator-search/screen/episodes/shared_c6__residual_lowfreq_softplus96_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | halfscore_fourier_skip96_l2 | [FAIL](runs/shared-discriminator-search/screen/episodes/shared_c6__halfscore_fourier_skip96_l2__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | additive_raw_fourier64_l2 | [FAIL](runs/shared-discriminator-search/screen/episodes/shared_c6__additive_raw_fourier64_l2__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | local_rbf64_direct | [FAIL](runs/shared-local-density-search/screen/episodes/shared_c6__local_rbf64_direct__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | local_rbf128_direct | [FAIL](runs/shared-local-density-search/screen/episodes/shared_c6__local_rbf128_direct__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | local_rbf256_direct | [FAIL](runs/shared-local-density-search/screen/episodes/shared_c6__local_rbf256_direct__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | local_rbf128_adaptive | [FAIL](runs/shared-local-density-search/screen/episodes/shared_c6__local_rbf128_adaptive__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | local_cauchy128_direct | [FAIL](runs/shared-local-density-search/screen/episodes/shared_c6__local_cauchy128_direct__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | local_rbf128_softplus64 | [FAIL](runs/shared-local-density-search/screen/episodes/shared_c6__local_rbf128_softplus64__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | local_quad16_width1 | [FAIL](runs/shared-local-density-search/screen/episodes/shared_c6__local_quad16_width1__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | local_quad32_width1 | [FAIL](runs/shared-local-density-search/screen/episodes/shared_c6__local_quad32_width1__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | local_quad32_width05 | [FAIL](runs/shared-local-density-search/screen/episodes/shared_c6__local_quad32_width05__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | local_quad32_adaptive | [FAIL](runs/shared-local-density-search/screen/episodes/shared_c6__local_quad32_adaptive__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | local_product_silu64_l2 | [FAIL](runs/shared-local-density-search/screen/episodes/shared_c6__local_product_silu64_l2__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | local_product_silu96_l2 | [FAIL](runs/shared-local-density-search/screen/episodes/shared_c6__local_product_silu96_l2__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | local_product_silu128_l2 | [FAIL](runs/shared-local-density-search/screen/episodes/shared_c6__local_product_silu128_l2__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | local_squared_silu64_l3 | [FAIL](runs/shared-local-density-search/screen/episodes/shared_c6__local_squared_silu64_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | local_squared_silu96_l3 | [FAIL](runs/shared-local-density-search/screen/episodes/shared_c6__local_squared_silu96_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | local_product_softplus96_l2 | [FAIL](runs/shared-local-density-search/screen/episodes/shared_c6__local_product_softplus96_l2__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | curvature_raw_silu128_l3_q32_w1p0 | [FAIL](runs/shared-local-density-search/refinement/episodes/shared_c6__curvature_raw_silu128_l3_q32_w1p0__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | curvature_raw_silu128_l3_q64_w1p0 | [FAIL](runs/shared-local-density-search/refinement/episodes/shared_c6__curvature_raw_silu128_l3_q64_w1p0__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | curvature_raw_silu128_l3_q32_w0p5 | [FAIL](runs/shared-local-density-search/refinement/episodes/shared_c6__curvature_raw_silu128_l3_q32_w0p5__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | curvature_raw_softplus96_l3_q32_w1p0 | [FAIL](runs/shared-local-density-search/refinement/episodes/shared_c6__curvature_raw_softplus96_l3_q32_w1p0__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | curvature_raw_softplus96_l3_q64_w1p0 | [FAIL](runs/shared-local-density-search/refinement/episodes/shared_c6__curvature_raw_softplus96_l3_q64_w1p0__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | curvature_raw_softplus96_l3_q32_w0p5 | [FAIL](runs/shared-local-density-search/refinement/episodes/shared_c6__curvature_raw_softplus96_l3_q32_w0p5__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_raw_silu64_l3 | [FAIL](runs/shared-width-search/screen/episodes/shared_c6__width_raw_silu64_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_raw_silu96_l3 | [FAIL](runs/shared-width-search/screen/episodes/shared_c6__width_raw_silu96_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_raw_silu160_l3 | [FAIL](runs/shared-width-search/screen/episodes/shared_c6__width_raw_silu160_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_raw_silu192_l3 | [FAIL](runs/shared-width-search/screen/episodes/shared_c6__width_raw_silu192_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_raw_silu128_l2 | [FAIL](runs/shared-width-search/screen/episodes/shared_c6__width_raw_silu128_l2__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_raw_silu128_l4 | [FAIL](runs/shared-width-search/screen/episodes/shared_c6__width_raw_silu128_l4__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_raw_silu96_l2 | [FAIL](runs/shared-width-search/screen/episodes/shared_c6__width_raw_silu96_l2__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_raw_silu96_l4 | [FAIL](runs/shared-width-search/screen/episodes/shared_c6__width_raw_silu96_l4__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_raw_silu160_l2 | [FAIL](runs/shared-width-search/screen/episodes/shared_c6__width_raw_silu160_l2__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_raw_silu160_l4 | [FAIL](runs/shared-width-search/screen/episodes/shared_c6__width_raw_silu160_l4__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_raw_silu192_l2 | [FAIL](runs/shared-width-search/screen/episodes/shared_c6__width_raw_silu192_l2__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_raw_silu128_l3_linear_skip | [FAIL](runs/shared-width-search/screen/episodes/shared_c6__width_raw_silu128_l3_linear_skip__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_residual_raw_silu128_l3 | [FAIL](runs/shared-width-search/screen/episodes/shared_c6__width_residual_raw_silu128_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_raw_softplus128_l3 | [FAIL](runs/shared-width-search/screen/episodes/shared_c6__width_raw_softplus128_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_raw_softplus2_128_l3 | [FAIL](runs/shared-width-search/screen/episodes/shared_c6__width_raw_softplus2_128_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_raw_softplus128_l4 | [FAIL](runs/shared-width-search/screen/episodes/shared_c6__width_raw_softplus128_l4__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_ref_softplus128_l3_head025 | [FAIL](runs/shared-width-search/refinement/episodes/shared_c6__width_ref_softplus128_l3_head025__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_ref_softplus128_l3_head05 | [FAIL](runs/shared-width-search/refinement/episodes/shared_c6__width_ref_softplus128_l3_head05__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_ref_softplus128_l3_head2 | [FAIL](runs/shared-width-search/refinement/episodes/shared_c6__width_ref_softplus128_l3_head2__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_ref_softplus1_128_l3 | [FAIL](runs/shared-width-search/refinement/episodes/shared_c6__width_ref_softplus1_128_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_ref_softplus10_128_l3 | [FAIL](runs/shared-width-search/refinement/episodes/shared_c6__width_ref_softplus10_128_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_ref_softplus128_l3_linear_skip | [FAIL](runs/shared-width-search/refinement/episodes/shared_c6__width_ref_softplus128_l3_linear_skip__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_ref_softplus128_l3_residual | [FAIL](runs/shared-width-search/refinement/episodes/shared_c6__width_ref_softplus128_l3_residual__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_ref_silu160_l2_head025 | [FAIL](runs/shared-width-search/refinement/episodes/shared_c6__width_ref_silu160_l2_head025__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_ref_silu160_l2_head05 | [FAIL](runs/shared-width-search/refinement/episodes/shared_c6__width_ref_silu160_l2_head05__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_ref_silu160_l2_head2 | [FAIL](runs/shared-width-search/refinement/episodes/shared_c6__width_ref_silu160_l2_head2__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_ref_silu160_l2_linear_skip | [FAIL](runs/shared-width-search/refinement/episodes/shared_c6__width_ref_silu160_l2_linear_skip__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_ref_silu160_l2_residual | [FAIL](runs/shared-width-search/refinement/episodes/shared_c6__width_ref_silu160_l2_residual__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_last_softplus3_128_l3 | [FAIL](runs/shared-width-search/last_refinement/episodes/shared_c6__width_last_softplus3_128_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_last_softplus4_128_l3 | [FAIL](runs/shared-width-search/last_refinement/episodes/shared_c6__width_last_softplus4_128_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_last_softplus6_128_l3 | [FAIL](runs/shared-width-search/last_refinement/episodes/shared_c6__width_last_softplus6_128_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_last_softplus8_128_l3 | [PASS](runs/shared-width-search/last_refinement/episodes/shared_c6__width_last_softplus8_128_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_last_silu160_l2_linear_skip_head05 | [FAIL](runs/shared-width-search/last_refinement/episodes/shared_c6__width_last_silu160_l2_linear_skip_head05__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | width_last_silu160_l2_linear_skip_head2 | [FAIL](runs/shared-width-search/last_refinement/episodes/shared_c6__width_last_silu160_l2_linear_skip_head2__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | ensemble2_equal_softplus64_l2 | [FAIL](runs/shared-ensemble-search/screen/episodes/shared_c6__ensemble2_equal_softplus64_l2__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | ensemble2_multiscale_softplus64_l2 | [FAIL](runs/shared-ensemble-search/screen/episodes/shared_c6__ensemble2_multiscale_softplus64_l2__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | ensemble3_multiscale_softplus64_l2 | [FAIL](runs/shared-ensemble-search/screen/episodes/shared_c6__ensemble3_multiscale_softplus64_l2__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | ensemble3_multiscale_silu64_l2 | [FAIL](runs/shared-ensemble-search/screen/episodes/shared_c6__ensemble3_multiscale_silu64_l2__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | ensemble2_multiscale_silu64_l3 | [FAIL](runs/shared-ensemble-search/screen/episodes/shared_c6__ensemble2_multiscale_silu64_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | ensemble3_broad_softplus48_l3 | [FAIL](runs/shared-ensemble-search/screen/episodes/shared_c6__ensemble3_broad_softplus48_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | ensemble4_broad_silu48_l2 | [FAIL](runs/shared-ensemble-search/screen/episodes/shared_c6__ensemble4_broad_silu48_l2__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | ensemble2_highscale_softplus96_l2 | [FAIL](runs/shared-ensemble-search/screen/episodes/shared_c6__ensemble2_highscale_softplus96_l2__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | ensemble2_mixed64_l2 | [FAIL](runs/shared-ensemble-search/screen/episodes/shared_c6__ensemble2_mixed64_l2__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | ensemble2_multiscale_halfscore64_l2 | [FAIL](runs/shared-ensemble-search/screen/episodes/shared_c6__ensemble2_multiscale_halfscore64_l2__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | ensemble2_multiscale_fullscore64_l2 | [FAIL](runs/shared-ensemble-search/screen/episodes/shared_c6__ensemble2_multiscale_fullscore64_l2__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | ensemble2_multiscale_spectrum64_l2 | [FAIL](runs/shared-ensemble-search/screen/episodes/shared_c6__ensemble2_multiscale_spectrum64_l2__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | pointnorm_layer_first_softplus96_l3 | [FAIL](runs/shared-pointnorm-search/screen/episodes/shared_c6__pointnorm_layer_first_softplus96_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | pointnorm_layer_all_softplus96_l3 | [FAIL](runs/shared-pointnorm-search/screen/episodes/shared_c6__pointnorm_layer_all_softplus96_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | pointnorm_layer_first_silu128_l3 | [FAIL](runs/shared-pointnorm-search/screen/episodes/shared_c6__pointnorm_layer_first_silu128_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | pointnorm_layer_all_silu128_l3 | [FAIL](runs/shared-pointnorm-search/screen/episodes/shared_c6__pointnorm_layer_all_silu128_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | pointnorm_rms_all_silu128_l3 | [FAIL](runs/shared-pointnorm-search/screen/episodes/shared_c6__pointnorm_rms_all_silu128_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | pointnorm_weight_all_silu128_l3 | [FAIL](runs/shared-pointnorm-search/screen/episodes/shared_c6__pointnorm_weight_all_silu128_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | pointnorm_layer_all_softplus96_beta2_l3 | [FAIL](runs/shared-pointnorm-search/refinement-width/episodes/shared_c6__pointnorm_layer_all_softplus96_beta2_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | pointnorm_layer_all_softplus96_skip_l3 | [FAIL](runs/shared-pointnorm-search/refinement-width/episodes/shared_c6__pointnorm_layer_all_softplus96_skip_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_overlap | raw_silu128_l3 | [FAIL](runs/shared-discriminator-search/cross/episodes/shared_c6__raw_silu128_l3__vector_overlap.json.gz) |
| shared_c6 | vector_overlap | quadratic_tanh96_l3 | [FAIL](runs/shared-discriminator-search/cross/episodes/shared_c6__quadratic_tanh96_l3__vector_overlap.json.gz) |
| shared_c6 | vector_overlap | raw_softplus96_l3 | [PASS](runs/shared-discriminator-search/cross/episodes/shared_c6__raw_softplus96_l3__vector_overlap.json.gz) |
| shared_c6 | vector_overlap | additive_raw_fourier64_l2 | [FAIL](runs/shared-discriminator-search/cross/episodes/shared_c6__additive_raw_fourier64_l2__vector_overlap.json.gz) |
| shared_c6 | vector_two_broad | raw_silu128_l3 | [FAIL](runs/shared-discriminator-search/completion/episodes/shared_c6__raw_silu128_l3__vector_two_broad.json.gz) |
| shared_c6 | vector_two_broad | raw_softplus96_l3 | [PASS](runs/shared-discriminator-search/completion/episodes/shared_c6__raw_softplus96_l3__vector_two_broad.json.gz) |
| shared_c6 | vector_anisotropic | raw_silu128_l3 | [PASS](runs/shared-discriminator-search/cross/episodes/shared_c6__raw_silu128_l3__vector_anisotropic.json.gz) |
| shared_c6 | vector_anisotropic | quadratic_tanh96_l3 | [FAIL](runs/shared-discriminator-search/cross/episodes/shared_c6__quadratic_tanh96_l3__vector_anisotropic.json.gz) |
| shared_c6 | vector_anisotropic | raw_softplus96_l3 | [FAIL](runs/shared-discriminator-search/cross/episodes/shared_c6__raw_softplus96_l3__vector_anisotropic.json.gz) |
| shared_c6 | vector_anisotropic | additive_raw_fourier64_l2 | [PASS](runs/shared-discriminator-search/cross/episodes/shared_c6__additive_raw_fourier64_l2__vector_anisotropic.json.gz) |
| shared_c6 | vector_spiral | raw_silu128_l3 | [FAIL](runs/shared-discriminator-search/completion/episodes/shared_c6__raw_silu128_l3__vector_spiral.json.gz) |
| shared_c6 | vector_spiral | raw_softplus96_l3 | [PASS](runs/shared-discriminator-search/completion/episodes/shared_c6__raw_softplus96_l3__vector_spiral.json.gz) |
| sched_h50_f01 | vector_unequal_mass | pointnorm_layer_all_softplus96_beta4_l3 | [FAIL](runs/shared-schedule-search/screen/episodes/sched_h50_f01__vector_unequal_mass.json.gz) |
| sched_h50_f01 | vector_unequal_mass | normstruct_center_fixed96_beta6 | [FAIL](runs/shared-schedule-search/beta6_screen/episodes/sched_h50_f01__vector_unequal_mass.json.gz) |
| sched_h40_f01 | vector_unequal_mass | pointnorm_layer_all_softplus96_beta4_l3 | [FAIL](runs/shared-schedule-search/screen/episodes/sched_h40_f01__vector_unequal_mass.json.gz) |
| sched_h40_f01 | vector_unequal_mass | normstruct_center_fixed96_beta6 | [FAIL](runs/shared-schedule-search/beta6_screen/episodes/sched_h40_f01__vector_unequal_mass.json.gz) |
| c6_profile_prior4 | vector_unequal_mass | pointnorm_layer_all_softplus96_beta4_l3 | [FAIL](runs/rare-global-optimizer-screen/episodes/c6_profile_prior4__vector_unequal_mass.json.gz) |
| sched_h30_f01 | vector_unequal_mass | pointnorm_layer_all_softplus96_beta4_l3 | [FAIL](runs/shared-schedule-search/screen/episodes/sched_h30_f01__vector_unequal_mass.json.gz) |
| sched_h30_f01 | vector_unequal_mass | normstruct_center_fixed96_beta6 | [FAIL](runs/shared-schedule-search/beta6_screen/episodes/sched_h30_f01__vector_unequal_mass.json.gz) |
| sched_h30_f05 | vector_unequal_mass | pointnorm_layer_all_softplus96_beta4_l3 | [FAIL](runs/shared-schedule-search/screen/episodes/sched_h30_f05__vector_unequal_mass.json.gz) |
| sched_h30_f05 | vector_unequal_mass | normstruct_center_fixed96_beta6 | [FAIL](runs/shared-schedule-search/beta6_screen/episodes/sched_h30_f05__vector_unequal_mass.json.gz) |
| c6_profile_prior3 | vector_unequal_mass | pointnorm_layer_all_softplus96_beta4_l3 | [FAIL](runs/rare-global-optimizer-screen/episodes/c6_profile_prior3__vector_unequal_mass.json.gz) |
| c6_profile_beta2_095 | vector_unequal_mass | pointnorm_layer_all_softplus96_beta4_l3 | [FAIL](runs/rare-global-optimizer-screen/episodes/c6_profile_beta2_095__vector_unequal_mass.json.gz) |
| c6_profile_beta1_01 | vector_unequal_mass | pointnorm_layer_all_softplus96_beta4_l3 | [FAIL](runs/rare-global-optimizer-screen/episodes/c6_profile_beta1_01__vector_unequal_mass.json.gz) |

</details>

## What stays fixed in the tests

Data, target metrics, thresholds, seed 0, generators, initialization rules, particle counts, batch sizes and update budgets are the frozen test setup. They match for every candidate. Each task keeps its existing reconstruction/identity/cover objective. Resource sizes differ between tests, but a candidate cannot change them to obtain a pass. These are development cases, not unseen holdouts.

Architecture remains separate from formulation: explicit discriminator variants are allowed under the same unchanged recipe, with all trials/failures recorded. The importer checks D-only changes against the frozen reference test and reports reference-profile performance separately. Legacy EMA measurement remains host-specific and never affects ranking.

## Join the search

[Contribution instructions and one-command run](../../../benchmarks/transfer_suite/UNADJUSTED_SEARCH.md). The runner accepts one global recipe card and runs all 19 tests by default. Preserve failed runs. Screening is allowed, but only a complete row can qualify.

[Current search findings and remaining failures](FINDINGS.md) · [Reproduce the leading recipes](leading_candidates.json).

[All metrics, convergence and separate EMA](leaderboard.json) · [Registered entries](entries.json) · [Validation](validation.json) · [Historical adjusted comparison](../default_comparison/README.md).
