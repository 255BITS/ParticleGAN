# Unadjusted ParticleGAN default leaderboard

**This is the primary comparison for selecting a shared default.** Each candidate uses one unchanged loss/regularization/optimizer recipe on every test. No per-example LR, Adam, prior-rate or loss-weight adjustments. The earlier adjusted 19/19 result does not compete on this leaderboard.

**Overall PASS requires 19/19 live behavioral tests**, each passing every metric for at least five final observations of a complete 24-point curve. EMA is separate. Missing cases stay in the denominator; partial rows cannot beat a completed candidate or qualify as winners.

| Candidate | Required | Data | Images | Live total | Reference D profile | Attempted | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| shared c6 (`shared_c6`) | 9/9 | 4/6 | 4/4 | **17/19** | 15/19 | 19/19 | **FAIL** |
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
| Proposed public default (`gan`) | 1/9 | 6/6 | 1/4 | **8/19** | 8/19 | 19/19 | **FAIL** |
| Current master default (`gan_legacy`) | 1/9 | 3/6 | 1/4 | **5/19** | 5/19 | 19/19 | **FAIL** |

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

## One recipe per row

| Candidate | G / D / particle LR | Adam betas | b_cap coefficient / κ | Spread weight | Update rule |
| --- | --- | --- | --- | ---: | --- |
| shared_c6 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 6 / 1.25 | 0.05 | Adam |
| lr00425_prior2 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 3 / 1.25 | 0.05 | Adam |
| ratio_g34_d68_p85 | 0.0034 / 0.0068 / 0.0085 | (0.0, 0.99) | 3 / 1.25 | 0.05 | Adam |
| equal_lr_00425 | 0.00425 / 0.00425 / 0.00425 | (0.0, 0.99) | 3 / 1.25 | 0.05 | Adam |
| equal_lr_0025 | 0.0025 / 0.0025 / 0.0025 | (0.0, 0.99) | 3 / 1.25 | 0.05 | Adam |
| shared_k075 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 3 / 0.75 | 0.05 | Adam |
| shared_spread01 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 3 / 1.25 | 0.01 | Adam |
| ratio_g425_d6375_p85 | 0.00425 / 0.006375 / 0.0085 | (0.0, 0.99) | 3 / 1.25 | 0.05 | Adam |
| equal_lr_0034 | 0.0034 / 0.0034 / 0.0034 | (0.0, 0.99) | 3 / 1.25 | 0.05 | Adam |
| lr0034_dprior125 | 0.0034 / 0.00425 / 0.00425 | (0.0, 0.99) | 3 / 1.25 | 0.05 | Adam |
| relative_cap_05 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 3 / 1.25 | 0.05 | Adam + relative step cap 0.05 |
| equal_lr_0017 | 0.0017 / 0.0017 / 0.0017 | (0.0, 0.99) | 3 / 1.25 | 0.05 | Adam |
| gan | 0.001 / 0.0015 / 0.01 | (0.0, 0.99) | 3 / 1.25 | 0.05 | Adam |
| gan_legacy | 0.0006 / 0.0009 / 0.006 | (0.0, 0.999) | 1 / 1 | 1 | Adam |
| shared_b05_999 | 0.00425 / 0.00425 / 0.0085 | (0.5, 0.999) | 3 / 1.25 | 0.05 | Adam |
| ratio_g25_d15_p85 | 0.0025 / 0.0015 / 0.0085 | (0.0, 0.99) | 3 / 1.25 | 0.05 | Adam |
| ratio_g425_d2125_p85 | 0.00425 / 0.002125 / 0.0085 | (0.0, 0.99) | 3 / 1.25 | 0.05 | Adam |
| ratio_g25_d375_p100 | 0.0025 / 0.00375 / 0.01 | (0.0, 0.99) | 3 / 1.25 | 0.05 | Adam |
| shared_k10 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 3 / 1 | 0.05 | Adam |
| shared_k15 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 3 / 1.5 | 0.05 | Adam |
| shared_c10 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 10 / 1.25 | 0.05 | Adam |
| shared_b0_999 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.999) | 3 / 1.25 | 0.05 | Adam |
| shared_p3 | 0.00425 / 0.00425 / 0.01275 | (0.0, 0.99) | 3 / 1.25 | 0.05 | Adam |
| ratio_g34_d255_p85 | 0.0034 / 0.00255 / 0.0085 | (0.0, 0.99) | 3 / 1.25 | 0.05 | Adam |
| relative_cap_025 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 3 / 1.25 | 0.05 | Adam + relative step cap 0.025 |
| shared_b05 | 0.00425 / 0.00425 / 0.0085 | (0.5, 0.99) | 3 / 1.25 | 0.05 | Adam |
| relative_cap_01 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 3 / 1.25 | 0.05 | Adam + relative step cap 0.01 |
| shared_c20 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 20 / 1.25 | 0.05 | Adam |
| shared_p4 | 0.00425 / 0.00425 / 0.017 | (0.0, 0.99) | 3 / 1.25 | 0.05 | Adam |

All current entries use Rp logistic, no particle L2, and the same schedule: hold for 60% of the budget, then cosine toward 5%. Rates above are absolute and are applied to every optimizer group, including directly optimized particles and AE prior groups.

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
| vector_unequal_mass | [FAIL](runs/shared-discriminator-search/screen/episodes/shared_c6__raw_silu128_l3__vector_unequal_mass.json.gz) | [FAIL](runs/round0-2/episodes/lr00425_prior2__vector_unequal_mass.json.gz) | [FAIL](runs/shared-ratio-search/screen/episodes/ratio_g34_d68_p85__vector_unequal_mass.json.gz) | [PASS](../default_comparison/proposed/episodes/proposed__vector_unequal_mass.json.gz) | [FAIL](../default_comparison/current/episodes/current__vector_unequal_mass.json.gz) |
| vector_unequal_width | [FAIL](runs/shared-discriminator-search/screen/episodes/shared_c6__raw_silu128_l3__vector_unequal_width.json.gz) | [FAIL](runs/round0-2/episodes/lr00425_prior2__vector_unequal_width.json.gz) | [FAIL](runs/shared-ratio-search/screen/episodes/ratio_g34_d68_p85__vector_unequal_width.json.gz) | [PASS](../default_comparison/proposed/episodes/proposed__vector_unequal_width.json.gz) | [FAIL](../default_comparison/current/episodes/current__vector_unequal_width.json.gz) |
| vector_anisotropic | [PASS](runs/shared-discriminator-search/cross/episodes/shared_c6__additive_raw_fourier64_l2__vector_anisotropic.json.gz) | [PASS](runs/round0-2/episodes/lr00425_prior2__vector_anisotropic.json.gz) | [FAIL](runs/shared-ratio-search/screen/episodes/ratio_g34_d68_p85__vector_anisotropic.json.gz) | [PASS](../default_comparison/proposed/episodes/proposed__vector_anisotropic.json.gz) | [PASS](../default_comparison/current/episodes/current__vector_anisotropic.json.gz) |
| vector_overlap | [PASS](runs/shared-discriminator-search/cross/episodes/shared_c6__raw_softplus96_l3__vector_overlap.json.gz) | [FAIL](runs/round0-2/episodes/lr00425_prior2__vector_overlap.json.gz) | [PASS](runs/shared-ratio-search/screen/episodes/ratio_g34_d68_p85__vector_overlap.json.gz) | [PASS](../default_comparison/proposed/episodes/proposed__vector_overlap.json.gz) | [FAIL](../default_comparison/current/episodes/current__vector_overlap.json.gz) |
| vector_spiral | [PASS](runs/completion-1/episodes/shared_c6__vector_spiral.json.gz) | [PASS](runs/round0-2/episodes/lr00425_prior2__vector_spiral.json.gz) | [PASS](runs/shared-ratio-search/completion/episodes/ratio_g34_d68_p85__vector_spiral.json.gz) | [PASS](../default_comparison/proposed/episodes/proposed__vector_spiral.json.gz) | [PASS](../default_comparison/current/episodes/current__vector_spiral.json.gz) |
| img_stripes2 | [PASS](runs/completion-1/episodes/shared_c6__img_stripes2.json.gz) | [PASS](runs/round0-2/episodes/lr00425_prior2__img_stripes2.json.gz) | [PASS](runs/shared-ratio-search/completion/episodes/ratio_g34_d68_p85__img_stripes2.json.gz) | [PASS](../default_comparison/proposed/episodes/proposed__img_stripes2.json.gz) | [PASS](../default_comparison/current/episodes/current__img_stripes2.json.gz) |
| img_bars4 | [PASS](runs/round1-0/episodes/shared_c6__img_bars4.json.gz) | [PASS](runs/round0-2/episodes/lr00425_prior2__img_bars4.json.gz) | [PASS](runs/shared-ratio-search/completion/episodes/ratio_g34_d68_p85__img_bars4.json.gz) | [FAIL](../default_comparison/proposed/episodes/proposed__img_bars4.json.gz) | [FAIL](../default_comparison/current/episodes/current__img_bars4.json.gz) |
| img_blobs4 | [PASS](runs/completion-1/episodes/shared_c6__img_blobs4.json.gz) | [PASS](runs/round0-2/episodes/lr00425_prior2__img_blobs4.json.gz) | [PASS](runs/shared-ratio-search/screen/episodes/ratio_g34_d68_p85__img_blobs4.json.gz) | [FAIL](../default_comparison/proposed/episodes/proposed__img_blobs4.json.gz) | [FAIL](../default_comparison/current/episodes/current__img_blobs4.json.gz) |
| img_intensity2 | [PASS](runs/completion-1/episodes/shared_c6__img_intensity2.json.gz) | [PASS](runs/round0-2/episodes/lr00425_prior2__img_intensity2.json.gz) | [PASS](runs/shared-ratio-search/completion/episodes/ratio_g34_d68_p85__img_intensity2.json.gz) | [FAIL](../default_comparison/proposed/episodes/proposed__img_intensity2.json.gz) | [FAIL](../default_comparison/current/episodes/current__img_intensity2.json.gz) |

## Remaining failures in the leading complete recipes

The final passing streak must reach five observations. A good last checkpoint alone does not pass.

| Recipe | Test | Final failing metrics (value; required bound) | Final passing streak |
| --- | --- | --- | ---: |
| shared_c6 | [vector_unequal_mass](runs/shared-discriminator-search/screen/episodes/shared_c6__raw_silu128_l3__vector_unequal_mass.json.gz) | component_min_eigen_ratio: 0.018911; needs >= 0.15 | 0/5 |
| shared_c6 | [vector_unequal_width](runs/shared-discriminator-search/screen/episodes/shared_c6__raw_silu128_l3__vector_unequal_width.json.gz) | Final metrics pass | 1/5 |
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
| shared_c6 | vector_unequal_width | raw_softplus96_l3 | [FAIL](runs/shared-discriminator-search/screen/episodes/shared_c6__raw_softplus96_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | raw_silu128_l3 | [FAIL](runs/shared-discriminator-search/screen/episodes/shared_c6__raw_silu128_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | quadratic_softplus96_l2 | [FAIL](runs/shared-discriminator-search/screen/episodes/shared_c6__quadratic_softplus96_l2__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | quadratic_tanh96_l3 | [FAIL](runs/shared-discriminator-search/screen/episodes/shared_c6__quadratic_tanh96_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | residual_raw_softplus96_l3 | [FAIL](runs/shared-discriminator-search/screen/episodes/shared_c6__residual_raw_softplus96_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | residual_lowfreq_softplus96_l3 | [FAIL](runs/shared-discriminator-search/screen/episodes/shared_c6__residual_lowfreq_softplus96_l3__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | halfscore_fourier_skip96_l2 | [FAIL](runs/shared-discriminator-search/screen/episodes/shared_c6__halfscore_fourier_skip96_l2__vector_unequal_width.json.gz) |
| shared_c6 | vector_unequal_width | additive_raw_fourier64_l2 | [FAIL](runs/shared-discriminator-search/screen/episodes/shared_c6__additive_raw_fourier64_l2__vector_unequal_width.json.gz) |
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

## What stays fixed in the tests

Data, target metrics, thresholds, seed 0, generators, initialization rules, particle counts, batch sizes and update budgets are the frozen test setup. They match for every candidate. Each task keeps its existing reconstruction/identity/cover objective. Resource sizes differ between tests, but a candidate cannot change them to obtain a pass. These are development cases, not unseen holdouts.

Architecture remains separate from formulation: explicit discriminator variants are allowed under the same unchanged recipe, with all trials/failures recorded. The importer checks D-only changes against the frozen reference test and reports reference-profile performance separately. Legacy EMA measurement remains host-specific and never affects ranking.

## Join the search

[Contribution instructions and one-command run](../../../benchmarks/transfer_suite/UNADJUSTED_SEARCH.md). The runner accepts one global recipe card and runs all 19 tests by default. Preserve failed runs. Screening is allowed, but only a complete row can qualify.

[Current search findings and remaining failures](FINDINGS.md) · [Reproduce the leading recipes](leading_candidates.json).

[All metrics, convergence and separate EMA](leaderboard.json) · [Registered entries](entries.json) · [Validation](validation.json) · [Historical adjusted comparison](../default_comparison/README.md).
