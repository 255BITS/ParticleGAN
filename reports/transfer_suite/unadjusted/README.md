# Unadjusted ParticleGAN default leaderboard

**This is the primary comparison for selecting a shared default.** Each candidate uses one unchanged loss/regularization/optimizer recipe on every test. No per-example LR, Adam, prior-rate or loss-weight adjustments. The earlier adjusted 19/19 result does not compete on this leaderboard.

**Overall PASS requires 19/19 live behavioral tests**, each passing every metric for at least five final observations of a complete 24-point curve. EMA is separate. Missing cases stay in the denominator; partial rows cannot beat a completed candidate or qualify as winners.

| Candidate | Required | Data | Images | Live total | Attempted | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| lr00425 prior2 (`lr00425_prior2`) | 8/9 | 3/6 | 4/4 | **15/19** | 19/19 | **FAIL** |
| shared c6 (`shared_c6`) | 9/9 | 2/6 | 4/4 | **15/19** | 19/19 | **FAIL** |
| equal lr 00425 (`equal_lr_00425`) | 8/9 | 3/6 | 3/4 | **14/19** | 19/19 | **FAIL** |
| equal lr 0025 (`equal_lr_0025`) | 7/9 | 3/6 | 4/4 | **14/19** | 19/19 | **FAIL** |
| shared k075 (`shared_k075`) | 7/9 | 4/6 | 3/4 | **14/19** | 19/19 | **FAIL** |
| shared spread01 (`shared_spread01`) | 9/9 | 1/6 | 3/4 | **13/19** | 19/19 | **FAIL** |
| equal lr 0034 (`equal_lr_0034`) | 8/9 | 3/6 | 2/4 | **13/19** | 19/19 | **FAIL** |
| lr0034 dprior125 (`lr0034_dprior125`) | 7/9 | 3/6 | 3/4 | **13/19** | 19/19 | **FAIL** |
| equal lr 0017 (`equal_lr_0017`) | 5/9 | 2/6 | 4/4 | **11/19** | 19/19 | **FAIL** |
| Proposed public default (`gan`) | 1/9 | 6/6 | 1/4 | **8/19** | 19/19 | **FAIL** |
| Current master default (`gan_legacy`) | 1/9 | 3/6 | 1/4 | **5/19** | 19/19 | **FAIL** |

Rank complete candidates by live passes, then lower normalized final metric shortfall. An all-pass candidate is the target; a partial improvement is not an all-pass stamp.

## Screening results — not ranked

| Candidate | Passing measured tests | Attempted | Status |
| --- | ---: | ---: | --- |
| shared_b05_999 | 3 | 6/19 | INCOMPLETE |
| shared_k10 | 2 | 6/19 | INCOMPLETE |
| shared_k15 | 2 | 6/19 | INCOMPLETE |
| shared_c10 | 2 | 6/19 | INCOMPLETE |
| shared_b0_999 | 2 | 6/19 | INCOMPLETE |
| shared_p3 | 2 | 6/19 | INCOMPLETE |
| shared_b05 | 1 | 6/19 | INCOMPLETE |
| shared_c20 | 1 | 6/19 | INCOMPLETE |
| shared_p4 | 0 | 6/19 | INCOMPLETE |

## One recipe per row

| Candidate | G / D / particle LR | Adam betas | b_cap coefficient / κ | Spread weight |
| --- | --- | --- | --- | ---: |
| lr00425_prior2 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 3 / 1.25 | 0.05 |
| shared_c6 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 6 / 1.25 | 0.05 |
| equal_lr_00425 | 0.00425 / 0.00425 / 0.00425 | (0.0, 0.99) | 3 / 1.25 | 0.05 |
| equal_lr_0025 | 0.0025 / 0.0025 / 0.0025 | (0.0, 0.99) | 3 / 1.25 | 0.05 |
| shared_k075 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 3 / 0.75 | 0.05 |
| shared_spread01 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 3 / 1.25 | 0.01 |
| equal_lr_0034 | 0.0034 / 0.0034 / 0.0034 | (0.0, 0.99) | 3 / 1.25 | 0.05 |
| lr0034_dprior125 | 0.0034 / 0.00425 / 0.00425 | (0.0, 0.99) | 3 / 1.25 | 0.05 |
| equal_lr_0017 | 0.0017 / 0.0017 / 0.0017 | (0.0, 0.99) | 3 / 1.25 | 0.05 |
| gan | 0.001 / 0.0015 / 0.01 | (0.0, 0.99) | 3 / 1.25 | 0.05 |
| gan_legacy | 0.0006 / 0.0009 / 0.006 | (0.0, 0.999) | 1 / 1 | 1 |
| shared_b05_999 | 0.00425 / 0.00425 / 0.0085 | (0.5, 0.999) | 3 / 1.25 | 0.05 |
| shared_k10 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 3 / 1 | 0.05 |
| shared_k15 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 3 / 1.5 | 0.05 |
| shared_c10 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 10 / 1.25 | 0.05 |
| shared_b0_999 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.999) | 3 / 1.25 | 0.05 |
| shared_p3 | 0.00425 / 0.00425 / 0.01275 | (0.0, 0.99) | 3 / 1.25 | 0.05 |
| shared_b05 | 0.00425 / 0.00425 / 0.0085 | (0.5, 0.99) | 3 / 1.25 | 0.05 |
| shared_c20 | 0.00425 / 0.00425 / 0.0085 | (0.0, 0.99) | 20 / 1.25 | 0.05 |
| shared_p4 | 0.00425 / 0.00425 / 0.017 | (0.0, 0.99) | 3 / 1.25 | 0.05 |

All current entries use Rp logistic, no particle L2, and the same schedule: hold for 60% of the budget, then cosine toward 5%. Rates above are absolute and are applied to every optimizer group, including directly optimized particles and AE prior groups.

## Every test: leading complete candidates and public baselines

All candidate metrics and EMA profiles remain in [leaderboard.json](leaderboard.json).

| Test | lr00425_prior2 | shared_c6 | equal_lr_00425 | gan | gan_legacy |
| --- | --- | --- | --- | --- | --- |
| two_pole | [PASS](runs/round0-2/episodes/lr00425_prior2__two_pole.json.gz) | [PASS](runs/round1-0/episodes/shared_c6__two_pole.json.gz) | [PASS](runs/round0-1/episodes/equal_lr_00425__two_pole.json.gz) | [FAIL](../default_comparison/proposed/episodes/proposed__two_pole.json.gz) | [FAIL](../default_comparison/current/episodes/current__two_pole.json.gz) |
| trajectory | [PASS](runs/round0-2/episodes/lr00425_prior2__trajectory.json.gz) | [PASS](runs/completion-1/episodes/shared_c6__trajectory.json.gz) | [PASS](runs/round0-1/episodes/equal_lr_00425__trajectory.json.gz) | [FAIL](../default_comparison/proposed/episodes/proposed__trajectory.json.gz) | [FAIL](../default_comparison/current/episodes/current__trajectory.json.gz) |
| residual_student | [PASS](runs/round0-2/episodes/lr00425_prior2__residual_student.json.gz) | [PASS](runs/completion-1/episodes/shared_c6__residual_student.json.gz) | [PASS](runs/round0-1/episodes/equal_lr_00425__residual_student.json.gz) | [FAIL](../default_comparison/proposed/episodes/proposed__residual_student.json.gz) | [FAIL](../default_comparison/current/episodes/current__residual_student.json.gz) |
| unipolar | [PASS](runs/round0-2/episodes/lr00425_prior2__unipolar.json.gz) | [PASS](runs/completion-1/episodes/shared_c6__unipolar.json.gz) | [PASS](runs/round0-1/episodes/equal_lr_00425__unipolar.json.gz) | [FAIL](../default_comparison/proposed/episodes/proposed__unipolar.json.gz) | [FAIL](../default_comparison/current/episodes/current__unipolar.json.gz) |
| ae_gan_hold | [PASS](runs/round0-2/episodes/lr00425_prior2__ae_gan_hold.json.gz) | [PASS](runs/completion-1/episodes/shared_c6__ae_gan_hold.json.gz) | [PASS](runs/round0-1/episodes/equal_lr_00425__ae_gan_hold.json.gz) | [PASS](../default_comparison/proposed/episodes/proposed__ae_gan_hold.json.gz) | [PASS](../default_comparison/current/episodes/current__ae_gan_hold.json.gz) |
| cover_leftover | [PASS](runs/round0-2/episodes/lr00425_prior2__cover_leftover.json.gz) | [PASS](runs/completion-1/episodes/shared_c6__cover_leftover.json.gz) | [PASS](runs/round0-1/episodes/equal_lr_00425__cover_leftover.json.gz) | [FAIL](../default_comparison/proposed/episodes/proposed__cover_leftover.json.gz) | [FAIL](../default_comparison/current/episodes/current__cover_leftover.json.gz) |
| unused_token_hold | [PASS](runs/round0-2/episodes/lr00425_prior2__unused_token_hold.json.gz) | [PASS](runs/completion-1/episodes/shared_c6__unused_token_hold.json.gz) | [PASS](runs/round0-1/episodes/equal_lr_00425__unused_token_hold.json.gz) | [FAIL](../default_comparison/proposed/episodes/proposed__unused_token_hold.json.gz) | [FAIL](../default_comparison/current/episodes/current__unused_token_hold.json.gz) |
| mid_scale_identity | [PASS](runs/round0-2/episodes/lr00425_prior2__mid_scale_identity.json.gz) | [PASS](runs/completion-1/episodes/shared_c6__mid_scale_identity.json.gz) | [PASS](runs/round0-1/episodes/equal_lr_00425__mid_scale_identity.json.gz) | [FAIL](../default_comparison/proposed/episodes/proposed__mid_scale_identity.json.gz) | [FAIL](../default_comparison/current/episodes/current__mid_scale_identity.json.gz) |
| mode_hold | [FAIL](runs/round0-2/episodes/lr00425_prior2__mode_hold.json.gz) | [PASS](runs/round1-0/episodes/shared_c6__mode_hold.json.gz) | [FAIL](runs/round0-1/episodes/equal_lr_00425__mode_hold.json.gz) | [FAIL](../default_comparison/proposed/episodes/proposed__mode_hold.json.gz) | [FAIL](../default_comparison/current/episodes/current__mode_hold.json.gz) |
| vector_two_broad | [PASS](runs/round0-2/episodes/lr00425_prior2__vector_two_broad.json.gz) | [PASS](runs/completion-1/episodes/shared_c6__vector_two_broad.json.gz) | [PASS](runs/round0-1/episodes/equal_lr_00425__vector_two_broad.json.gz) | [PASS](../default_comparison/proposed/episodes/proposed__vector_two_broad.json.gz) | [PASS](../default_comparison/current/episodes/current__vector_two_broad.json.gz) |
| vector_unequal_mass | [FAIL](runs/round0-2/episodes/lr00425_prior2__vector_unequal_mass.json.gz) | [FAIL](runs/round1-0/episodes/shared_c6__vector_unequal_mass.json.gz) | [FAIL](runs/round0-1/episodes/equal_lr_00425__vector_unequal_mass.json.gz) | [PASS](../default_comparison/proposed/episodes/proposed__vector_unequal_mass.json.gz) | [FAIL](../default_comparison/current/episodes/current__vector_unequal_mass.json.gz) |
| vector_unequal_width | [FAIL](runs/round0-2/episodes/lr00425_prior2__vector_unequal_width.json.gz) | [FAIL](runs/round1-0/episodes/shared_c6__vector_unequal_width.json.gz) | [FAIL](runs/round0-1/episodes/equal_lr_00425__vector_unequal_width.json.gz) | [PASS](../default_comparison/proposed/episodes/proposed__vector_unequal_width.json.gz) | [FAIL](../default_comparison/current/episodes/current__vector_unequal_width.json.gz) |
| vector_anisotropic | [PASS](runs/round0-2/episodes/lr00425_prior2__vector_anisotropic.json.gz) | [FAIL](runs/completion-1/episodes/shared_c6__vector_anisotropic.json.gz) | [PASS](runs/round0-1/episodes/equal_lr_00425__vector_anisotropic.json.gz) | [PASS](../default_comparison/proposed/episodes/proposed__vector_anisotropic.json.gz) | [PASS](../default_comparison/current/episodes/current__vector_anisotropic.json.gz) |
| vector_overlap | [FAIL](runs/round0-2/episodes/lr00425_prior2__vector_overlap.json.gz) | [FAIL](runs/round1-0/episodes/shared_c6__vector_overlap.json.gz) | [FAIL](runs/round0-1/episodes/equal_lr_00425__vector_overlap.json.gz) | [PASS](../default_comparison/proposed/episodes/proposed__vector_overlap.json.gz) | [FAIL](../default_comparison/current/episodes/current__vector_overlap.json.gz) |
| vector_spiral | [PASS](runs/round0-2/episodes/lr00425_prior2__vector_spiral.json.gz) | [PASS](runs/completion-1/episodes/shared_c6__vector_spiral.json.gz) | [PASS](runs/round0-1/episodes/equal_lr_00425__vector_spiral.json.gz) | [PASS](../default_comparison/proposed/episodes/proposed__vector_spiral.json.gz) | [PASS](../default_comparison/current/episodes/current__vector_spiral.json.gz) |
| img_stripes2 | [PASS](runs/round0-2/episodes/lr00425_prior2__img_stripes2.json.gz) | [PASS](runs/completion-1/episodes/shared_c6__img_stripes2.json.gz) | [FAIL](runs/round0-1/episodes/equal_lr_00425__img_stripes2.json.gz) | [PASS](../default_comparison/proposed/episodes/proposed__img_stripes2.json.gz) | [PASS](../default_comparison/current/episodes/current__img_stripes2.json.gz) |
| img_bars4 | [PASS](runs/round0-2/episodes/lr00425_prior2__img_bars4.json.gz) | [PASS](runs/round1-0/episodes/shared_c6__img_bars4.json.gz) | [PASS](runs/round0-1/episodes/equal_lr_00425__img_bars4.json.gz) | [FAIL](../default_comparison/proposed/episodes/proposed__img_bars4.json.gz) | [FAIL](../default_comparison/current/episodes/current__img_bars4.json.gz) |
| img_blobs4 | [PASS](runs/round0-2/episodes/lr00425_prior2__img_blobs4.json.gz) | [PASS](runs/completion-1/episodes/shared_c6__img_blobs4.json.gz) | [PASS](runs/round0-1/episodes/equal_lr_00425__img_blobs4.json.gz) | [FAIL](../default_comparison/proposed/episodes/proposed__img_blobs4.json.gz) | [FAIL](../default_comparison/current/episodes/current__img_blobs4.json.gz) |
| img_intensity2 | [PASS](runs/round0-2/episodes/lr00425_prior2__img_intensity2.json.gz) | [PASS](runs/completion-1/episodes/shared_c6__img_intensity2.json.gz) | [PASS](runs/round0-1/episodes/equal_lr_00425__img_intensity2.json.gz) | [FAIL](../default_comparison/proposed/episodes/proposed__img_intensity2.json.gz) | [FAIL](../default_comparison/current/episodes/current__img_intensity2.json.gz) |

## What stays fixed in the tests

Data, target metrics, thresholds, seed 0, architectures, initializations, particle counts, batch sizes and update budgets are the frozen test setup. They match for every candidate. Each task keeps its existing reconstruction/identity/cover objective. Resource sizes differ between tests, but a candidate cannot change them to obtain a pass. These are development cases, not unseen holdouts.

This table currently compares the common reference architecture profile. Architecture remains separate from formulation: discriminator variants are allowed under the same unchanged recipe, with all trials/failures recorded. The current runner/importer validates the reference profile; explicit variant support is needed before importing another profile. Legacy EMA measurement remains host-specific and never affects ranking.

## Join the search

[Contribution instructions and one-command run](../../../benchmarks/transfer_suite/UNADJUSTED_SEARCH.md). The runner accepts one global recipe card and runs all 19 tests by default. Preserve failed runs. Screening is allowed, but only a complete row can qualify.

[Current search findings and remaining failures](FINDINGS.md) · [Reproduce the leading recipes](leading_candidates.json).

[All metrics, convergence and separate EMA](leaderboard.json) · [Registered entries](entries.json) · [Validation](validation.json) · [Historical adjusted comparison](../default_comparison/README.md).
