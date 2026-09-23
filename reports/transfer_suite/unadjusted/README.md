# Unadjusted ParticleGAN default leaderboard

**This is the primary comparison for selecting a shared default.** Each candidate uses one unchanged loss/regularization/optimizer recipe on every test. No per-example LR, Adam, prior-rate or loss-weight adjustments. The earlier adjusted 19/19 result does not compete on this leaderboard.

**Overall PASS requires 19/19 live behavioral tests**, each passing every metric for at least five final observations of a complete 24-point curve. EMA is separate. Missing cases stay in the denominator; partial rows cannot beat a completed candidate or qualify as winners.

| Candidate | Required | Data | Images | Live total | Attempted | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Proposed public default (`gan`) | 1/9 | 6/6 | 1/4 | **8/19** | 19/19 | **FAIL** |
| Current master default (`gan_legacy`) | 1/9 | 3/6 | 1/4 | **5/19** | 19/19 | **FAIL** |

Rank complete candidates by live passes, then lower normalized final metric shortfall. An all-pass candidate is the target; a partial improvement is not an all-pass stamp.

## One recipe per row

| Candidate | G / D / particle LR | Adam betas | b_cap coefficient / κ | Spread weight |
| --- | --- | --- | --- | ---: |
| gan | 0.001 / 0.0015 / 0.01 | (0.0, 0.99) | 3 / 1.25 | 0.05 |
| gan_legacy | 0.0006 / 0.0009 / 0.006 | (0.0, 0.999) | 1 / 1 | 1 |

All current entries use Rp logistic, no particle L2, and the same schedule: hold for 60% of the budget, then cosine toward 5%. Rates above are absolute and are applied to every optimizer group, including directly optimized particles and AE prior groups.

## Every test

| Test | gan | gan_legacy |
| --- | --- | --- |
| two_pole | [FAIL](../default_comparison/proposed/episodes/proposed__two_pole.json.gz) | [FAIL](../default_comparison/current/episodes/current__two_pole.json.gz) |
| trajectory | [FAIL](../default_comparison/proposed/episodes/proposed__trajectory.json.gz) | [FAIL](../default_comparison/current/episodes/current__trajectory.json.gz) |
| residual_student | [FAIL](../default_comparison/proposed/episodes/proposed__residual_student.json.gz) | [FAIL](../default_comparison/current/episodes/current__residual_student.json.gz) |
| unipolar | [FAIL](../default_comparison/proposed/episodes/proposed__unipolar.json.gz) | [FAIL](../default_comparison/current/episodes/current__unipolar.json.gz) |
| ae_gan_hold | [PASS](../default_comparison/proposed/episodes/proposed__ae_gan_hold.json.gz) | [PASS](../default_comparison/current/episodes/current__ae_gan_hold.json.gz) |
| cover_leftover | [FAIL](../default_comparison/proposed/episodes/proposed__cover_leftover.json.gz) | [FAIL](../default_comparison/current/episodes/current__cover_leftover.json.gz) |
| unused_token_hold | [FAIL](../default_comparison/proposed/episodes/proposed__unused_token_hold.json.gz) | [FAIL](../default_comparison/current/episodes/current__unused_token_hold.json.gz) |
| mid_scale_identity | [FAIL](../default_comparison/proposed/episodes/proposed__mid_scale_identity.json.gz) | [FAIL](../default_comparison/current/episodes/current__mid_scale_identity.json.gz) |
| mode_hold | [FAIL](../default_comparison/proposed/episodes/proposed__mode_hold.json.gz) | [FAIL](../default_comparison/current/episodes/current__mode_hold.json.gz) |
| vector_two_broad | [PASS](../default_comparison/proposed/episodes/proposed__vector_two_broad.json.gz) | [PASS](../default_comparison/current/episodes/current__vector_two_broad.json.gz) |
| vector_unequal_mass | [PASS](../default_comparison/proposed/episodes/proposed__vector_unequal_mass.json.gz) | [FAIL](../default_comparison/current/episodes/current__vector_unequal_mass.json.gz) |
| vector_unequal_width | [PASS](../default_comparison/proposed/episodes/proposed__vector_unequal_width.json.gz) | [FAIL](../default_comparison/current/episodes/current__vector_unequal_width.json.gz) |
| vector_anisotropic | [PASS](../default_comparison/proposed/episodes/proposed__vector_anisotropic.json.gz) | [PASS](../default_comparison/current/episodes/current__vector_anisotropic.json.gz) |
| vector_overlap | [PASS](../default_comparison/proposed/episodes/proposed__vector_overlap.json.gz) | [FAIL](../default_comparison/current/episodes/current__vector_overlap.json.gz) |
| vector_spiral | [PASS](../default_comparison/proposed/episodes/proposed__vector_spiral.json.gz) | [PASS](../default_comparison/current/episodes/current__vector_spiral.json.gz) |
| img_stripes2 | [PASS](../default_comparison/proposed/episodes/proposed__img_stripes2.json.gz) | [PASS](../default_comparison/current/episodes/current__img_stripes2.json.gz) |
| img_bars4 | [FAIL](../default_comparison/proposed/episodes/proposed__img_bars4.json.gz) | [FAIL](../default_comparison/current/episodes/current__img_bars4.json.gz) |
| img_blobs4 | [FAIL](../default_comparison/proposed/episodes/proposed__img_blobs4.json.gz) | [FAIL](../default_comparison/current/episodes/current__img_blobs4.json.gz) |
| img_intensity2 | [FAIL](../default_comparison/proposed/episodes/proposed__img_intensity2.json.gz) | [FAIL](../default_comparison/current/episodes/current__img_intensity2.json.gz) |

## What stays fixed in the tests

Data, target metrics, thresholds, seed 0, architectures, initializations, particle counts, batch sizes and update budgets are the frozen test setup. They match for every candidate. Each task keeps its existing reconstruction/identity/cover objective. Resource sizes differ between tests, but a candidate cannot change them to obtain a pass. These are development cases, not unseen holdouts.

The selected architecture for each test is shared by all candidates. A different architecture study must be reported separately; no candidate may silently cherry-pick a different network per result. Legacy EMA measurement remains host-specific and never affects ranking.

## Join the search

[Contribution instructions and one-command run](../../../benchmarks/transfer_suite/UNADJUSTED_SEARCH.md). The runner accepts one global recipe card and runs all 19 tests by default. Preserve failed runs. Screening is allowed, but only a complete row can qualify.

[All metrics, convergence and separate EMA](leaderboard.json) · [Registered entries](entries.json) · [Validation](validation.json) · [Historical adjusted comparison](../default_comparison/README.md).
