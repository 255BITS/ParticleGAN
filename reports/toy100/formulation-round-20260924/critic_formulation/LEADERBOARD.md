# Critic formulation search

Ranking: sustained primary passes, then lower mean frozen final normalized shortfall. Partial scores are diagnostic; no candidate is qualified without its own 22 gates and continuation.

| Rank | Candidate | Sustained primary | Ring modes / HQ | Ring suffix | Unequal eigen ratio | Unequal covariance error | Unequal suffix | Full 22 | Own-state stability |
|---:|---|---:|---|---:|---:|---:|---:|---|---|
| 1 | c05_ra_r1r2 | 1/2 | 8 / 1 | 5/5 | 0.38229 | 1.3411 | 0/5 | NOT_RUN | NOT_RUN |
| 2 | c06_ra_r1_only | 1/2 | 5 / 1 | 0/5 | 0.33534 | 0.25129 | 6/5 | NOT_RUN | NOT_RUN |
| 3 | c02_ra_cap | 0/2 | 7 / 0.99561 | 0/5 | 0.57846 | 0.88795 | 0/5 | NOT_RUN | NOT_RUN |
| 4 | c01_r1r2 | 0/2 | 8 / 0.90137 | 1/5 | 0.022336 | 0.79466 | 0/5 | NOT_RUN | NOT_RUN |
| 5 | c03_smooth_cap | 0/2 | 2 / 0.25952 | 0/5 | 0 | 0.48501 | 0/5 | NOT_RUN | NOT_RUN |
| 6 | c04_hinge_cap | 0/2 | 5 / 1 | 0/5 | 0.0068896 | 2.3112 | 0/5 | NOT_RUN | NOT_RUN |

All acquisition verdicts use the frozen sustained gate, including at least five passing observations at the end. Every primary screen uses its full 1,200 D + 1,200 G/prior updates. No seed experiments.

Gate matrix:

| Gate | c05_ra_r1r2 | c06_ra_r1_only | c02_ra_cap | c01_r1r2 | c03_smooth_cap | c04_hinge_cap |
|---|---|---|---|---|---|---|
| mode_hold | PASS | FAIL | FAIL | FAIL | FAIL | FAIL |
| vector_unequal_mass | FAIL | PASS | FAIL | FAIL | FAIL | FAIL |
| trajectory | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| img_intensity2 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| img_bars4 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| img_blobs4 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| two_pole | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| unipolar | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| mid_scale_identity | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| cover_leftover | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| residual_student | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| img_stripes2 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| vector_overlap | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| vector_unequal_width | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| vector_two_broad | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| vector_anisotropic | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| vector_spiral | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| ae_gan_hold | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| unused_token_hold | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| grid100 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| rotated100 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| staggered100 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
