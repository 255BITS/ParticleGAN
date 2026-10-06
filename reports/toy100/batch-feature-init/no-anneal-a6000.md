# Merged batch_feature_zero with the learning-rate anneal removed

Question: does the merged `batch_feature_zero` init still pass the fixed toy
gates and the long hold when the learning-rate anneal is removed?

**No.** Suite **13/22**. Priority **2/8**. Long hold **NOT_CONVERGED**.

Scored git SHA `1ababbb0056221b185a9773599a799ec728bfb20` on `develop`. PR #194
merge `c720645e` is an ancestor. Init was the `batch_feature_zero` hook, not a
fixture copy.

## Anneal removal, local to that run

Anneal removal was local to that run, not in this PR. `learning_rate_scale`
still checked its arguments, then returned `1.0` for every step. Generator,
critic, and prior
stayed at the recipe initial rates (`0.00425`, prior `0.0085`) for all 3,600
steps. `mode_hold` multipliers stayed `1.0` through step 1199. No salt, width,
or seed changes.

## Suite

| Task | Result | Recorded detail |
|---|---|---|
| two_pole | PASS | |
| trajectory | FAIL | identity MSE 1.014, limit 0.02 |
| residual_student | FAIL | identity MSE 0.197, success 0 |
| unipolar | PASS | |
| ae_gan_hold | PASS | |
| cover_leftover | PASS | |
| unused_token_hold | PASS | |
| mid_scale_identity | PASS | |
| mode_hold (ring) | PASS | |
| vector_two_broad | PASS | |
| vector_unequal_mass | FAIL | component covariance error 4.97, limit 0.85 |
| vector_unequal_width | FAIL | min eigenratio 0.146, need 0.15 |
| vector_anisotropic | FAIL | stable suffix 2/5 |
| vector_overlap | PASS | |
| vector_spiral | PASS | |
| img_stripes2 | PASS | |
| img_bars4 | PASS | |
| img_blobs4 | FAIL | stable suffix 1/5 |
| img_intensity2 | PASS | |
| grid100 | FAIL | final 4 modes, HQ 0.081, 0/5 terminal |
| rotated100 | FAIL | 100 modes, 0/5 terminal quality |
| staggered100 | FAIL | final 2 modes, HQ 0.035 |

## Stress

Long hold **NOT_CONVERGED** (longest streak 165/200, 0 hold checks; step 6300
was 1 mode, HQ 0.002). Pre-shift STAY (steps 1210–2400) **81/120 FAIL** (min 1
mode). Shifted-target recovery **FAIL**, 0/120, deadline missed, final 2 modes,
HQ 0.198.

## Priority convention at the repo seed

| Cell | Result |
|---|---|
| ring | PASS |
| stripes | PASS |
| unequal | FAIL |
| blobs | FAIL |
| hold | FAIL |
| stay | FAIL |
| grid100 | FAIL |
| rotated100 | FAIL |

Priority **2/8**.

## Hardware

Both RTX A6000s, cap 4 jobs per GPU. Not a CPU score.

## Reading

This section is a reading, not a new result. The published 22/22 still depends
on the anneal. This note does not propose or start a follow-up run.
