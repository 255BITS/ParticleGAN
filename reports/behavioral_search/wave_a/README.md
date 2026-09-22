# Behavioral baseline — live weights

> Side search on a CPU that does not reproduce the published ring. Do not promote these rows over the [PR #38 leaderboard](../../behavioral_baseline/README.md). Summary: [../README.md](../README.md).

**Passing live baseline: `r1_r2_0_1_l2_005`.**

Protocol `behavior-v1` · CPU · seed 0 · fixed host budgets · final live weights.

**Overall PASS requires all 29 numerical bounds on all 9 trained toys, plus the 10 shared behavioral/integration checks.** EMA is diagnostic and cannot rescue a live failure. Config equality checks are excluded. Shared checks are run once: they do not depend on the GAN config and do not contribute to its rank.

Rows rank by passed toys, then passed numerical bounds; equal counts tie. Missing/nonfinite results and errors cannot pass. Thresholds and budgets are frozen before config search.

| Rank | Config | Live toys | Live bounds | Overall | Failed live targets |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | `r1_r2_0_1_l2_005` | 9/9 | 29/29 | **PASS** | None |
| 2 | `r1_r2_0_15` | 8/9 | 28/29 | **FAIL** | trajectory |
| 3 | `r1_r2_0_05` | 8/9 | 27/29 | **FAIL** | mode_hold |
| 3 | `r1_r2_0_1_cover_1` | 8/9 | 27/29 | **FAIL** | mode_hold |
| 5 | `r1_r2_0_08` | 7/9 | 27/29 | **FAIL** | trajectory, mode_hold |
| 5 | `r1_r2_0_1_l2_001` | 7/9 | 27/29 | **FAIL** | trajectory, mode_hold |
| 7 | `r1_r2_0_12` | 7/9 | 26/29 | **FAIL** | trajectory, mode_hold |
| 7 | `r1_r2_0_1_no_l2` | 7/9 | 26/29 | **FAIL** | trajectory, mode_hold |
| 7 | `r1_r2_0_1_no_vicreg` | 7/9 | 26/29 | **FAIL** | trajectory, mode_hold |
| 10 | `r1_r2_0_1_cover_0` | 7/9 | 23/29 | **FAIL** | cover_leftover, mode_hold |
| 11 | `r1_r2_0_2` | 6/9 | 23/29 | **FAIL** | trajectory, residual_student, mode_hold |

## Live toy matrix

| Config | two_pole | trajectory | residual_student | unipolar | ae_gan_hold | cover_leftover | unused_token_hold | mid_scale_identity | mode_hold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `r1_r2_0_1_l2_005` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `r1_r2_0_15` | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `r1_r2_0_05` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | FAIL |
| `r1_r2_0_1_cover_1` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | FAIL |
| `r1_r2_0_08` | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | FAIL |
| `r1_r2_0_1_l2_001` | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | FAIL |
| `r1_r2_0_12` | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | FAIL |
| `r1_r2_0_1_no_l2` | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | FAIL |
| `r1_r2_0_1_no_vicreg` | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | FAIL |
| `r1_r2_0_1_cover_0` | PASS | PASS | PASS | PASS | PASS | FAIL | PASS | PASS | FAIL |
| `r1_r2_0_2` | PASS | FAIL | FAIL | PASS | PASS | PASS | PASS | PASS | FAIL |

## EMA diagnostics

Only the ring and cover/leftover hosts maintain EMA. These results are separate from the live ranking.

| Config | Ring modes | Ring HQ | Ring bounds | Cover/leftover bounds |
| --- | ---: | ---: | --- | --- |
| `r1_r2_0_1_l2_005` | 7/8 | 91.89% | 2/2 | 6/6 |
| `r1_r2_0_15` | 8/8 | 83.72% | 1/2 | 6/6 |
| `r1_r2_0_05` | 5/8 | 66.92% | 0/2 | 6/6 |
| `r1_r2_0_1_cover_1` | 6/8 | 50.22% | 0/2 | 6/6 |
| `r1_r2_0_08` | 5/8 | 82.96% | 0/2 | 6/6 |
| `r1_r2_0_1_l2_001` | 6/8 | 74.58% | 0/2 | 6/6 |
| `r1_r2_0_12` | 1/8 | 7.67% | 0/2 | 6/6 |
| `r1_r2_0_1_no_l2` | 6/8 | 83.91% | 0/2 | 6/6 |
| `r1_r2_0_1_no_vicreg` | 6/8 | 91.58% | 1/2 | 6/6 |
| `r1_r2_0_1_cover_0` | 6/8 | 50.22% | 0/2 | 2/6 |
| `r1_r2_0_2` | 5/8 | 65.77% | 0/2 | 6/6 |

## Shared behavioral and integration checks

These measure fixed geometry, checkpoint selection, frozen-critic behavior, LoRA targeting and DSL execution. They run against the pinned conceptmod application; no candidate receives extra ranking points for them.

| Check | Status | Evidence |
| --- | --- | --- |
| orbit_hold | PASS | direction_cos=0.99203 (>= 0.98); speed_rel=0.00803434 (<= 0.05); radius_rel=4.74158e-17 (<= 0.02) |
| erase_keep_backend | PASS | teacher_leak=0 (<= 0.2); u_kept=1 (>= 0.85); pole_rel_err=0 (<= 0.2); same_dir=0 (<= 0.25) |
| late_collapse | PASS | selected_val_error=0.06 (<= 0.25); selected_collapsed=0 (<= 0); last_val_error=1.4 (>= 0.25); best_train_val_error=1.4 (>= 0.25) |
| keep_critic | PASS | weight_delta=0 (<= 0.0) |
| lm_target | PASS | expr_gain=1 (>= 0.85); struct_hold=1 (>= 0.95); identity_mse=0 (<= 1e-08) |
| field_lift | PASS | plane: PASS; tilted: PASS |
| path_suffix_lora | PASS | suffix: PASS; regex: PASS |
| dsl_macro_expand | PASS | documented expansion PASS; 7/7 incorrect expansions rejected |
| dsl_phrase_jobs | PASS | neutralize: PASS; bipolar: PASS; remap: PASS; keep_erase: PASS; mix: PASS; isolate: PASS |
| dsl_game_geometry | PASS | write: PASS; erase_esd: PASS; exaggerate: PASS; erase_esd_freeze: PASS |

## Every live metric

Margins are in each metric's own units: nonnegative meets the bound. No averaging across metrics.

<details><summary>r1_r2_0_1_l2_005: PASS</summary>

```json
{
  "name": "r1_r2_0_1_l2_005",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "a_r1r2",
  "reg_coeff": 0.1,
  "reg_kappa": 1.0,
  "particle_l2": 0.005,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 1.0
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.45249978 | >= 0.3 | +0.1525 | PASS |
| two_pole | grad_med | 0.70941538 | <= 1.0 | +0.2906 | PASS |
| trajectory | identity_mse | 0.0014659077 | <= 0.02 | +0.01853 | PASS |
| residual_student | identity_mse | 0.0013384549 | <= 0.02 | +0.01866 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.99981427 | >= 0.85 | +0.1498 | PASS |
| unipolar | off_caption | 2.3531376e-10 | <= 0.05 | +0.05 | PASS |
| unipolar | neu_hold | 0.99962805 | >= 0.85 | +0.1496 | PASS |
| ae_gan_hold | recon_mse | 0.0025427183 | <= 0.05 | +0.04746 | PASS |
| ae_gan_hold | hold | 0.023057705 | <= 0.35 | +0.3269 | PASS |
| cover_leftover | u_kept | 0.99401515 | >= 0.85 | +0.144 | PASS |
| cover_leftover | content_kept | 1.0039576 | >= 0.75 | +0.254 | PASS |
| cover_leftover | leak_ratio | 0.0019926781 | <= 0.2 | +0.198 | PASS |
| cover_leftover | pole_rel_err_plus | 0.0043984647 | <= 0.2 | +0.1956 | PASS |
| cover_leftover | pole_rel_err_minus | 0.0031145136 | <= 0.2 | +0.1969 | PASS |
| cover_leftover | same_dir | 0.0016501368 | <= 0.25 | +0.2483 | PASS |
| unused_token_hold | unused_hold | 0.99430069 | >= 0.85 | +0.1443 | PASS |
| unused_token_hold | concept_move | 0.99949992 | >= 0.85 | +0.1495 | PASS |
| mid_scale_identity | concept_cos_plus | 1 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_cos_minus | 1 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0000033 | >= 0.75 | +0.25 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0000033 | <= 1.25 | +0.25 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0000005 | >= 0.75 | +0.25 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0000005 | <= 1.25 | +0.25 | PASS |
| mid_scale_identity | identity_at_0 | 0.99999837 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | identity_at_mid | 0.99998613 | >= 0.85 | +0.15 | PASS |
| mode_hold | modes | 8 | >= 7 | +1 | PASS |
| mode_hold | hq | 1 | >= 0.9 | +0.1 | PASS |

</details>

<details><summary>r1_r2_0_15: FAIL</summary>

```json
{
  "name": "r1_r2_0_15",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "a_r1r2",
  "reg_coeff": 0.15,
  "reg_kappa": 1.0,
  "particle_l2": 0.02,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 1.0
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.41680124 | >= 0.3 | +0.1168 | PASS |
| two_pole | grad_med | 0.50002998 | <= 1.0 | +0.5 | PASS |
| trajectory | identity_mse | 0.29674104 | <= 0.02 | -0.2767 | FAIL |
| residual_student | identity_mse | 0.0011495517 | <= 0.02 | +0.01885 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.99930161 | >= 0.85 | +0.1493 | PASS |
| unipolar | off_caption | 2.5162807e-11 | <= 0.05 | +0.05 | PASS |
| unipolar | neu_hold | 0.99868863 | >= 0.85 | +0.1487 | PASS |
| ae_gan_hold | recon_mse | 0.003295908 | <= 0.05 | +0.0467 | PASS |
| ae_gan_hold | hold | 0.020326147 | <= 0.35 | +0.3297 | PASS |
| cover_leftover | u_kept | 0.99655413 | >= 0.85 | +0.1466 | PASS |
| cover_leftover | content_kept | 1.0028858 | >= 0.75 | +0.2529 | PASS |
| cover_leftover | leak_ratio | 0.0014989826 | <= 0.2 | +0.1985 | PASS |
| cover_leftover | pole_rel_err_plus | 0.002690976 | <= 0.2 | +0.1973 | PASS |
| cover_leftover | pole_rel_err_minus | 0.0021493989 | <= 0.2 | +0.1979 | PASS |
| cover_leftover | same_dir | 0.0013874896 | <= 0.25 | +0.2486 | PASS |
| unused_token_hold | unused_hold | 0.99119209 | >= 0.85 | +0.1412 | PASS |
| unused_token_hold | concept_move | 0.99790549 | >= 0.85 | +0.1479 | PASS |
| mid_scale_identity | concept_cos_plus | 1 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_cos_minus | 1 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_mag_plus | 1.000006 | >= 0.75 | +0.25 | PASS |
| mid_scale_identity | concept_mag_plus | 1.000006 | <= 1.25 | +0.25 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0000021 | >= 0.75 | +0.25 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0000021 | <= 1.25 | +0.25 | PASS |
| mid_scale_identity | identity_at_0 | 0.99998277 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | identity_at_mid | 0.99984514 | >= 0.85 | +0.1498 | PASS |
| mode_hold | modes | 8 | >= 7 | +1 | PASS |
| mode_hold | hq | 1 | >= 0.9 | +0.1 | PASS |

</details>

<details><summary>r1_r2_0_05: FAIL</summary>

```json
{
  "name": "r1_r2_0_05",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "a_r1r2",
  "reg_coeff": 0.05,
  "reg_kappa": 1.0,
  "particle_l2": 0.02,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 1.0
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.46114373 | >= 0.3 | +0.1611 | PASS |
| two_pole | grad_med | 0.95730174 | <= 1.0 | +0.0427 | PASS |
| trajectory | identity_mse | 0.0028654875 | <= 0.02 | +0.01713 | PASS |
| residual_student | identity_mse | 0.0019388889 | <= 0.02 | +0.01806 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.99999887 | >= 0.85 | +0.15 | PASS |
| unipolar | off_caption | 1.3276096e-11 | <= 0.05 | +0.05 | PASS |
| unipolar | neu_hold | 0.99997381 | >= 0.85 | +0.15 | PASS |
| ae_gan_hold | recon_mse | 0.0031279949 | <= 0.05 | +0.04687 | PASS |
| ae_gan_hold | hold | 0.0084408261 | <= 0.35 | +0.3416 | PASS |
| cover_leftover | u_kept | 0.98523425 | >= 0.85 | +0.1352 | PASS |
| cover_leftover | content_kept | 1.0044472 | >= 0.75 | +0.2544 | PASS |
| cover_leftover | leak_ratio | 0.0028048001 | <= 0.2 | +0.1972 | PASS |
| cover_leftover | pole_rel_err_plus | 0.010033817 | <= 0.2 | +0.19 | PASS |
| cover_leftover | pole_rel_err_minus | 0.0076605212 | <= 0.2 | +0.1923 | PASS |
| cover_leftover | same_dir | 0.0021707254 | <= 0.25 | +0.2478 | PASS |
| unused_token_hold | unused_hold | 0.99621605 | >= 0.85 | +0.1462 | PASS |
| unused_token_hold | concept_move | 0.99995017 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_cos_plus | 0.99999487 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_cos_minus | 1 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_mag_plus | 0.99865431 | >= 0.75 | +0.2487 | PASS |
| mid_scale_identity | concept_mag_plus | 0.99865431 | <= 1.25 | +0.2513 | PASS |
| mid_scale_identity | concept_mag_minus | 0.99998492 | >= 0.75 | +0.25 | PASS |
| mid_scale_identity | concept_mag_minus | 0.99998492 | <= 1.25 | +0.25 | PASS |
| mid_scale_identity | identity_at_0 | 0.99990821 | >= 0.85 | +0.1499 | PASS |
| mid_scale_identity | identity_at_mid | 0.99421846 | >= 0.85 | +0.1442 | PASS |
| mode_hold | modes | 5 | >= 7 | -2 | FAIL |
| mode_hold | hq | 0.73876953 | >= 0.9 | -0.1612 | FAIL |

</details>

<details><summary>r1_r2_0_1_cover_1: FAIL</summary>

```json
{
  "name": "r1_r2_0_1_cover_1",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "a_r1r2",
  "reg_coeff": 0.1,
  "reg_kappa": 1.0,
  "particle_l2": 0.02,
  "vicreg_weight": 0.05,
  "cover_weight": 1.0,
  "lr_multiplier": 1.0
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.45243272 | >= 0.3 | +0.1524 | PASS |
| two_pole | grad_med | 0.73340946 | <= 1.0 | +0.2666 | PASS |
| trajectory | identity_mse | 0.0028052721 | <= 0.02 | +0.01719 | PASS |
| residual_student | identity_mse | 0.0019482638 | <= 0.02 | +0.01805 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.99981427 | >= 0.85 | +0.1498 | PASS |
| unipolar | off_caption | 2.3531376e-10 | <= 0.05 | +0.05 | PASS |
| unipolar | neu_hold | 0.99962805 | >= 0.85 | +0.1496 | PASS |
| ae_gan_hold | recon_mse | 0.0035041561 | <= 0.05 | +0.0465 | PASS |
| ae_gan_hold | hold | 0.033878949 | <= 0.35 | +0.3161 | PASS |
| cover_leftover | u_kept | 0.9877231 | >= 0.85 | +0.1377 | PASS |
| cover_leftover | content_kept | 1.0043882 | >= 0.75 | +0.2544 | PASS |
| cover_leftover | leak_ratio | 0.002043791 | <= 0.2 | +0.198 | PASS |
| cover_leftover | pole_rel_err_plus | 0.0083526084 | <= 0.2 | +0.1916 | PASS |
| cover_leftover | pole_rel_err_minus | 0.0065590478 | <= 0.2 | +0.1934 | PASS |
| cover_leftover | same_dir | 0.0019953843 | <= 0.25 | +0.248 | PASS |
| unused_token_hold | unused_hold | 0.99430069 | >= 0.85 | +0.1443 | PASS |
| unused_token_hold | concept_move | 0.99949992 | >= 0.85 | +0.1495 | PASS |
| mid_scale_identity | concept_cos_plus | 1 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_cos_minus | 1 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0000727 | >= 0.75 | +0.2501 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0000727 | <= 1.25 | +0.2499 | PASS |
| mid_scale_identity | concept_mag_minus | 0.99995333 | >= 0.75 | +0.25 | PASS |
| mid_scale_identity | concept_mag_minus | 0.99995333 | <= 1.25 | +0.25 | PASS |
| mid_scale_identity | identity_at_0 | 0.99995134 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | identity_at_mid | 0.99958894 | >= 0.85 | +0.1496 | PASS |
| mode_hold | modes | 1 | >= 7 | -6 | FAIL |
| mode_hold | hq | 0.083251953 | >= 0.9 | -0.8167 | FAIL |

</details>

<details><summary>r1_r2_0_08: FAIL</summary>

```json
{
  "name": "r1_r2_0_08",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "a_r1r2",
  "reg_coeff": 0.08,
  "reg_kappa": 1.0,
  "particle_l2": 0.02,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 1.0
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.45605361 | >= 0.3 | +0.1561 | PASS |
| two_pole | grad_med | 0.84002757 | <= 1.0 | +0.16 | PASS |
| trajectory | identity_mse | 0.022175701 | <= 0.02 | -0.002176 | FAIL |
| residual_student | identity_mse | 0.0023159345 | <= 0.02 | +0.01768 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.99994135 | >= 0.85 | +0.1499 | PASS |
| unipolar | off_caption | 2.8948553e-11 | <= 0.05 | +0.05 | PASS |
| unipolar | neu_hold | 0.99987647 | >= 0.85 | +0.1499 | PASS |
| ae_gan_hold | recon_mse | 0.0054269983 | <= 0.05 | +0.04457 | PASS |
| ae_gan_hold | hold | 0.02950941 | <= 0.35 | +0.3205 | PASS |
| cover_leftover | u_kept | 0.99207347 | >= 0.85 | +0.1421 | PASS |
| cover_leftover | content_kept | 1.0040683 | >= 0.75 | +0.2541 | PASS |
| cover_leftover | leak_ratio | 0.0022023436 | <= 0.2 | +0.1978 | PASS |
| cover_leftover | pole_rel_err_plus | 0.0056210649 | <= 0.2 | +0.1944 | PASS |
| cover_leftover | pole_rel_err_minus | 0.0040014074 | <= 0.2 | +0.196 | PASS |
| cover_leftover | same_dir | 0.0018888497 | <= 0.25 | +0.2481 | PASS |
| unused_token_hold | unused_hold | 0.99599892 | >= 0.85 | +0.146 | PASS |
| unused_token_hold | concept_move | 0.99991721 | >= 0.85 | +0.1499 | PASS |
| mid_scale_identity | concept_cos_plus | 1 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_cos_minus | 1 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0000696 | >= 0.75 | +0.2501 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0000696 | <= 1.25 | +0.2499 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0000172 | >= 0.75 | +0.25 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0000172 | <= 1.25 | +0.25 | PASS |
| mid_scale_identity | identity_at_0 | 0.99999155 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | identity_at_mid | 0.99970566 | >= 0.85 | +0.1497 | PASS |
| mode_hold | modes | 5 | >= 7 | -2 | FAIL |
| mode_hold | hq | 1 | >= 0.9 | +0.1 | PASS |

</details>

<details><summary>r1_r2_0_1_l2_001: FAIL</summary>

```json
{
  "name": "r1_r2_0_1_l2_001",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "a_r1r2",
  "reg_coeff": 0.1,
  "reg_kappa": 1.0,
  "particle_l2": 0.01,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 1.0
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.45271191 | >= 0.3 | +0.1527 | PASS |
| two_pole | grad_med | 0.71018612 | <= 1.0 | +0.2898 | PASS |
| trajectory | identity_mse | 0.26175204 | <= 0.02 | -0.2418 | FAIL |
| residual_student | identity_mse | 0.0014402666 | <= 0.02 | +0.01856 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.99981427 | >= 0.85 | +0.1498 | PASS |
| unipolar | off_caption | 2.3531376e-10 | <= 0.05 | +0.05 | PASS |
| unipolar | neu_hold | 0.99962805 | >= 0.85 | +0.1496 | PASS |
| ae_gan_hold | recon_mse | 0.0051971157 | <= 0.05 | +0.0448 | PASS |
| ae_gan_hold | hold | 0.016089249 | <= 0.35 | +0.3339 | PASS |
| cover_leftover | u_kept | 0.99406194 | >= 0.85 | +0.1441 | PASS |
| cover_leftover | content_kept | 1.0039024 | >= 0.75 | +0.2539 | PASS |
| cover_leftover | leak_ratio | 0.001919321 | <= 0.2 | +0.1981 | PASS |
| cover_leftover | pole_rel_err_plus | 0.0043490659 | <= 0.2 | +0.1957 | PASS |
| cover_leftover | pole_rel_err_minus | 0.0031057096 | <= 0.2 | +0.1969 | PASS |
| cover_leftover | same_dir | 0.0015497793 | <= 0.25 | +0.2485 | PASS |
| unused_token_hold | unused_hold | 0.99430069 | >= 0.85 | +0.1443 | PASS |
| unused_token_hold | concept_move | 0.99949992 | >= 0.85 | +0.1495 | PASS |
| mid_scale_identity | concept_cos_plus | 1 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_cos_minus | 1 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0000033 | >= 0.75 | +0.25 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0000033 | <= 1.25 | +0.25 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0000005 | >= 0.75 | +0.25 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0000005 | <= 1.25 | +0.25 | PASS |
| mid_scale_identity | identity_at_0 | 0.99999837 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | identity_at_mid | 0.99998613 | >= 0.85 | +0.15 | PASS |
| mode_hold | modes | 6 | >= 7 | -1 | FAIL |
| mode_hold | hq | 1 | >= 0.9 | +0.1 | PASS |

</details>

<details><summary>r1_r2_0_12: FAIL</summary>

```json
{
  "name": "r1_r2_0_12",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "a_r1r2",
  "reg_coeff": 0.12,
  "reg_kappa": 1.0,
  "particle_l2": 0.02,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 1.0
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.44034246 | >= 0.3 | +0.1403 | PASS |
| two_pole | grad_med | 0.61827528 | <= 1.0 | +0.3817 | PASS |
| trajectory | identity_mse | 0.26135871 | <= 0.02 | -0.2414 | FAIL |
| residual_student | identity_mse | 0.0011664898 | <= 0.02 | +0.01883 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.99954361 | >= 0.85 | +0.1495 | PASS |
| unipolar | off_caption | 2.6876604e-11 | <= 0.05 | +0.05 | PASS |
| unipolar | neu_hold | 0.99915157 | >= 0.85 | +0.1492 | PASS |
| ae_gan_hold | recon_mse | 0.0024723606 | <= 0.05 | +0.04753 | PASS |
| ae_gan_hold | hold | 0.019146556 | <= 0.35 | +0.3309 | PASS |
| cover_leftover | u_kept | 0.99567144 | >= 0.85 | +0.1457 | PASS |
| cover_leftover | content_kept | 1.0032168 | >= 0.75 | +0.2532 | PASS |
| cover_leftover | leak_ratio | 0.0014004048 | <= 0.2 | +0.1986 | PASS |
| cover_leftover | pole_rel_err_plus | 0.0032334167 | <= 0.2 | +0.1968 | PASS |
| cover_leftover | pole_rel_err_minus | 0.0026939334 | <= 0.2 | +0.1973 | PASS |
| cover_leftover | same_dir | 0.0013826035 | <= 0.25 | +0.2486 | PASS |
| unused_token_hold | unused_hold | 0.99137845 | >= 0.85 | +0.1414 | PASS |
| unused_token_hold | concept_move | 0.99889472 | >= 0.85 | +0.1489 | PASS |
| mid_scale_identity | concept_cos_plus | 1 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_cos_minus | 1 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0000045 | >= 0.75 | +0.25 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0000045 | <= 1.25 | +0.25 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0000064 | >= 0.75 | +0.25 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0000064 | <= 1.25 | +0.25 | PASS |
| mid_scale_identity | identity_at_0 | 0.99994636 | >= 0.85 | +0.1499 | PASS |
| mid_scale_identity | identity_at_mid | 0.99971162 | >= 0.85 | +0.1497 | PASS |
| mode_hold | modes | 3 | >= 7 | -4 | FAIL |
| mode_hold | hq | 0.59106445 | >= 0.9 | -0.3089 | FAIL |

</details>

<details><summary>r1_r2_0_1_no_l2: FAIL</summary>

```json
{
  "name": "r1_r2_0_1_no_l2",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "a_r1r2",
  "reg_coeff": 0.1,
  "reg_kappa": 1.0,
  "particle_l2": 0.0,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 1.0
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.45215765 | >= 0.3 | +0.1522 | PASS |
| two_pole | grad_med | 0.71325541 | <= 1.0 | +0.2867 | PASS |
| trajectory | identity_mse | 0.25768968 | <= 0.02 | -0.2377 | FAIL |
| residual_student | identity_mse | 0.0014335552 | <= 0.02 | +0.01857 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.99981427 | >= 0.85 | +0.1498 | PASS |
| unipolar | off_caption | 2.3531376e-10 | <= 0.05 | +0.05 | PASS |
| unipolar | neu_hold | 0.99962805 | >= 0.85 | +0.1496 | PASS |
| ae_gan_hold | recon_mse | 0.0031004217 | <= 0.05 | +0.0469 | PASS |
| ae_gan_hold | hold | 0.002231841 | <= 0.35 | +0.3478 | PASS |
| cover_leftover | u_kept | 0.99414473 | >= 0.85 | +0.1441 | PASS |
| cover_leftover | content_kept | 1.0042237 | >= 0.75 | +0.2542 | PASS |
| cover_leftover | leak_ratio | 0.0018800111 | <= 0.2 | +0.1981 | PASS |
| cover_leftover | pole_rel_err_plus | 0.0043318411 | <= 0.2 | +0.1957 | PASS |
| cover_leftover | pole_rel_err_minus | 0.0031601849 | <= 0.2 | +0.1968 | PASS |
| cover_leftover | same_dir | 0.0017076932 | <= 0.25 | +0.2483 | PASS |
| unused_token_hold | unused_hold | 0.99430069 | >= 0.85 | +0.1443 | PASS |
| unused_token_hold | concept_move | 0.99949992 | >= 0.85 | +0.1495 | PASS |
| mid_scale_identity | concept_cos_plus | 1 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_cos_minus | 1 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0000033 | >= 0.75 | +0.25 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0000033 | <= 1.25 | +0.25 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0000005 | >= 0.75 | +0.25 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0000005 | <= 1.25 | +0.25 | PASS |
| mid_scale_identity | identity_at_0 | 0.99999837 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | identity_at_mid | 0.99998613 | >= 0.85 | +0.15 | PASS |
| mode_hold | modes | 1 | >= 7 | -6 | FAIL |
| mode_hold | hq | 0.079345703 | >= 0.9 | -0.8207 | FAIL |

</details>

<details><summary>r1_r2_0_1_no_vicreg: FAIL</summary>

```json
{
  "name": "r1_r2_0_1_no_vicreg",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "a_r1r2",
  "reg_coeff": 0.1,
  "reg_kappa": 1.0,
  "particle_l2": 0.02,
  "vicreg_weight": 0.0,
  "cover_weight": 1.5,
  "lr_multiplier": 1.0
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.45243272 | >= 0.3 | +0.1524 | PASS |
| two_pole | grad_med | 0.73340946 | <= 1.0 | +0.2666 | PASS |
| trajectory | identity_mse | 0.24995297 | <= 0.02 | -0.23 | FAIL |
| residual_student | identity_mse | 0.0029907599 | <= 0.02 | +0.01701 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.99981427 | >= 0.85 | +0.1498 | PASS |
| unipolar | off_caption | 2.3531376e-10 | <= 0.05 | +0.05 | PASS |
| unipolar | neu_hold | 0.99962805 | >= 0.85 | +0.1496 | PASS |
| ae_gan_hold | recon_mse | 0.0025832078 | <= 0.05 | +0.04742 | PASS |
| ae_gan_hold | hold | 0.0016914558 | <= 0.35 | +0.3483 | PASS |
| cover_leftover | u_kept | 0.99459939 | >= 0.85 | +0.1446 | PASS |
| cover_leftover | content_kept | 1.0022153 | >= 0.75 | +0.2522 | PASS |
| cover_leftover | leak_ratio | 0.0016908651 | <= 0.2 | +0.1983 | PASS |
| cover_leftover | pole_rel_err_plus | 0.0038171012 | <= 0.2 | +0.1962 | PASS |
| cover_leftover | pole_rel_err_minus | 0.0032795335 | <= 0.2 | +0.1967 | PASS |
| cover_leftover | same_dir | 0.0015465199 | <= 0.25 | +0.2485 | PASS |
| unused_token_hold | unused_hold | 0.99430069 | >= 0.85 | +0.1443 | PASS |
| unused_token_hold | concept_move | 0.99949992 | >= 0.85 | +0.1495 | PASS |
| mid_scale_identity | concept_cos_plus | 1 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_cos_minus | 1 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0000033 | >= 0.75 | +0.25 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0000033 | <= 1.25 | +0.25 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0000005 | >= 0.75 | +0.25 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0000005 | <= 1.25 | +0.25 | PASS |
| mid_scale_identity | identity_at_0 | 0.99999837 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | identity_at_mid | 0.99998613 | >= 0.85 | +0.15 | PASS |
| mode_hold | modes | 2 | >= 7 | -5 | FAIL |
| mode_hold | hq | 0.33251953 | >= 0.9 | -0.5675 | FAIL |

</details>

<details><summary>r1_r2_0_1_cover_0: FAIL</summary>

```json
{
  "name": "r1_r2_0_1_cover_0",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "a_r1r2",
  "reg_coeff": 0.1,
  "reg_kappa": 1.0,
  "particle_l2": 0.02,
  "vicreg_weight": 0.05,
  "cover_weight": 0.0,
  "lr_multiplier": 1.0
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.45243272 | >= 0.3 | +0.1524 | PASS |
| two_pole | grad_med | 0.73340946 | <= 1.0 | +0.2666 | PASS |
| trajectory | identity_mse | 0.0017796927 | <= 0.02 | +0.01822 | PASS |
| residual_student | identity_mse | 0.0015484063 | <= 0.02 | +0.01845 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.99981427 | >= 0.85 | +0.1498 | PASS |
| unipolar | off_caption | 2.3531376e-10 | <= 0.05 | +0.05 | PASS |
| unipolar | neu_hold | 0.99962805 | >= 0.85 | +0.1496 | PASS |
| ae_gan_hold | recon_mse | 0.0026361679 | <= 0.05 | +0.04736 | PASS |
| ae_gan_hold | hold | 0.0011342166 | <= 0.35 | +0.3489 | PASS |
| cover_leftover | u_kept | 0.61506652 | >= 0.85 | -0.2349 | FAIL |
| cover_leftover | content_kept | 0.67415881 | >= 0.75 | -0.07584 | FAIL |
| cover_leftover | leak_ratio | 0.057033124 | <= 0.2 | +0.143 | PASS |
| cover_leftover | pole_rel_err_plus | 0.28180423 | <= 0.2 | -0.0818 | FAIL |
| cover_leftover | pole_rel_err_minus | 0.2819283 | <= 0.2 | -0.08193 | FAIL |
| cover_leftover | same_dir | 0.012579933 | <= 0.25 | +0.2374 | PASS |
| unused_token_hold | unused_hold | 0.99430069 | >= 0.85 | +0.1443 | PASS |
| unused_token_hold | concept_move | 0.99949992 | >= 0.85 | +0.1495 | PASS |
| mid_scale_identity | concept_cos_plus | 1 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_cos_minus | 0.99999994 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_mag_plus | 0.99975485 | >= 0.75 | +0.2498 | PASS |
| mid_scale_identity | concept_mag_plus | 0.99975485 | <= 1.25 | +0.2502 | PASS |
| mid_scale_identity | concept_mag_minus | 0.99970591 | >= 0.75 | +0.2497 | PASS |
| mid_scale_identity | concept_mag_minus | 0.99970591 | <= 1.25 | +0.2503 | PASS |
| mid_scale_identity | identity_at_0 | 0.99978813 | >= 0.85 | +0.1498 | PASS |
| mid_scale_identity | identity_at_mid | 0.9997916 | >= 0.85 | +0.1498 | PASS |
| mode_hold | modes | 1 | >= 7 | -6 | FAIL |
| mode_hold | hq | 0.083251953 | >= 0.9 | -0.8167 | FAIL |

</details>

<details><summary>r1_r2_0_2: FAIL</summary>

```json
{
  "name": "r1_r2_0_2",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "a_r1r2",
  "reg_coeff": 0.2,
  "reg_kappa": 1.0,
  "particle_l2": 0.02,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 1.0
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.37161395 | >= 0.3 | +0.07161 | PASS |
| two_pole | grad_med | 0.38548204 | <= 1.0 | +0.6145 | PASS |
| trajectory | identity_mse | 0.48885044 | <= 0.02 | -0.4689 | FAIL |
| residual_student | identity_mse | 0.059231684 | <= 0.02 | -0.03923 | FAIL |
| residual_student | success_rate | 0.58333331 | >= 1.0 | -0.4167 | FAIL |
| residual_student | wrong_pad_rate | 0.58333331 | <= 0.0 | -0.5833 | FAIL |
| unipolar | cover | 0.99882287 | >= 0.85 | +0.1488 | PASS |
| unipolar | off_caption | 1.0997802e-10 | <= 0.05 | +0.05 | PASS |
| unipolar | neu_hold | 0.99771 | >= 0.85 | +0.1477 | PASS |
| ae_gan_hold | recon_mse | 0.0030230817 | <= 0.05 | +0.04698 | PASS |
| ae_gan_hold | hold | 0.0019355965 | <= 0.35 | +0.3481 | PASS |
| cover_leftover | u_kept | 0.99783282 | >= 0.85 | +0.1478 | PASS |
| cover_leftover | content_kept | 1.0026064 | >= 0.75 | +0.2526 | PASS |
| cover_leftover | leak_ratio | 0.0012107332 | <= 0.2 | +0.1988 | PASS |
| cover_leftover | pole_rel_err_plus | 0.0019235663 | <= 0.2 | +0.1981 | PASS |
| cover_leftover | pole_rel_err_minus | 0.0019777736 | <= 0.2 | +0.198 | PASS |
| cover_leftover | same_dir | 0.0014928235 | <= 0.25 | +0.2485 | PASS |
| unused_token_hold | unused_hold | 0.99720029 | >= 0.85 | +0.1472 | PASS |
| unused_token_hold | concept_move | 0.99619026 | >= 0.85 | +0.1462 | PASS |
| mid_scale_identity | concept_cos_plus | 0.99999994 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_cos_minus | 1 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_mag_plus | 0.99991614 | >= 0.75 | +0.2499 | PASS |
| mid_scale_identity | concept_mag_plus | 0.99991614 | <= 1.25 | +0.2501 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0000114 | >= 0.75 | +0.25 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0000114 | <= 1.25 | +0.25 | PASS |
| mid_scale_identity | identity_at_0 | 0.99997518 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | identity_at_mid | 0.99940027 | >= 0.85 | +0.1494 | PASS |
| mode_hold | modes | 3 | >= 7 | -4 | FAIL |
| mode_hold | hq | 0.40625 | >= 0.9 | -0.4938 | FAIL |

</details>

## Reproduce or compare another approach

```bash
python -m benchmarks.locked_shared.baseline --reference /path/to/conceptmod
python -m benchmarks.locked_shared.baseline --configs my_configs.json --reference /path/to/conceptmod --output reports/my_search
# Resume only with exactly matching source, runtime and config fingerprints:
python -m benchmarks.locked_shared.baseline --resume --reference /path/to/conceptmod
```

Use [passing_configs.json](passing_configs.json) to rerun the passing baseline alone, or [configs.json](configs.json) for the comparison set. Each setting applies wherever that loss exists; there are no per-toy overrides. The fixed protocol and actual source hashes are in [protocol.json](protocol.json); all measurements, timings and shared-check evidence are in [results.json](results.json). See [scope and configuration mapping](../../benchmarks/locked_shared/BASELINE.md). Exit code 0 means at least one full PASS; 1 means no full PASS. Output is saved after each toy. Without `--reference`, candidate training still runs but the full result is INCOMPLETE.

This is one fixed-seed CPU regression baseline, not evidence of downstream transfer or robustness across random initializations.
