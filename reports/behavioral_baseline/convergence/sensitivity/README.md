# Behavioral baseline — live weights

**Regression PASS: `cap3_cosine_start0p6_lr0p85`, `sensitivity_reg_kappa_1p1`, `sensitivity_lr_multiplier_1p1`, `sensitivity_vicreg_weight_1p1`, `sensitivity_reg_kappa_0p9`.**

Protocol `behavior-v2` · CPU · seed 0 · fixed host budgets · final live weights.

**Overall PASS requires all 29 numerical bounds on all 9 trained toys, plus the 10 shared behavioral/integration checks.** EMA is diagnostic and cannot rescue a live failure. Config equality checks are excluded. Shared checks are run once: they do not depend on the GAN config and do not contribute to its rank.

**Regression PASS is a minimum bar for default selection.** The original ring bar permits 7/8 modes. Actual coverage, HQ, balance and checkpoint stability are visible below; a final-step PASS does not establish a stable ParticleGAN default. The [default-selection analysis](../../default_selection.md) also compares the stock recipe on the ring host.

Rows rank by passed toys, then passed numerical bounds, then live ring coverage, HQ and effective modes. Missing/nonfinite results and errors cannot pass. Thresholds and budgets are frozen before config search.

| Rank | Config | Live toys | Live bounds | Ring modes | Ring HQ | Effective modes | Regression |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `cap3_cosine_start0p6_lr0p85` (b_cap) | 9/9 | 29/29 | 8/8 | 100.00% | 7.54/8 | **PASS** |
| 2 | `sensitivity_reg_kappa_1p1` (b_cap) | 9/9 | 29/29 | 8/8 | 100.00% | 7.26/8 | **PASS** |
| 3 | `sensitivity_lr_multiplier_1p1` (b_cap) | 9/9 | 29/29 | 8/8 | 100.00% | 7.19/8 | **PASS** |
| 4 | `sensitivity_vicreg_weight_1p1` (b_cap) | 9/9 | 29/29 | 8/8 | 100.00% | 7.17/8 | **PASS** |
| 5 | `sensitivity_reg_kappa_0p9` (b_cap) | 9/9 | 29/29 | 8/8 | 100.00% | 6.99/8 | **PASS** |
| 6 | `sensitivity_lr_multiplier_0p9` (b_cap) | 8/9 | 28/29 | 7/8 | 100.00% | 6.43/8 | **FAIL** |
| 7 | `sensitivity_vicreg_weight_0p9` (b_cap) | 8/9 | 28/29 | 7/8 | 83.06% | 6.60/8 | **FAIL** |
| 8 | `sensitivity_reg_coeff_1p1` (b_cap) | 8/9 | 27/29 | 6/8 | 83.25% | 5.42/8 | **FAIL** |
| 9 | `sensitivity_reg_coeff_0p9` (b_cap) | 8/9 | 27/29 | 4/8 | 48.73% | 3.77/8 | **FAIL** |

Effective modes measures balance among high-quality outputs; eight balanced modes gives 8. HQ measures quality of generated samples and does not penalize a missing target cluster. Thus 100% HQ can coexist with 7/8 coverage.

## Live stability and exact particle coverage

Final-step selection is unchanged. For default selection, report the five observations at steps 1,000, 1,050, 1,100, 1,150 and 1,200. These are sampled checkpoints, not a claim about every intervening step. Enumerating all 12 equally likely particles additionally distinguishes a missing mode from an unlucky 4,096-sample evaluation.

| Config | Worst tail modes | Worst tail HQ | Tail checks with 8/8 and HQ ≥90% | Final HQ particles by mode (0–7) |
| --- | ---: | ---: | ---: | --- |
| `cap3_cosine_start0p6_lr0p85` | 8/8 | 90.99% | 5/5 | [1, 2, 1, 1, 2, 1, 2, 2] |
| `sensitivity_reg_kappa_1p1` | 8/8 | 91.38% | 5/5 | [1, 2, 2, 1, 1, 1, 3, 1] |
| `sensitivity_lr_multiplier_1p1` | 8/8 | 100.00% | 5/5 | [3, 1, 2, 2, 1, 1, 1, 1] |
| `sensitivity_vicreg_weight_1p1` | 8/8 | 83.45% | 4/5 | [3, 2, 1, 1, 1, 1, 2, 1] |
| `sensitivity_reg_kappa_0p9` | 8/8 | 92.33% | 5/5 | [1, 1, 3, 1, 3, 1, 1, 1] |
| `sensitivity_lr_multiplier_0p9` | 6/8 | 82.96% | 0/5 | [3, 1, 2, 0, 2, 1, 2, 1] |
| `sensitivity_vicreg_weight_0p9` | 5/8 | 58.25% | 0/5 | [2, 1, 0, 2, 1, 1, 2, 1] |
| `sensitivity_reg_coeff_1p1` | 5/8 | 75.59% | 0/5 | [0, 1, 3, 1, 2, 2, 1, 0] |
| `sensitivity_reg_coeff_0p9` | 2/8 | 23.85% | 0/5 | [2, 0, 0, 0, 2, 0, 1, 1] |

## Convergence speed

Each host has 24 evenly spaced observations. Sustained PASS requires a complete curve, at least five consecutive passing observations, and no later failure through the final budget. Ring convergence requires all eight modes and HQ ≥90%; original regression bounds stay unchanged. Stable-from is the start of that final passing stretch; confirmation is its fifth observation. Times include setup and measurement overhead. These certify observations, not every intervening update. Historical runs without timing are not assigned estimated speeds.

| Config | Sustained toys | Ring first PASS step | Ring stable from step | Ring confirmed step | Ring confirmed seconds | All toys wall seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cap3_cosine_start0p6_lr0p85` | 9/9 | 850 | 850 | 1050 | 5.76 | 21.30 |
| `sensitivity_reg_kappa_1p1` | 9/9 | 900 | 1000 | 1200 | 6.38 | 19.54 |
| `sensitivity_lr_multiplier_1p1` | 9/9 | 850 | 850 | 1050 | 5.54 | 19.26 |
| `sensitivity_vicreg_weight_1p1` | 8/9 | 950 | Not reached | Not reached | Not reached | 19.48 |
| `sensitivity_reg_kappa_0p9` | 9/9 | 950 | 950 | 1150 | 6.08 | 19.33 |
| `sensitivity_lr_multiplier_0p9` | 7/9 | Not reached | Not reached | Not reached | Not reached | 19.71 |
| `sensitivity_vicreg_weight_0p9` | 8/9 | Not reached | Not reached | Not reached | Not reached | 19.56 |
| `sensitivity_reg_coeff_1p1` | 8/9 | Not reached | Not reached | Not reached | Not reached | 19.48 |
| `sensitivity_reg_coeff_0p9` | 8/9 | Not reached | Not reached | Not reached | Not reached | 19.31 |

## Stock-recipe ring comparison

Both rows use the same 20,000 particles, 7,000 steps, optimizer and cosine schedule; only the penalty changes. They are compared with each other and are not included in the 12-particle regression rank. This is the ring host, not the 100-Gaussian benchmark. Tail observations cover the final 200 steps at 50-step intervals.

| Penalty | Live modes | Live HQ | Effective modes | EMA modes / HQ | Worst tail live HQ | Full-coverage/HQ tail checks |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| b_cap coeff 1.0 | 8/8 | 99.05% | 7.95/8 | 8/8 / 99.66% | 98.51% | 5/5 |
| a_r1r2 coeff 0.1 | 8/8 | 99.32% | 7.94/8 | 8/8 / 99.44% | 99.24% | 5/5 |

Both stock-recipe penalties sustain full coverage at the measured late checkpoints. The current `b_cap` recipe and its EMA pass; the small-host R1+R2 result does not justify replacing the default or disabling EMA. Raw evidence and its original source fingerprint are in [stock_ring.json](../../stock_ring.json).


## Live toy matrix

| Config | two_pole | trajectory | residual_student | unipolar | ae_gan_hold | cover_leftover | unused_token_hold | mid_scale_identity | mode_hold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cap3_cosine_start0p6_lr0p85` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `sensitivity_reg_kappa_1p1` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `sensitivity_lr_multiplier_1p1` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `sensitivity_vicreg_weight_1p1` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `sensitivity_reg_kappa_0p9` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `sensitivity_lr_multiplier_0p9` | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `sensitivity_vicreg_weight_0p9` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | FAIL |
| `sensitivity_reg_coeff_1p1` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | FAIL |
| `sensitivity_reg_coeff_0p9` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | FAIL |

## EMA diagnostics

Only the ring and cover/leftover hosts maintain EMA. These results are separate from the live ranking.

| Config | Ring modes | Ring HQ | Ring bounds | Cover/leftover bounds |
| --- | ---: | ---: | --- | --- |
| `cap3_cosine_start0p6_lr0p85` | 8/8 | 100.00% | 2/2 | 6/6 |
| `sensitivity_reg_kappa_1p1` | 8/8 | 100.00% | 2/2 | 6/6 |
| `sensitivity_lr_multiplier_1p1` | 8/8 | 100.00% | 2/2 | 6/6 |
| `sensitivity_vicreg_weight_1p1` | 8/8 | 92.07% | 2/2 | 6/6 |
| `sensitivity_reg_kappa_0p9` | 8/8 | 100.00% | 2/2 | 6/6 |
| `sensitivity_lr_multiplier_0p9` | 7/8 | 100.00% | 2/2 | 6/6 |
| `sensitivity_vicreg_weight_0p9` | 7/8 | 75.39% | 1/2 | 6/6 |
| `sensitivity_reg_coeff_1p1` | 5/8 | 67.31% | 0/2 | 6/6 |
| `sensitivity_reg_coeff_0p9` | 3/8 | 39.87% | 0/2 | 6/6 |

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

<details><summary>cap3_cosine_start0p6_lr0p85: PASS</summary>

```json
{
  "name": "cap3_cosine_start0p6_lr0p85",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "b_cap",
  "reg_coeff": 3.0,
  "reg_kappa": 1.25,
  "particle_l2": 0.0,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 0.85,
  "lr_schedule": "cosine",
  "lr_anneal_start": 0.6,
  "lr_floor": 0.05
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.35178924 | >= 0.3 | +0.05179 | PASS |
| two_pole | grad_med | 0.39726675 | <= 1.0 | +0.6027 | PASS |
| trajectory | identity_mse | 2.0105224e-05 | <= 0.02 | +0.01998 | PASS |
| residual_student | identity_mse | 1.6864316e-05 | <= 0.02 | +0.01998 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.98261771 | >= 0.85 | +0.1326 | PASS |
| unipolar | off_caption | 0.00068092852 | <= 0.05 | +0.04932 | PASS |
| unipolar | neu_hold | 0.97055284 | >= 0.85 | +0.1206 | PASS |
| ae_gan_hold | recon_mse | 0.0097926352 | <= 0.05 | +0.04021 | PASS |
| ae_gan_hold | hold | 0.0014229559 | <= 0.35 | +0.3486 | PASS |
| cover_leftover | u_kept | 0.99445545 | >= 0.85 | +0.1445 | PASS |
| cover_leftover | content_kept | 0.99390267 | >= 0.75 | +0.2439 | PASS |
| cover_leftover | leak_ratio | 0.0020671902 | <= 0.2 | +0.1979 | PASS |
| cover_leftover | pole_rel_err_plus | 0.0067389295 | <= 0.2 | +0.1933 | PASS |
| cover_leftover | pole_rel_err_minus | 0.0034179795 | <= 0.2 | +0.1966 | PASS |
| cover_leftover | same_dir | 0.0027803755 | <= 0.25 | +0.2472 | PASS |
| unused_token_hold | unused_hold | 0.9919152 | >= 0.85 | +0.1419 | PASS |
| unused_token_hold | concept_move | 0.96520639 | >= 0.85 | +0.1152 | PASS |
| mid_scale_identity | concept_cos_plus | 0.99998099 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_cos_minus | 0.99999762 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_mag_plus | 0.99937016 | >= 0.75 | +0.2494 | PASS |
| mid_scale_identity | concept_mag_plus | 0.99937016 | <= 1.25 | +0.2506 | PASS |
| mid_scale_identity | concept_mag_minus | 1.000617 | >= 0.75 | +0.2506 | PASS |
| mid_scale_identity | concept_mag_minus | 1.000617 | <= 1.25 | +0.2494 | PASS |
| mid_scale_identity | identity_at_0 | 0.99875459 | >= 0.85 | +0.1488 | PASS |
| mid_scale_identity | identity_at_mid | 0.99771367 | >= 0.85 | +0.1477 | PASS |
| mode_hold | modes | 8 | >= 7 | +1 | PASS |
| mode_hold | hq | 1 | >= 0.9 | +0.1 | PASS |

</details>

<details><summary>sensitivity_reg_kappa_1p1: PASS</summary>

```json
{
  "name": "sensitivity_reg_kappa_1p1",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "b_cap",
  "reg_coeff": 3.0,
  "reg_kappa": 1.375,
  "particle_l2": 0.0,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 0.85,
  "lr_schedule": "cosine",
  "lr_anneal_start": 0.6,
  "lr_floor": 0.05
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.36238006 | >= 0.3 | +0.06238 | PASS |
| two_pole | grad_med | 0.4550128 | <= 1.0 | +0.545 | PASS |
| trajectory | identity_mse | 1.4748763e-05 | <= 0.02 | +0.01999 | PASS |
| residual_student | identity_mse | 1.6449392e-05 | <= 0.02 | +0.01998 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.93421639 | >= 0.85 | +0.08422 | PASS |
| unipolar | off_caption | 4.1099943e-06 | <= 0.05 | +0.05 | PASS |
| unipolar | neu_hold | 0.94118043 | >= 0.85 | +0.09118 | PASS |
| ae_gan_hold | recon_mse | 0.010705932 | <= 0.05 | +0.03929 | PASS |
| ae_gan_hold | hold | 0.0012587575 | <= 0.35 | +0.3487 | PASS |
| cover_leftover | u_kept | 0.98903804 | >= 0.85 | +0.139 | PASS |
| cover_leftover | content_kept | 0.99750885 | >= 0.75 | +0.2475 | PASS |
| cover_leftover | leak_ratio | 0.003292564 | <= 0.2 | +0.1967 | PASS |
| cover_leftover | pole_rel_err_plus | 0.0089537604 | <= 0.2 | +0.191 | PASS |
| cover_leftover | pole_rel_err_minus | 0.003909451 | <= 0.2 | +0.1961 | PASS |
| cover_leftover | same_dir | 0.0049832545 | <= 0.25 | +0.245 | PASS |
| unused_token_hold | unused_hold | 0.99131802 | >= 0.85 | +0.1413 | PASS |
| unused_token_hold | concept_move | 0.95348361 | >= 0.85 | +0.1035 | PASS |
| mid_scale_identity | concept_cos_plus | 0.99990171 | >= 0.85 | +0.1499 | PASS |
| mid_scale_identity | concept_cos_minus | 0.99984783 | >= 0.85 | +0.1498 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0149978 | >= 0.75 | +0.265 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0149978 | <= 1.25 | +0.235 | PASS |
| mid_scale_identity | concept_mag_minus | 0.9889369 | >= 0.75 | +0.2389 | PASS |
| mid_scale_identity | concept_mag_minus | 0.9889369 | <= 1.25 | +0.2611 | PASS |
| mid_scale_identity | identity_at_0 | 0.98248187 | >= 0.85 | +0.1325 | PASS |
| mid_scale_identity | identity_at_mid | 0.99903094 | >= 0.85 | +0.149 | PASS |
| mode_hold | modes | 8 | >= 7 | +1 | PASS |
| mode_hold | hq | 1 | >= 0.9 | +0.1 | PASS |

</details>

<details><summary>sensitivity_lr_multiplier_1p1: PASS</summary>

```json
{
  "name": "sensitivity_lr_multiplier_1p1",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "b_cap",
  "reg_coeff": 3.0,
  "reg_kappa": 1.25,
  "particle_l2": 0.0,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 0.935,
  "lr_schedule": "cosine",
  "lr_anneal_start": 0.6,
  "lr_floor": 0.05
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.40202332 | >= 0.3 | +0.102 | PASS |
| two_pole | grad_med | 0.4503893 | <= 1.0 | +0.5496 | PASS |
| trajectory | identity_mse | 1.8523851e-05 | <= 0.02 | +0.01998 | PASS |
| residual_student | identity_mse | 3.0011186e-05 | <= 0.02 | +0.01997 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.93235193 | >= 0.85 | +0.08235 | PASS |
| unipolar | off_caption | 0.0009688352 | <= 0.05 | +0.04903 | PASS |
| unipolar | neu_hold | 0.93343086 | >= 0.85 | +0.08343 | PASS |
| ae_gan_hold | recon_mse | 0.010881466 | <= 0.05 | +0.03912 | PASS |
| ae_gan_hold | hold | 0.0026164879 | <= 0.35 | +0.3474 | PASS |
| cover_leftover | u_kept | 0.99161588 | >= 0.85 | +0.1416 | PASS |
| cover_leftover | content_kept | 0.99505856 | >= 0.75 | +0.2451 | PASS |
| cover_leftover | leak_ratio | 0.0051862467 | <= 0.2 | +0.1948 | PASS |
| cover_leftover | pole_rel_err_plus | 0.0075348564 | <= 0.2 | +0.1925 | PASS |
| cover_leftover | pole_rel_err_minus | 0.0021529365 | <= 0.2 | +0.1978 | PASS |
| cover_leftover | same_dir | 0.004651641 | <= 0.25 | +0.2453 | PASS |
| unused_token_hold | unused_hold | 0.99262107 | >= 0.85 | +0.1426 | PASS |
| unused_token_hold | concept_move | 0.92765437 | >= 0.85 | +0.07765 | PASS |
| mid_scale_identity | concept_cos_plus | 0.99999893 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_cos_minus | 0.99999887 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0003219 | >= 0.75 | +0.2503 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0003219 | <= 1.25 | +0.2497 | PASS |
| mid_scale_identity | concept_mag_minus | 0.99942124 | >= 0.75 | +0.2494 | PASS |
| mid_scale_identity | concept_mag_minus | 0.99942124 | <= 1.25 | +0.2506 | PASS |
| mid_scale_identity | identity_at_0 | 0.99803608 | >= 0.85 | +0.148 | PASS |
| mid_scale_identity | identity_at_mid | 0.99808019 | >= 0.85 | +0.1481 | PASS |
| mode_hold | modes | 8 | >= 7 | +1 | PASS |
| mode_hold | hq | 1 | >= 0.9 | +0.1 | PASS |

</details>

<details><summary>sensitivity_vicreg_weight_1p1: PASS</summary>

```json
{
  "name": "sensitivity_vicreg_weight_1p1",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "b_cap",
  "reg_coeff": 3.0,
  "reg_kappa": 1.25,
  "particle_l2": 0.0,
  "vicreg_weight": 0.055,
  "cover_weight": 1.5,
  "lr_multiplier": 0.85,
  "lr_schedule": "cosine",
  "lr_anneal_start": 0.6,
  "lr_floor": 0.05
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.35178924 | >= 0.3 | +0.05179 | PASS |
| two_pole | grad_med | 0.39726675 | <= 1.0 | +0.6027 | PASS |
| trajectory | identity_mse | 1.8896239e-05 | <= 0.02 | +0.01998 | PASS |
| residual_student | identity_mse | 1.6694357e-05 | <= 0.02 | +0.01998 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.98261771 | >= 0.85 | +0.1326 | PASS |
| unipolar | off_caption | 0.00068092852 | <= 0.05 | +0.04932 | PASS |
| unipolar | neu_hold | 0.97055284 | >= 0.85 | +0.1206 | PASS |
| ae_gan_hold | recon_mse | 0.0097926352 | <= 0.05 | +0.04021 | PASS |
| ae_gan_hold | hold | 0.0014229559 | <= 0.35 | +0.3486 | PASS |
| cover_leftover | u_kept | 0.98994612 | >= 0.85 | +0.1399 | PASS |
| cover_leftover | content_kept | 0.9976428 | >= 0.75 | +0.2476 | PASS |
| cover_leftover | leak_ratio | 0.0037357336 | <= 0.2 | +0.1963 | PASS |
| cover_leftover | pole_rel_err_plus | 0.0073291245 | <= 0.2 | +0.1927 | PASS |
| cover_leftover | pole_rel_err_minus | 0.0029036808 | <= 0.2 | +0.1971 | PASS |
| cover_leftover | same_dir | 0.0043573483 | <= 0.25 | +0.2456 | PASS |
| unused_token_hold | unused_hold | 0.9919152 | >= 0.85 | +0.1419 | PASS |
| unused_token_hold | concept_move | 0.96520639 | >= 0.85 | +0.1152 | PASS |
| mid_scale_identity | concept_cos_plus | 0.99998099 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_cos_minus | 0.99999762 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_mag_plus | 0.99937016 | >= 0.75 | +0.2494 | PASS |
| mid_scale_identity | concept_mag_plus | 0.99937016 | <= 1.25 | +0.2506 | PASS |
| mid_scale_identity | concept_mag_minus | 1.000617 | >= 0.75 | +0.2506 | PASS |
| mid_scale_identity | concept_mag_minus | 1.000617 | <= 1.25 | +0.2494 | PASS |
| mid_scale_identity | identity_at_0 | 0.99875459 | >= 0.85 | +0.1488 | PASS |
| mid_scale_identity | identity_at_mid | 0.99771367 | >= 0.85 | +0.1477 | PASS |
| mode_hold | modes | 8 | >= 7 | +1 | PASS |
| mode_hold | hq | 1 | >= 0.9 | +0.1 | PASS |

</details>

<details><summary>sensitivity_reg_kappa_0p9: PASS</summary>

```json
{
  "name": "sensitivity_reg_kappa_0p9",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "b_cap",
  "reg_coeff": 3.0,
  "reg_kappa": 1.125,
  "particle_l2": 0.0,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 0.85,
  "lr_schedule": "cosine",
  "lr_anneal_start": 0.6,
  "lr_floor": 0.05
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.33736363 | >= 0.3 | +0.03736 | PASS |
| two_pole | grad_med | 0.33723226 | <= 1.0 | +0.6628 | PASS |
| trajectory | identity_mse | 2.5616862e-05 | <= 0.02 | +0.01997 | PASS |
| residual_student | identity_mse | 9.1800548e-06 | <= 0.02 | +0.01999 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.95841981 | >= 0.85 | +0.1084 | PASS |
| unipolar | off_caption | 0.00014295695 | <= 0.05 | +0.04986 | PASS |
| unipolar | neu_hold | 0.94192824 | >= 0.85 | +0.09193 | PASS |
| ae_gan_hold | recon_mse | 0.011518259 | <= 0.05 | +0.03848 | PASS |
| ae_gan_hold | hold | 0.0015525327 | <= 0.35 | +0.3484 | PASS |
| cover_leftover | u_kept | 0.99352824 | >= 0.85 | +0.1435 | PASS |
| cover_leftover | content_kept | 0.99735865 | >= 0.75 | +0.2474 | PASS |
| cover_leftover | leak_ratio | 0.0049076613 | <= 0.2 | +0.1951 | PASS |
| cover_leftover | pole_rel_err_plus | 0.0057038902 | <= 0.2 | +0.1943 | PASS |
| cover_leftover | pole_rel_err_minus | 0.0034605793 | <= 0.2 | +0.1965 | PASS |
| cover_leftover | same_dir | 0.0043675221 | <= 0.25 | +0.2456 | PASS |
| unused_token_hold | unused_hold | 0.99237681 | >= 0.85 | +0.1424 | PASS |
| unused_token_hold | concept_move | 0.96687665 | >= 0.85 | +0.1169 | PASS |
| mid_scale_identity | concept_cos_plus | 0.99999452 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_cos_minus | 0.99999648 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_mag_plus | 1.002403 | >= 0.75 | +0.2524 | PASS |
| mid_scale_identity | concept_mag_plus | 1.002403 | <= 1.25 | +0.2476 | PASS |
| mid_scale_identity | concept_mag_minus | 0.99783796 | >= 0.75 | +0.2478 | PASS |
| mid_scale_identity | concept_mag_minus | 0.99783796 | <= 1.25 | +0.2522 | PASS |
| mid_scale_identity | identity_at_0 | 0.99888615 | >= 0.85 | +0.1489 | PASS |
| mid_scale_identity | identity_at_mid | 0.99805483 | >= 0.85 | +0.1481 | PASS |
| mode_hold | modes | 8 | >= 7 | +1 | PASS |
| mode_hold | hq | 1 | >= 0.9 | +0.1 | PASS |

</details>

<details><summary>sensitivity_lr_multiplier_0p9: FAIL</summary>

```json
{
  "name": "sensitivity_lr_multiplier_0p9",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "b_cap",
  "reg_coeff": 3.0,
  "reg_kappa": 1.25,
  "particle_l2": 0.0,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 0.765,
  "lr_schedule": "cosine",
  "lr_anneal_start": 0.6,
  "lr_floor": 0.05
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.2998752 | >= 0.3 | -0.0001248 | FAIL |
| two_pole | grad_med | 0.33886185 | <= 1.0 | +0.6611 | PASS |
| trajectory | identity_mse | 1.4081634e-05 | <= 0.02 | +0.01999 | PASS |
| residual_student | identity_mse | 8.6424907e-06 | <= 0.02 | +0.01999 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.99492195 | >= 0.85 | +0.1449 | PASS |
| unipolar | off_caption | 0.00028742515 | <= 0.05 | +0.04971 | PASS |
| unipolar | neu_hold | 0.97257401 | >= 0.85 | +0.1226 | PASS |
| ae_gan_hold | recon_mse | 0.017379232 | <= 0.05 | +0.03262 | PASS |
| ae_gan_hold | hold | 0.00084572792 | <= 0.35 | +0.3492 | PASS |
| cover_leftover | u_kept | 0.98259603 | >= 0.85 | +0.1326 | PASS |
| cover_leftover | content_kept | 0.98694659 | >= 0.75 | +0.2369 | PASS |
| cover_leftover | leak_ratio | 0.0042816652 | <= 0.2 | +0.1957 | PASS |
| cover_leftover | pole_rel_err_plus | 0.01352505 | <= 0.2 | +0.1865 | PASS |
| cover_leftover | pole_rel_err_minus | 0.0086264322 | <= 0.2 | +0.1914 | PASS |
| cover_leftover | same_dir | 0.0038055168 | <= 0.25 | +0.2462 | PASS |
| unused_token_hold | unused_hold | 0.99756439 | >= 0.85 | +0.1476 | PASS |
| unused_token_hold | concept_move | 0.86858691 | >= 0.85 | +0.01859 | PASS |
| mid_scale_identity | concept_cos_plus | 0.99991482 | >= 0.85 | +0.1499 | PASS |
| mid_scale_identity | concept_cos_minus | 0.99989539 | >= 0.85 | +0.1499 | PASS |
| mid_scale_identity | concept_mag_plus | 0.9880451 | >= 0.75 | +0.238 | PASS |
| mid_scale_identity | concept_mag_plus | 0.9880451 | <= 1.25 | +0.262 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0201077 | >= 0.75 | +0.2701 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0201077 | <= 1.25 | +0.2299 | PASS |
| mid_scale_identity | identity_at_0 | 0.98345386 | >= 0.85 | +0.1335 | PASS |
| mid_scale_identity | identity_at_mid | 0.99454748 | >= 0.85 | +0.1445 | PASS |
| mode_hold | modes | 7 | >= 7 | +0 | PASS |
| mode_hold | hq | 1 | >= 0.9 | +0.1 | PASS |

</details>

<details><summary>sensitivity_vicreg_weight_0p9: FAIL</summary>

```json
{
  "name": "sensitivity_vicreg_weight_0p9",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "b_cap",
  "reg_coeff": 3.0,
  "reg_kappa": 1.25,
  "particle_l2": 0.0,
  "vicreg_weight": 0.045,
  "cover_weight": 1.5,
  "lr_multiplier": 0.85,
  "lr_schedule": "cosine",
  "lr_anneal_start": 0.6,
  "lr_floor": 0.05
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.35178924 | >= 0.3 | +0.05179 | PASS |
| two_pole | grad_med | 0.39726675 | <= 1.0 | +0.6027 | PASS |
| trajectory | identity_mse | 1.4525137e-05 | <= 0.02 | +0.01999 | PASS |
| residual_student | identity_mse | 1.0369814e-05 | <= 0.02 | +0.01999 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.98261771 | >= 0.85 | +0.1326 | PASS |
| unipolar | off_caption | 0.00068092852 | <= 0.05 | +0.04932 | PASS |
| unipolar | neu_hold | 0.97055284 | >= 0.85 | +0.1206 | PASS |
| ae_gan_hold | recon_mse | 0.0097926352 | <= 0.05 | +0.04021 | PASS |
| ae_gan_hold | hold | 0.0014229559 | <= 0.35 | +0.3486 | PASS |
| cover_leftover | u_kept | 0.99124913 | >= 0.85 | +0.1412 | PASS |
| cover_leftover | content_kept | 0.99001167 | >= 0.75 | +0.24 | PASS |
| cover_leftover | leak_ratio | 0.0062300465 | <= 0.2 | +0.1938 | PASS |
| cover_leftover | pole_rel_err_plus | 0.0086838268 | <= 0.2 | +0.1913 | PASS |
| cover_leftover | pole_rel_err_minus | 0.0043431004 | <= 0.2 | +0.1957 | PASS |
| cover_leftover | same_dir | 0.0041077912 | <= 0.25 | +0.2459 | PASS |
| unused_token_hold | unused_hold | 0.9919152 | >= 0.85 | +0.1419 | PASS |
| unused_token_hold | concept_move | 0.96520639 | >= 0.85 | +0.1152 | PASS |
| mid_scale_identity | concept_cos_plus | 0.99998099 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_cos_minus | 0.99999762 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_mag_plus | 0.99937016 | >= 0.75 | +0.2494 | PASS |
| mid_scale_identity | concept_mag_plus | 0.99937016 | <= 1.25 | +0.2506 | PASS |
| mid_scale_identity | concept_mag_minus | 1.000617 | >= 0.75 | +0.2506 | PASS |
| mid_scale_identity | concept_mag_minus | 1.000617 | <= 1.25 | +0.2494 | PASS |
| mid_scale_identity | identity_at_0 | 0.99875459 | >= 0.85 | +0.1488 | PASS |
| mid_scale_identity | identity_at_mid | 0.99771367 | >= 0.85 | +0.1477 | PASS |
| mode_hold | modes | 7 | >= 7 | +0 | PASS |
| mode_hold | hq | 0.83056641 | >= 0.9 | -0.06943 | FAIL |

</details>

<details><summary>sensitivity_reg_coeff_1p1: FAIL</summary>

```json
{
  "name": "sensitivity_reg_coeff_1p1",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "b_cap",
  "reg_coeff": 3.3,
  "reg_kappa": 1.25,
  "particle_l2": 0.0,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 0.85,
  "lr_schedule": "cosine",
  "lr_anneal_start": 0.6,
  "lr_floor": 0.05
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.35152304 | >= 0.3 | +0.05152 | PASS |
| two_pole | grad_med | 0.3974089 | <= 1.0 | +0.6026 | PASS |
| trajectory | identity_mse | 1.2558331e-05 | <= 0.02 | +0.01999 | PASS |
| residual_student | identity_mse | 1.0995264e-05 | <= 0.02 | +0.01999 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.94784604 | >= 0.85 | +0.09785 | PASS |
| unipolar | off_caption | 0.00029029582 | <= 0.05 | +0.04971 | PASS |
| unipolar | neu_hold | 0.94263948 | >= 0.85 | +0.09264 | PASS |
| ae_gan_hold | recon_mse | 0.010254644 | <= 0.05 | +0.03975 | PASS |
| ae_gan_hold | hold | 0.0015817159 | <= 0.35 | +0.3484 | PASS |
| cover_leftover | u_kept | 0.99231951 | >= 0.85 | +0.1423 | PASS |
| cover_leftover | content_kept | 0.99436867 | >= 0.75 | +0.2444 | PASS |
| cover_leftover | leak_ratio | 0.0036856368 | <= 0.2 | +0.1963 | PASS |
| cover_leftover | pole_rel_err_plus | 0.0066647478 | <= 0.2 | +0.1933 | PASS |
| cover_leftover | pole_rel_err_minus | 0.0033764804 | <= 0.2 | +0.1966 | PASS |
| cover_leftover | same_dir | 0.0035797545 | <= 0.25 | +0.2464 | PASS |
| unused_token_hold | unused_hold | 0.99189882 | >= 0.85 | +0.1419 | PASS |
| unused_token_hold | concept_move | 0.95797311 | >= 0.85 | +0.108 | PASS |
| mid_scale_identity | concept_cos_plus | 0.99999177 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_cos_minus | 0.99996132 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0004942 | >= 0.75 | +0.2505 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0004942 | <= 1.25 | +0.2495 | PASS |
| mid_scale_identity | concept_mag_minus | 0.99220461 | >= 0.75 | +0.2422 | PASS |
| mid_scale_identity | concept_mag_minus | 0.99220461 | <= 1.25 | +0.2578 | PASS |
| mid_scale_identity | identity_at_0 | 0.98988303 | >= 0.85 | +0.1399 | PASS |
| mid_scale_identity | identity_at_mid | 0.99189073 | >= 0.85 | +0.1419 | PASS |
| mode_hold | modes | 6 | >= 7 | -1 | FAIL |
| mode_hold | hq | 0.83251953 | >= 0.9 | -0.06748 | FAIL |

</details>

<details><summary>sensitivity_reg_coeff_0p9: FAIL</summary>

```json
{
  "name": "sensitivity_reg_coeff_0p9",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "b_cap",
  "reg_coeff": 2.7,
  "reg_kappa": 1.25,
  "particle_l2": 0.0,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 0.85,
  "lr_schedule": "cosine",
  "lr_anneal_start": 0.6,
  "lr_floor": 0.05
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.35203421 | >= 0.3 | +0.05203 | PASS |
| two_pole | grad_med | 0.39657915 | <= 1.0 | +0.6034 | PASS |
| trajectory | identity_mse | 1.3757531e-05 | <= 0.02 | +0.01999 | PASS |
| residual_student | identity_mse | 1.8596706e-05 | <= 0.02 | +0.01998 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.96563516 | >= 0.85 | +0.1156 | PASS |
| unipolar | off_caption | 4.5274419e-06 | <= 0.05 | +0.05 | PASS |
| unipolar | neu_hold | 0.93888996 | >= 0.85 | +0.08889 | PASS |
| ae_gan_hold | recon_mse | 0.01060233 | <= 0.05 | +0.0394 | PASS |
| ae_gan_hold | hold | 0.0018078414 | <= 0.35 | +0.3482 | PASS |
| cover_leftover | u_kept | 0.99388843 | >= 0.85 | +0.1439 | PASS |
| cover_leftover | content_kept | 0.99576298 | >= 0.75 | +0.2458 | PASS |
| cover_leftover | leak_ratio | 0.0054538777 | <= 0.2 | +0.1945 | PASS |
| cover_leftover | pole_rel_err_plus | 0.0090216901 | <= 0.2 | +0.191 | PASS |
| cover_leftover | pole_rel_err_minus | 0.0066598454 | <= 0.2 | +0.1933 | PASS |
| cover_leftover | same_dir | 0.0026303975 | <= 0.25 | +0.2474 | PASS |
| unused_token_hold | unused_hold | 0.99190317 | >= 0.85 | +0.1419 | PASS |
| unused_token_hold | concept_move | 0.96673547 | >= 0.85 | +0.1167 | PASS |
| mid_scale_identity | concept_cos_plus | 0.99990612 | >= 0.85 | +0.1499 | PASS |
| mid_scale_identity | concept_cos_minus | 0.9998883 | >= 0.85 | +0.1499 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0044074 | >= 0.75 | +0.2544 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0044074 | <= 1.25 | +0.2456 | PASS |
| mid_scale_identity | concept_mag_minus | 0.99630028 | >= 0.75 | +0.2463 | PASS |
| mid_scale_identity | concept_mag_minus | 0.99630028 | <= 1.25 | +0.2537 | PASS |
| mid_scale_identity | identity_at_0 | 0.98400363 | >= 0.85 | +0.134 | PASS |
| mid_scale_identity | identity_at_mid | 0.99788143 | >= 0.85 | +0.1479 | PASS |
| mode_hold | modes | 4 | >= 7 | -3 | FAIL |
| mode_hold | hq | 0.48730469 | >= 0.9 | -0.4127 | FAIL |

</details>

## Reproduce or compare another approach

```bash
python -m benchmarks.locked_shared.baseline --reference /path/to/conceptmod
python -m benchmarks.locked_shared.baseline --configs my_configs.json --reference /path/to/conceptmod --output reports/my_search
# Resume only with exactly matching source, runtime and config fingerprints:
python -m benchmarks.locked_shared.baseline --resume --reference /path/to/conceptmod
```

Use [passing_configs.json](passing_configs.json) to rerun the passing baseline alone, or [configs.json](configs.json) for the comparison set. Each setting applies wherever that loss exists; there are no per-toy overrides. The fixed protocol and actual source hashes are in [protocol.json](protocol.json); all measurements, timings and shared-check evidence are in [results.json](results.json). See [scope and configuration mapping](../../../../benchmarks/locked_shared/BASELINE.md). Exit code 0 means at least one full PASS; 1 means no full PASS. Output is saved after each toy. Without `--reference`, candidate training still runs but the full result is INCOMPLETE.

This is one fixed-seed CPU regression baseline, not evidence of downstream transfer or robustness across random initializations.
