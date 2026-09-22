# Behavioral baseline — live weights

**Regression PASS: `bcap_k1p25_c3p0_lr0p85`, `cap3_cosine_start0p6_lr0p85`, `cap3_cosine_start0p8_lr0p85`, `cap3_cosine_start0p6_lr1p15`.**

Protocol `behavior-v2` · CPU · seed 0 · fixed host budgets · final live weights.

**Overall PASS requires all 29 numerical bounds on all 9 trained toys, plus the 10 shared behavioral/integration checks.** EMA is diagnostic and cannot rescue a live failure. Config equality checks are excluded. Shared checks are run once: they do not depend on the GAN config and do not contribute to its rank.

**Regression PASS is a minimum bar for default selection.** The original ring bar permits 7/8 modes. Actual coverage, HQ, balance and checkpoint stability are visible below; a final-step PASS does not establish a stable ParticleGAN default. The [default-selection analysis](../../default_selection.md) also compares the stock recipe on the ring host.

Rows rank by passed toys, then passed numerical bounds, then live ring coverage, HQ and effective modes. Missing/nonfinite results and errors cannot pass. Thresholds and budgets are frozen before config search.

| Rank | Config | Live toys | Live bounds | Ring modes | Ring HQ | Effective modes | Regression |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `bcap_k1p25_c3p0_lr0p85` (b_cap) | 9/9 | 29/29 | 8/8 | 100.00% | 7.54/8 | **PASS** |
| 1 | `cap3_cosine_start0p6_lr0p85` (b_cap) | 9/9 | 29/29 | 8/8 | 100.00% | 7.54/8 | **PASS** |
| 1 | `cap3_cosine_start0p8_lr0p85` (b_cap) | 9/9 | 29/29 | 8/8 | 100.00% | 7.54/8 | **PASS** |
| 4 | `cap3_cosine_start0p6_lr1p15` (b_cap) | 9/9 | 29/29 | 8/8 | 100.00% | 7.53/8 | **PASS** |
| 5 | `cap3_cosine_start0p4_lr0p85` (b_cap) | 8/9 | 28/29 | 8/8 | 92.33% | 7.50/8 | **FAIL** |
| 6 | `cap3_cosine_start0p8_lr1p15` (b_cap) | 8/9 | 28/29 | 7/8 | 75.20% | 6.56/8 | **FAIL** |
| 7 | `cap3_cosine_start0p6_lr1p0` (b_cap) | 8/9 | 27/29 | 4/8 | 50.95% | 3.75/8 | **FAIL** |
| 8 | `cap3_cosine_start0p8_lr1p0` (b_cap) | 8/9 | 27/29 | 4/8 | 50.51% | 3.73/8 | **FAIL** |

Effective modes measures balance among high-quality outputs; eight balanced modes gives 8. HQ measures quality of generated samples and does not penalize a missing target cluster. Thus 100% HQ can coexist with 7/8 coverage.

## Live stability and exact particle coverage

Final-step selection is unchanged. For default selection, report the five observations at steps 1,000, 1,050, 1,100, 1,150 and 1,200. These are sampled checkpoints, not a claim about every intervening step. Enumerating all 12 equally likely particles additionally distinguishes a missing mode from an unlucky 4,096-sample evaluation.

| Config | Worst tail modes | Worst tail HQ | Tail checks with 8/8 and HQ ≥90% | Final HQ particles by mode (0–7) |
| --- | ---: | ---: | ---: | --- |
| `bcap_k1p25_c3p0_lr0p85` | 8/8 | 83.91% | 4/5 | [1, 2, 1, 1, 2, 1, 2, 2] |
| `cap3_cosine_start0p6_lr0p85` | 8/8 | 90.99% | 5/5 | [1, 2, 1, 1, 2, 1, 2, 2] |
| `cap3_cosine_start0p8_lr0p85` | 8/8 | 100.00% | 5/5 | [1, 2, 1, 1, 2, 1, 2, 2] |
| `cap3_cosine_start0p6_lr1p15` | 6/8 | 74.71% | 3/5 | [2, 1, 2, 1, 2, 1, 2, 1] |
| `cap3_cosine_start0p4_lr0p85` | 8/8 | 90.99% | 5/5 | [1, 1, 1, 1, 2, 1, 2, 2] |
| `cap3_cosine_start0p8_lr1p15` | 5/8 | 59.11% | 1/5 | [0, 1, 1, 1, 2, 1, 2, 1] |
| `cap3_cosine_start0p6_lr1p0` | 4/8 | 50.95% | 0/5 | [2, 2, 0, 0, 0, 0, 1, 1] |
| `cap3_cosine_start0p8_lr1p0` | 4/8 | 50.51% | 0/5 | [2, 2, 1, 0, 1, 0, 0, 0] |

## Convergence speed

Each host has 24 evenly spaced observations. Sustained PASS requires a complete curve, at least five consecutive passing observations, and no later failure through the final budget. Ring convergence requires all eight modes and HQ ≥90%; original regression bounds stay unchanged. Stable-from is the start of that final passing stretch; confirmation is its fifth observation. Times include setup and measurement overhead. These certify observations, not every intervening update. Historical runs without timing are not assigned estimated speeds.

| Config | Sustained toys | Ring first PASS step | Ring stable from step | Ring confirmed step | Ring confirmed seconds | All toys wall seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bcap_k1p25_c3p0_lr0p85` | 8/9 | 1000 | Not reached | Not reached | Not reached | 21.24 |
| `cap3_cosine_start0p6_lr0p85` | 9/9 | 850 | 850 | 1050 | 5.61 | 20.10 |
| `cap3_cosine_start0p8_lr0p85` | 9/9 | 1000 | 1000 | 1200 | 6.54 | 20.18 |
| `cap3_cosine_start0p6_lr1p15` | 8/9 | 950 | Not reached | Not reached | Not reached | 20.09 |
| `cap3_cosine_start0p4_lr0p85` | 8/9 | 700 | 800 | 1000 | 5.44 | 20.16 |
| `cap3_cosine_start0p8_lr1p15` | 7/9 | 1100 | Not reached | Not reached | Not reached | 20.12 |
| `cap3_cosine_start0p6_lr1p0` | 8/9 | Not reached | Not reached | Not reached | Not reached | 20.08 |
| `cap3_cosine_start0p8_lr1p0` | 8/9 | Not reached | Not reached | Not reached | Not reached | 19.95 |

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
| `bcap_k1p25_c3p0_lr0p85` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `cap3_cosine_start0p6_lr0p85` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `cap3_cosine_start0p8_lr0p85` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `cap3_cosine_start0p6_lr1p15` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `cap3_cosine_start0p4_lr0p85` | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `cap3_cosine_start0p8_lr1p15` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | FAIL |
| `cap3_cosine_start0p6_lr1p0` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | FAIL |
| `cap3_cosine_start0p8_lr1p0` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | FAIL |

## EMA diagnostics

Only the ring and cover/leftover hosts maintain EMA. These results are separate from the live ranking.

| Config | Ring modes | Ring HQ | Ring bounds | Cover/leftover bounds |
| --- | ---: | ---: | --- | --- |
| `bcap_k1p25_c3p0_lr0p85` | 8/8 | 100.00% | 2/2 | 6/6 |
| `cap3_cosine_start0p6_lr0p85` | 8/8 | 100.00% | 2/2 | 6/6 |
| `cap3_cosine_start0p8_lr0p85` | 8/8 | 100.00% | 2/2 | 6/6 |
| `cap3_cosine_start0p6_lr1p15` | 8/8 | 100.00% | 2/2 | 6/6 |
| `cap3_cosine_start0p4_lr0p85` | 8/8 | 100.00% | 2/2 | 6/6 |
| `cap3_cosine_start0p8_lr1p15` | 8/8 | 92.33% | 2/2 | 6/6 |
| `cap3_cosine_start0p6_lr1p0` | 7/8 | 100.00% | 2/2 | 6/6 |
| `cap3_cosine_start0p8_lr1p0` | 4/8 | 50.68% | 0/2 | 6/6 |

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

<details><summary>bcap_k1p25_c3p0_lr0p85: PASS</summary>

```json
{
  "name": "bcap_k1p25_c3p0_lr0p85",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "b_cap",
  "reg_coeff": 3.0,
  "reg_kappa": 1.25,
  "particle_l2": 0.0,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 0.85,
  "lr_schedule": "host",
  "lr_anneal_start": 0.6,
  "lr_floor": 0.05
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.46814358 | >= 0.3 | +0.1681 | PASS |
| two_pole | grad_med | 0.51399529 | <= 1.0 | +0.486 | PASS |
| trajectory | identity_mse | 0.0024807875 | <= 0.02 | +0.01752 | PASS |
| residual_student | identity_mse | 0.0018423192 | <= 0.02 | +0.01816 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.9400924 | >= 0.85 | +0.09009 | PASS |
| unipolar | off_caption | 2.5285453e-07 | <= 0.05 | +0.05 | PASS |
| unipolar | neu_hold | 0.9424445 | >= 0.85 | +0.09244 | PASS |
| ae_gan_hold | recon_mse | 0.008615057 | <= 0.05 | +0.04138 | PASS |
| ae_gan_hold | hold | 0.014944904 | <= 0.35 | +0.3351 | PASS |
| cover_leftover | u_kept | 0.97393875 | >= 0.85 | +0.1239 | PASS |
| cover_leftover | content_kept | 1.014431 | >= 0.75 | +0.2644 | PASS |
| cover_leftover | leak_ratio | 0.00047995616 | <= 0.2 | +0.1995 | PASS |
| cover_leftover | pole_rel_err_plus | 0.017975062 | <= 0.2 | +0.182 | PASS |
| cover_leftover | pole_rel_err_minus | 0.018130351 | <= 0.2 | +0.1819 | PASS |
| cover_leftover | same_dir | 0.0062595584 | <= 0.25 | +0.2437 | PASS |
| unused_token_hold | unused_hold | 0.99618372 | >= 0.85 | +0.1462 | PASS |
| unused_token_hold | concept_move | 0.90526522 | >= 0.85 | +0.05527 | PASS |
| mid_scale_identity | concept_cos_plus | 0.99991602 | >= 0.85 | +0.1499 | PASS |
| mid_scale_identity | concept_cos_minus | 0.99995482 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0017906 | >= 0.75 | +0.2518 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0017906 | <= 1.25 | +0.2482 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0046275 | >= 0.75 | +0.2546 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0046275 | <= 1.25 | +0.2454 | PASS |
| mid_scale_identity | identity_at_0 | 0.99416178 | >= 0.85 | +0.1442 | PASS |
| mid_scale_identity | identity_at_mid | 0.99793291 | >= 0.85 | +0.1479 | PASS |
| mode_hold | modes | 8 | >= 7 | +1 | PASS |
| mode_hold | hq | 1 | >= 0.9 | +0.1 | PASS |

</details>

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

<details><summary>cap3_cosine_start0p8_lr0p85: PASS</summary>

```json
{
  "name": "cap3_cosine_start0p8_lr0p85",
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
  "lr_anneal_start": 0.8,
  "lr_floor": 0.05
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.41416061 | >= 0.3 | +0.1142 | PASS |
| two_pole | grad_med | 0.4605042 | <= 1.0 | +0.5395 | PASS |
| trajectory | identity_mse | 1.9266308e-05 | <= 0.02 | +0.01998 | PASS |
| residual_student | identity_mse | 1.7971675e-05 | <= 0.02 | +0.01998 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.97072746 | >= 0.85 | +0.1207 | PASS |
| unipolar | off_caption | 0.00029451375 | <= 0.05 | +0.04971 | PASS |
| unipolar | neu_hold | 0.96395746 | >= 0.85 | +0.114 | PASS |
| ae_gan_hold | recon_mse | 0.0049802689 | <= 0.05 | +0.04502 | PASS |
| ae_gan_hold | hold | 0.0017966061 | <= 0.35 | +0.3482 | PASS |
| cover_leftover | u_kept | 1.0049364 | >= 0.85 | +0.1549 | PASS |
| cover_leftover | content_kept | 1.0200347 | >= 0.75 | +0.27 | PASS |
| cover_leftover | leak_ratio | 0.0010265377 | <= 0.2 | +0.199 | PASS |
| cover_leftover | pole_rel_err_plus | 0.0080237286 | <= 0.2 | +0.192 | PASS |
| cover_leftover | pole_rel_err_minus | 0.010189686 | <= 0.2 | +0.1898 | PASS |
| cover_leftover | same_dir | 0.0043961899 | <= 0.25 | +0.2456 | PASS |
| unused_token_hold | unused_hold | 0.99440599 | >= 0.85 | +0.1444 | PASS |
| unused_token_hold | concept_move | 0.90350697 | >= 0.85 | +0.05351 | PASS |
| mid_scale_identity | concept_cos_plus | 0.99998862 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_cos_minus | 0.99999672 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0057001 | >= 0.75 | +0.2557 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0057001 | <= 1.25 | +0.2443 | PASS |
| mid_scale_identity | concept_mag_minus | 0.99403298 | >= 0.75 | +0.244 | PASS |
| mid_scale_identity | concept_mag_minus | 0.99403298 | <= 1.25 | +0.256 | PASS |
| mid_scale_identity | identity_at_0 | 0.99744935 | >= 0.85 | +0.1474 | PASS |
| mid_scale_identity | identity_at_mid | 0.99935594 | >= 0.85 | +0.1494 | PASS |
| mode_hold | modes | 8 | >= 7 | +1 | PASS |
| mode_hold | hq | 1 | >= 0.9 | +0.1 | PASS |

</details>

<details><summary>cap3_cosine_start0p6_lr1p15: PASS</summary>

```json
{
  "name": "cap3_cosine_start0p6_lr1p15",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "b_cap",
  "reg_coeff": 3.0,
  "reg_kappa": 1.25,
  "particle_l2": 0.0,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 1.15,
  "lr_schedule": "cosine",
  "lr_anneal_start": 0.6,
  "lr_floor": 0.05
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.52348679 | >= 0.3 | +0.2235 | PASS |
| two_pole | grad_med | 0.57029259 | <= 1.0 | +0.4297 | PASS |
| trajectory | identity_mse | 0.01865313 | <= 0.02 | +0.001347 | PASS |
| residual_student | identity_mse | 8.6173695e-06 | <= 0.02 | +0.01999 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.98349203 | >= 0.85 | +0.1335 | PASS |
| unipolar | off_caption | 5.8077611e-05 | <= 0.05 | +0.04994 | PASS |
| unipolar | neu_hold | 0.96334084 | >= 0.85 | +0.1133 | PASS |
| ae_gan_hold | recon_mse | 0.0022739228 | <= 0.05 | +0.04773 | PASS |
| ae_gan_hold | hold | 0.00042286396 | <= 0.35 | +0.3496 | PASS |
| cover_leftover | u_kept | 1.0016929 | >= 0.85 | +0.1517 | PASS |
| cover_leftover | content_kept | 1.0144973 | >= 0.75 | +0.2645 | PASS |
| cover_leftover | leak_ratio | 0.0041780471 | <= 0.2 | +0.1958 | PASS |
| cover_leftover | pole_rel_err_plus | 0.0063611655 | <= 0.2 | +0.1936 | PASS |
| cover_leftover | pole_rel_err_minus | 0.0059697628 | <= 0.2 | +0.194 | PASS |
| cover_leftover | same_dir | 0.0035245083 | <= 0.25 | +0.2465 | PASS |
| unused_token_hold | unused_hold | 0.99155043 | >= 0.85 | +0.1416 | PASS |
| unused_token_hold | concept_move | 0.97986357 | >= 0.85 | +0.1299 | PASS |
| mid_scale_identity | concept_cos_plus | 0.99999964 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_cos_minus | 1 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0000262 | >= 0.75 | +0.25 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0000262 | <= 1.25 | +0.25 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0004054 | >= 0.75 | +0.2504 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0004054 | <= 1.25 | +0.2496 | PASS |
| mid_scale_identity | identity_at_0 | 0.999934 | >= 0.85 | +0.1499 | PASS |
| mid_scale_identity | identity_at_mid | 0.99925039 | >= 0.85 | +0.1493 | PASS |
| mode_hold | modes | 8 | >= 7 | +1 | PASS |
| mode_hold | hq | 1 | >= 0.9 | +0.1 | PASS |

</details>

<details><summary>cap3_cosine_start0p4_lr0p85: FAIL</summary>

```json
{
  "name": "cap3_cosine_start0p4_lr0p85",
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
  "lr_anneal_start": 0.4,
  "lr_floor": 0.05
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.28688332 | >= 0.3 | -0.01312 | FAIL |
| two_pole | grad_med | 0.3222093 | <= 1.0 | +0.6778 | PASS |
| trajectory | identity_mse | 2.2470513e-05 | <= 0.02 | +0.01998 | PASS |
| residual_student | identity_mse | 1.3242202e-05 | <= 0.02 | +0.01999 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.9991443 | >= 0.85 | +0.1491 | PASS |
| unipolar | off_caption | 0.00018461872 | <= 0.05 | +0.04982 | PASS |
| unipolar | neu_hold | 0.98327398 | >= 0.85 | +0.1333 | PASS |
| ae_gan_hold | recon_mse | 0.0064886673 | <= 0.05 | +0.04351 | PASS |
| ae_gan_hold | hold | 0.0013810679 | <= 0.35 | +0.3486 | PASS |
| cover_leftover | u_kept | 0.98847662 | >= 0.85 | +0.1385 | PASS |
| cover_leftover | content_kept | 0.99429768 | >= 0.75 | +0.2443 | PASS |
| cover_leftover | leak_ratio | 0.0016445627 | <= 0.2 | +0.1984 | PASS |
| cover_leftover | pole_rel_err_plus | 0.0090535013 | <= 0.2 | +0.1909 | PASS |
| cover_leftover | pole_rel_err_minus | 0.0086793108 | <= 0.2 | +0.1913 | PASS |
| cover_leftover | same_dir | 0.0018899599 | <= 0.25 | +0.2481 | PASS |
| unused_token_hold | unused_hold | 0.99844377 | >= 0.85 | +0.1484 | PASS |
| unused_token_hold | concept_move | 0.86215721 | >= 0.85 | +0.01216 | PASS |
| mid_scale_identity | concept_cos_plus | 0.99993044 | >= 0.85 | +0.1499 | PASS |
| mid_scale_identity | concept_cos_minus | 0.99995363 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0062248 | >= 0.75 | +0.2562 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0062248 | <= 1.25 | +0.2438 | PASS |
| mid_scale_identity | concept_mag_minus | 0.9988845 | >= 0.75 | +0.2489 | PASS |
| mid_scale_identity | concept_mag_minus | 0.9988845 | <= 1.25 | +0.2511 | PASS |
| mid_scale_identity | identity_at_0 | 0.98895591 | >= 0.85 | +0.139 | PASS |
| mid_scale_identity | identity_at_mid | 0.99754201 | >= 0.85 | +0.1475 | PASS |
| mode_hold | modes | 8 | >= 7 | +1 | PASS |
| mode_hold | hq | 0.92333984 | >= 0.9 | +0.02334 | PASS |

</details>

<details><summary>cap3_cosine_start0p8_lr1p15: FAIL</summary>

```json
{
  "name": "cap3_cosine_start0p8_lr1p15",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "b_cap",
  "reg_coeff": 3.0,
  "reg_kappa": 1.25,
  "particle_l2": 0.0,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 1.15,
  "lr_schedule": "cosine",
  "lr_anneal_start": 0.8,
  "lr_floor": 0.05
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.59968644 | >= 0.3 | +0.2997 | PASS |
| two_pole | grad_med | 0.60322648 | <= 1.0 | +0.3968 | PASS |
| trajectory | identity_mse | 0.018593602 | <= 0.02 | +0.001406 | PASS |
| residual_student | identity_mse | 1.1449551e-05 | <= 0.02 | +0.01999 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.98254978 | >= 0.85 | +0.1325 | PASS |
| unipolar | off_caption | 0.0013578588 | <= 0.05 | +0.04864 | PASS |
| unipolar | neu_hold | 0.95855 | >= 0.85 | +0.1085 | PASS |
| ae_gan_hold | recon_mse | 0.0033893692 | <= 0.05 | +0.04661 | PASS |
| ae_gan_hold | hold | 0.0011173075 | <= 0.35 | +0.3489 | PASS |
| cover_leftover | u_kept | 0.99337279 | >= 0.85 | +0.1434 | PASS |
| cover_leftover | content_kept | 1.0064424 | >= 0.75 | +0.2564 | PASS |
| cover_leftover | leak_ratio | 0.00078730776 | <= 0.2 | +0.1992 | PASS |
| cover_leftover | pole_rel_err_plus | 0.0055896486 | <= 0.2 | +0.1944 | PASS |
| cover_leftover | pole_rel_err_minus | 0.0041341209 | <= 0.2 | +0.1959 | PASS |
| cover_leftover | same_dir | 0.0057463447 | <= 0.25 | +0.2443 | PASS |
| unused_token_hold | unused_hold | 0.99786677 | >= 0.85 | +0.1479 | PASS |
| unused_token_hold | concept_move | 0.90091382 | >= 0.85 | +0.05091 | PASS |
| mid_scale_identity | concept_cos_plus | 0.9999994 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_cos_minus | 0.99999952 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_mag_plus | 0.99964041 | >= 0.75 | +0.2496 | PASS |
| mid_scale_identity | concept_mag_plus | 0.99964041 | <= 1.25 | +0.2504 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0003929 | >= 0.75 | +0.2504 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0003929 | <= 1.25 | +0.2496 | PASS |
| mid_scale_identity | identity_at_0 | 0.9989606 | >= 0.85 | +0.149 | PASS |
| mid_scale_identity | identity_at_mid | 0.99965148 | >= 0.85 | +0.1497 | PASS |
| mode_hold | modes | 7 | >= 7 | +0 | PASS |
| mode_hold | hq | 0.75195312 | >= 0.9 | -0.148 | FAIL |

</details>

<details><summary>cap3_cosine_start0p6_lr1p0: FAIL</summary>

```json
{
  "name": "cap3_cosine_start0p6_lr1p0",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "b_cap",
  "reg_coeff": 3.0,
  "reg_kappa": 1.25,
  "particle_l2": 0.0,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 1.0,
  "lr_schedule": "cosine",
  "lr_anneal_start": 0.6,
  "lr_floor": 0.05
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.44000587 | >= 0.3 | +0.14 | PASS |
| two_pole | grad_med | 0.49025095 | <= 1.0 | +0.5097 | PASS |
| trajectory | identity_mse | 1.4446846e-05 | <= 0.02 | +0.01999 | PASS |
| residual_student | identity_mse | 1.6061667e-05 | <= 0.02 | +0.01998 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.96124028 | >= 0.85 | +0.1112 | PASS |
| unipolar | off_caption | 0.00018739885 | <= 0.05 | +0.04981 | PASS |
| unipolar | neu_hold | 0.93935586 | >= 0.85 | +0.08936 | PASS |
| ae_gan_hold | recon_mse | 0.005015953 | <= 0.05 | +0.04498 | PASS |
| ae_gan_hold | hold | 0.00066700461 | <= 0.35 | +0.3493 | PASS |
| cover_leftover | u_kept | 0.99597191 | >= 0.85 | +0.146 | PASS |
| cover_leftover | content_kept | 1.0028141 | >= 0.75 | +0.2528 | PASS |
| cover_leftover | leak_ratio | 0.0078636699 | <= 0.2 | +0.1921 | PASS |
| cover_leftover | pole_rel_err_plus | 0.0063037197 | <= 0.2 | +0.1937 | PASS |
| cover_leftover | pole_rel_err_minus | 0.0041076494 | <= 0.2 | +0.1959 | PASS |
| cover_leftover | same_dir | 0.004685835 | <= 0.25 | +0.2453 | PASS |
| unused_token_hold | unused_hold | 0.99746328 | >= 0.85 | +0.1475 | PASS |
| unused_token_hold | concept_move | 0.88312189 | >= 0.85 | +0.03312 | PASS |
| mid_scale_identity | concept_cos_plus | 0.99998879 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_cos_minus | 0.99999607 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_mag_plus | 0.99967206 | >= 0.75 | +0.2497 | PASS |
| mid_scale_identity | concept_mag_plus | 0.99967206 | <= 1.25 | +0.2503 | PASS |
| mid_scale_identity | concept_mag_minus | 0.99936366 | >= 0.75 | +0.2494 | PASS |
| mid_scale_identity | concept_mag_minus | 0.99936366 | <= 1.25 | +0.2506 | PASS |
| mid_scale_identity | identity_at_0 | 0.99617728 | >= 0.85 | +0.1462 | PASS |
| mid_scale_identity | identity_at_mid | 0.9986137 | >= 0.85 | +0.1486 | PASS |
| mode_hold | modes | 4 | >= 7 | -3 | FAIL |
| mode_hold | hq | 0.50952148 | >= 0.9 | -0.3905 | FAIL |

</details>

<details><summary>cap3_cosine_start0p8_lr1p0: FAIL</summary>

```json
{
  "name": "cap3_cosine_start0p8_lr1p0",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "b_cap",
  "reg_coeff": 3.0,
  "reg_kappa": 1.25,
  "particle_l2": 0.0,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 1.0,
  "lr_schedule": "cosine",
  "lr_anneal_start": 0.8,
  "lr_floor": 0.05
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.50979072 | >= 0.3 | +0.2098 | PASS |
| two_pole | grad_med | 0.55325258 | <= 1.0 | +0.4467 | PASS |
| trajectory | identity_mse | 2.0757165e-05 | <= 0.02 | +0.01998 | PASS |
| residual_student | identity_mse | 3.289762e-05 | <= 0.02 | +0.01997 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.94582991 | >= 0.85 | +0.09583 | PASS |
| unipolar | off_caption | 9.8796141e-05 | <= 0.05 | +0.0499 | PASS |
| unipolar | neu_hold | 0.93018427 | >= 0.85 | +0.08018 | PASS |
| ae_gan_hold | recon_mse | 0.0023045605 | <= 0.05 | +0.0477 | PASS |
| ae_gan_hold | hold | 0.0012908196 | <= 0.35 | +0.3487 | PASS |
| cover_leftover | u_kept | 0.99789887 | >= 0.85 | +0.1479 | PASS |
| cover_leftover | content_kept | 1.0131335 | >= 0.75 | +0.2631 | PASS |
| cover_leftover | leak_ratio | 0.0015148314 | <= 0.2 | +0.1985 | PASS |
| cover_leftover | pole_rel_err_plus | 0.0051716114 | <= 0.2 | +0.1948 | PASS |
| cover_leftover | pole_rel_err_minus | 0.0073328712 | <= 0.2 | +0.1927 | PASS |
| cover_leftover | same_dir | 0.0052864192 | <= 0.25 | +0.2447 | PASS |
| unused_token_hold | unused_hold | 0.99283371 | >= 0.85 | +0.1428 | PASS |
| unused_token_hold | concept_move | 0.9629927 | >= 0.85 | +0.113 | PASS |
| mid_scale_identity | concept_cos_plus | 1 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_cos_minus | 0.99999988 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0008641 | >= 0.75 | +0.2509 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0008641 | <= 1.25 | +0.2491 | PASS |
| mid_scale_identity | concept_mag_minus | 0.99898851 | >= 0.75 | +0.249 | PASS |
| mid_scale_identity | concept_mag_minus | 0.99898851 | <= 1.25 | +0.251 | PASS |
| mid_scale_identity | identity_at_0 | 0.99921799 | >= 0.85 | +0.1492 | PASS |
| mid_scale_identity | identity_at_mid | 0.99901858 | >= 0.85 | +0.149 | PASS |
| mode_hold | modes | 4 | >= 7 | -3 | FAIL |
| mode_hold | hq | 0.50512695 | >= 0.9 | -0.3949 | FAIL |

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
