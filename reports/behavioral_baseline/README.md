# Behavioral baseline — live weights

**Latest result: delayed cosine decay makes the cap candidate sustain all nine
behavioral toys, with 8/8 ring modes and 100% final HQ.**
Read the [convergence study](convergence/README.md) for the current comparison,
[resolved candidate](convergence/leading_config.json),
[full schedule leaderboard](convergence/schedule/README.md), sensitivity tests
and [actual 100-Gaussian results](convergence/grid/README.md).

| Formulation | Final live bounds | Sustained toys | Final ring modes / HQ |
| --- | ---: | ---: | --- |
| Original locked shared | 26/29 | 7/9 | 5/8 / 82.30% |
| R1+R2 0.1 | 29/29 | 8/9 | 7/8 / 100% |
| Cap 1.25, coeff 3, LR ×0.85, original schedules | 29/29 | 8/9 | 8/8 / 100% |
| **Same cap + cosine from 60%, floor 5%** | **29/29** | **9/9** | **8/8 / 100%** |

Sustained means at least five consecutive final passing observations in a
complete 24-point curve, with no later observed failure. It adds a full-coverage
ring target without weakening the original bounds. All ten shared behavioral
checks pass separately. EMA never rescues a live failure.

On the actual 100-Gaussian task, stock and the transferred candidate both sustain
100/100 modes from step 6,000, confirmed at 7,000. The candidate is sensitive to
nearby settings and data units. Keep stock production defaults; use
`get_recipe("gan_behavioral")` as an explicit candidate. The
[training helper](../../docs/api.md#gantrainer) applies either recipe consistently.

The [new PR #39 candidates](convergence/README.md#additional-candidates-from-pr-39)
were also rerun: none passes every bound here, with or without the shared cosine
schedule. The L2 .005 R1+R2 row reaches 8/8 modes at 100% HQ but fails trajectory.
The external runtime's different outcomes remain documented separately.

A [trained generic LR controller](../learned_lr/README.md) was also tested on
this full suite after fitting only on separate distributions. It reaches 27/29
bounds and seven sustained toys; cosine retains 29/29 and all nine. Its fitting
objective improves, but it does not replace the scheduled cap baseline.

## Historical final-step search

The following tables preserve the original protocol and its measurements;
the updated convergence results above supersede its default-selection ranking.

**Regression PASS: `bcap_k1p25_c3p0_lr0p85`, `b_cap_k1_25_c2_lr0_85`, `b_cap_k1_25_c2_0_no_l2`, `bcap_k1p25_c2p0_lr0p8`, `r1_r2_0_1`.**

Protocol `behavior-v1` · CPU · seed 0 · fixed host budgets · final live weights.

**Overall PASS requires all 29 numerical bounds on all 9 trained toys, plus the 10 shared behavioral/integration checks.** EMA is diagnostic and cannot rescue a live failure. Config equality checks are excluded. Shared checks are run once: they do not depend on the GAN config and do not contribute to its rank.

**Regression PASS is a minimum bar for default selection.** The original ring bar permits 7/8 modes. Actual coverage, HQ, balance and checkpoint stability are visible below; a final-step PASS does not establish a stable ParticleGAN default. The [default-selection analysis](default_selection.md) also compares the stock recipe on the ring host.

Rows rank by passed toys, then passed numerical bounds, then live ring coverage, HQ and effective modes. Missing/nonfinite results and errors cannot pass. Thresholds and budgets are frozen before config search.

| Rank | Config | Live toys | Live bounds | Ring modes | Ring HQ | Effective modes | Regression |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `bcap_k1p25_c3p0_lr0p85` (b_cap) | 9/9 | 29/29 | 8/8 | 100.00% | 7.54/8 | **PASS** |
| 2 | `b_cap_k1_25_c2_lr0_85` (b_cap) | 9/9 | 29/29 | 8/8 | 100.00% | 7.51/8 | **PASS** |
| 3 | `b_cap_k1_25_c2_0_no_l2` (b_cap) | 9/9 | 29/29 | 8/8 | 91.75% | 7.21/8 | **PASS** |
| 4 | `bcap_k1p25_c2p0_lr0p8` (b_cap) | 9/9 | 29/29 | 8/8 | 91.67% | 7.48/8 | **PASS** |
| 5 | `r1_r2_0_1` (a_r1r2) | 9/9 | 29/29 | 7/8 | 100.00% | 6.37/8 | **PASS** |
| 6 | `r1_r2_0_1_no_l2` (a_r1r2) | 8/9 | 28/29 | 8/8 | 100.00% | 7.61/8 | **FAIL** |
| 7 | `no_particle_l2` (b_cap) | 8/9 | 28/29 | 8/8 | 74.05% | 7.72/8 | **FAIL** |
| 8 | `b_cap_no_l2_coeff_5` (b_cap) | 8/9 | 27/29 | 4/8 | 48.12% | 3.80/8 | **FAIL** |
| 9 | `b_cap_no_l2_coeff_2` (b_cap) | 7/9 | 26/29 | 6/8 | 65.84% | 5.73/8 | **FAIL** |
| 10 | `locked_shared` (b_cap) | 7/9 | 26/29 | 5/8 | 82.30% | 4.73/8 | **FAIL** |
| 11 | `b_cap_no_l2_lr_half` (b_cap) | 6/9 | 26/29 | 7/8 | 92.33% | 6.71/8 | **FAIL** |
| 12 | `b_cap_no_l2_lr_quarter` (b_cap) | 2/9 | 17/29 | 5/8 | 75.29% | 4.58/8 | **FAIL** |

Effective modes measures balance among high-quality outputs; eight balanced modes gives 8. HQ measures quality of generated samples and does not penalize a missing target cluster. Thus 100% HQ can coexist with 7/8 coverage.

## Live stability and exact particle coverage

Final-step selection is unchanged. For default selection, report the five observations at steps 1,000, 1,050, 1,100, 1,150 and 1,200. These are sampled checkpoints, not a claim about every intervening step. Enumerating all 12 equally likely particles additionally distinguishes a missing mode from an unlucky 4,096-sample evaluation.

| Config | Worst tail modes | Worst tail HQ | Tail checks with 8/8 and HQ ≥90% | Final HQ particles by mode (0–7) |
| --- | ---: | ---: | ---: | --- |
| `bcap_k1p25_c3p0_lr0p85` | 8/8 | 83.91% | 4/5 | [1, 2, 1, 1, 2, 1, 2, 2] |
| `b_cap_k1_25_c2_lr0_85` | 3/8 | 41.89% | 3/5 | [2, 1, 1, 2, 1, 2, 2, 1] |
| `b_cap_k1_25_c2_0_no_l2` | 7/8 | 75.22% | 3/5 | [3, 2, 1, 1, 1, 1, 1, 1] |
| `bcap_k1p25_c2p0_lr0p8` | 3/8 | 34.40% | 2/5 | [2, 1, 1, 1, 2, 1, 1, 2] |
| `r1_r2_0_1` | 2/8 | 16.60% | 0/5 | [1, 0, 3, 1, 1, 2, 2, 2] |
| `r1_r2_0_1_no_l2` | 5/8 | 42.94% | 3/5 | [2, 1, 2, 1, 1, 2, 1, 2] |
| `no_particle_l2` | 6/8 | 57.76% | 0/5 | [2, 1, 1, 1, 1, 1, 1, 1] |
| `b_cap_no_l2_coeff_5` | 1/8 | 8.33% | 1/5 | [2, 1, 0, 0, 0, 0, 2, 1] |
| `b_cap_no_l2_coeff_2` | 3/8 | 42.65% | 0/5 | [1, 1, 2, 0, 0, 2, 1, 1] |
| `locked_shared` | 2/8 | 17.43% | 0/5 | [2, 1, 2, 2, 0, 0, 0, 3] |
| `b_cap_no_l2_lr_half` | 4/8 | 58.59% | 1/5 | [0, 2, 2, 1, 1, 2, 1, 2] |
| `b_cap_no_l2_lr_quarter` | 2/8 | 40.41% | 0/5 | [2, 3, 1, 2, 0, 0, 0, 1] |

## Stock-recipe ring comparison

Both rows use the same 20,000 particles, 7,000 steps, optimizer and cosine schedule; only the penalty changes. They are compared with each other and are not included in the 12-particle regression rank. This is the ring host, not the 100-Gaussian benchmark. Tail observations cover the final 200 steps at 50-step intervals.

| Penalty | Live modes | Live HQ | Effective modes | EMA modes / HQ | Worst tail live HQ | Full-coverage/HQ tail checks |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| b_cap coeff 1.0 | 8/8 | 99.05% | 7.95/8 | 8/8 / 99.66% | 98.51% | 5/5 |
| a_r1r2 coeff 0.1 | 8/8 | 99.32% | 7.94/8 | 8/8 / 99.44% | 99.24% | 5/5 |

Both stock-recipe penalties sustain full coverage at the measured late checkpoints. The current `b_cap` recipe and its EMA pass; the small-host R1+R2 result does not justify replacing the default or disabling EMA. Raw evidence and its original source fingerprint are in [stock_ring.json](stock_ring.json).


## Live toy matrix

| Config | two_pole | trajectory | residual_student | unipolar | ae_gan_hold | cover_leftover | unused_token_hold | mid_scale_identity | mode_hold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bcap_k1p25_c3p0_lr0p85` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `b_cap_k1_25_c2_lr0_85` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `b_cap_k1_25_c2_0_no_l2` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `bcap_k1p25_c2p0_lr0p8` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `r1_r2_0_1` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `r1_r2_0_1_no_l2` | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `no_particle_l2` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | FAIL |
| `b_cap_no_l2_coeff_5` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | FAIL |
| `b_cap_no_l2_coeff_2` | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | FAIL |
| `locked_shared` | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | FAIL |
| `b_cap_no_l2_lr_half` | FAIL | FAIL | PASS | PASS | PASS | FAIL | PASS | PASS | PASS |
| `b_cap_no_l2_lr_quarter` | FAIL | FAIL | PASS | FAIL | PASS | FAIL | FAIL | FAIL | FAIL |

## EMA diagnostics

Only the ring and cover/leftover hosts maintain EMA. These results are separate from the live ranking.

| Config | Ring modes | Ring HQ | Ring bounds | Cover/leftover bounds |
| --- | ---: | ---: | --- | --- |
| `bcap_k1p25_c3p0_lr0p85` | 8/8 | 100.00% | 2/2 | 6/6 |
| `b_cap_k1_25_c2_lr0_85` | 8/8 | 100.00% | 2/2 | 6/6 |
| `b_cap_k1_25_c2_0_no_l2` | 8/8 | 100.00% | 2/2 | 6/6 |
| `bcap_k1p25_c2p0_lr0p8` | 8/8 | 100.00% | 2/2 | 6/6 |
| `r1_r2_0_1` | 5/8 | 65.16% | 0/2 | 6/6 |
| `r1_r2_0_1_no_l2` | 7/8 | 83.33% | 1/2 | 6/6 |
| `no_particle_l2` | 8/8 | 100.00% | 2/2 | 6/6 |
| `b_cap_no_l2_coeff_5` | 8/8 | 91.38% | 2/2 | 6/6 |
| `b_cap_no_l2_coeff_2` | 8/8 | 100.00% | 2/2 | 6/6 |
| `locked_shared` | 3/8 | 49.49% | 0/2 | 6/6 |
| `b_cap_no_l2_lr_half` | 7/8 | 83.91% | 1/2 | 5/6 |
| `b_cap_no_l2_lr_quarter` | 3/8 | 49.32% | 0/2 | 2/6 |

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
  "lr_multiplier": 0.85
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

<details><summary>b_cap_k1_25_c2_lr0_85: PASS</summary>

```json
{
  "name": "b_cap_k1_25_c2_lr0_85",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "b_cap",
  "reg_coeff": 2.0,
  "reg_kappa": 1.25,
  "particle_l2": 0.0,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 0.85
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.46855497 | >= 0.3 | +0.1686 | PASS |
| two_pole | grad_med | 0.5033989 | <= 1.0 | +0.4966 | PASS |
| trajectory | identity_mse | 0.0021119271 | <= 0.02 | +0.01789 | PASS |
| residual_student | identity_mse | 0.0039597214 | <= 0.02 | +0.01604 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.89915878 | >= 0.85 | +0.04916 | PASS |
| unipolar | off_caption | 4.3612086e-06 | <= 0.05 | +0.05 | PASS |
| unipolar | neu_hold | 0.97800527 | >= 0.85 | +0.128 | PASS |
| ae_gan_hold | recon_mse | 0.0055781878 | <= 0.05 | +0.04442 | PASS |
| ae_gan_hold | hold | 0.014450962 | <= 0.35 | +0.3355 | PASS |
| cover_leftover | u_kept | 0.97437864 | >= 0.85 | +0.1244 | PASS |
| cover_leftover | content_kept | 1.0162716 | >= 0.75 | +0.2663 | PASS |
| cover_leftover | leak_ratio | 0.00062306155 | <= 0.2 | +0.1994 | PASS |
| cover_leftover | pole_rel_err_plus | 0.017903289 | <= 0.2 | +0.1821 | PASS |
| cover_leftover | pole_rel_err_minus | 0.019645127 | <= 0.2 | +0.1804 | PASS |
| cover_leftover | same_dir | 0.0056517286 | <= 0.25 | +0.2443 | PASS |
| unused_token_hold | unused_hold | 0.9951797 | >= 0.85 | +0.1452 | PASS |
| unused_token_hold | concept_move | 0.91561018 | >= 0.85 | +0.06561 | PASS |
| mid_scale_identity | concept_cos_plus | 0.99999088 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_cos_minus | 0.99999118 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0222633 | >= 0.75 | +0.2723 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0222633 | <= 1.25 | +0.2277 | PASS |
| mid_scale_identity | concept_mag_minus | 0.99590272 | >= 0.75 | +0.2459 | PASS |
| mid_scale_identity | concept_mag_minus | 0.99590272 | <= 1.25 | +0.2541 | PASS |
| mid_scale_identity | identity_at_0 | 0.99481656 | >= 0.85 | +0.1448 | PASS |
| mid_scale_identity | identity_at_mid | 0.98868487 | >= 0.85 | +0.1387 | PASS |
| mode_hold | modes | 8 | >= 7 | +1 | PASS |
| mode_hold | hq | 1 | >= 0.9 | +0.1 | PASS |

</details>

<details><summary>b_cap_k1_25_c2_0_no_l2: PASS</summary>

```json
{
  "name": "b_cap_k1_25_c2_0_no_l2",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "b_cap",
  "reg_coeff": 2.0,
  "reg_kappa": 1.25,
  "particle_l2": 0.0,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 1.0
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.57043242 | >= 0.3 | +0.2704 | PASS |
| two_pole | grad_med | 0.58388507 | <= 1.0 | +0.4161 | PASS |
| trajectory | identity_mse | 0.001477993 | <= 0.02 | +0.01852 | PASS |
| residual_student | identity_mse | 0.0021327541 | <= 0.02 | +0.01787 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.97132541 | >= 0.85 | +0.1213 | PASS |
| unipolar | off_caption | 2.5244235e-05 | <= 0.05 | +0.04997 | PASS |
| unipolar | neu_hold | 0.95682654 | >= 0.85 | +0.1068 | PASS |
| ae_gan_hold | recon_mse | 0.0031173849 | <= 0.05 | +0.04688 | PASS |
| ae_gan_hold | hold | 0.0044206111 | <= 0.35 | +0.3456 | PASS |
| cover_leftover | u_kept | 0.98192518 | >= 0.85 | +0.1319 | PASS |
| cover_leftover | content_kept | 1.0032236 | >= 0.75 | +0.2532 | PASS |
| cover_leftover | leak_ratio | 0.0013062411 | <= 0.2 | +0.1987 | PASS |
| cover_leftover | pole_rel_err_plus | 0.012282635 | <= 0.2 | +0.1877 | PASS |
| cover_leftover | pole_rel_err_minus | 0.00768692 | <= 0.2 | +0.1923 | PASS |
| cover_leftover | same_dir | 0.00506778 | <= 0.25 | +0.2449 | PASS |
| unused_token_hold | unused_hold | 0.99569036 | >= 0.85 | +0.1457 | PASS |
| unused_token_hold | concept_move | 0.89507558 | >= 0.85 | +0.04508 | PASS |
| mid_scale_identity | concept_cos_plus | 0.99992085 | >= 0.85 | +0.1499 | PASS |
| mid_scale_identity | concept_cos_minus | 0.99989218 | >= 0.85 | +0.1499 | PASS |
| mid_scale_identity | concept_mag_plus | 0.99775535 | >= 0.75 | +0.2478 | PASS |
| mid_scale_identity | concept_mag_plus | 0.99775535 | <= 1.25 | +0.2522 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0007049 | >= 0.75 | +0.2507 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0007049 | <= 1.25 | +0.2493 | PASS |
| mid_scale_identity | identity_at_0 | 0.98274424 | >= 0.85 | +0.1327 | PASS |
| mid_scale_identity | identity_at_mid | 0.98470469 | >= 0.85 | +0.1347 | PASS |
| mode_hold | modes | 8 | >= 7 | +1 | PASS |
| mode_hold | hq | 0.91748047 | >= 0.9 | +0.01748 | PASS |

</details>

<details><summary>bcap_k1p25_c2p0_lr0p8: PASS</summary>

```json
{
  "name": "bcap_k1p25_c2p0_lr0p8",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "b_cap",
  "reg_coeff": 2.0,
  "reg_kappa": 1.25,
  "particle_l2": 0.0,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 0.8
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.4333936 | >= 0.3 | +0.1334 | PASS |
| two_pole | grad_med | 0.47120562 | <= 1.0 | +0.5288 | PASS |
| trajectory | identity_mse | 0.0015909426 | <= 0.02 | +0.01841 | PASS |
| residual_student | identity_mse | 0.0011406759 | <= 0.02 | +0.01886 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.92676975 | >= 0.85 | +0.07677 | PASS |
| unipolar | off_caption | 0.00016951858 | <= 0.05 | +0.04983 | PASS |
| unipolar | neu_hold | 0.9626383 | >= 0.85 | +0.1126 | PASS |
| ae_gan_hold | recon_mse | 0.021579783 | <= 0.05 | +0.02842 | PASS |
| ae_gan_hold | hold | 0.014317389 | <= 0.35 | +0.3357 | PASS |
| cover_leftover | u_kept | 0.956896 | >= 0.85 | +0.1069 | PASS |
| cover_leftover | content_kept | 1.0134042 | >= 0.75 | +0.2634 | PASS |
| cover_leftover | leak_ratio | 0.0017273081 | <= 0.2 | +0.1983 | PASS |
| cover_leftover | pole_rel_err_plus | 0.028861713 | <= 0.2 | +0.1711 | PASS |
| cover_leftover | pole_rel_err_minus | 0.030062083 | <= 0.2 | +0.1699 | PASS |
| cover_leftover | same_dir | 0.0045240104 | <= 0.25 | +0.2455 | PASS |
| unused_token_hold | unused_hold | 0.99920878 | >= 0.85 | +0.1492 | PASS |
| unused_token_hold | concept_move | 0.88934269 | >= 0.85 | +0.03934 | PASS |
| mid_scale_identity | concept_cos_plus | 0.99986529 | >= 0.85 | +0.1499 | PASS |
| mid_scale_identity | concept_cos_minus | 0.99994218 | >= 0.85 | +0.1499 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0265924 | >= 0.75 | +0.2766 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0265924 | <= 1.25 | +0.2234 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0023004 | >= 0.75 | +0.2523 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0023004 | <= 1.25 | +0.2477 | PASS |
| mid_scale_identity | identity_at_0 | 0.9923136 | >= 0.85 | +0.1423 | PASS |
| mid_scale_identity | identity_at_mid | 0.99078122 | >= 0.85 | +0.1408 | PASS |
| mode_hold | modes | 8 | >= 7 | +1 | PASS |
| mode_hold | hq | 0.91674805 | >= 0.9 | +0.01675 | PASS |

</details>

<details><summary>r1_r2_0_1: PASS</summary>

```json
{
  "name": "r1_r2_0_1",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "a_r1r2",
  "reg_coeff": 0.1,
  "reg_kappa": 1.0,
  "particle_l2": 0.02,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 1.0
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.45241687 | >= 0.3 | +0.1524 | PASS |
| two_pole | grad_med | 0.73421687 | <= 1.0 | +0.2658 | PASS |
| trajectory | identity_mse | 0.0037677779 | <= 0.02 | +0.01623 | PASS |
| residual_student | identity_mse | 0.0018429351 | <= 0.02 | +0.01816 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.99981421 | >= 0.85 | +0.1498 | PASS |
| unipolar | off_caption | 2.3555513e-10 | <= 0.05 | +0.05 | PASS |
| unipolar | neu_hold | 0.99962804 | >= 0.85 | +0.1496 | PASS |
| ae_gan_hold | recon_mse | 0.007062762 | <= 0.05 | +0.04294 | PASS |
| ae_gan_hold | hold | 0.013251475 | <= 0.35 | +0.3367 | PASS |
| cover_leftover | u_kept | 0.99400919 | >= 0.85 | +0.144 | PASS |
| cover_leftover | content_kept | 1.0038434 | >= 0.75 | +0.2538 | PASS |
| cover_leftover | leak_ratio | 0.001957112 | <= 0.2 | +0.198 | PASS |
| cover_leftover | pole_rel_err_plus | 0.0043786368 | <= 0.2 | +0.1956 | PASS |
| cover_leftover | pole_rel_err_minus | 0.0031655058 | <= 0.2 | +0.1968 | PASS |
| cover_leftover | same_dir | 0.0015631897 | <= 0.25 | +0.2484 | PASS |
| unused_token_hold | unused_hold | 0.99430081 | >= 0.85 | +0.1443 | PASS |
| unused_token_hold | concept_move | 0.99949992 | >= 0.85 | +0.1495 | PASS |
| mid_scale_identity | concept_cos_plus | 1 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_cos_minus | 1 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0000033 | >= 0.75 | +0.25 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0000033 | <= 1.25 | +0.25 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0000005 | >= 0.75 | +0.25 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0000005 | <= 1.25 | +0.25 | PASS |
| mid_scale_identity | identity_at_0 | 0.99999848 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | identity_at_mid | 0.99998602 | >= 0.85 | +0.15 | PASS |
| mode_hold | modes | 7 | >= 7 | +0 | PASS |
| mode_hold | hq | 1 | >= 0.9 | +0.1 | PASS |

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
| two_pole | mean_abs | 0.45210361 | >= 0.3 | +0.1521 | PASS |
| two_pole | grad_med | 0.73248017 | <= 1.0 | +0.2675 | PASS |
| trajectory | identity_mse | 0.25167292 | <= 0.02 | -0.2317 | FAIL |
| residual_student | identity_mse | 0.0025260719 | <= 0.02 | +0.01747 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.99981421 | >= 0.85 | +0.1498 | PASS |
| unipolar | off_caption | 2.3555513e-10 | <= 0.05 | +0.05 | PASS |
| unipolar | neu_hold | 0.99962804 | >= 0.85 | +0.1496 | PASS |
| ae_gan_hold | recon_mse | 0.0035263118 | <= 0.05 | +0.04647 | PASS |
| ae_gan_hold | hold | 0.0040334305 | <= 0.35 | +0.346 | PASS |
| cover_leftover | u_kept | 0.99399017 | >= 0.85 | +0.144 | PASS |
| cover_leftover | content_kept | 1.0043715 | >= 0.75 | +0.2544 | PASS |
| cover_leftover | leak_ratio | 0.0019741945 | <= 0.2 | +0.198 | PASS |
| cover_leftover | pole_rel_err_plus | 0.0044583483 | <= 0.2 | +0.1955 | PASS |
| cover_leftover | pole_rel_err_minus | 0.0030616836 | <= 0.2 | +0.1969 | PASS |
| cover_leftover | same_dir | 0.0018064787 | <= 0.25 | +0.2482 | PASS |
| unused_token_hold | unused_hold | 0.99430081 | >= 0.85 | +0.1443 | PASS |
| unused_token_hold | concept_move | 0.99949992 | >= 0.85 | +0.1495 | PASS |
| mid_scale_identity | concept_cos_plus | 1 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_cos_minus | 1 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0000033 | >= 0.75 | +0.25 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0000033 | <= 1.25 | +0.25 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0000005 | >= 0.75 | +0.25 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0000005 | <= 1.25 | +0.25 | PASS |
| mid_scale_identity | identity_at_0 | 0.99999848 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | identity_at_mid | 0.99998602 | >= 0.85 | +0.15 | PASS |
| mode_hold | modes | 8 | >= 7 | +1 | PASS |
| mode_hold | hq | 1 | >= 0.9 | +0.1 | PASS |

</details>

<details><summary>no_particle_l2: FAIL</summary>

```json
{
  "name": "no_particle_l2",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "b_cap",
  "reg_coeff": 1.0,
  "reg_kappa": 1.0,
  "particle_l2": 0.0,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 1.0
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.52926749 | >= 0.3 | +0.2293 | PASS |
| two_pole | grad_med | 0.41900352 | <= 1.0 | +0.581 | PASS |
| trajectory | identity_mse | 0.0028351143 | <= 0.02 | +0.01716 | PASS |
| residual_student | identity_mse | 0.0014887018 | <= 0.02 | +0.01851 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.96062029 | >= 0.85 | +0.1106 | PASS |
| unipolar | off_caption | 3.8675364e-06 | <= 0.05 | +0.05 | PASS |
| unipolar | neu_hold | 0.96343626 | >= 0.85 | +0.1134 | PASS |
| ae_gan_hold | recon_mse | 0.0030494039 | <= 0.05 | +0.04695 | PASS |
| ae_gan_hold | hold | 0.0013684166 | <= 0.35 | +0.3486 | PASS |
| cover_leftover | u_kept | 0.9797536 | >= 0.85 | +0.1298 | PASS |
| cover_leftover | content_kept | 0.99335745 | >= 0.75 | +0.2434 | PASS |
| cover_leftover | leak_ratio | 0.0021324642 | <= 0.2 | +0.1979 | PASS |
| cover_leftover | pole_rel_err_plus | 0.013850644 | <= 0.2 | +0.1861 | PASS |
| cover_leftover | pole_rel_err_minus | 0.010945952 | <= 0.2 | +0.1891 | PASS |
| cover_leftover | same_dir | 0.0024600484 | <= 0.25 | +0.2475 | PASS |
| unused_token_hold | unused_hold | 0.99144782 | >= 0.85 | +0.1414 | PASS |
| unused_token_hold | concept_move | 0.96768949 | >= 0.85 | +0.1177 | PASS |
| mid_scale_identity | concept_cos_plus | 0.9999643 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_cos_minus | 0.99999696 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0213077 | >= 0.75 | +0.2713 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0213077 | <= 1.25 | +0.2287 | PASS |
| mid_scale_identity | concept_mag_minus | 0.97101057 | >= 0.75 | +0.221 | PASS |
| mid_scale_identity | concept_mag_minus | 0.97101057 | <= 1.25 | +0.279 | PASS |
| mid_scale_identity | identity_at_0 | 0.99663862 | >= 0.85 | +0.1466 | PASS |
| mid_scale_identity | identity_at_mid | 0.99949249 | >= 0.85 | +0.1495 | PASS |
| mode_hold | modes | 8 | >= 7 | +1 | PASS |
| mode_hold | hq | 0.74047852 | >= 0.9 | -0.1595 | FAIL |

</details>

<details><summary>b_cap_no_l2_coeff_5: FAIL</summary>

```json
{
  "name": "b_cap_no_l2_coeff_5",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "b_cap",
  "reg_coeff": 5.0,
  "reg_kappa": 1.0,
  "particle_l2": 0.0,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 1.0
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.52802056 | >= 0.3 | +0.228 | PASS |
| two_pole | grad_med | 0.45490339 | <= 1.0 | +0.5451 | PASS |
| trajectory | identity_mse | 0.0024610851 | <= 0.02 | +0.01754 | PASS |
| residual_student | identity_mse | 0.0014070906 | <= 0.02 | +0.01859 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.90922668 | >= 0.85 | +0.05923 | PASS |
| unipolar | off_caption | 3.6856582e-06 | <= 0.05 | +0.05 | PASS |
| unipolar | neu_hold | 0.94606873 | >= 0.85 | +0.09607 | PASS |
| ae_gan_hold | recon_mse | 0.0025178646 | <= 0.05 | +0.04748 | PASS |
| ae_gan_hold | hold | 0.0041451715 | <= 0.35 | +0.3459 | PASS |
| cover_leftover | u_kept | 0.98793309 | >= 0.85 | +0.1379 | PASS |
| cover_leftover | content_kept | 1.0055135 | >= 0.75 | +0.2555 | PASS |
| cover_leftover | leak_ratio | 0.0058985651 | <= 0.2 | +0.1941 | PASS |
| cover_leftover | pole_rel_err_plus | 0.0098108826 | <= 0.2 | +0.1902 | PASS |
| cover_leftover | pole_rel_err_minus | 0.0087727215 | <= 0.2 | +0.1912 | PASS |
| cover_leftover | same_dir | 0.0033299449 | <= 0.25 | +0.2467 | PASS |
| unused_token_hold | unused_hold | 0.99809678 | >= 0.85 | +0.1481 | PASS |
| unused_token_hold | concept_move | 0.92488609 | >= 0.85 | +0.07489 | PASS |
| mid_scale_identity | concept_cos_plus | 0.99991286 | >= 0.85 | +0.1499 | PASS |
| mid_scale_identity | concept_cos_minus | 0.99993062 | >= 0.85 | +0.1499 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0133743 | >= 0.75 | +0.2634 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0133743 | <= 1.25 | +0.2366 | PASS |
| mid_scale_identity | concept_mag_minus | 0.9883914 | >= 0.75 | +0.2384 | PASS |
| mid_scale_identity | concept_mag_minus | 0.9883914 | <= 1.25 | +0.2616 | PASS |
| mid_scale_identity | identity_at_0 | 0.9933349 | >= 0.85 | +0.1433 | PASS |
| mid_scale_identity | identity_at_mid | 0.9990331 | >= 0.85 | +0.149 | PASS |
| mode_hold | modes | 4 | >= 7 | -3 | FAIL |
| mode_hold | hq | 0.48120117 | >= 0.9 | -0.4188 | FAIL |

</details>

<details><summary>b_cap_no_l2_coeff_2: FAIL</summary>

```json
{
  "name": "b_cap_no_l2_coeff_2",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "b_cap",
  "reg_coeff": 2.0,
  "reg_kappa": 1.0,
  "particle_l2": 0.0,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 1.0
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.53122997 | >= 0.3 | +0.2312 | PASS |
| two_pole | grad_med | 0.42677635 | <= 1.0 | +0.5732 | PASS |
| trajectory | identity_mse | 0.022777891 | <= 0.02 | -0.002778 | FAIL |
| residual_student | identity_mse | 0.0048443354 | <= 0.02 | +0.01516 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.99601719 | >= 0.85 | +0.146 | PASS |
| unipolar | off_caption | 0.00030724931 | <= 0.05 | +0.04969 | PASS |
| unipolar | neu_hold | 0.96036072 | >= 0.85 | +0.1104 | PASS |
| ae_gan_hold | recon_mse | 0.0058257319 | <= 0.05 | +0.04417 | PASS |
| ae_gan_hold | hold | 0.017145395 | <= 0.35 | +0.3329 | PASS |
| cover_leftover | u_kept | 0.98192017 | >= 0.85 | +0.1319 | PASS |
| cover_leftover | content_kept | 0.99649384 | >= 0.75 | +0.2465 | PASS |
| cover_leftover | leak_ratio | 0.0055331056 | <= 0.2 | +0.1945 | PASS |
| cover_leftover | pole_rel_err_plus | 0.012918538 | <= 0.2 | +0.1871 | PASS |
| cover_leftover | pole_rel_err_minus | 0.0095484238 | <= 0.2 | +0.1905 | PASS |
| cover_leftover | same_dir | 0.0036500148 | <= 0.25 | +0.2463 | PASS |
| unused_token_hold | unused_hold | 0.99467418 | >= 0.85 | +0.1447 | PASS |
| unused_token_hold | concept_move | 0.91720127 | >= 0.85 | +0.0672 | PASS |
| mid_scale_identity | concept_cos_plus | 0.99992043 | >= 0.85 | +0.1499 | PASS |
| mid_scale_identity | concept_cos_minus | 0.99998301 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_mag_plus | 0.99564666 | >= 0.75 | +0.2456 | PASS |
| mid_scale_identity | concept_mag_plus | 0.99564666 | <= 1.25 | +0.2544 | PASS |
| mid_scale_identity | concept_mag_minus | 1.009899 | >= 0.75 | +0.2599 | PASS |
| mid_scale_identity | concept_mag_minus | 1.009899 | <= 1.25 | +0.2401 | PASS |
| mid_scale_identity | identity_at_0 | 0.99459483 | >= 0.85 | +0.1446 | PASS |
| mid_scale_identity | identity_at_mid | 0.98553742 | >= 0.85 | +0.1355 | PASS |
| mode_hold | modes | 6 | >= 7 | -1 | FAIL |
| mode_hold | hq | 0.65844727 | >= 0.9 | -0.2416 | FAIL |

</details>

<details><summary>locked_shared: FAIL</summary>

```json
{
  "name": "locked_shared",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "b_cap",
  "reg_coeff": 1.0,
  "reg_kappa": 1.0,
  "particle_l2": 0.02,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 1.0
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.51437211 | >= 0.3 | +0.2144 | PASS |
| two_pole | grad_med | 0.41965193 | <= 1.0 | +0.5803 | PASS |
| trajectory | identity_mse | 0.23389013 | <= 0.02 | -0.2139 | FAIL |
| residual_student | identity_mse | 0.0021875103 | <= 0.02 | +0.01781 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.96062029 | >= 0.85 | +0.1106 | PASS |
| unipolar | off_caption | 3.8675364e-06 | <= 0.05 | +0.05 | PASS |
| unipolar | neu_hold | 0.96343626 | >= 0.85 | +0.1134 | PASS |
| ae_gan_hold | recon_mse | 0.0073521044 | <= 0.05 | +0.04265 | PASS |
| ae_gan_hold | hold | 0.018749481 | <= 0.35 | +0.3313 | PASS |
| cover_leftover | u_kept | 0.98943173 | >= 0.85 | +0.1394 | PASS |
| cover_leftover | content_kept | 0.99730197 | >= 0.75 | +0.2473 | PASS |
| cover_leftover | leak_ratio | 0.0032522324 | <= 0.2 | +0.1967 | PASS |
| cover_leftover | pole_rel_err_plus | 0.0075671989 | <= 0.2 | +0.1924 | PASS |
| cover_leftover | pole_rel_err_minus | 0.010743973 | <= 0.2 | +0.1893 | PASS |
| cover_leftover | same_dir | 0.0025605498 | <= 0.25 | +0.2474 | PASS |
| unused_token_hold | unused_hold | 0.99144782 | >= 0.85 | +0.1414 | PASS |
| unused_token_hold | concept_move | 0.96768949 | >= 0.85 | +0.1177 | PASS |
| mid_scale_identity | concept_cos_plus | 0.9999643 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_cos_minus | 0.99999696 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0213077 | >= 0.75 | +0.2713 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0213077 | <= 1.25 | +0.2287 | PASS |
| mid_scale_identity | concept_mag_minus | 0.97101057 | >= 0.75 | +0.221 | PASS |
| mid_scale_identity | concept_mag_minus | 0.97101057 | <= 1.25 | +0.279 | PASS |
| mid_scale_identity | identity_at_0 | 0.99663862 | >= 0.85 | +0.1466 | PASS |
| mid_scale_identity | identity_at_mid | 0.99949249 | >= 0.85 | +0.1495 | PASS |
| mode_hold | modes | 5 | >= 7 | -2 | FAIL |
| mode_hold | hq | 0.82299805 | >= 0.9 | -0.077 | FAIL |

</details>

<details><summary>b_cap_no_l2_lr_half: FAIL</summary>

```json
{
  "name": "b_cap_no_l2_lr_half",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "b_cap",
  "reg_coeff": 1.0,
  "reg_kappa": 1.0,
  "particle_l2": 0.0,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 0.5
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.19244264 | >= 0.3 | -0.1076 | FAIL |
| two_pole | grad_med | 0.15308526 | <= 1.0 | +0.8469 | PASS |
| trajectory | identity_mse | 0.25957972 | <= 0.02 | -0.2396 | FAIL |
| residual_student | identity_mse | 0.00079482124 | <= 0.02 | +0.01921 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.98442579 | >= 0.85 | +0.1344 | PASS |
| unipolar | off_caption | 7.6192119e-12 | <= 0.05 | +0.05 | PASS |
| unipolar | neu_hold | 0.98129234 | >= 0.85 | +0.1313 | PASS |
| ae_gan_hold | recon_mse | 0.0070036817 | <= 0.05 | +0.043 | PASS |
| ae_gan_hold | hold | 0.0011552859 | <= 0.35 | +0.3488 | PASS |
| cover_leftover | u_kept | 0.79516863 | >= 0.85 | -0.05483 | FAIL |
| cover_leftover | content_kept | 0.98580348 | >= 0.75 | +0.2358 | PASS |
| cover_leftover | leak_ratio | 0.0011033958 | <= 0.2 | +0.1989 | PASS |
| cover_leftover | pole_rel_err_plus | 0.1350904 | <= 0.2 | +0.06491 | PASS |
| cover_leftover | pole_rel_err_minus | 0.14488973 | <= 0.2 | +0.05511 | PASS |
| cover_leftover | same_dir | 0.0096827246 | <= 0.25 | +0.2403 | PASS |
| unused_token_hold | unused_hold | 0.9929916 | >= 0.85 | +0.143 | PASS |
| unused_token_hold | concept_move | 0.95017514 | >= 0.85 | +0.1002 | PASS |
| mid_scale_identity | concept_cos_plus | 0.99857926 | >= 0.85 | +0.1486 | PASS |
| mid_scale_identity | concept_cos_minus | 0.99982661 | >= 0.85 | +0.1498 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0379159 | >= 0.75 | +0.2879 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0379159 | <= 1.25 | +0.2121 | PASS |
| mid_scale_identity | concept_mag_minus | 0.99410224 | >= 0.75 | +0.2441 | PASS |
| mid_scale_identity | concept_mag_minus | 0.99410224 | <= 1.25 | +0.2559 | PASS |
| mid_scale_identity | identity_at_0 | 0.99867873 | >= 0.85 | +0.1487 | PASS |
| mid_scale_identity | identity_at_mid | 0.97305924 | >= 0.85 | +0.1231 | PASS |
| mode_hold | modes | 7 | >= 7 | +0 | PASS |
| mode_hold | hq | 0.92333984 | >= 0.9 | +0.02334 | PASS |

</details>

<details><summary>b_cap_no_l2_lr_quarter: FAIL</summary>

```json
{
  "name": "b_cap_no_l2_lr_quarter",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "b_cap",
  "reg_coeff": 1.0,
  "reg_kappa": 1.0,
  "particle_l2": 0.0,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 0.25
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.085981376 | >= 0.3 | -0.214 | FAIL |
| two_pole | grad_med | 0.073521733 | <= 1.0 | +0.9265 | PASS |
| trajectory | identity_mse | 0.21579856 | <= 0.02 | -0.1958 | FAIL |
| residual_student | identity_mse | 0.0003418063 | <= 0.02 | +0.01966 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.85161501 | >= 0.85 | +0.001615 | PASS |
| unipolar | off_caption | 1.4334314e-09 | <= 0.05 | +0.05 | PASS |
| unipolar | neu_hold | 0.82733293 | >= 0.85 | -0.02267 | FAIL |
| ae_gan_hold | recon_mse | 0.02370568 | <= 0.05 | +0.02629 | PASS |
| ae_gan_hold | hold | 0.0033645702 | <= 0.35 | +0.3466 | PASS |
| cover_leftover | u_kept | 0.56019127 | >= 0.85 | -0.2898 | FAIL |
| cover_leftover | content_kept | 0.79840946 | >= 0.75 | +0.04841 | PASS |
| cover_leftover | leak_ratio | 0.004276606 | <= 0.2 | +0.1957 | PASS |
| cover_leftover | pole_rel_err_plus | 0.29893193 | <= 0.2 | -0.09893 | FAIL |
| cover_leftover | pole_rel_err_minus | 0.32240656 | <= 0.2 | -0.1224 | FAIL |
| cover_leftover | same_dir | 0.025182223 | <= 0.25 | +0.2248 | PASS |
| unused_token_hold | unused_hold | 0.99631054 | >= 0.85 | +0.1463 | PASS |
| unused_token_hold | concept_move | 0.57893144 | >= 0.85 | -0.2711 | FAIL |
| mid_scale_identity | concept_cos_plus | 0.9912864 | >= 0.85 | +0.1413 | PASS |
| mid_scale_identity | concept_cos_minus | 0.96317011 | >= 0.85 | +0.1132 | PASS |
| mid_scale_identity | concept_mag_plus | 0.6172393 | >= 0.75 | -0.1328 | FAIL |
| mid_scale_identity | concept_mag_plus | 0.6172393 | <= 1.25 | +0.6328 | PASS |
| mid_scale_identity | concept_mag_minus | 0.6477735 | >= 0.75 | -0.1022 | FAIL |
| mid_scale_identity | concept_mag_minus | 0.6477735 | <= 1.25 | +0.6022 | PASS |
| mid_scale_identity | identity_at_0 | 0.73561207 | >= 0.85 | -0.1144 | FAIL |
| mid_scale_identity | identity_at_mid | 0.93794788 | >= 0.85 | +0.08795 | PASS |
| mode_hold | modes | 5 | >= 7 | -2 | FAIL |
| mode_hold | hq | 0.75292969 | >= 0.9 | -0.1471 | FAIL |

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
