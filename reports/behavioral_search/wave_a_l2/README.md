# Behavioral baseline — live weights

**Passing live baseline: `r1_r2_0_1_l2_004`, `r1_r2_0_1_l2_005`, `r1_r2_0_1_l2_007`.**

Protocol `behavior-v1` · CPU · seed 0 · fixed host budgets · final live weights.

**Overall PASS requires all 29 numerical bounds on all 9 trained toys, plus the 10 shared behavioral/integration checks.** EMA is diagnostic and cannot rescue a live failure. Config equality checks are excluded. Shared checks are run once: they do not depend on the GAN config and do not contribute to its rank.

Rows rank by passed toys, then passed numerical bounds; equal counts tie. Missing/nonfinite results and errors cannot pass. Thresholds and budgets are frozen before config search.

| Rank | Config | Live toys | Live bounds | Overall | Failed live targets |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | `r1_r2_0_1_l2_004` | 9/9 | 29/29 | **PASS** | None |
| 1 | `r1_r2_0_1_l2_005` | 9/9 | 29/29 | **PASS** | None |
| 1 | `r1_r2_0_1_l2_007` | 9/9 | 29/29 | **PASS** | None |
| 4 | `r1_r2_0_1_l2_006` | 8/9 | 28/29 | **FAIL** | mode_hold |
| 5 | `r1_r2_0_1_l2_003` | 8/9 | 27/29 | **FAIL** | mode_hold |

## Live toy matrix

| Config | two_pole | trajectory | residual_student | unipolar | ae_gan_hold | cover_leftover | unused_token_hold | mid_scale_identity | mode_hold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `r1_r2_0_1_l2_004` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `r1_r2_0_1_l2_005` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `r1_r2_0_1_l2_007` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `r1_r2_0_1_l2_006` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | FAIL |
| `r1_r2_0_1_l2_003` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | FAIL |

## EMA diagnostics

Only the ring and cover/leftover hosts maintain EMA. These results are separate from the live ranking.

| Config | Ring modes | Ring HQ | Ring bounds | Cover/leftover bounds |
| --- | ---: | ---: | --- | --- |
| `r1_r2_0_1_l2_004` | 8/8 | 100.00% | 2/2 | 6/6 |
| `r1_r2_0_1_l2_005` | 7/8 | 91.89% | 2/2 | 6/6 |
| `r1_r2_0_1_l2_007` | 6/8 | 75.22% | 0/2 | 6/6 |
| `r1_r2_0_1_l2_006` | 7/8 | 100.00% | 2/2 | 6/6 |
| `r1_r2_0_1_l2_003` | 3/8 | 24.95% | 0/2 | 6/6 |

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

<details><summary>r1_r2_0_1_l2_004: PASS</summary>

```json
{
  "name": "r1_r2_0_1_l2_004",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "a_r1r2",
  "reg_coeff": 0.1,
  "reg_kappa": 1.0,
  "particle_l2": 0.004,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 1.0
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.45246306 | >= 0.3 | +0.1525 | PASS |
| two_pole | grad_med | 0.71091789 | <= 1.0 | +0.2891 | PASS |
| trajectory | identity_mse | 0.0011282545 | <= 0.02 | +0.01887 | PASS |
| residual_student | identity_mse | 0.00058902975 | <= 0.02 | +0.01941 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.99981427 | >= 0.85 | +0.1498 | PASS |
| unipolar | off_caption | 2.3531376e-10 | <= 0.05 | +0.05 | PASS |
| unipolar | neu_hold | 0.99962805 | >= 0.85 | +0.1496 | PASS |
| ae_gan_hold | recon_mse | 0.0034516677 | <= 0.05 | +0.04655 | PASS |
| ae_gan_hold | hold | 0.016199844 | <= 0.35 | +0.3338 | PASS |
| cover_leftover | u_kept | 0.99387841 | >= 0.85 | +0.1439 | PASS |
| cover_leftover | content_kept | 1.0038319 | >= 0.75 | +0.2538 | PASS |
| cover_leftover | leak_ratio | 0.0020549445 | <= 0.2 | +0.1979 | PASS |
| cover_leftover | pole_rel_err_plus | 0.0044788779 | <= 0.2 | +0.1955 | PASS |
| cover_leftover | pole_rel_err_minus | 0.0030362497 | <= 0.2 | +0.197 | PASS |
| cover_leftover | same_dir | 0.0017224613 | <= 0.25 | +0.2483 | PASS |
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

<details><summary>r1_r2_0_1_l2_007: PASS</summary>

```json
{
  "name": "r1_r2_0_1_l2_007",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "a_r1r2",
  "reg_coeff": 0.1,
  "reg_kappa": 1.0,
  "particle_l2": 0.007,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 1.0
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.45267427 | >= 0.3 | +0.1527 | PASS |
| two_pole | grad_med | 0.73103487 | <= 1.0 | +0.269 | PASS |
| trajectory | identity_mse | 0.0059904587 | <= 0.02 | +0.01401 | PASS |
| residual_student | identity_mse | 0.0016942549 | <= 0.02 | +0.01831 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.99981427 | >= 0.85 | +0.1498 | PASS |
| unipolar | off_caption | 2.3531376e-10 | <= 0.05 | +0.05 | PASS |
| unipolar | neu_hold | 0.99962805 | >= 0.85 | +0.1496 | PASS |
| ae_gan_hold | recon_mse | 0.005422662 | <= 0.05 | +0.04458 | PASS |
| ae_gan_hold | hold | 0.011501991 | <= 0.35 | +0.3385 | PASS |
| cover_leftover | u_kept | 0.99399464 | >= 0.85 | +0.144 | PASS |
| cover_leftover | content_kept | 1.0039017 | >= 0.75 | +0.2539 | PASS |
| cover_leftover | leak_ratio | 0.0019374823 | <= 0.2 | +0.1981 | PASS |
| cover_leftover | pole_rel_err_plus | 0.0043905526 | <= 0.2 | +0.1956 | PASS |
| cover_leftover | pole_rel_err_minus | 0.0031224382 | <= 0.2 | +0.1969 | PASS |
| cover_leftover | same_dir | 0.0016717639 | <= 0.25 | +0.2483 | PASS |
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
| mode_hold | hq | 0.9206543 | >= 0.9 | +0.02065 | PASS |

</details>

<details><summary>r1_r2_0_1_l2_006: FAIL</summary>

```json
{
  "name": "r1_r2_0_1_l2_006",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "a_r1r2",
  "reg_coeff": 0.1,
  "reg_kappa": 1.0,
  "particle_l2": 0.006,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 1.0
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.45264384 | >= 0.3 | +0.1526 | PASS |
| two_pole | grad_med | 0.73036408 | <= 1.0 | +0.2696 | PASS |
| trajectory | identity_mse | 0.0052132574 | <= 0.02 | +0.01479 | PASS |
| residual_student | identity_mse | 0.0017355178 | <= 0.02 | +0.01826 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.99981427 | >= 0.85 | +0.1498 | PASS |
| unipolar | off_caption | 2.3531376e-10 | <= 0.05 | +0.05 | PASS |
| unipolar | neu_hold | 0.99962805 | >= 0.85 | +0.1496 | PASS |
| ae_gan_hold | recon_mse | 0.0027074469 | <= 0.05 | +0.04729 | PASS |
| ae_gan_hold | hold | 0.010990562 | <= 0.35 | +0.339 | PASS |
| cover_leftover | u_kept | 0.99410014 | >= 0.85 | +0.1441 | PASS |
| cover_leftover | content_kept | 1.0040125 | >= 0.75 | +0.254 | PASS |
| cover_leftover | leak_ratio | 0.0019725849 | <= 0.2 | +0.198 | PASS |
| cover_leftover | pole_rel_err_plus | 0.0043501975 | <= 0.2 | +0.1956 | PASS |
| cover_leftover | pole_rel_err_minus | 0.0031091494 | <= 0.2 | +0.1969 | PASS |
| cover_leftover | same_dir | 0.0016006507 | <= 0.25 | +0.2484 | PASS |
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
| mode_hold | hq | 0.92163086 | >= 0.9 | +0.02163 | PASS |

</details>

<details><summary>r1_r2_0_1_l2_003: FAIL</summary>

```json
{
  "name": "r1_r2_0_1_l2_003",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "a_r1r2",
  "reg_coeff": 0.1,
  "reg_kappa": 1.0,
  "particle_l2": 0.003,
  "vicreg_weight": 0.05,
  "cover_weight": 1.5,
  "lr_multiplier": 1.0
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.45235458 | >= 0.3 | +0.1524 | PASS |
| two_pole | grad_med | 0.73414689 | <= 1.0 | +0.2659 | PASS |
| trajectory | identity_mse | 0.0028225079 | <= 0.02 | +0.01718 | PASS |
| residual_student | identity_mse | 0.0024902744 | <= 0.02 | +0.01751 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.99981427 | >= 0.85 | +0.1498 | PASS |
| unipolar | off_caption | 2.3531376e-10 | <= 0.05 | +0.05 | PASS |
| unipolar | neu_hold | 0.99962805 | >= 0.85 | +0.1496 | PASS |
| ae_gan_hold | recon_mse | 0.0032639566 | <= 0.05 | +0.04674 | PASS |
| ae_gan_hold | hold | 0.0091674626 | <= 0.35 | +0.3408 | PASS |
| cover_leftover | u_kept | 0.9939134 | >= 0.85 | +0.1439 | PASS |
| cover_leftover | content_kept | 1.0039813 | >= 0.75 | +0.254 | PASS |
| cover_leftover | leak_ratio | 0.0021042626 | <= 0.2 | +0.1979 | PASS |
| cover_leftover | pole_rel_err_plus | 0.0044828663 | <= 0.2 | +0.1955 | PASS |
| cover_leftover | pole_rel_err_minus | 0.003124008 | <= 0.2 | +0.1969 | PASS |
| cover_leftover | same_dir | 0.0016362893 | <= 0.25 | +0.2484 | PASS |
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
| mode_hold | modes | 3 | >= 7 | -4 | FAIL |
| mode_hold | hq | 0.31933594 | >= 0.9 | -0.5807 | FAIL |

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
