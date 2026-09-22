# Behavioral baseline — live weights

> Side search on a CPU that does not reproduce the published ring. Do not promote these rows over the [PR #38 leaderboard](../../behavioral_baseline/README.md). Summary: [../README.md](../README.md).

**No complete passing live configuration recorded.**

Protocol `behavior-v1` · CPU · seed 0 · fixed host budgets · final live weights.

**Overall PASS requires all 29 numerical bounds on all 9 trained toys, plus the 10 shared behavioral/integration checks.** EMA is diagnostic and cannot rescue a live failure. Config equality checks are excluded. Shared checks are run once: they do not depend on the GAN config and do not contribute to its rank.

Rows rank by passed toys, then passed numerical bounds; equal counts tie. Missing/nonfinite results and errors cannot pass. Thresholds and budgets are frozen before config search.

| Rank | Config | Live toys | Live bounds | Overall | Failed live targets |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | `music_cover` | 8/9 | 27/29 | **FAIL** | mode_hold |
| 2 | `no_vicreg` | 7/9 | 27/29 | **FAIL** | trajectory, mode_hold |
| 3 | `no_l2_no_vicreg` | 7/9 | 26/29 | **FAIL** | trajectory, mode_hold |
| 4 | `base_regularization` | 7/9 | 24/29 | **FAIL** | cover_leftover, mode_hold |
| 5 | `no_cover` | 7/9 | 23/29 | **FAIL** | cover_leftover, mode_hold |

## Live toy matrix

| Config | two_pole | trajectory | residual_student | unipolar | ae_gan_hold | cover_leftover | unused_token_hold | mid_scale_identity | mode_hold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `music_cover` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | FAIL |
| `no_vicreg` | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | FAIL |
| `no_l2_no_vicreg` | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | FAIL |
| `base_regularization` | PASS | PASS | PASS | PASS | PASS | FAIL | PASS | PASS | FAIL |
| `no_cover` | PASS | PASS | PASS | PASS | PASS | FAIL | PASS | PASS | FAIL |

## EMA diagnostics

Only the ring and cover/leftover hosts maintain EMA. These results are separate from the live ranking.

| Config | Ring modes | Ring HQ | Ring bounds | Cover/leftover bounds |
| --- | ---: | ---: | --- | --- |
| `music_cover` | 8/8 | 100.00% | 2/2 | 6/6 |
| `no_vicreg` | 7/8 | 91.72% | 2/2 | 6/6 |
| `no_l2_no_vicreg` | 4/8 | 51.51% | 0/2 | 6/6 |
| `base_regularization` | 8/8 | 100.00% | 2/2 | 3/6 |
| `no_cover` | 8/8 | 100.00% | 2/2 | 2/6 |

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

<details><summary>music_cover: FAIL</summary>

```json
{
  "name": "music_cover",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "b_cap",
  "reg_coeff": 1.0,
  "reg_kappa": 1.0,
  "particle_l2": 0.02,
  "vicreg_weight": 0.05,
  "cover_weight": 1.0,
  "lr_multiplier": 1.0
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.51437217 | >= 0.3 | +0.2144 | PASS |
| two_pole | grad_med | 0.41965187 | <= 1.0 | +0.5803 | PASS |
| trajectory | identity_mse | 0.0019263001 | <= 0.02 | +0.01807 | PASS |
| residual_student | identity_mse | 0.0037978187 | <= 0.02 | +0.0162 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.96393463 | >= 0.85 | +0.1139 | PASS |
| unipolar | off_caption | 0.00053965067 | <= 0.05 | +0.04946 | PASS |
| unipolar | neu_hold | 0.95567976 | >= 0.85 | +0.1057 | PASS |
| ae_gan_hold | recon_mse | 0.004570175 | <= 0.05 | +0.04543 | PASS |
| ae_gan_hold | hold | 0.0071683167 | <= 0.35 | +0.3428 | PASS |
| cover_leftover | u_kept | 0.96853923 | >= 0.85 | +0.1185 | PASS |
| cover_leftover | content_kept | 0.99711275 | >= 0.75 | +0.2471 | PASS |
| cover_leftover | leak_ratio | 0.0057400301 | <= 0.2 | +0.1943 | PASS |
| cover_leftover | pole_rel_err_plus | 0.021647649 | <= 0.2 | +0.1784 | PASS |
| cover_leftover | pole_rel_err_minus | 0.017902549 | <= 0.2 | +0.1821 | PASS |
| cover_leftover | same_dir | 0.0038497247 | <= 0.25 | +0.2462 | PASS |
| unused_token_hold | unused_hold | 0.99144792 | >= 0.85 | +0.1414 | PASS |
| unused_token_hold | concept_move | 0.96768949 | >= 0.85 | +0.1177 | PASS |
| mid_scale_identity | concept_cos_plus | 0.99966246 | >= 0.85 | +0.1497 | PASS |
| mid_scale_identity | concept_cos_minus | 0.99972296 | >= 0.85 | +0.1497 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0099963 | >= 0.75 | +0.26 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0099963 | <= 1.25 | +0.24 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0043446 | >= 0.75 | +0.2543 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0043446 | <= 1.25 | +0.2457 | PASS |
| mid_scale_identity | identity_at_0 | 0.97614622 | >= 0.85 | +0.1261 | PASS |
| mid_scale_identity | identity_at_mid | 0.99074362 | >= 0.85 | +0.1407 | PASS |
| mode_hold | modes | 4 | >= 7 | -3 | FAIL |
| mode_hold | hq | 0.49243164 | >= 0.9 | -0.4076 | FAIL |

</details>

<details><summary>no_vicreg: FAIL</summary>

```json
{
  "name": "no_vicreg",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "b_cap",
  "reg_coeff": 1.0,
  "reg_kappa": 1.0,
  "particle_l2": 0.02,
  "vicreg_weight": 0.0,
  "cover_weight": 1.5,
  "lr_multiplier": 1.0
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.51437217 | >= 0.3 | +0.2144 | PASS |
| two_pole | grad_med | 0.41965187 | <= 1.0 | +0.5803 | PASS |
| trajectory | identity_mse | 0.022007309 | <= 0.02 | -0.002007 | FAIL |
| residual_student | identity_mse | 0.0019983638 | <= 0.02 | +0.018 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.96393463 | >= 0.85 | +0.1139 | PASS |
| unipolar | off_caption | 0.00053965067 | <= 0.05 | +0.04946 | PASS |
| unipolar | neu_hold | 0.95567976 | >= 0.85 | +0.1057 | PASS |
| ae_gan_hold | recon_mse | 0.0038015451 | <= 0.05 | +0.0462 | PASS |
| ae_gan_hold | hold | 0.023890372 | <= 0.35 | +0.3261 | PASS |
| cover_leftover | u_kept | 0.99393504 | >= 0.85 | +0.1439 | PASS |
| cover_leftover | content_kept | 1.0069095 | >= 0.75 | +0.2569 | PASS |
| cover_leftover | leak_ratio | 0.0023346203 | <= 0.2 | +0.1977 | PASS |
| cover_leftover | pole_rel_err_plus | 0.0052751391 | <= 0.2 | +0.1947 | PASS |
| cover_leftover | pole_rel_err_minus | 0.0039236774 | <= 0.2 | +0.1961 | PASS |
| cover_leftover | same_dir | 0.0029971468 | <= 0.25 | +0.247 | PASS |
| unused_token_hold | unused_hold | 0.99144792 | >= 0.85 | +0.1414 | PASS |
| unused_token_hold | concept_move | 0.96768949 | >= 0.85 | +0.1177 | PASS |
| mid_scale_identity | concept_cos_plus | 0.99986649 | >= 0.85 | +0.1499 | PASS |
| mid_scale_identity | concept_cos_minus | 0.99996555 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_mag_plus | 0.99932432 | >= 0.75 | +0.2493 | PASS |
| mid_scale_identity | concept_mag_plus | 0.99932432 | <= 1.25 | +0.2507 | PASS |
| mid_scale_identity | concept_mag_minus | 0.99813956 | >= 0.75 | +0.2481 | PASS |
| mid_scale_identity | concept_mag_minus | 0.99813956 | <= 1.25 | +0.2519 | PASS |
| mid_scale_identity | identity_at_0 | 0.98746178 | >= 0.85 | +0.1375 | PASS |
| mid_scale_identity | identity_at_mid | 0.99776103 | >= 0.85 | +0.1478 | PASS |
| mode_hold | modes | 7 | >= 7 | +0 | PASS |
| mode_hold | hq | 0.82714844 | >= 0.9 | -0.07285 | FAIL |

</details>

<details><summary>no_l2_no_vicreg: FAIL</summary>

```json
{
  "name": "no_l2_no_vicreg",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "b_cap",
  "reg_coeff": 1.0,
  "reg_kappa": 1.0,
  "particle_l2": 0.0,
  "vicreg_weight": 0.0,
  "cover_weight": 1.5,
  "lr_multiplier": 1.0
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.52926749 | >= 0.3 | +0.2293 | PASS |
| two_pole | grad_med | 0.41900396 | <= 1.0 | +0.581 | PASS |
| trajectory | identity_mse | 0.23717575 | <= 0.02 | -0.2172 | FAIL |
| residual_student | identity_mse | 0.0012719053 | <= 0.02 | +0.01873 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.96393463 | >= 0.85 | +0.1139 | PASS |
| unipolar | off_caption | 0.00053965067 | <= 0.05 | +0.04946 | PASS |
| unipolar | neu_hold | 0.95567976 | >= 0.85 | +0.1057 | PASS |
| ae_gan_hold | recon_mse | 0.002700039 | <= 0.05 | +0.0473 | PASS |
| ae_gan_hold | hold | 0.0079781385 | <= 0.35 | +0.342 | PASS |
| cover_leftover | u_kept | 0.9966362 | >= 0.85 | +0.1466 | PASS |
| cover_leftover | content_kept | 1.0046214 | >= 0.75 | +0.2546 | PASS |
| cover_leftover | leak_ratio | 0.0006640139 | <= 0.2 | +0.1993 | PASS |
| cover_leftover | pole_rel_err_plus | 0.0028621699 | <= 0.2 | +0.1971 | PASS |
| cover_leftover | pole_rel_err_minus | 0.0021742189 | <= 0.2 | +0.1978 | PASS |
| cover_leftover | same_dir | 0.001454698 | <= 0.25 | +0.2485 | PASS |
| unused_token_hold | unused_hold | 0.99144792 | >= 0.85 | +0.1414 | PASS |
| unused_token_hold | concept_move | 0.96768949 | >= 0.85 | +0.1177 | PASS |
| mid_scale_identity | concept_cos_plus | 0.99986649 | >= 0.85 | +0.1499 | PASS |
| mid_scale_identity | concept_cos_minus | 0.99996555 | >= 0.85 | +0.15 | PASS |
| mid_scale_identity | concept_mag_plus | 0.99932432 | >= 0.75 | +0.2493 | PASS |
| mid_scale_identity | concept_mag_plus | 0.99932432 | <= 1.25 | +0.2507 | PASS |
| mid_scale_identity | concept_mag_minus | 0.99813956 | >= 0.75 | +0.2481 | PASS |
| mid_scale_identity | concept_mag_minus | 0.99813956 | <= 1.25 | +0.2519 | PASS |
| mid_scale_identity | identity_at_0 | 0.98746178 | >= 0.85 | +0.1375 | PASS |
| mid_scale_identity | identity_at_mid | 0.99776103 | >= 0.85 | +0.1478 | PASS |
| mode_hold | modes | 6 | >= 7 | -1 | FAIL |
| mode_hold | hq | 0.82373047 | >= 0.9 | -0.07627 | FAIL |

</details>

<details><summary>base_regularization: FAIL</summary>

```json
{
  "name": "base_regularization",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "b_cap",
  "reg_coeff": 1.0,
  "reg_kappa": 1.0,
  "particle_l2": 0.0,
  "vicreg_weight": 1.0,
  "cover_weight": 0.0,
  "lr_multiplier": 1.0
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.52926749 | >= 0.3 | +0.2293 | PASS |
| two_pole | grad_med | 0.41900396 | <= 1.0 | +0.581 | PASS |
| trajectory | identity_mse | 0.0018108109 | <= 0.02 | +0.01819 | PASS |
| residual_student | identity_mse | 0.0049625398 | <= 0.02 | +0.01504 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.96393463 | >= 0.85 | +0.1139 | PASS |
| unipolar | off_caption | 0.00053965067 | <= 0.05 | +0.04946 | PASS |
| unipolar | neu_hold | 0.95567976 | >= 0.85 | +0.1057 | PASS |
| ae_gan_hold | recon_mse | 0.0016371319 | <= 0.05 | +0.04836 | PASS |
| ae_gan_hold | hold | 0.0053778985 | <= 0.35 | +0.3446 | PASS |
| cover_leftover | u_kept | 0.62790173 | >= 0.85 | -0.2221 | FAIL |
| cover_leftover | content_kept | 0.81691134 | >= 0.75 | +0.06691 | PASS |
| cover_leftover | leak_ratio | 0.064707769 | <= 0.2 | +0.1353 | PASS |
| cover_leftover | pole_rel_err_plus | 0.25890329 | <= 0.2 | -0.0589 | FAIL |
| cover_leftover | pole_rel_err_minus | 0.26535425 | <= 0.2 | -0.06535 | FAIL |
| cover_leftover | same_dir | 0.026140481 | <= 0.25 | +0.2239 | PASS |
| unused_token_hold | unused_hold | 0.99144792 | >= 0.85 | +0.1414 | PASS |
| unused_token_hold | concept_move | 0.96768949 | >= 0.85 | +0.1177 | PASS |
| mid_scale_identity | concept_cos_plus | 0.99962425 | >= 0.85 | +0.1496 | PASS |
| mid_scale_identity | concept_cos_minus | 0.99971741 | >= 0.85 | +0.1497 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0167801 | >= 0.75 | +0.2668 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0167801 | <= 1.25 | +0.2332 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0024039 | >= 0.75 | +0.2524 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0024039 | <= 1.25 | +0.2476 | PASS |
| mid_scale_identity | identity_at_0 | 0.98759996 | >= 0.85 | +0.1376 | PASS |
| mid_scale_identity | identity_at_mid | 0.96537068 | >= 0.85 | +0.1154 | PASS |
| mode_hold | modes | 6 | >= 7 | -1 | FAIL |
| mode_hold | hq | 0.58251953 | >= 0.9 | -0.3175 | FAIL |

</details>

<details><summary>no_cover: FAIL</summary>

```json
{
  "name": "no_cover",
  "loss_type": "logistic",
  "gan_mode": "rp",
  "reg_arm": "b_cap",
  "reg_coeff": 1.0,
  "reg_kappa": 1.0,
  "particle_l2": 0.02,
  "vicreg_weight": 0.05,
  "cover_weight": 0.0,
  "lr_multiplier": 1.0
}
```

| Toy | Metric | Actual | Required | Margin | Result |
| --- | --- | ---: | --- | ---: | --- |
| two_pole | mean_abs | 0.51437217 | >= 0.3 | +0.2144 | PASS |
| two_pole | grad_med | 0.41965187 | <= 1.0 | +0.5803 | PASS |
| trajectory | identity_mse | 0.0013241419 | <= 0.02 | +0.01868 | PASS |
| residual_student | identity_mse | 0.0042032846 | <= 0.02 | +0.0158 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.96393463 | >= 0.85 | +0.1139 | PASS |
| unipolar | off_caption | 0.00053965067 | <= 0.05 | +0.04946 | PASS |
| unipolar | neu_hold | 0.95567976 | >= 0.85 | +0.1057 | PASS |
| ae_gan_hold | recon_mse | 0.0020023088 | <= 0.05 | +0.048 | PASS |
| ae_gan_hold | hold | 0.00034526698 | <= 0.35 | +0.3497 | PASS |
| cover_leftover | u_kept | 0.57155895 | >= 0.85 | -0.2784 | FAIL |
| cover_leftover | content_kept | 0.61655813 | >= 0.75 | -0.1334 | FAIL |
| cover_leftover | leak_ratio | 0.092818009 | <= 0.2 | +0.1072 | PASS |
| cover_leftover | pole_rel_err_plus | 0.31773275 | <= 0.2 | -0.1177 | FAIL |
| cover_leftover | pole_rel_err_minus | 0.30923119 | <= 0.2 | -0.1092 | FAIL |
| cover_leftover | same_dir | 0.029139953 | <= 0.25 | +0.2209 | PASS |
| unused_token_hold | unused_hold | 0.99144792 | >= 0.85 | +0.1414 | PASS |
| unused_token_hold | concept_move | 0.96768949 | >= 0.85 | +0.1177 | PASS |
| mid_scale_identity | concept_cos_plus | 0.99962425 | >= 0.85 | +0.1496 | PASS |
| mid_scale_identity | concept_cos_minus | 0.99971741 | >= 0.85 | +0.1497 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0167801 | >= 0.75 | +0.2668 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0167801 | <= 1.25 | +0.2332 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0024039 | >= 0.75 | +0.2524 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0024039 | <= 1.25 | +0.2476 | PASS |
| mid_scale_identity | identity_at_0 | 0.98759996 | >= 0.85 | +0.1376 | PASS |
| mid_scale_identity | identity_at_mid | 0.96537068 | >= 0.85 | +0.1154 | PASS |
| mode_hold | modes | 4 | >= 7 | -3 | FAIL |
| mode_hold | hq | 0.49243164 | >= 0.9 | -0.4076 | FAIL |

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
