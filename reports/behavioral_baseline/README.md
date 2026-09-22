# Behavioral baseline — live weights

**Passing live baseline: `r1_r2_0_1`.**

Protocol `behavior-v1` · CPU · seed 0 · fixed host budgets · final live weights.

**Overall PASS requires all 29 numerical bounds on all 9 trained toys, plus the 10 shared behavioral/integration checks.** EMA is diagnostic and cannot rescue a live failure. Config equality checks are excluded. Shared checks are run once: they do not depend on the GAN config and do not contribute to its rank.

Rows rank by passed toys, then passed numerical bounds; equal counts tie. Missing/nonfinite results and errors cannot pass. Thresholds and budgets are frozen before config search.

| Rank | Config | Live toys | Live bounds | Overall | Failed live targets |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | `r1_r2_0_1` | 9/9 | 29/29 | **PASS** | None |
| 2 | `no_particle_l2` | 8/9 | 28/29 | **FAIL** | mode_hold |
| 3 | `locked_shared` | 7/9 | 26/29 | **FAIL** | trajectory, mode_hold |

## Live toy matrix

| Config | two_pole | trajectory | residual_student | unipolar | ae_gan_hold | cover_leftover | unused_token_hold | mid_scale_identity | mode_hold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `r1_r2_0_1` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `no_particle_l2` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | FAIL |
| `locked_shared` | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | FAIL |

## EMA diagnostics

Only the ring and cover/leftover hosts maintain EMA. These results are separate from the live ranking.

| Config | Ring modes | Ring HQ | Ring bounds | Cover/leftover bounds |
| --- | ---: | ---: | --- | --- |
| `r1_r2_0_1` | 5/8 | 65.16% | 0/2 | 6/6 |
| `no_particle_l2` | 8/8 | 100.00% | 2/2 | 6/6 |
| `locked_shared` | 3/8 | 49.49% | 0/2 | 6/6 |

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

## Reproduce or compare another approach

```bash
python -m benchmarks.locked_shared.baseline --reference /path/to/conceptmod
python -m benchmarks.locked_shared.baseline --configs my_configs.json --reference /path/to/conceptmod --output reports/my_search
# Resume only with exactly matching source, runtime and config fingerprints:
python -m benchmarks.locked_shared.baseline --resume --reference /path/to/conceptmod
```

Use [passing_configs.json](passing_configs.json) to rerun the passing baseline alone, or [configs.json](configs.json) for the comparison set. Each setting applies wherever that loss exists; there are no per-toy overrides. The fixed protocol and actual source hashes are in [protocol.json](protocol.json); all measurements, timings and shared-check evidence are in [results.json](results.json). See [scope and configuration mapping](../../benchmarks/locked_shared/BASELINE.md). Exit code 0 means at least one full PASS; 1 means no full PASS. Output is saved after each toy. Without `--reference`, candidate training still runs but the full result is INCOMPLETE.

This is one fixed-seed CPU regression baseline, not evidence of downstream transfer or robustness across random initializations.
