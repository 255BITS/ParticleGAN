# Behavioral baseline — live weights

**Regression PASS: `cap3_cosine_start0p6_lr0p85`.**

Protocol `behavior-v2` · CPU · seed 0 · fixed host budgets · final live weights.

**Overall PASS requires all 29 numerical bounds on all 9 trained toys, plus the 10 shared behavioral/integration checks.** EMA is diagnostic and cannot rescue a live failure. Config equality checks are excluded. Shared checks are run once: they do not depend on the GAN config and do not contribute to its rank.

**Regression PASS is a minimum bar for default selection.** The original ring bar permits 7/8 modes. Actual coverage, HQ, balance and checkpoint stability are visible below; a final-step PASS does not establish a stable ParticleGAN default. The [default-selection analysis](../../default_selection.md) also compares the stock recipe on the ring host.

Rows rank by passed toys, then passed numerical bounds, then live ring coverage, HQ and effective modes. Missing/nonfinite results and errors cannot pass. Thresholds and budgets are frozen before config search.

| Rank | Config | Live toys | Live bounds | Ring modes | Ring HQ | Effective modes | Regression |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `cap3_cosine_start0p6_lr0p85` (b_cap) | 9/9 | 29/29 | 8/8 | 100.00% | 7.54/8 | **PASS** |

Effective modes measures balance among high-quality outputs; eight balanced modes gives 8. HQ measures quality of generated samples and does not penalize a missing target cluster. Thus 100% HQ can coexist with 7/8 coverage.

## Live stability and exact particle coverage

Final-step selection is unchanged. For default selection, report the five observations at steps 1,000, 1,050, 1,100, 1,150 and 1,200. These are sampled checkpoints, not a claim about every intervening step. Enumerating all 12 equally likely particles additionally distinguishes a missing mode from an unlucky 4,096-sample evaluation.

| Config | Worst tail modes | Worst tail HQ | Tail checks with 8/8 and HQ ≥90% | Final HQ particles by mode (0–7) |
| --- | ---: | ---: | ---: | --- |
| `cap3_cosine_start0p6_lr0p85` | 8/8 | 90.99% | 5/5 | [1, 2, 1, 1, 2, 1, 2, 2] |

## Convergence speed

Each host has 24 evenly spaced observations. Sustained PASS requires a complete curve, at least five consecutive passing observations, and no later failure through the final budget. Ring convergence requires all eight modes and HQ ≥90%; original regression bounds stay unchanged. Stable-from is the start of that final passing stretch; confirmation is its fifth observation. Times include setup and measurement overhead. These certify observations, not every intervening update. Historical runs without timing are not assigned estimated speeds.

| Config | Sustained toys | Ring first PASS step | Ring stable from step | Ring confirmed step | Ring confirmed seconds | All toys wall seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cap3_cosine_start0p6_lr0p85` | 9/9 | 850 | 850 | 1050 | 5.91 | 21.55 |

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

## EMA diagnostics

Only the ring and cover/leftover hosts maintain EMA. These results are separate from the live ranking.

| Config | Ring modes | Ring HQ | Ring bounds | Cover/leftover bounds |
| --- | ---: | ---: | --- | --- |
| `cap3_cosine_start0p6_lr0p85` | 8/8 | 100.00% | 2/2 | 6/6 |

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

## Reproduce or compare another approach

```bash
python -m benchmarks.locked_shared.baseline --reference /path/to/conceptmod
python -m benchmarks.locked_shared.baseline --configs my_configs.json --reference /path/to/conceptmod --output reports/my_search
# Resume only with exactly matching source, runtime and config fingerprints:
python -m benchmarks.locked_shared.baseline --resume --reference /path/to/conceptmod
```

Use [passing_configs.json](passing_configs.json) to rerun the passing baseline alone, or [configs.json](configs.json) for the comparison set. Each setting applies wherever that loss exists; there are no per-toy overrides. The fixed protocol and actual source hashes are in [protocol.json](protocol.json); all measurements, timings and shared-check evidence are in [results.json](results.json). See [scope and configuration mapping](../../../../benchmarks/locked_shared/BASELINE.md). Exit code 0 means at least one full PASS; 1 means no full PASS. Output is saved after each toy. Without `--reference`, candidate training still runs but the full result is INCOMPLETE.

This is one fixed-seed CPU regression baseline, not evidence of downstream transfer or robustness across random initializations.
