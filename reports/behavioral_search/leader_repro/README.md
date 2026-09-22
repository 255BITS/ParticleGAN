# Behavioral baseline — live weights

> Reproduction of Codex’s leading config `bcap_k1p25_c3p0_lr0p85` on this CPU. Codex recorded a full PASS with a live ring of 8/8 at 100% HQ. This run does not match that ring or trajectory. Their leaderboard is unchanged. Summary: [../README.md](../README.md).

**No complete passing live configuration recorded.**

Protocol `behavior-v1` · CPU · seed 0 · fixed host budgets · final live weights.

**Overall PASS requires all 29 numerical bounds on all 9 trained toys, plus the 10 shared behavioral/integration checks.** EMA is diagnostic and cannot rescue a live failure. Config equality checks are excluded. Shared checks are run once: they do not depend on the GAN config and do not contribute to its rank.

**Regression PASS is a minimum bar for default selection.** The original ring bar permits 7/8 modes. Actual coverage, HQ, balance and checkpoint stability are visible below; a final-step PASS does not establish a stable ParticleGAN default. The [default-selection analysis](default_selection.md) also compares the stock recipe on the ring host.

Rows rank by passed toys, then passed numerical bounds, then live ring coverage, HQ and effective modes. Missing/nonfinite results and errors cannot pass. Thresholds and budgets are frozen before config search.

| Rank | Config | Live toys | Live bounds | Ring modes | Ring HQ | Effective modes | Regression |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `bcap_k1p25_c3p0_lr0p85` (b_cap) | 7/9 | 26/29 | 5/8 | 66.67% | 4.79/8 | **FAIL** |

Effective modes measures balance among high-quality outputs; eight balanced modes gives 8. HQ measures quality of generated samples and does not penalize a missing target cluster. Thus 100% HQ can coexist with 7/8 coverage.

## Live stability and exact particle coverage

Final-step selection is unchanged. For default selection, report the five observations at steps 1,000, 1,050, 1,100, 1,150 and 1,200. These are sampled checkpoints, not a claim about every intervening step. Enumerating all 12 equally likely particles additionally distinguishes a missing mode from an unlucky 4,096-sample evaluation.

| Config | Worst tail modes | Worst tail HQ | Tail checks with 8/8 and HQ ≥90% | Final HQ particles by mode (0–7) |
| --- | ---: | ---: | ---: | --- |
| `bcap_k1p25_c3p0_lr0p85` | 4/8 | 40.89% | 0/5 | [2, 1, 2, 1, 0, 2, 0, 0] |

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
| `bcap_k1p25_c3p0_lr0p85` | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | FAIL |

## EMA diagnostics

Only the ring and cover/leftover hosts maintain EMA. These results are separate from the live ranking.

| Config | Ring modes | Ring HQ | Ring bounds | Cover/leftover bounds |
| --- | ---: | ---: | --- | --- |
| `bcap_k1p25_c3p0_lr0p85` | 6/8 | 82.96% | 0/2 | 6/6 |

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

<details><summary>bcap_k1p25_c3p0_lr0p85: FAIL</summary>

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
| two_pole | grad_med | 0.51399523 | <= 1.0 | +0.486 | PASS |
| trajectory | identity_mse | 0.021599449 | <= 0.02 | -0.001599 | FAIL |
| residual_student | identity_mse | 0.0018423294 | <= 0.02 | +0.01816 | PASS |
| residual_student | success_rate | 1 | >= 1.0 | +0 | PASS |
| residual_student | wrong_pad_rate | 0 | <= 0.0 | +0 | PASS |
| unipolar | cover | 0.91498068 | >= 0.85 | +0.06498 | PASS |
| unipolar | off_caption | 0.00018813263 | <= 0.05 | +0.04981 | PASS |
| unipolar | neu_hold | 0.95050065 | >= 0.85 | +0.1005 | PASS |
| ae_gan_hold | recon_mse | 0.0034568086 | <= 0.05 | +0.04654 | PASS |
| ae_gan_hold | hold | 0.0015421447 | <= 0.35 | +0.3485 | PASS |
| cover_leftover | u_kept | 0.97220581 | >= 0.85 | +0.1222 | PASS |
| cover_leftover | content_kept | 1.0087114 | >= 0.75 | +0.2587 | PASS |
| cover_leftover | leak_ratio | 0.0010659429 | <= 0.2 | +0.1989 | PASS |
| cover_leftover | pole_rel_err_plus | 0.018618492 | <= 0.2 | +0.1814 | PASS |
| cover_leftover | pole_rel_err_minus | 0.018833438 | <= 0.2 | +0.1812 | PASS |
| cover_leftover | same_dir | 0.0023988673 | <= 0.25 | +0.2476 | PASS |
| unused_token_hold | unused_hold | 0.99626632 | >= 0.85 | +0.1463 | PASS |
| unused_token_hold | concept_move | 0.90559014 | >= 0.85 | +0.05559 | PASS |
| mid_scale_identity | concept_cos_plus | 0.99992609 | >= 0.85 | +0.1499 | PASS |
| mid_scale_identity | concept_cos_minus | 0.99979967 | >= 0.85 | +0.1498 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0001982 | >= 0.75 | +0.2502 | PASS |
| mid_scale_identity | concept_mag_plus | 1.0001982 | <= 1.25 | +0.2498 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0061932 | >= 0.75 | +0.2562 | PASS |
| mid_scale_identity | concept_mag_minus | 1.0061932 | <= 1.25 | +0.2438 | PASS |
| mid_scale_identity | identity_at_0 | 0.98375969 | >= 0.85 | +0.1338 | PASS |
| mid_scale_identity | identity_at_mid | 0.99455658 | >= 0.85 | +0.1446 | PASS |
| mode_hold | modes | 5 | >= 7 | -2 | FAIL |
| mode_hold | hq | 0.66674805 | >= 0.9 | -0.2333 | FAIL |

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
