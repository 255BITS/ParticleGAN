# Principal: img_vh_bars2_transpose_vs_residual16

**Track:** application / image suite (unpaired domain-translation-style; not geometry GMM)

**Claim:** The published ParticleGAN image path (transpose conv G, width 12, RpGAN + b_cap coeff 3 kappa 1.25 + 32 particles) fails a novel unpaired vertical-vs-horizontal bar-domain task. The same formulation with an in-family architecture change — residual nearest-neighbor upsample, width 16 — sustains PASS under matched seed/budget/CPU.

**Novelty:** Two 8×8 grayscale modes that share intensity mass but differ in bar orientation (vertical domain vs horizontal domain). Product-adjacent to unpaired translation / domain-transfer demos. Distinct from:
- mask_inpaint2 (shared border + interior ramps)
- sparse_obs2 / edge-conditioned fills
- soft_ring2 / intensity2 / diag_ramp2 / colorize_lr2

This is NOT a diffusion/non-GAN baseline. Both arms stay inside ParticleGAN RpGAN + b_cap + particles. Batchfeat/shared_c6 vector D does not wire into the image suite; the documented published path for images is the transpose12 baseline.

**Tip:** `510e0054b2499ce725482de8e7a4ae8d72fd8f25` (`origin/particle-finetune/base`)

**HIT gate:** winner (baseline_transpose12) FAIL + control (residual16) PASS on img_vh_bars2.

## Results
| Arm | Verdict | Modes | HQ | mean RMSE | confirmed_step |
|-----|---------|-------|----|-----------|----------------|
| baseline_transpose12 | FAIL | 0 | 0.031 | ≈0.078 | — |
| residual16 | PASS | 2 | 1.0 | ≈0.022 | 575 |

## Reproduce
```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  python -u reports/transfer_suite/img_vh_bars2_transpose_vs_residual16/reproduce_arms.py
```
