# Principal: img_mask_inpaint2_transpose_vs_residual16

**Track:** application / image suite (masked reconstruction / inpainting-style; not geometry GMM)

**Claim:** The published ParticleGAN image path (transpose conv G, width 12, RpGAN + b_cap coeff 3 kappa 1.25 + 32 particles) fails a novel masked-reconstruction multimodal task. The same formulation with an in-family architecture change — residual nearest-neighbor upsample, width 16 — sustains PASS under matched seed/budget/CPU.

**Novelty:** Two 8x8 grayscale modes that share a fixed observed 1px border (intensity 0.75) and differ only in the masked interior 6x6:
- mode0: soft diagonal ramp `(i-1)+(j-1)/12` in the interior
- mode1: soft anti-diagonal ramp `(i-1)+(7-j)/12` in the interior
Product-adjacent to masked reconstruction / inpainting with multimodal completions under shared context. Distinct from:
- diag_ramp2 (full-image diagonal vs complement; no mask frame)
- sparse_obs2 (sparse 2x2 sensor blobs, no border context)
- intensity2 / soft_ring2 / colorize_lr2 / corner fills

This is NOT a diffusion/non-GAN baseline. Both arms stay inside ParticleGAN RpGAN + b_cap + particles. Batchfeat/shared_c6 vector D does not wire into the image suite; the documented published path for images is the transpose12 baseline.

**Tip:** `510e0054b2499ce725482de8e7a4ae8d72fd8f25` (`origin/particle-finetune/base`)

**HIT gate:** winner (baseline_transpose12) FAIL + control (residual16) PASS on img_mask_inpaint2.
