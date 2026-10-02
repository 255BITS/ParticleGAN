# Principal: img_sparse_obs2_transpose_vs_residual16

**Track:** application / image suite (sparse observation; not geometry GMM)

**Claim:** The published ParticleGAN image path (transpose conv G, width 12, RpGAN + b_cap coeff 3 kappa 1.25 + 32 particles) fails a novel sparse-observation occupancy task. The same formulation with an in-family architecture change — residual nearest-neighbor upsample, width 16 — sustains PASS under matched seed/budget/CPU.

**Novelty:** Two 8x8 grayscale modes that differ only by which sparse 2x2 sensor blobs are lit:
- mode0 (corners): NW+SE patches at intensity 0.9
- mode1 (edges): NE+SW patches at intensity 0.9
Product-adjacent sparse observation / occupancy-map multimodal generation. Distinct from soft_ring2 (topology annulus), intensity2 (single central patch), vignette2, gradient2, and gray_stripes2.

This is NOT a diffusion/non-GAN baseline. Both arms stay inside ParticleGAN RpGAN + b_cap + particles. Batchfeat/shared_c6 vector D does not wire into the image suite; the documented published path for images is the transpose12 baseline.

**Tip:** `510e0054b2499ce725482de8e7a4ae8d72fd8f25` (`origin/particle-finetune/base`)

**HIT gate:** winner (baseline_transpose12) FAIL + control (residual16) PASS on img_sparse_obs2.
