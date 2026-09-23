# Principal: img_diag_ramp2_transpose_vs_residual16

**Track:** application / image suite (masked / directional reconstruction; not geometry GMM)

**Claim:** The published ParticleGAN image path (transpose conv G, width 12, RpGAN + b_cap coeff 3 kappa 1.25 + 32 particles) fails a novel soft diagonal-ramp multimodal task. The same formulation with an in-family architecture change — residual nearest-neighbor upsample, width 16 — sustains PASS under matched seed/budget/CPU.

**Novelty:** Two 8x8 grayscale modes that are soft *diagonal* intensity ramps (i+j)/14 vs its complement 1-(i+j)/14. Product-adjacent to directional / anisotropic masked reconstruction and unpaired orientation-conditioned generation. Distinct from:
- gradient2 (axis-aligned horizontal ramp; probe only, never PR)
- vignette2 (radial center-bright/dark)
- soft_ring2 / intensity2 / sparse_obs2 (claimed HITs)
- gray_stripes2 (horizontal phase stripes)

This is NOT a diffusion/non-GAN baseline. Both arms stay inside ParticleGAN RpGAN + b_cap + particles. Batchfeat/shared_c6 vector D does not wire into the image suite; the documented published path for images is the transpose12 baseline.

**Tip:**  ()

**HIT gate:** winner (baseline_transpose12) FAIL + control (residual16) PASS on img_diag_ramp2.
