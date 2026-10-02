# Principal: img_radial_wedge2_transpose_vs_residual16

**Track:** application / image suite (angular wedge orientation; not geometry GMM)

**Claim:** The published ParticleGAN image path (transpose conv G, width 12, RpGAN + b_cap coeff 3 kappa 1.25 + 32 particles) fails a novel angular half-disk wedge task. The same formulation with an in-family architecture change — residual nearest-neighbor upsample, width 16 — sustains PASS under matched seed/budget/CPU.

**Novelty:** Two 8×8 grayscale modes that share radius (~3.2) but differ in angular support:
- mode0: upper half-disk (`atan2(y,x) >= 0`)
- mode1: left half-disk (`|atan2(y,x)| >= π/2`)

Product-adjacent to orientation / angular-structure discrimination. Distinct from:
- mask_inpaint2, soft_ring2, intensity2, sparse_obs2, diag_ramp2, colorize_lr2
- vh_bars2 (orthogonal bar domains)
- digit/letter topology probes
- row/col fill and sparse-points→field soft probes this fire

This is NOT a diffusion/non-GAN baseline. Both arms stay inside ParticleGAN RpGAN + b_cap + particles.

**Tip:** `510e0054b2499ce725482de8e7a4ae8d72fd8f25` (`origin/particle-finetune/base`)

**HIT gate:** winner (baseline_transpose12) FAIL + control (residual16) PASS on img_radial_wedge2.
