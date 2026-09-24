# Principal: img_moire_beat2_transpose_vs_residual16

**Track:** application / image suite (display/print interference)

**Claim:** The published ParticleGAN image path (transpose conv G, width 12, RpGAN + b_cap + particles) fails a novel dual-frequency moiré beat multimodal task. The same formulation with residual nearest-neighbor upsample width 16 sustains PASS under matched seed/budget/CPU.

**Task:** Two 8x8 grayscale modes: (0) period-2 vertical stripes; (1) period-3 vertical stripes. Product-adjacent to display aliasing, print interference, and frequency discrimination. Distinct from vh_bars (axis swap), halftone screen angles, intensity, and glyph farms.

**Winner:** published image suite baseline transpose12.
**Control:** residual_upsample width 16 (same shared_c6 / RpGAN+b_cap+particles family).
**Solvability gate:** HIT only if winner FAIL and control PASS.

**Results @ tip ac83ce32b99bd488c2dd700786c0c93f0bb1f27b:**
- baseline_transpose12: FAIL (modes 0, HQ 0.0, RMSE≈0.0911, no sustain)
- residual16: PASS (modes 2, HQ 1.0, RMSE≈0.0227, confirmed_step 500)
