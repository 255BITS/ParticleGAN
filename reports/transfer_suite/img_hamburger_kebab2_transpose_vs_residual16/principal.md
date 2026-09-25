# Principal: img_hamburger_kebab2_transpose_vs_residual16

**Track:** application / image suite (UI menu icons)

**Claim:** The published ParticleGAN image path (transpose conv G, width 12, RpGAN + b_cap + particles) fails a novel hamburger-vs-kebab menu icon multimodal task. The same formulation with residual nearest-neighbor upsample width 16 sustains PASS under matched seed/budget/CPU.

**Task:** Two 8x8 grayscale modes: (0) hamburger menu — three horizontal bars; (1) kebab menu — three vertical square dots. Product-adjacent to mobile/desktop overflow menus and iconography discrimination. Distinct from play/pause, traffic stack, sort chevrons, and glyph farms.

**Winner:** published image suite baseline transpose12.
**Control:** residual_upsample width 16 (same shared_c6 / RpGAN+b_cap+particles family).
**Solvability gate:** HIT only if winner FAIL and control PASS.

**Results @ tip ac83ce32b99bd488c2dd700786c0c93f0bb1f27b:**
- baseline_transpose12: FAIL (modes 1, HQ 0.5, RMSE≈0.0506, no sustain)
- residual16: PASS (modes 2, HQ 1.0, RMSE≈0.0301, confirmed_step 575)
