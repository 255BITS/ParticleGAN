# Principal: img_play_pause2_transpose_vs_residual16

**Track:** application / image suite (media-control UI glyphs)

**Claim:** The published ParticleGAN image path (transpose conv G, width 12, RpGAN + b_cap + particles) fails a novel play-triangle vs pause-bars multimodal task. The same formulation with residual nearest-neighbor upsample width 16 sustains PASS under matched seed/budget/CPU.

**Task:** Two 8x8 grayscale modes: (0) right-pointing play triangle; (1) pause double vertical bars. Product-adjacent to media players, transport controls, and glyph-level UI icon discrimination. Distinct from traffic-stack vertical order, folder-tab L/R, EQ boost bands, and geometry GMM breaks.

**Winner:** published image suite baseline transpose12.
**Control:** residual_upsample width 16 (same shared_c6 / RpGAN+b_cap+particles family).
**Solvability gate:** HIT only if winner FAIL and control PASS.

**Results @ tip 510e0054b2499ce725482de8e7a4ae8d72fd8f25:**
- baseline_transpose12: FAIL (modes 0, HQ 0.0, RMSE≈0.0991, no sustain)
- residual16: PASS (modes 2, HQ 1.0, RMSE≈0.0306, confirmed_step 575)
