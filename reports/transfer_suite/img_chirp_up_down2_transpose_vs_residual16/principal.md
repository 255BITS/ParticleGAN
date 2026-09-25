# Principal: img_chirp_up_down2_transpose_vs_residual16

**Track:** application / image suite (DSP spectrogram)

**Claim:** The published ParticleGAN image path (transpose conv G, width 12, RpGAN + b_cap + particles) fails a novel rising-vs-falling chirp spectrogram multimodal task. The same formulation with residual nearest-neighbor upsample width 16 sustains PASS under matched seed/budget/CPU.

**Task:** Two 8x8 grayscale modes: (0) rising chirp — frequency ridge y grows with x; (1) falling chirp — ridge y shrinks with x. Product-adjacent to audio/sonogram UIs and radar chirp polarity. Distinct from moiré beat, wave polarity, and glyph farms.

**Winner:** published image suite baseline transpose12.
**Control:** residual_upsample width 16 (same shared_c6 / RpGAN+b_cap+particles family).
**Solvability gate:** HIT only if winner FAIL and control PASS.

**Results @ tip ac83ce32b99bd488c2dd700786c0c93f0bb1f27b:**
- baseline_transpose12: FAIL (modes 2, HQ 0.71875, RMSE≈0.0460, no sustain)
- residual16: PASS (modes 2, HQ 0.90625, RMSE≈0.0381, confirmed_step 600)
