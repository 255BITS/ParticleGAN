# Principal: img_sonar_echo_near_far2_transpose_vs_residual16

**Track:** application / image suite (active sonar / A-scope)

**Claim:** The published ParticleGAN image path (transpose conv G, width 12, RpGAN + b_cap + particles) fails a novel near-vs-far sonar echo multimodal task. The same formulation with residual nearest-neighbor upsample width 16 sustains PASS under matched seed/budget/CPU.

**Task:** Two 8x8 grayscale modes representing an active-sonar A-scope: (0) strong **near** echo (early return, bright blob at low time index); (1) strong **far** echo (late return, blob at high time index); shared weak TX ping at t=0. Product-adjacent to radar/sonar ranging UIs and echo-delay discrimination. Distinct from chirp polarity, diffraction-order LR, and kymograph streaks.

**Winner:** published image suite baseline transpose12.
**Control:** residual_upsample width 16 (same shared_c6 / RpGAN+b_cap+particles family).
**Solvability gate:** HIT only if winner FAIL and control PASS.

**Results @ tip ac83ce32b99bd488c2dd700786c0c93f0bb1f27b:**
- baseline_transpose12: FAIL (modes 0, HQ 0.125, RMSE≈0.0676, no sustain)
- residual16: PASS (modes 2, HQ 0.9375, RMSE≈0.0324, confirmed_step 575)
