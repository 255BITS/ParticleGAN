# Principal: img_barcode_quiet_lr2_transpose_vs_residual16

**Track:** application / image suite (retail barcode / quiet-zone margin)

**Claim:** The published ParticleGAN image path (transpose conv G, width 12, RpGAN + b_cap + particles) fails a novel barcode quiet-zone LEFT vs RIGHT multimodal task. The same formulation with residual nearest-neighbor upsample width 16 sustains PASS under matched seed/budget/CPU.

**Task:** Two 8x8 grayscale modes representing a UPC-style bar pattern with a quiet-zone blank margin: (0) quiet zone on the **left** (blank cols 0–1, bars start at col 2); (1) quiet zone on the **right** (bars in cols 0–5, blank cols 6–7). Shared bar bit pattern `[1,0,1,1,0,1]`. Product-adjacent to retail barcode scanners and quiet-zone / margin discrimination. Distinct from sonar TOF blobs, chirp ridges, diffraction-order LR, moiré beat periods, and Manchester edge polarity.

**Winner:** published image suite baseline transpose12.
**Control:** residual_upsample width 16 (same shared_c6 / RpGAN+b_cap+particles family).
**Solvability gate:** HIT only if winner FAIL and control PASS.

**Results @ tip ac83ce32b99bd488c2dd700786c0c93f0bb1f27b:**
- baseline_transpose12: FAIL (modes 0, HQ 0.34375, RMSE≈0.0556, no sustain)
- residual16: PASS (modes 2, HQ 1.0, RMSE≈0.0282, confirmed_step 525)
