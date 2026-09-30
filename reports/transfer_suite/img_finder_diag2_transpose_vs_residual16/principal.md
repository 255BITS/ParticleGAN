# Principal: img_finder_diag2_transpose_vs_residual16

**Track:** application / image suite (QR-like finder pairing)

**Claim:** The published ParticleGAN image path (transpose conv G, width 12, RpGAN + b_cap + particles) fails a novel QR-like finder-pattern diagonal-pairing multimodal task. The same formulation with residual nearest-neighbor upsample width 16 sustains PASS under matched seed/budget/CPU.

**Task:** Two 8x8 grayscale modes with matched mass: (0) finder squares at TL+BR (main diagonal); (1) finder squares at TR+BL (anti-diagonal). Each finder is a 3x3 bright ring with dark center (classic QR finder motif). Product-adjacent to barcode/QR localization and landmark pairing. Distinct from corner_pair_diag2 soft (both PASS), T-junction / letterbox farms, and geometry GMM breaks.

**Winner:** published image suite baseline transpose12.
**Control:** residual_upsample width 16 (same shared_c6 / RpGAN+b_cap+particles family).
**Solvability gate:** HIT only if winner FAIL and control PASS.

**Results @ tip 510e0054b2499ce725482de8e7a4ae8d72fd8f25:**
- baseline_transpose12: FAIL (modes 0, HQ 0.09375, RMSE≈0.0669, no sustain)
- residual16: PASS (modes 2, HQ 0.96875, RMSE≈0.0313, confirmed_step 575)
