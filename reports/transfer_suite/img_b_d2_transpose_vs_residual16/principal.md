# Principal: img_b_d2_transpose_vs_residual16

**Track:** application / image suite (letter b vs d mirror chirality)

**Claim:** The published ParticleGAN image path (transpose conv G, width 12, RpGAN + b_cap + particles) fails a novel letter-b-vs-d multimodal task. The same formulation with residual nearest-neighbor upsample width 16 sustains PASS under matched seed/budget/CPU.

**Task:** Two 8x8 grayscale modes: letter b (stem left + bottom bowl) vs letter d (stem right + bottom bowl). Product-adjacent to glyph / OCR-lite chirality discrimination. Distinct from L_chirality2, smile_frown2, T_junction2, stairs_asc_desc2, K_lr2 soft, and geometry GMM breaks.

**Winner:** published image suite baseline transpose12.
**Control:** residual_upsample width 16 (same shared_c6 / RpGAN+b_cap+particles family).
**Solvability gate:** HIT only if winner FAIL and control PASS.
