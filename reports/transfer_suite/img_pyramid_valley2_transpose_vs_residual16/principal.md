# Principal: img_pyramid_valley2_transpose_vs_residual16

**Track:** application / image suite (radial intensity polarity)

**Claim:** The published ParticleGAN image path (transpose conv G, width 12, RpGAN + b_cap + particles) fails a novel bright-center pyramid vs dark-center valley multimodal task. The same formulation with residual nearest-neighbor upsample width 16 sustains PASS under matched seed/budget/CPU.

**Task:** Two 8x8 grayscale modes on a shared disk support: (0) radial pyramid — intensity peaks at center and falls with radius; (1) radial valley — intensity dark at center and rises with radius. Product-adjacent to photometric / shading polarity and depth-from-shading cues. Distinct from glyph chirality breaks (b_d2, L_chirality2, stairs, swirl), geometry GMM/gauge breaks, and soft shade_lr2 (both FAIL).

**Winner:** published image suite baseline transpose12.
**Control:** residual_upsample width 16 (same shared_c6 / RpGAN+b_cap+particles family).
**Solvability gate:** HIT only if winner FAIL and control PASS.
