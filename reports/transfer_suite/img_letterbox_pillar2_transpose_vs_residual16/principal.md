# Principal: img_letterbox_pillar2_transpose_vs_residual16

**Track:** application / image suite (cinematic aspect framing)

**Claim:** The published ParticleGAN image path (transpose conv G, width 12, RpGAN + b_cap + particles) fails a novel letterbox vs pillarbox multimodal task. The same formulation with residual nearest-neighbor upsample width 16 sustains PASS under matched seed/budget/CPU.

**Task:** Two 8x8 grayscale modes with matched bright content: (0) letterbox — dark horizontal bars top/bottom (widescreen framing); (1) pillarbox — dark vertical bars left/right (tall framing). Product-adjacent to video/letterbox detection and aspect-ratio conditioning. Distinct from vh_bars glyph orientation farm, geometry GMM/gauge breaks, and soft scanline_hv2 (both FAIL).

**Winner:** published image suite baseline transpose12.
**Control:** residual_upsample width 16 (same shared_c6 / RpGAN+b_cap+particles family).
**Solvability gate:** HIT only if winner FAIL and control PASS.

**Results @ tip 510e0054b2499ce725482de8e7a4ae8d72fd8f25:**
- baseline_transpose12: FAIL (modes 0, HQ 0.03125, RMSE≈0.0848, no sustain)
- residual16: PASS (modes 2, HQ 1.0, RMSE≈0.0268, confirmed_step 525)
