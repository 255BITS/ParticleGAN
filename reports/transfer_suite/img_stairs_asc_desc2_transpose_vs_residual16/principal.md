# Principal: img_stairs_asc_desc2_transpose_vs_residual16

**Track:** application / image suite (ascending vs descending staircase polarity)

**Claim:** The published ParticleGAN image path (transpose conv G, width 12, RpGAN + b_cap + particles) fails a novel ascending-vs-descending staircase multimodal task. The same formulation with residual nearest-neighbor upsample width 16 sustains PASS under matched seed/budget/CPU.

**Task:** Two 8x8 grayscale modes: ascending L→R thick staircase vs descending L→R staircase. Product-adjacent to discrete path / polarity discrimination. Distinct from dots_count23, barcode stripes, L_chirality2, ramp/flip leftovers, and geometry GMM breaks.

**Winner:** published image suite baseline transpose12.
**Control:** residual_upsample width 16 (same shared_c6 / RpGAN+b_cap+particles family).
**Solvability gate:** HIT only if winner FAIL and control PASS.
