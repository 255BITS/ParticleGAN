# Principal: img_dots_count23_transpose_vs_residual16

**Track:** application / image suite (discrete cardinality: 2 corner dots vs 3 triangle dots)

**Claim:** The published ParticleGAN image path (transpose conv G, width 12, RpGAN + b_cap + particles) fails a novel 2-vs-3 dot counting multimodal task. The same formulation with residual nearest-neighbor upsample width 16 sustains PASS under matched seed/budget/CPU.

**Task:** Two 8x8 grayscale modes: two 2x2 blobs at NW+SE vs three 2x2 blobs at NW+NE+S-center (triangle). Product-adjacent to discrete cardinality / sparse count discrimination. Distinct from sparse_obs2, sparse_pts_field2, blob_count2, corner_pair layouts, L_chirality2, and geometry GMM breaks.

**Winner:** published image suite baseline transpose12.
**Control:** residual_upsample width 16 (same shared_c6 / RpGAN+b_cap+particles family).
**Solvability gate:** HIT only if winner FAIL and control PASS.
