# Principal: img_swirl_cw2_transpose_vs_residual16

**Track:** application / image suite (continuous field chirality; not geometry GMM)

**Claim:** The published ParticleGAN image path (transpose conv G, width 12, RpGAN + b_cap coeff 3 kappa 1.25 + 32 particles) fails a novel clockwise vs counterclockwise spiral-intensity multimodal task. The same formulation with an in-family architecture change — residual nearest-neighbor upsample, width 16 — sustains PASS under matched seed/budget/CPU.

**Task:** Two 8x8 grayscale modes: spiral intensity field with clockwise angular phase vs the same field with counterclockwise angular phase (`sin(±2θ + 1.1 r)`). Product-adjacent to orientation / handedness discrimination on continuous fields. Distinct from L_chirality2 (discrete mirror L), soft_ring2, intensity2, radial_wedge2, smile_frown2, dof_center_edge2, mask_inpaint2, sparse_obs2, colorize_lr2, and soft probes spokes_rings / unpaired_rot90 / edge_shade / blob_count / sparse_cols.

**Winner:** published image suite baseline transpose12.
**Control:** residual_upsample width 16 (same shared_c6 / RpGAN+b_cap+particles family).
**Solvability gate:** HIT only if winner FAIL and control PASS.
