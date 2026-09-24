# Principal: img_dof_center_edge2_transpose_vs_residual16

**Track:** application / image suite (depth-of-field / focus-plane; not geometry GMM)

**Claim:** The published ParticleGAN image path (transpose conv G, width 12, RpGAN + b_cap coeff 3 kappa 1.25 + 32 particles) fails a novel sharp-center vs sharp-edge multimodal task. The same formulation with an in-family architecture change — residual nearest-neighbor upsample, width 16 — sustains PASS under matched seed/budget/CPU.

**Task:** Two 8x8 grayscale modes: a bright sharp center disk with soft falloff (center focus) vs a bright outer ring with dark center (edge / peripheral focus). Product-adjacent to depth-of-field / focus-plane discrimination. Distinct from soft_ring2, intensity2, ramp_corner2, diag_ramp2, radial_wedge2, smile_frown2, mask_inpaint2, sparse_obs2, colorize_lr2, T_junction2, vh_bars2, and soft probes X_plus / L_chirality / tri_ud / shear_lr / hourglass_bowtie / diag_bar_orient / halfmoon / blob_count.

**Winner:** published image suite baseline transpose12.
**Control:** residual_upsample width 16 (same shared_c6 / RpGAN+b_cap+particles family).
**Solvability gate:** HIT only if winner FAIL and control PASS.
