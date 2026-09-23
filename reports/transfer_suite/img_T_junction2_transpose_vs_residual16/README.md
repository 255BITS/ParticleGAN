# Principal: img_T_junction2_transpose_vs_residual16

**Track:** application / image suite (T-junction orientation / structure; not geometry GMM)

**Claim:** The published ParticleGAN image path (transpose conv G, width 12, RpGAN + b_cap coeff 3 kappa 1.25 + 32 particles) fails a novel T-junction orientation multimodal task. The same formulation with an in-family architecture change — residual nearest-neighbor upsample, width 16 — sustains PASS under matched seed/budget/CPU.

**Task:** Two 8x8 grayscale modes: T pointing down (horizontal bar on top + vertical stem) vs T pointing right (vertical bar on left + horizontal stem). Product-adjacent to orientation / junction-structure discrimination. Distinct from radial_wedge2, vh_bars2, soft_ring2, intensity2, mask_inpaint2, sparse_obs2, diag_ramp2, colorize_lr2, fg_bg_invert2, yin_yang2, and digit/letter topology.

**Winner:** published image suite baseline transpose12.
**Control:** residual_upsample width 16 (same shared_c6 / RpGAN+b_cap+particles family).
**Solvability gate:** HIT only if winner FAIL and control PASS.
