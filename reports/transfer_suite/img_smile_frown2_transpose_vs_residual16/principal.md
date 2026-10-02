# Principal: img_smile_frown2_transpose_vs_residual16

**Track:** application / image suite (expression / curve chirality; not geometry GMM)

**Claim:** The published ParticleGAN image path (transpose conv G, width 12, RpGAN + b_cap coeff 3 kappa 1.25 + 32 particles) fails a novel smile-vs-frown soft-curve multimodal task. The same formulation with an in-family architecture change — residual nearest-neighbor upsample, width 16 — sustains PASS under matched seed/budget/CPU.

**Task:** Two 8x8 grayscale modes: soft smile curve (parabola with bright band lower in the center) vs soft frown curve (parabola with bright band higher / upper on screen). Product-adjacent to expression / curve-chirality discrimination. Distinct from T_junction2, radial_wedge2, vh_bars2, soft_ring2, intensity2, mask_inpaint2, sparse_obs2, diag_ramp2, colorize_lr2, yin_yang2, U_orient2, arrow2, zigzag2, pacman2, and digit/letter topology.

**Winner:** published image suite baseline transpose12.
**Control:** residual_upsample width 16 (same shared_c6 / RpGAN+b_cap+particles family).
**Solvability gate:** HIT only if winner FAIL and control PASS.
