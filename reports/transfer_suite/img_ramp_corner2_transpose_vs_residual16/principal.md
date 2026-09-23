# Principal: img_ramp_corner2_transpose_vs_residual16

**Track:** application / image suite (lighting-direction corner ramp; not geometry GMM)

**Claim:** The published ParticleGAN image path (transpose conv G, width 12, RpGAN + b_cap coeff 3 kappa 1.25 + 32 particles) fails a novel NW-vs-SE intensity corner-ramp multimodal task. The same formulation with an in-family architecture change — residual nearest-neighbor upsample, width 16 — sustains PASS under matched seed/budget/CPU.

**Task:** Two 8x8 grayscale modes: brightness falling off from the NW corner vs falling off from the SE corner. Product-adjacent to lighting-direction / shading discrimination. Distinct from diag_ramp2 (edge diagonal ramp), intensity2, soft_ring2, smile_frown2, mask_inpaint2, sparse_obs2, colorize_lr2, T_junction2, radial_wedge2, vh_bars2, and the soft probes halfmoon_lr2 / C_open_lr2 / wave_phase2 / ring_gap_tb2 / S_soft2 / blob_count2.

**Winner:** published image suite baseline transpose12.
**Control:** residual_upsample width 16 (same shared_c6 / RpGAN+b_cap+particles family).
**Solvability gate:** HIT only if winner FAIL and control PASS.
