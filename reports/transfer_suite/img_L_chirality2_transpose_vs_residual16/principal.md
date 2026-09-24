# Principal: img_L_chirality2_transpose_vs_residual16

**Track:** application / image suite (discrete letter chirality / corner-joint handedness; not geometry GMM)

**Claim:** The published ParticleGAN image path (transpose conv G, width 12, RpGAN + b_cap coeff 3 kappa 1.25 + 32 particles) fails a novel L vs mirrored-L multimodal task. The same formulation with an in-family architecture change — residual nearest-neighbor upsample, width 16 — sustains PASS under matched seed/budget/CPU.

**Task:** Two 8x8 grayscale modes: canonical L (vertical stem left + bottom foot) vs mirrored L (vertical stem right + bottom foot). Product-adjacent to discrete shape handedness / chirality. Distinct from swirl_cw2 (continuous spiral field chirality), smile_frown2, radial_wedge2, T_junction2, dof_center_edge2, mask_inpaint2, soft_ring2, intensity2, and soft probes.

**Winner:** published image suite baseline transpose12.
**Control:** residual_upsample width 16 (same shared_c6 / RpGAN+b_cap+particles family).
**Solvability gate:** HIT only if winner FAIL and control PASS.
