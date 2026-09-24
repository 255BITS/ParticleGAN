# Principal: img_traffic_stack_rg2_transpose_vs_residual16

**Track:** application / image suite (traffic-light vertical order)

**Claim:** The published ParticleGAN image path (transpose conv G, width 12, RpGAN + b_cap + particles) fails a novel traffic-light bright-top vs bright-bottom multimodal task. The same formulation with residual nearest-neighbor upsample width 16 sustains PASS under matched seed/budget/CPU.

**Task:** Two 8x8 grayscale modes with shared housing column: (0) bright lamp on TOP of three stacked sockets (red-on-top order as intensity); (1) bright lamp on BOTTOM (green-on-top order as intensity). Mid sockets stay dim. Product-adjacent to status stacks, vertical order cues, and UI traffic/alert indicators. Distinct from wifi_bars height counts, EQ mid/hi boosts, and geometry GMM breaks.

**Winner:** published image suite baseline transpose12.
**Control:** residual_upsample width 16 (same shared_c6 / RpGAN+b_cap+particles family).
**Solvability gate:** HIT only if winner FAIL and control PASS.

**Results @ tip 510e0054b2499ce725482de8e7a4ae8d72fd8f25:**
- baseline_transpose12: FAIL (modes 1, HQ 0.5, RMSE≈0.0581, no sustain)
- residual16: PASS (modes 2, HQ 1.0, RMSE≈0.0193, confirmed_step 425)
