# Principal: img_vh_bars2_transpose_vs_residual16

**Track:** application / image suite (unpaired domain-translation-style toy)

**Claim:** The published ParticleGAN image path (transpose conv G, width 12, RpGAN + b_cap + 32 particles) fails a novel unpaired V↔H bar-domain task. The same formulation with an in-family architecture change — residual nearest-neighbor upsample, width 16 — sustains PASS under matched seed/budget/CPU.

**Novelty:** Two 8×8 grayscale modes with matched intensity mass but orthogonal bar orientation:
- mode0: two vertical bars (cols 1–2 and 5–6 at 0.9)
- mode1: two horizontal bars (rows 1–2 and 5–6 at 0.9)

Product-adjacent to unpaired domain translation / orientation-transfer demos. Distinct from mask_inpaint2, sparse_obs2, soft_ring2, intensity2, diag_ramp2, colorize_lr2, and geometry GMM breaks.

**Tip:** `510e0054b2499ce725482de8e7a4ae8d72fd8f25` (`origin/particle-finetune/base`)

**HIT gate:** winner (baseline_transpose12) FAIL + control (residual16) PASS on img_vh_bars2.
