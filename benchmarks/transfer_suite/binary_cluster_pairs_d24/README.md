# Binary-cluster pairs d24: hierarchical lengthscale trap

Two vertical pairs (intra-gap 4.0, local σ=1.0) with inter-cluster distance 24.

| Arm | Architecture | init_std | Verdict |
|-----|--------------|----------|---------|
| winner_published_absolute | batchfeat absolute kernel lengths (published gan_v3/shared_c6) | 0.5 | **FAIL** (suffix 0; collapses to 2/4 modes) |
| control_mlp_matched_init | lengthscale-free SimpleMLP | 6.0 | **PASS** (suffix 9) |

Tip: `ac83ce32b99bd488c2dd700786c0c93f0bb1f27b` (`origin/particle-finetune/base`).

Distinct from wide_gap (2 loose islands), line4 (uniform 1D), four_corner
(uniform square lattice), and N-gon / hub-spoke farms: this is a two-scale
hierarchy (fine pair structure + coarse inter-cluster gap).
