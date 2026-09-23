# Principal: img_radial_wedge2_transpose_vs_residual16

**Track:** application / image suite (angular half-disk wedges)

**Claim:** The published ParticleGAN image path (transpose conv G, width 12, RpGAN + b_cap + 32 particles) fails a novel upper-half vs left-half pie-disk task. The same formulation with residual nearest-neighbor upsample width 16 sustains PASS.

**Novelty:** Same radius disk, different angular support (upper half-plane vs left half-plane). Distinct from vh_bars2, mask_inpaint2, soft_ring2, sparse_obs2, intensity2, diag_ramp2, colorize_lr2, and letter/digit topology.

**Tip:** `510e0054b2499ce725482de8e7a4ae8d72fd8f25` (`origin/particle-finetune/base`)

## Results
| Arm | Verdict | Modes | HQ | mean RMSE | confirmed_step |
|-----|---------|-------|----|-----------|----------------|
| baseline_transpose12 | FAIL | 2 | 0.75 | ≈0.047 | — |
| residual16 | PASS | 2 | 0.9375 | ≈0.032 | 575 |

## Reproduce
```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  python -u reports/transfer_suite/img_radial_wedge2_transpose_vs_residual16/reproduce_arms.py
```
