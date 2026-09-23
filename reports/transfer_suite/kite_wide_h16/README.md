# kite_wide_h16

Asymmetric kite: apex `(0, 16)`, wings `(±10, 0)`, short tail `(0, -5)`.
Four equal-mass isotropic islands (local σ=1). Not a regular polygon, diamond,
tall spike, or chevron.

## Result @ tip `510e005` (origin/particle-finetune/base)

| arm | status | notes |
|-----|--------|-------|
| winner_published_absolute (batchfeat kernel_scales [0.1,0.25,0.5,1.0]) | **FAIL** | apex mass ~3.3% vs 25%; shortfall ≈ 0.18 |
| control_mlp_matched_init (init_std = reach = 8) | **PASS** | suffix 8; masses roughly balanced |

Fairness: same recipe/optimizer budget; control only drops absolute RBF lengths and matches prior init to reach.

Reproduce:

```bash
cd reports/transfer_suite/kite_wide_h16
MKL_CBWR=AVX2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=../../.. python -u reproduce_arms.py
```
