# boomerang_wide_r16

Classic 3-mode boomerang: left wing `(-16, 8)`, apex `(0, 16)`, drooping right `(16, -2)`.
Equal-mass isotropic islands (local σ=1). Curved sparse support — not a kite, chevron,
trapezoid, arch semicircle, or island-gap layout.

## Result @ tip `510e005` (origin/particle-finetune/base)

| arm | status | notes |
|-----|--------|-------|
| winner_published_absolute (batchfeat kernel_scales [0.1,0.25,0.5,1.0]) | **FAIL** | left wing mass ~11.5% vs 33%; shortfall ≈ 0.19 |
| control_mlp_matched_init (init_std = reach = 8) | **PASS** | suffix 11; masses roughly balanced |

Fairness: same recipe/optimizer budget; control only drops absolute RBF lengths and matches prior init to reach.

Reproduce:

```bash
cd reports/transfer_suite/boomerang_wide_r16
MKL_CBWR=AVX2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=../../.. python -u reproduce_arms.py
```
