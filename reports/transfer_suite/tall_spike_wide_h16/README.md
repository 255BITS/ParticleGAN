# tall_spike_wide_h16

Three isotropic islands on a tall isosceles spike: base tips at (±6, 0), apex at (0, 16).
Base NN = 12; legs ≈ 17.1. Narrower base than `chevron_wide_d16` (base 16); not equilateral.

## Result @ tip `510e005` (origin/particle-finetune/base)

| arm | status | notes |
|-----|--------|-------|
| winner_published_absolute (batchfeat kernel_scales [0.1,0.25,0.5,1.0]) | **FAIL** | apex mass ~3.9% vs 33%; shortfall ≈ 0.39 |
| control_mlp_matched_init (init_std = reach = 8) | **PASS** | suffix 11; mass balanced |

Fairness: same recipe/optimizer budget; control only drops absolute RBF lengths and matches prior init to reach.

Reproduce:

```bash
cd reports/transfer_suite/tall_spike_wide_h16
MKL_CBWR=AVX2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=../../.. python -u reproduce_arms.py
```
