# chevron_wide_d16

Three isotropic islands on a sharp chevron (boomerang): tips at (±8, 8), apex at (0, −16).
Base gap = 16 (HIT-family spacing); legs ≈ 25.3 (acute apex — not equilateral).

## Result @ tip `510e005` (origin/particle-finetune/base)

| arm | status | notes |
|-----|--------|-------|
| winner_published_absolute (batchfeat kernel_scales [0.1,0.25,0.5,1.0]) | **FAIL** | apex mass collapsed (~0.3% vs 33%); shortfall ≈ 0.69 |
| control_mlp_matched_init (init_std = reach = 8) | **PASS** | suffix 9; mass balanced |

Fairness: same recipe/optimizer budget; control only drops absolute RBF lengths and matches prior init to reach.

Reproduce:

```bash
cd break_tests/chevron_wide_d16
MKL_CBWR=AVX2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u reproduce_arms.py
```
