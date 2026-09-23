# trapezoid_wide_h16

Isosceles trapezoid: short top at `(±8, 16)`, wide base at `(±16, 0)`.
Top NN = 16; base NN = 32; height = 16; legs ≈ 17.1. Local σ = 1.0 unscaled.

## Result @ tip `510e005` (origin/particle-finetune/base)

| arm | status | notes |
|-----|--------|-------|
| winner_published_absolute (batchfeat kernel_scales [0.1,0.25,0.5,1.0]) | **FAIL** | right base dropped (mass% ~50/37/13/0); shortfall ≈ 0.66 |
| control_mlp_matched_init (init_std = reach = 8) | **PASS** | suffix 9; mass roughly balanced |

Fairness: same recipe/optimizer budget; control only drops absolute RBF lengths and matches prior init to reach.

Distinct from `kite_wide_h16` (apex+wings+tail), `tall_spike_wide_h16` (triangle), `four_corner_wide_g8` (square), `diamond_wide_r16` (plus arms), `chevron_wide_d16` (acute V).

Reproduce:

```bash
cd reports/transfer_suite/trapezoid_wide_h16
MKL_CBWR=AVX2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=../../.. python -u reproduce_arms.py
```
