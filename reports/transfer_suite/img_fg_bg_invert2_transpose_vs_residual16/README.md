# Break: img_fg_bg_invert2 — transpose12 FAIL / residual16 PASS

Tip: `510e0054b2499ce725482de8e7a4ae8d72fd8f25` (`origin/particle-finetune/base`)

| Arm | Role | Verdict | Modes | HQ | mean RMSE | confirmed_step |
|-----|------|---------|-------|----|-----------|----------------|
| baseline_transpose12 | published image path | FAIL | 1 | 0.46875 | ≈0.062 | — |
| residual16 | in-formulation control | PASS | 2 | 1.0 | ≈0.027 | 550 |

## Reproduce

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  python -u reports/transfer_suite/img_fg_bg_invert2_transpose_vs_residual16/reproduce_arms.py
```

CPU-only. Solvability gate: same GAN formulation under residual16 PASSes.
