# Break: img_intensity2 transpose12 vs residual16

**Track:** application / image suite (not geometry GMM)

**Tip:** `510e0054b2499ce725482de8e7a4ae8d72fd8f25` (`origin/particle-finetune/base`)

## Principal

The published ParticleGAN image path (transpose-conv G, width 12, RpGAN + b_cap coeff 3 / kappa 1.25 + 32 particles, seed 0, 1 CPU thread) **fails** healthy ranking task `img_intensity2` (central patch intensity 0.35 vs 0.85).

The **same formulation** with an in-family architecture change — residual nearest-neighbor upsample, width 16 — **sustains PASS** on the same task under matched budget/seed/CPU.

Batchfeat / shared_c6 vector D does not wire into the image suite; the documented published path for images is the transpose12 baseline in `benchmarks/transfer_suite/image_tasks.py` / `image_solvability.py`.

## Arms (seed 0, CPU)

| Arm | Role | Verdict | modes | HQ | confirmed_step | mean RMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| baseline_transpose12 | published winner path | **FAIL** | 0/2 | 0.0% | — | 0.165 |
| residual16 | in-formulation arch control | **PASS** | 2/2 | 100% | 575 | 0.021 |

## Solvability

HIT = winner FAIL + in-formulation control PASS. Residual16 stays inside ParticleGAN RpGAN + b_cap + particles; this is not a diffusion/non-GAN baseline.

## Reproduce

```bash
cd ParticleGAN
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  python -u reports/transfer_suite/img_intensity2_transpose_vs_residual16/reproduce_arms.py
```

Use Python ≥3.10 with torch (union types in particlegan).
