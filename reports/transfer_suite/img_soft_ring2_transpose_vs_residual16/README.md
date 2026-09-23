# Principal: img_soft_ring2_transpose_vs_residual16

**Track:** application / image suite (not geometry GMM)

**Claim:** The published ParticleGAN image path (transpose conv G, width 12, RpGAN + b_cap coeff 3 kappa 1.25 + 32 particles) fails a novel soft-intensity ring-vs-disk task. The same formulation with an in-family architecture change — residual nearest-neighbor upsample, width 16 — sustains PASS under matched seed/budget/CPU.

**Novelty:** Soft annular ring at intensity 0.4 (hollow interior) vs filled disk at 0.85 with the same outer bbox. Combines topology and photometric fidelity. Distinct from stock `img_intensity2` (uniform central patches at 0.35/0.85) and from binary ring/disk (which both arms pass).

This is NOT a diffusion/non-GAN baseline. Both arms stay inside ParticleGAN RpGAN + b_cap + particles. Batchfeat/shared_c6 vector D does not wire into the image suite; the documented published path for images is the transpose12 baseline.

**Tip:** `510e0054b2499ce725482de8e7a4ae8d72fd8f25` (`origin/particle-finetune/base`)

**HIT gate:** winner (baseline_transpose12) FAIL + control (residual16) PASS on img_soft_ring2.

## Reproduce

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  python -u reports/transfer_suite/img_soft_ring2_transpose_vs_residual16/reproduce_arms.py
```

(Python ≥3.10 with torch; `conceptmod` conda env on pop-os.)
