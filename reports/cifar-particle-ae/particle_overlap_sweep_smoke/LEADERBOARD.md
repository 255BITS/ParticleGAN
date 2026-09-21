# Smoke only: reduced sample counts, not benchmark scores

# Frozen particle noise sweep

| Checkpoint | Noise scale | FID50k | Density | Coverage | Latent confusion |
|---|---:|---:|---:|---:|---:|
| 80000 | 1.0 | 141.8019 | 0.8063 | 0.9219 | 0.00000% |
| 80000 | 0.5 | 140.7719 | 0.7703 | 0.9062 | 0.00000% |

Inference-only changes, not training improvements. Coverage uses 10k real/fake, k5. Same draws across scales.
