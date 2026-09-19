# Frozen particle noise sweep

| Checkpoint | Noise scale | FID50k | Density | Coverage | Latent confusion |
|---|---:|---:|---:|---:|---:|
| 80000 | 1.0 | 15.7527 | 0.6469 | 0.6426 | 0.00916% |
| 80000 | 0.75 | 15.9090 | 0.6472 | 0.6403 | 0.00000% |
| 80000 | 0.5 | 16.2898 | 0.6418 | 0.6319 | 0.00000% |
| 160000 | 1.0 | 19.0584 | 0.5904 | 0.5803 | 2.67029% |
| 160000 | 0.75 | 19.7838 | 0.5824 | 0.5697 | 0.28992% |
| 160000 | 0.5 | 20.9044 | 0.5836 | 0.5462 | 0.00000% |

Inference-only changes, not training improvements. Coverage uses 10k real/fake, k5. Same draws across scales.
