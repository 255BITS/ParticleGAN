# Endpoint information and quality

Two new probes certified; existing4096 endpoint diagnostic reused. All checkpoints40k.

| Particles | FID50k | Bits / available | Density | Coverage | Between-sibling variance |
|---|---:|---:|---:|---:|---:|
| 4096 | 17.2350 | 1.4854 / 2 | 0.6148 | 0.6139 | 12.29% |
| 8192 | 17.7683 | 1.9164 / 3 | 0.5956 | 0.5915 | 9.49% |
| 16384 | 17.0982 | 2.6886 / 4 | 0.6160 | 0.6217 | 11.06% |

Bits are a held-out restricted-decoder estimate, not exact entropy. Review shuffled-label controls/per-parent scores in results.json. Feature distinctions can reflect artifacts; interpret with density/coverage and FID. Same cached10000-image real reference and10000generated samples,k5.
