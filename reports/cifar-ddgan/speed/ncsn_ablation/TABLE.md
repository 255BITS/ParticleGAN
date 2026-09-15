| Run | GPU | Steps × batch | Steady samples/s | Train min | Peak train GiB | FID | FID samples |
|---|---:|---:|---:|---:|---:|---:|---:|
| bundle_lazy4_ncsnpp | 1 | 1000 × 64 | 212.2 | 5.11 | 14.05 | 151.632 | 5000 |
| cache_only_ncsnpp | 0 | 1000 × 64 | 208.0 | 5.21 | 14.06 | 82.942 | 5000 |
| fused_only_ncsnpp | 0 | 1000 × 64 | 198.3 | 5.46 | 14.05 | 195.833 | 5000 |
| channels_last_only_ncsnpp | 1 | 1000 × 64 | 166.0 | 6.51 | 14.05 | 331.490 | 5000 |
