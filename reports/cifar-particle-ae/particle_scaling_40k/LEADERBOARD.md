# Larger particle-count continuation

2/2 new runs certified. Existing4096 trajectory reused as benchmark.

| Particles | FID25k | FID30k | FID35k | FID40k | Train minutes |
|---|---:|---:|---:|---:|---:|
| 4096 | 17.8876 | 17.4796 | 16.5033 | 17.2350 | 14.66 |
| 8192 | 17.5680 | 17.6849 | 17.7297 | 17.7683 | 14.43 |
| 16384 | 17.3601 | 17.5677 | 17.4324 | 17.0982 | 14.77 |

All runs continue their respective20k checkpoints with unchanged rates, optimizer/EMA/RNG and oneD update. FID50k uses the established protocol; no seed repeats or automatic promotion past40k.
