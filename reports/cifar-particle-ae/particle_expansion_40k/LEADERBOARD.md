# Particle expansion persistence check

2/2 certified; both respective 20k scout endpoints continued with unchanged optimizer/EMA/RNG.

| Arm | FID25k | FID30k | FID35k | FID40k | Train minutes |
|---|---:|---:|---:|---:|---:|
| split_4096 | 17.8876 | 17.4796 | 16.5033 | 17.2350 | 14.66 |
| control_1024 | 21.8973 | 21.0193 | 21.3182 | 26.0216 | 14.43 |

Control minus expanded FID (positive favors expansion): +4.0097, +3.5396, +4.8149, +8.7866.

Expanded center count does not establish semantic coverage. Review sibling grids/variation alongside FID. No automatic promotion to 200k.
