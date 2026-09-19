# 32k_40k training result

Certified full-state continuation; all evaluation scores FID50k.

| Step | FID50k |
|---|---:|
| 15000 | 18.5834 |
| 20000 | 17.3152 |
| 25000 | 16.9717 |
| 30000 | 16.3545 |
| 35000 | 16.2066 |
| 40000 | 16.3451 |

Existing16k at40k: 17.0982. New32k endpoint minus16k: -0.7532.

Train minutes 22.59. Best sampled point 16.2066 at35000; final 16.3451. Target<13. No automatic next training stage.
