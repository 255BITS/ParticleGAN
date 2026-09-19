# Wider deconv with GroupNorm, 16k particles

| Step | Wide + GN FID50k | Small no norm FID50k | Difference |
|---|---:|---:|---:|
| 5000 | 50.4709 | 77.2284 | -26.7575 |
| 10000 | 29.9273 | 46.6522 | -16.7249 |
| 15000 | 22.7725 | 37.6181 | -14.8456 |
| 20000 | 22.3330 | 33.3280 | -10.9950 |
| 25000 | 22.0368 | 30.6990 | -8.6622 |
| 30000 | 20.9390 | 28.8087 | -7.8697 |
| 35000 | 20.9987 | 27.0096 | -6.0109 |
| 40000 | 18.2285 | 26.2232 | -7.9946 |

Best sampled 18.2285 at 40000; final 18.2285. Training minutes 24.84.

Matched scratch prior initialization, D/E initialization, shared sigma, recipe and evaluation steps. This tests width plus normalization jointly and cannot separate their individual effects. No automatic promotion queued.
