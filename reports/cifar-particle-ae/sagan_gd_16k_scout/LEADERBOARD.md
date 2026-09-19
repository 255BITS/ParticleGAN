# SAGAN-style attention in G and D, 16k particles

| Step | Attention FID50k | No-attention wide GN FID50k | Difference |
|---|---:|---:|---:|
| 5000 | 48.7583 | 50.4709 | -1.7126 |
| 10000 | 35.7416 | 29.9273 | +5.8143 |
| 15000 | 29.7097 | 22.7725 | +6.9372 |
| 20000 | 26.0541 | 22.3330 | +3.7212 |
| 25000 | 21.0066 | 22.0368 | -1.0303 |
| 30000 | 22.2370 | 20.9390 | +1.2980 |
| 35000 | 20.4561 | 20.9987 | -0.5425 |
| 40000 | 19.6425 | 18.2285 | +1.4140 |

Best sampled 19.6425 at 40000; final 19.6425. Training minutes 28.42.

Same scratch particle initialization and existing base G/D/E weights, sigma and recipe. Attention is active from initialization with a fixed unit residual coefficient. This jointly changes G and D; it does not isolate which side helps. SAGAN-style attention adaptation, not a reproduction of the original paper. No automatic promotion.
