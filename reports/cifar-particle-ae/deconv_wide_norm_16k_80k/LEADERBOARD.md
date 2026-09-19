# Wider GroupNorm deconv 16k, continuation to 80k

| Step | FID50k |
|---|---:|
| 5000 | 50.4709 |
| 10000 | 29.9273 |
| 15000 | 22.7725 |
| 20000 | 22.3330 |
| 25000 | 22.0368 |
| 30000 | 20.9390 |
| 35000 | 20.9987 |
| 40000 | 18.2285 |
| 45000 | 17.4961 |
| 50000 | 19.3232 |
| 55000 | 18.7246 |
| 60000 | 18.5205 |
| 65000 | 18.4515 |
| 70000 | 18.6665 |
| 75000 | 17.8848 |
| 80000 | 17.6565 |

Best sampled: 17.4961 at 45000; final: 17.6565. Continuation train minutes: 24.93.

Full-state resume, unchanged recipe and constant learning rates. No further stage queued. Historical CNN best FID50k15.7527 at80k uses a different prior initialization history.
