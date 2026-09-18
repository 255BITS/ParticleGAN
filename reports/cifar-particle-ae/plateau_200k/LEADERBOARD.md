# Two-D continuation — stopped by user

200k was not completed. Stopped to test capacity growth using both GPUs.

| Step | FID50k | Test MSE |
|---:|---:|---:|
| 70000 | 19.1510 | 0.04069 |
| 80000 | 18.8932 | 0.04077 |
| 90000 | 19.1343 | 0.03992 |
| 100000 | 19.1420 | 0.03952 |
| 110000 | 18.8515 | 0.04121 |
| 120000 | 19.2600 | 0.03901 |
| 130000 | 18.3879 | 0.03868 |
| 140000 | 19.0329 | 0.03829 |
| 150000 | 18.7748 | 0.03782 |
| 160000 | 18.4886 | 0.03763 |
| 170000 | 18.3010 | 0.03781 |

Best evaluated FID: 18.3010 at 170000. Last logged update: 172100. Saved checkpoints retained.

The run remained around FID 18–19 despite additional discriminator updates. Capacity scouts are the next test; this curve does not establish the cause of the plateau.
