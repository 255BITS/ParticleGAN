SMOKE ONLY: FID128 is not a benchmark.

# Particle overlap checkpoint interventions

| Arm | Step | FID | Median nearest distance | Latent confusion |
|---|---:|---:|---:|---:|
| freeze_centers | 80016 | 141.8541 | 2.1637 | 0.0092% |
| reduced_noise | 80016 | 141.0991 | 2.1648 | 0.0000% |

Existing unchanged control: 80k 15.7527, 90k 15.7901, 100k 16.4609.
Same parent, learning rates, seed and objective. Freeze arm fixes live and EMA centers separately; prior optimizer state is retained but gets no updates.
Reduced-noise arm changes sigma for training and evaluation; initial sampling effects must be separated from learning.
No automatic continuation beyond 100k.
