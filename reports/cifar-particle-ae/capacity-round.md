# Capacity and duration round

Target: CIFAR-10 FID50k below 13. Historical width32 baseline: 19.611 at 10k,
20.105 at 20k, 19.439 at 30k. See [audited curve](lazy-long/README.md).

| GPU | Configuration | Training | Generator parameters |
|---:|---|---|---:|
| 0 | width32, depth1, N=8 | Resume 30k → 100k | 645,123 |
| 1 | width64, depth1, N=8 | Fresh 20k | 2,291,715 |
| 1 | width64, depth2, N=8 | Fresh 20k, queued after width64 | 4,072,963 |

Only G capacity changes in the scouts. D/E width32, scratch encoder, frozen
pretrained discriminator features, reconstruction weight, optimizers and lazy
bcap settings stay fixed. Depth2 adds one residual block at each of the three
output resolutions. All arms use seed24002 as a configuration control; no seed
sweeps. D/E initialization is identical across the generator variants.

The duration run restores the 30k checkpoint's models, EMA, both optimizers,
training random streams and CPU/CUDA RNG states. Checkpoint hash:
`b7e5e974be8053514cb04fcff1d9baa11e43a3e2d6c6dcf3649289b9c3a007af`.
The parent checkpoint and historical trainer remain intact. A new standalone
trainer preserves historical source certificates. The performance bundle is
not used, following its worse 5k FID scout.

FID50k and reconstruction on all 10k test images run every 10k updates, including
the final checkpoint. Checkpoints are retained. New invocation time and cumulative
training time are recorded separately. The duration track should take roughly
one hour based on the prior 22 updates/s; capacity timing will be estimated from
the running scouts. Per-run training cap: two hours, excluding evaluation.

Validation: six tests passed, including deterministic full-state replay against
uninterrupted training and D/E initialization invariance. Real 32-update smoke
runs passed for both capacity variants and the original 30k checkpoint; resumed
optimizer states advanced to update30032. Grid dry-runs and shell syntax passed.
Replay testing uses deterministic settings; normal training retains the previous
fast CUDA settings, so it does not promise bitwise replay under production kernels.

Launch:
```bash
bash experiments/cifar_ae_capacity_pipeline.sh duration_100k
bash experiments/cifar_ae_capacity_pipeline.sh capacity_scout
```

Follow both tracks:
```bash
tail -F runs/cifar_particle_ae/{duration_100k,capacity_scout}/PIPELINE.log
```

Each pipeline writes a certified leaderboard, FID curve table, timings and
recommendations when its queue finishes:

- [Duration report](duration_100k/LEADERBOARD.md)
- [Capacity report](capacity_scout/LEADERBOARD.md)

Compare capacity scouts at 20k against the historical 20k reference. Compare the
duration curve against its actual 30k starting point. Favor further training
when FID improvement and cost support it; reconstruction improvement alone is
insufficient. Review both tracks before selecting the next long run.
