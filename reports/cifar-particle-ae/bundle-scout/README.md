# Optimization bundle FID scout on GPU 1

Two sequential 5,000-step runs compare baseline with the GPU1 profiler's winning
bundle: batched real/fake discriminator forwards, exact bcap-logit reuse, and
foreach EMA. Both use the scratch encoder, N=8, batch64, seed24002, and identical
optimizer/data settings. There is no seed sweep. Both use two CPU threads.

Final ranking uses FID50k against CIFAR train50k, with TF-compatible Inception.
FID5k at step2,500 is a diagnostic; it cannot be compared directly with the final
50k score as a learning trend. Reconstruction uses 1,000 test images. Evaluation
checkpoints are retained. The pipeline generates `LEADERBOARD.md` and
`leaderboard.json`, including the measured FID difference and training speedup.

The experiment uses `experiments/train_cifar_ae_bundle.py`, a standalone snapshot
of the production trainer with a `speed_bundle` switch. This keeps the active
GPU0 experiment's source fingerprint intact. The paired baseline uses this same
snapshot, and the same thread count as the bundle. Shared CPU/desktop activity
can still affect GPU1 throughput; prioritize the paired comparison.

```sh
bash experiments/cifar_ae_bundle_pipeline.sh
tail -F runs/cifar_particle_ae/bundle_scout/PIPELINE.log
```

Before launch, CPU tests compare the actual scout helper's loss and D gradients
on ordinary and active-bcap steps, verify the reduction from four D forwards to
one on regularized updates, and check EMA parameter/buffer agreement. Separate
32-step GPU smoke runs cover both paths, reconstruction, and checkpoint saving.

The microbenchmark reported 8.6–10.8% higher throughput; this scout tests whether
quality holds over actual training. Neither outcome changes the running GPU0
trainer automatically. Once both results are certified, inspect the generation
grids and weigh the measured FID difference against saved training time before
promoting an implementation change.
