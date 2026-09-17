# Particle autoencoder continuation handoff

## Current status: first CIFAR pair complete (2026-09-17)

The user authorized a basic matched CIFAR image experiment and a config queue
using both GPUs. Implemented and completed; no more training is queued.

- Read [CIFAR report](../reports/cifar-particle-ae/README.md) and its protocol.
- Direct unconditional CIFAR32, K1024, latent64, fixed sigma0.212616; same
  initialization/data/prior draws. GAN versus bounded particle autoencoder+GAN,
  10k updates each, one shared seed24002. Existing queue used unchanged with
  `--workers_per_gpu 1 --gpus 0,1`; combined log under
  `runs/cifar_particle_ae/scout.live.log`.
- Final FID50k: GAN81.411, bounded33.266. Training7.99 vs8.61min (+7.8%).
  Main pair total including eval21.2 summed process minutes,10.9 concurrent
  wall minutes. Peak5.32GiB including Inception.
- Important: both regress late. Same-count FID5k at7500->10000:
  GAN25.951->85.361; bounded25.267->37.918. Final-checkpoint read-only audit
  confirms this is not an evaluation sample-count artifact. Do not describe
  the large endpoint margin as an established stable generation advantage.
- Bounded testMSE.063365,PSNR18.00; blurry reconstructions. Zero-offset
  MSE.364833 (5.76x); shuffledparticle.073319 (+15.7%). Offsets carry much of
  the information, unlike the toy. Used363/effective109.5 of1024; offsetRMS2.04,
  11.3% saturation; no aggregate matching to the sampling Gaussian.
- Recommend next: same pair and budget, halve all learning rates to test
  stability; retain numbered checkpoints (currently only latest survives).
  This recommendation has not been launched. Continuous-encoder control is
  useful after baseline stability. Keep oracle training deferred.
- Scripts: `train_cifar_particle_ae.py`, `analyze_cifar_particle_ae.py`,
  `audit_cifar_particle_ae.py`; model `lib/image_particle_autoencoder.py`;
  configs `configs/cifar_particle_ae/{pilot,scout}`. Runs/checkpoints/source
  archives are in `runs/cifar_particle_ae/`; reports contain portable results.
- 62 tests +13 subtests pass. Analyzer verifies certificates, matched hashes,
  fixed sigma/frozen D features, complete budgets and reconstruction errors.
  First bounded audit terminated code143, preserved then evaluation-only retry
  succeeded. Two short pilots sharedGPU0 accidentally; full scouts used0/1.
- Main worktree source remains untouched. Continue in this feature worktree.

## Previous direction (implemented by the CIFAR round above)

User requested a commit and pause for compaction before further experiments.
Next: test the scalable bounded particle autoencoder + GAN on simple images,
against a matched MoG GAN with reconstruction disabled. No image experiment
has started. Joint oracle-supervised GAN training is deferred.

```text
Reconstruct: E(X) -> (k, u) -> p[k] + fixed_sigma * 3*tanh(u/3) -> G -> X_hat
Generate:   uniform k + Gaussian noise -> p[k] + fixed_sigma * noise -> G -> X_new
Train:      reconstruction + GAN objective + existing particle regularizer
```

This is a particle autoencoder + GAN, without a per-example VAE posterior or KL.
Hard selection uses nearest particle to an encoder query; a soft routing
surrogate supplies the query gradient. The baseline uses the original global
surrogate, not the later local-routing, balancing, or oracle additions.
Routing still costs O(batch × particles × latent dimension); it is not free,
but avoids exhaustive comparisons in image space.

Choose dataset, latent dimension, image architecture, loss scaling, budget, and
image-appropriate quality/diversity metrics before launching. A small digit-image
task is a candidate, not a locked protocol. Inspect existing image infrastructure
and available data first. Match G/D/prior, initialization, data streams, and update
budgets across arms. Calibrate sigma once for the new prior, then keep it fixed.
Record held-out reconstruction, generation quality/diversity, hard particle usage,
offset RMS and zero/random-offset ablations, runtime, and peak memory. Keep flushed
logs and publish a leaderboard plus interpretation. No seed-only experiments.

## Completed evidence

- Nine-arm 2D 100-Gaussian scout: bounded offsets give 92/100 covered modes,
  82.23% HQ, reconstruction MSE 0.002775. Best HQ/reconstruction; local routing
  reaches 93 modes with worse quality. Other metrics have different winners.
- Exhaustive zero-offset oracle: decode all 400 centers and pick the closest
  output to each observed X. Bounded encoder MSE 0.00280830 versus oracle
  0.00188026 on 100k held-out examples. Exact only among current center outputs;
  not an optimum over offsets, future decoders, or unconditional generation.
- Frozen encoder fitting: two 6,000-update arms from the bounded checkpoint,
  with G/prior/sigma frozen, zero offsets, fresh Adam, and identical continuation
  data. Oracle query regression gives MSE 0.00209430: 25.42% improvement and
  76.94% of the available gap closed. Reconstruction control gives 0.02755613;
  6.247% wrong-grid cases account for 94.10% of its error. Generation is unchanged.
- Oracle fitting demonstrates learnability of much of the selection gap. It
  does not establish image-scale performance. Oracle search was cheap here
  because 400 tiny decoded centers could be cached with G frozen.

## Workspace and artifacts

- Worktree: `/home/martyn/dev/ParticleGAN-mog-autoencoder`
- Branch: `feature/mog-autoencoder`; leave the main worktree untouched.
- Python: `/home/martyn/dev/ParticleGAN/.venv/bin/python`
- Commits: `40a55a7` initial scout; `9235c3b` local routing; `bb6cc0a` oracle audit;
  `c04615d` frozen encoder fitting. This handoff is committed separately.
- Reports: [scout](../reports/mog-autoencoder/README.md),
  [oracle](../reports/mog-autoencoder/ORACLE.md),
  [encoder fitting](../reports/mog-autoencoder/encoder-fit/README.md).
- Runs/checkpoints: `runs/mog_autoencoder/scout/` and
  `runs/mog_autoencoder/encoder_fit/` (gitignored; retained locally).
- Latest logs: `runs/mog_autoencoder/encoder_fit.console.log` and
  `runs/mog_autoencoder/encoder_fit.audit.log`; use `tail -F`.
- Existing image entry points include `experiments/train_cifar_ddgan.py`,
  `lib/image_ddgan.py`, and `lib/cifar_metrics.py`; suitability for a simple-image
  particle autoencoder has not yet been assessed.
- Latest code verification: 38 tests pass across `test_mog_encoder_fit.py`,
  `test_mog_oracle.py`, `test_mog_autoencoder.py`, `test_mog.py`, `test_mog_api.py`.
  Frozen state hashes, original checkpoint integrity, matched continuation RNG,
  complete budgets, and saved-source hashes were verified. No runs are active.
