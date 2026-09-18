# Direct CIFAR particle autoencoder baseline

Prespecified before the first pilot, 2026-09-17. Worktree `feature/mog-autoencoder`.

Question: does adding bounded particle reconstruction improve unconditional
CIFAR-10 generation, and does its code carry image-specific information?

## Matched arms

- `gan`: direct decoder `G(p[k] + sigma * GaussianNoise)`; uniform particle draws.
- `bounded`: same sampling and GAN objective, plus
  `E(X) -> (query, offset) -> p[nearest(query)] + sigma * 3*tanh(offset/3) -> G`.
  Add pixel mean squared reconstruction error with coefficient 1 in [-1,1].

The only between-arm configuration differences are arm and output directory.
All models (including the unused control encoder) are initialized in the same
order, with hashes recorded. Separate identical data/prior RNG streams; D and G
receive independent real minibatches. The reconstruction arm uses G's existing
real batch. Both arms therefore consume identical data, flips, and prior draws.
One shared seed 24002; no seed-only experiments.

## Architecture and optimization

Unconditional CIFAR-10 at native 32x32. Labels are unused. Direct generator takes
only the latent code, projects to 4x4, then upsamples through three residual
blocks (widths 128,64,32); no image input, diffusion, or encoder skip connections.
Reuse the CIFAR ResBlock implementation. The existing pretrained ResNet18 feature
critic is adapted to one scalar output with constant zero context and one dummy
class/time index. Frozen feature weights/BN statistics stay frozen; no real image
information passes through context. Reuse the feature extractor and adversarial
objective, without a classification auxiliary objective.

1024 learned particles in 64 dimensions. Differentiable per-coordinate prior
standardization; sigma = .025 times the initial median nearest-neighbor distance,
calibrated once and saved as a buffer. Gaussian sampling offsets; deterministic
bounded encoding offsets. No KL, oracle supervision, usage balancing, or offset
distribution matching. Sigma itself never learns.

Encoder: three stride-2 convolutions with GroupNorm, then separate query and
offset linear heads. Query uses non-affine per-example LayerNorm to fix its
initial scale; offset head starts at zero. Unlike the toy there is no known
spatial input skip. Hard nearest-particle forward, global softmax surrogate
backward to query; only selected rows receive the direct reconstruction gradient
(prior standardization also couples rows). Softmax uses *mean squared* latent
distance / .125. This equals the toy .25 summed-distance temperature in 2D and
fixes per-coordinate units in the new dimension. Matrix distances avoid B*K*D
storage. This is a starting protocol, not a tuned or proven optimum.

Rp logistic GAN; exact bcap coefficient/kappa 1, lazy every fourth D update at
4x weight; full-table particle VICReg weight 1. Adam G/E .0006, D .0009, betas
(0,.999); prior .006, betas(.5,.999). Fused Adam, float32/TF32, constant rates.
Batch 64; 10,000 updates. EMA .995 for G/E/prior, evaluated together. Final
checkpoint only; no best-checkpoint selection. GPU numerical nondeterminism is
possible, even with matched initial state and RNG streams.

## Budget and stages

First queue two 200-update pilot configs on GPUs 0 and 1, without FID; evaluate
512 test reconstructions and save sample grids. Check finite losses, update
throughput, memory, source/config integrity, matched initialization/RNG hashes,
and frozen sigma/feature weights. Pilot verifies plumbing and cost, not quality.
Then queue the two 10k scouts if projected aggregate GPU time fits approximately
30–60 minutes including evaluation. Each process has a 30-minute training cap;
a cap failure does not emit a completed-run summary. No automatic promotion to
50k, architecture sweep, or oracle training.

## Evaluation

- Generation: existing cached CIFAR train50k / torch-fidelity Inception FID.
  Diagnostic FID5k at 2500/5000/7500 updates; final FID50k at 10k. Compare only
  equal sample counts. Report Inception covariance trace / real trace as a coarse
  spread diagnostic, not a coverage certificate. Random sample grids are unlabeled.
- Reconstruction: all 10k **test** images, no augmentation. Pixel MSE in [-1,1],
  PSNR from aggregate MSE, per-image p90/p99. EMA encoder/decoder/prior together.
  Report hard used/effective particles, hard-uniform and hard-soft usage TV,
  offset RMS/saturation and per-particle conditional offset mean RMS.
- With the same checkpoint, zero offsets, replace offsets with Gaussian noise,
  and shuffle selected particle IDs across the held-out dataset while retaining
  each image's offset. Save errors/IDs/counts and grids (columns input, predicted,
  zero, random, shuffled). These ablations need no extra training.
- Report training and total runtime separately, peak allocated GPU memory, prior
  movement, and fixed-state integrity. Evaluation uses dedicated RNGs.

Rank final scouts by FID50k ascending, alongside reconstruction, usage, and cost.
The historical DDGAN FID is context only: it uses labels, diffusion, another
generator, and a different particle table. A poor direct baseline would make
this comparison inconclusive about the broader method; it is not evidence that
the established DDGAN quality transfers to this new setup.

## Queue and logs

From the feature worktree, using the existing queue and log follower:

```sh
/home/martyn/dev/ParticleGAN/.venv/bin/python -u experiments/follow_grid.py \
  --root runs/cifar_particle_ae/scout --log runs/cifar_particle_ae/scout.live.log -- \
  --configs 'configs/cifar_particle_ae/scout/*.yaml' --gpus 0,1 --workers_per_gpu 1 \
  --python /home/martyn/dev/ParticleGAN/.venv/bin/python \
  --trainer experiments/train_cifar_particle_ae.py
tail -F runs/cifar_particle_ae/scout.live.log
```

Use `pilot` in place of `scout` for the pilot queue. Explicitly request one
process per GPU (the runner defaults to five). The runner verifies config, source and summary digests before marking
a job complete or reusing it; failed attempts are preserved. Each job saves
flushed logs, metrics, exact config, source archive/hashes, full checkpoint with
optimizer/RNG state, and a completion certificate. Exact resume is not implemented
in this new trainer; rerunning an unfinished config starts a preserved new attempt.
