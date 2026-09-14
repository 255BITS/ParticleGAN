# Agreed next experiments after compact

User accepted this order after discussing pretrained D, spatial particle
injection, and training duration/annealing. This supersedes the previous
width32-versus-width64 recommendation. User authorized committing the baseline
and launching this schedule pair on both GPUs on 2026-09-14.

## First: 30k updates, cosine versus constant LR

Full validated configs: `configs/cifar_ddgan/schedule_30k/cosine.yaml` and
`constant.yaml`; manifest in the same folder. Both start fresh from the same
seed24002. Use both GPUs, one run each. No seed sweep.

Both retain the winning width32 U-Net / GroupNorm D, UCD, learned20k x128 prior,
Gaussian step noise, T4 and the exact train_denoising.py adversarial diffusion
formulation. Same batch64, G LR .0006, D LR .0009, prior LR .006, bcap1/kappa1,
VICReg1, CE .02, Adam(0,.999), EMA .995. No architecture or objective changes.

- Cosine: steps30000, lr_anneal_start .6, lr_floor .05. Decay begins after18k.
- Constant: steps30000, lr_floor1.0, which already disables decay for all groups.
- Only lr_floor and out_dir differ between the pair.
- Keep diagnostic5k FID every2k, final50k FID, class-row grids and checkpoints.
- Compare at matched updates and elapsed training time; prior10k winner is a
  duration reference with a different planned annealing horizon.
- At10k, each particle is sampled about32 times for G training on average;
  at30k about96. These are sampling counts, not counts of Adam momentum updates.
- Training updates (`steps`) are separate from diffusion T, which stays4.

Launch after the user resumes experiments:

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python -u experiments/follow_grid.py \
  --root results/cifar_ddgan/schedule_30k \
  --log results/cifar_ddgan/live.log \
  --runner-log results/cifar_ddgan/schedule_30k.runner.log -- \
  --config_manifest configs/cifar_ddgan/schedule_30k/manifest.json \
  --trainer experiments/train_cifar_ddgan.py --gpus 0,1 --workers_per_gpu 1
```

Tail: `tail -F results/cifar_ddgan/live.log`.
Estimate roughly35–40 minutes including frequent FID; measure actual throughput.
The analyzer now reports update and sample counts from the actual configs. Do not edit trainer,
runner or lib while a certified grid is active. No trainer changes are needed
for this schedule experiment. Strict resume cannot extend the10k horizon.

## Second: give particles a spatial path into G

Currently z goes through an MLP and contributes global scale/shift modulation in
every G block; it is not directly a spatial feature map. First test an added
projection z -> [channels,4,4], concatenated at the U-Net bottleneck before the
decoder, retaining encoder skips and existing modulation. Keep128-dimensional
particles initially to isolate injection. The shared posterior and D stay fixed.

Then test smaller particles (e.g.16/32 dimensions) separately. Smaller dimension
also changes VICReg covariance pressure, so don't attribute gains solely to
latent capacity. xt and Gaussian reverse noise still provide stochasticity.
StyleGAN3 Fourier input refers to spatial coordinate features with latent-
controlled transforms, distinct from Fourier-encoding z. Consider coordinates
only after the simple learned spatial projection.

## Third: pretrained discriminator features

Consider frozen pretrained image features plus trainable xt/time conditioning
and ten UCD heads. Preserve candidate-only bcap through the frozen extractor
and the existing Rp loss. Feature weights can be frozen while input gradients
remain enabled. Noisy transitions may not match clean-image pretraining, and
bcap adds computation through the backbone. This is a hypothesis to test.

Sources reviewed:
- https://arxiv.org/abs/2111.01007 (Projected GANs)
- https://github.com/NVlabs/stylegan3/blob/main/training/networks_stylegan3.py

Width scaling and transformers remain later options. No further experiment
configs beyond the schedule pair are implemented yet.
