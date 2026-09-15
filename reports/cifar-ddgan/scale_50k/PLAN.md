# Longer training and the official DDGAN generator

Two independent 50,000-update runs, one per GPU. Both start from scratch so their
training source and evaluation protocol match. GPU 0 retains the winning U-Net32;
GPU 1 uses the official NCSN++ width128 generator adapted to class conditioning
and our learned128D particles. Neither uses pretrained generator weights.

Both retain frozen ResNet18 + pixel D, joint timestep/class UCD, 20,000 particles,
Gaussian step noise, T=4, the existing posterior, Rp logistic, bcap, unique-row
VICReg, constant learning rates, batch64, EMA .995 and seed24002. No seed sweep.
The upstream architecture uses PixelNorm in its particle mapping; this removes
radial information at G input, but gradients still reach the learned particle
coordinates. The particle table and its regularizer remain unchanged.

The new generator retains upstream adaptive GroupNorm, BigGAN residual blocks,
FIR resampling, positional timestep embedding, attention at16 plus the middle
block, residual input pyramid, width multipliers[1,2,2,2], two encoder and three
decoder blocks per resolution, z embedding256, and n_mlp4. Class embeddings are
added to the processed timestep embedding. Native upstream PyTorch FIR runs on
CUDA without a custom extension. Source commit and licenses are recorded in
lib/ddgan_ncsnpp/UPSTREAM.md. This is an architecture adaptation, not an exact
reproduction of the paper's training recipe.

50k updates represent64 epoch-equivalents for each optimizer,128 total real-data
pass equivalents because G and D draw separate batches with replacement.
Evaluation:5k diagnostic samples every10k updates; final50k samples. Rank using
final50k FID and compare diagnostic curves only at matching sample counts.
Compare equal updates and GPU training time; architecture compute differs.

Configs: configs/cifar_ddgan/scale_50k/{unet32,ncsnpp128}.yaml.
Preflight:100-update runs of both full-size architectures on their target GPUs.
Full runs are left uninterrupted until both finish. No intermediate manual
metric inspection or adaptive hyperparameter changes. Defaults stay at the
previous measured10k winner until completed evidence supports promotion.

Launch:
```
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python -u experiments/follow_grid.py \
 --root results/cifar_ddgan/scale_50k --log results/cifar_ddgan/live.log \
 --runner-log results/cifar_ddgan/scale_50k.runner.log -- \
 --config_manifest configs/cifar_ddgan/scale_50k/manifest.json \
 --trainer experiments/train_cifar_ddgan.py --gpus 0,1 --workers_per_gpu 1
```
Tail: `tail -F results/cifar_ddgan/live.log`.
