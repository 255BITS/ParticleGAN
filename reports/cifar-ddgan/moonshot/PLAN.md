# Two architecture moonshots, 2026-09-14

User requested two distinct 10k-update moonshots on both GPUs, retaining joint
UCD, ParticleGAN and the shared train_denoising.py formulation. This supersedes
the previously queued 30k class-vs-joint confirmation. Additional scouts are
authorized if results warrant them. No seed-only repetitions.

## First-principles bets

1. **pretrained_d / GPU 0:** retain width32 U-Net G and add a frozen ImageNet
   ResNet18 feature branch to the existing width32 GroupNorm pixel critic.
   The pixel branch can model noisy transitions; learned feature heads can use
   visual structure without having to discover it from scratch. Candidate and
   xt are resized to 64, normalized with ImageNet mean/std without clamping,
   processed through frozen eval-mode layer1/2/3, and concatenated per stage.
   Each trainable projection head returns40 joint logits. Combined logits are
   `(pixel + sum(feature_heads)/sqrt(3))/sqrt(2)`. This is one critic and one
   existing loss, not an added perceptual loss. bcap differentiates the entire
   combined critic with respect to candidate pixels only. Frozen extractor
   state is hashed in environment.json and saved in checkpoints.
   This is inspired by Projected GAN, not a reproduction. Pretraining uses
   external ImageNet data; do not label this a from-scratch result.
2. **flat_particle_g / GPU 1:** retain width32 GroupNorm pixel D and replace G
   with a six-block,128-channel local/global hybrid. Lossless pixel unshuffle
   converts3x32x32 into12x16x16; a pointwise projection expands to128 channels.
   All blocks retain 256 tokens x128 channels. Each has depthwise local mixing,
   four-head global attention and a2x expanding MLP with residual connections.
   No pooling, token merging, compressed image latent, or class token.
   A learned128D particle supplies global class/time modulation and a separate
   Linear->16x16x16 spatial path, expanded to128 channels. Output projects to12
   channels and pixel-shuffles toRGB. Tanh predicts cleanx0 as before.
   The image path has no intermediate dimensional bottleneck; normalization
   and learned transforms are not claimed to be mathematically invertible.
   Multiple changes form this moonshot: follow-up ablations would be required
   to credit spatial injection versus attention/width/local mixing.

## Controls and evaluation

Both use20,000 learned128D particles, fresh particle per reverse step,
Gaussian diffusion-step noise,T4 with alpha_bar=[1,.9,.5,.05,.0001], batch 64,
seed 24002, constant G/D/priorLR=.0006/.0009/.006, Adam(0,.999), EMA.995,
Rp logistic, bcap1/kappa1, UCD CE.02, unique-particle VICReg1, horizontal flips.
Generator target, posterior, sampler and all training losses stay unchanged.
Joint40-way UCD selects `(t-1)*10+c`; neither t nor c enters D features.

Full configs: configs/cifar_ddgan/moonshot/{pretrained_d,flat_particle_g}.yaml.
100-step GPU smoke configs are in moonshot_smoke; their20-sample FIDs have no
quality interpretation. Scouts train10k updates with5k-sample diagnostic FID
at 10k and final50k-sample FID. Keep checkpoints for analysis. Training speed
and total wall time must accompany quality; these are equal-update, not
equal-compute comparisons. A50k FID avoids confusing an update budget with
sample-count bias. Historical width32 class-only10k finalFID62.819; there is
not yet a matched constant-LR joint-UCD10k control. Do not attribute changes
solely to architecture without that control.

Tail: `tail -F results/cifar_ddgan/live.log`.

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python -u experiments/follow_grid.py \
 --root results/cifar_ddgan/moonshot --log results/cifar_ddgan/live.log \
 --runner-log results/cifar_ddgan/moonshot.runner.log -- \
 --config_manifest configs/cifar_ddgan/moonshot/manifest.json \
 --trainer experiments/train_cifar_ddgan.py --gpus 0,1 --workers_per_gpu 1
```

Sources: [Projected GAN](https://arxiv.org/abs/2111.01007),
[TransGAN](https://arxiv.org/abs/2102.07074),
[ResNet18 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).

## Adaptive follow-up, after first pair completed

Final50k FIDs at 10k updates: pretrained_D31.741, flat_particle_G52.156.
Both completed/certified; first pair 21.6min wall time. Pretrained D also gives
visibly more coherent object/animal structure. Proceed with a matched joint
control (GPU0) and combined flat_G+pretrained_D (GPU1), each10k updates,
configs/cifar_ddgan/moonshot_followup/manifest.json. No source changes between
pairs. This fills a 2x2 architecture comparison at one seed, not a seed sweep.
