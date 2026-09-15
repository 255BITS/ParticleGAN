# CIFAR moonshots: pretrained D produces the step change

At 10,000 updates, adding pretrained visual features to D lowers final FID from
54.931 to **31.741**, a **42.2% reduction** against the matched joint-UCD control.
Training takes 20.05 rather than 10.25 minutes. This is the selected fast baseline.
All four full runs completed and were certified. Two sequential GPU pairs took
21.6 and 27.4 minutes. No seed sweep or failed full runs.

| Rank | Generator | Discriminator | FID, 50k samples | Training min | Total min |
|---|---|---|---:|---:|---:|
| 1 | U-Net 32 | Pixel + pretrained ResNet18 features | **31.741** | 20.05 | 21.50 |
| 2 | Flat particle hybrid 128 | Pixel + pretrained ResNet18 features | 38.719 | 25.49 | 27.31 |
| 3 | Flat particle hybrid 128 | Pixel | 52.156 | 16.56 | 18.42 |
| 4 | U-Net 32 | Pixel | 54.931 | 10.25 | 11.73 |

Every row uses joint timestep/class UCD, 20,000 learned 128D latent particles,
Gaussian step noise, four reverse steps, constant learning rates, and the same
DDGAN/ParticleGAN losses. Batch 64 and seed 24002 are fixed. These compare equal
updates, not equal compute. The pretrained branch uses external ImageNet data;
this is not a from-scratch result. Production CUDA kernels are not bitwise
deterministic, and there are no repeat-seed confidence intervals.

The 5k-sample diagnostic FIDs were35.630 /42.975 /56.881 /58.554 in rank order.
Use the 50k results above for ranking. Historical class-only UCD at30k updates
achieved43.678; the new winner is better even at one-third the updates, although
that historical comparison also changes UCD and D architecture.

## What changed and what we learned

**Pretrained D:** frozen eval-mode ResNet18 through layer 3 supplies three feature
scales. Candidate and xt are bilinearly resized to 64, normalized with ImageNet
statistics without clamping, and processed by the same frozen extractor.
Trainable heads concatenate candidate/xt features and produce 40 joint logits.
Their combined logits are added to the original pixel critic, then the same
Rp logistic, real/fake joint CE, and candidate-only bcap are applied. There is
no added perceptual/reconstruction loss. Frozen features retain input gradients
and support bcap double backward. The checkpoint feature hash exactly matches
startup, including batch-normalization buffers, after 10k updates.

**Flat particle G:** lossless2x2 pixel rearrangement expands12 channels to128;
all six local-convolution/global-attention blocks retain 256 tokens. No pooling,
token merging, compressed image latent, or intermediate narrow image path.
Particles enter through both global modulation and a spatial16x16 feature map.
This G alone improves FID by only 5.1% while costing61.6% more training time. With
pretrained D it is 22.0% worse in FID than the U-Net and 27.2% slower. It remains a
config-selectable research option, but does not earn promotion at this budget.
This is an architecture package, not an isolated test of spatial injection.

The grids agree with the ranking: pretrained-D/U-Net samples have clearer object
structure, birds, faces and vehicles. Animal anatomy and fine detail remain
imperfect. Global FID does not independently measure requested-class accuracy.

## Particle and gradient diagnostics

Read-only probes use 256 training images and the non-EMA checkpoints. Holding xt,
class and time fixed, changing the sampled particle gives these clean-output MSEs:

| Model | t=1 | t=2 | t=3 | t=4 |
|---|---:|---:|---:|---:|
| U-Net / pixel control | .00074 | .00203 | .00782 | .01825 |
| U-Net / pretrained D | .00158 | .00370 | .00943 | .01386 |
| Flat G / pixel | .00672 | .03066 | .15314 | .28187 |

Particles affect the trained outputs in every case. Much greater particle
sensitivity in flat G does not by itself imply better quality or diversity.
This does not isolate the benefit of learning particles versus Gaussian or fixed
priors. Candidate-gradient norm at t3 rises from .314 in the control to .838
with pretrained D. That is consistent with a stronger intermediate-noise
learning signal, not causal proof. Full per-step probes, including combined,
are in [probes](probes).

## Promotion and reproduction

`configs/cifar_ddgan/default.yaml` and trainer DEFAULTS now exactly match the
measured31.741 recipe except output directory: **10k updates**, U-Net 32,
`d_backbone: pretrained_resnet18`, `ucd_target: time_class`, constant LR,
G/D/particle rates.0006/.0009/.006, learned particles, Gaussian step noise.
No-argument command: `.venv/bin/python experiments/train_cifar_ddgan.py`.
The previous 30k default budget is shortened to the measured fast baseline.
Historical repository configs explicitly select `d_backbone: pixel` so the
new inherited default cannot silently alter old experiments.

First pair configs are in `configs/cifar_ddgan/moonshot`; matched control and
combined are in `configs/cifar_ddgan/moonshot_followup`. Each contains complete
YAMLs and a manifest. The saved run config/source archive is authoritative for
strict resume; changing defaults after completion changes the source hash.
Current code follows the same training equations. No long training is queued.

Review [comparison.png](comparison.png), [TABLE.md](TABLE.md), per-run exported
configs/metrics/provenance/certificates, [PLAN.md](PLAN.md), and
[validation.txt](validation.txt). Checkpoints and source archives remain in
ignored `results/cifar_ddgan/` directories. Tail log:
`tail -F results/cifar_ddgan/live.log`.

Next useful bet: scale or deepen the retained U-Net against this stronger D,
then separately test spatial particle injection without replacing its multiscale
image path. Longer training can establish the new ceiling. More pretraining
variants are plausible, but current evidence does not call for a transformer
or MoE migration. These are options, not a queued experiment plan.

Related primary sources: [Projected GAN](https://arxiv.org/abs/2111.01007),
[TransGAN](https://arxiv.org/abs/2102.07074),
[official ResNet18 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
These runs are adaptations to the shared particle DDGAN formulation, not
reproductions of those papers.
