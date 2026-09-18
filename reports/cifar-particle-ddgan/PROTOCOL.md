# CIFAR direct / DDGAN particle autoencoder comparison

Four arms, one shared seed24002, 10,000 updates, batch64. No seed sweep.
Each architecture has a GAN control and a bounded particle autoencoder+GAN.
The encoder arms are particle AE-GAN and DDGAN + particle AE. Their encoders
are deterministic; neither learns a stochastic posterior or optimizes a KL,
ELBO, or aggregate posterior matching objective.

```
E(X) -> (query, u) -> nearest particle k
z_X = p[k] + fixed_sigma * 3*tanh(u/3)
Direct reconstruction: G(z_X) -> X_hat
DDGAN reconstruction: G(z_X, X_t, t) -> X_hat
Generation: uniform particle k + sigma * Gaussian noise; no encoder
```

The first DDGAN encoder deliberately sees only clean X, with the same encoder
architecture as the direct arm. Its code is shared across noise levels.
DDGAN trains adversarial reverse transitions with randomly sampled prior codes;
the separate reconstruction forward uses the encoded code and mean clean-image
MSE, coefficient1, averaged over uniformly sampled t=1..4. Reconstruction does
not replace the adversarial prior draw. The critic sees (X_(t-1), X_t, t).
The noise schedule is alpha_bar=[1,.9,.5,.05,.0001]. Generation takes four reverse
steps with independent MoG codes and Gaussian transition noise.

All arms are unconditional (no CIFAR labels), K1024, z64, width32, sigma_rel.025
calibrated once (sigma .2126164287). Prior means use differentiable per-coordinate
standardization; full-table raw particle VICReg coefficient1. Hard nearest
forward/global soft routing backward temperature.125 in mean-squared units.
Fixed-sigma offsets remain bounded to +/-3 during encoding; no usage balancing.

G/E Adam .0003, D .00045, prior .003; G/E/D betas(0,.999), prior(.5,.999).
These rates are half the previous direct scout's rates, selected before results
because both previous arms regressed late. ConstantLR, fusedAdam, EMA.995.
Rp logistic loss, exact bcap coefficient/kappa1 with lazy4. Frozen ResNet18 feature
critic plus pixel branch, GroupNorm. DDGAN uses four time heads, direct one head;
no auxiliary classification loss. G is the existing small residual U-Net for
DDGAN and the existing latent-only upsampling generator for direct.
Thus encoder/no-encoder comparisons within an architecture isolate the encoder
objective; cross-architecture rankings compare systems, not equal parameters or
sampling compute. Historical class-conditioned DDGAN has different prior,
regularization, labels and learning rates and is listed only as context.

Modules, including unused encoder, initialize in identical order within each
architecture pair. Data/prior/time/corruption/reverse RNGs are separate from
evaluation and are matched within pairs. Data and prior draws also match across
architectures. Two fresh real batches per update; iid CIFAR train50k with random
horizontal flips. Evaluation uses EMA G/E/prior and test images without flips.

Queue: existing follow_grid/run_grid, one worker per GPU on GPUs0,1.
200-step no-FID pilots precede full runs. Each run has a 30-minute training cap;
no automatic promotion, resume, or longer training. Numbered full checkpoints
are retained at2500/5000/7500/10000, checkpoint.pt links to latest. Exact completed
config/source fingerprints are reusable; failed attempts are archived by runner.

Primary leaderboard: FID50k versus CIFAR train50k at exactly10k updates, no best
checkpoint selection. Also FID5k at every saved checkpoint (including final;
final5k uses first5k images of50k), feature covariance trace/real, training and
total runtime, and peak CUDA allocated memory. Standard cached Inception protocol
in lib/cifar_metrics.py; no new real feature statistics or downloads.

Direct reconstruction: all10k CIFAR test images, MSE[-1,1], PSNR, offset and
particle ablations, usage/offset statistics. DDGAN: all10k test images, clean
prediction MSE separately at each t with fixed X_t across code ablations.
Ablations replace code with prior, shuffled whole encoded code, selected center,
or shuffled particle retaining the original offset. Baseline DDGAN reports prior
code prediction MSE. DDGAN denoising MSE is not latent-only autoencoder MSE.
Variation follow-up holds noisy input and timestep fixed while varying latent;
noise injected into the code is an inference probe, not a learned posterior.

Tail combined log from this worktree:

```
tail -F runs/cifar_particle_ddgan/scout.live.log
```

Run command:

```
/home/martyn/dev/ParticleGAN/.venv/bin/python -u experiments/follow_grid.py \
 --root runs/cifar_particle_ddgan/scout --log runs/cifar_particle_ddgan/scout.live.log -- \
 --configs 'configs/cifar_particle_ddgan/scout/*.yaml' --gpus 0,1 --workers_per_gpu 1 \
 --python /home/martyn/dev/ParticleGAN/.venv/bin/python \
 --trainer experiments/train_cifar_particle_ddgan.py
```

After training, numerical variation is evaluated separately on the two DDGAN
checkpoints and the direct encoder checkpoint. No image grids are inspected or
written by this audit. Example (use a fresh output directory):

```
CUDA_VISIBLE_DEVICES=0 /home/martyn/dev/ParticleGAN/.venv/bin/python -u \
 experiments/measure_cifar_ddgan_variation.py \
 --run runs/cifar_particle_ddgan/scout/ddgan_bounded \
 --out runs/cifar_particle_ddgan/variation/ddgan_bounded
```

The audit uses 512 held-out inputs and eight draws each. Encoded codes receive
0/.5/1 times sigma Gaussian noise without clipping. DDGAN also gets eight prior
codes per input. Each noisy image and timestep is fixed across draws/conditions;
the outputs measured are clean predictions before reverse-transition noise.
Noisy-input hashes must match across the two DDGAN checkpoints. Exact uint8
uniqueness, pair pixel RMSE, normalized Inception feature cosine distance,
reconstruction MSE, and nearest deterministic-reconstruction retrieval are saved
per input. Retrieval does not establish semantic identity or calibrated posterior
coverage. No end-to-end encoded DDGAN reverse-chain reconstruction is claimed.

The report analyzer verifies completion certificates, config/source fingerprints,
checkpoint hashes, per-image saved metrics, paired initialization/training RNGs,
and matched noisy inputs before publishing tables and learning curves:

```
/home/martyn/dev/ParticleGAN/.venv/bin/python experiments/analyze_cifar_particle_ddgan.py
```
