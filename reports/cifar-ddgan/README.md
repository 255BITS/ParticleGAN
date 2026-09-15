# CIFAR-10 U-Net DDGAN baseline

Latest completed round: the [Anima transplant](anima/READOUT.md) reaches
FID50k 30.263 at 10k, versus32.550 for its matched frozen random control. It
takes19.98 training minutes and does not beat the faster attention U-Net29.327.
The experiment lives on `experiment/anima-transplant`; no defaults promoted.
The user explicitly wants constant learning rates; the decay proposal is withdrawn.
All six new runs completed; both GPUs are free.

The preceding round: the fast plain U-Net reaches **FID 25.397 at 50k
updates in 45.73 training minutes**. Attention improves the 10k scout to
29.327, but finishes 50k at 26.334 in 53.09 minutes. Its diagnostic FID is best
at 30k and then worsens. See [attention and duration results](attention_duration/READOUT.md).
No-argument training keeps the plain U-Net and 10k budget; optional attention
is config controlled. Both GPUs are free; no runs are queued.

The [preceding capacity round](capacity/READOUT.md) found no improvement from
widening G or deeper ResNet34 features. The [speed round](speed/READOUT.md)
promoted exact lazy-4 bcap, frozen-condition feature caching and fused Adam.
That faster recipe is now validated at 50k, versus the historical every-step
50k result 26.680 in 99.34 training minutes. See the [handoff](RUNBOOK.md).

Run from the repository root. Install the `images` extra with a torchvision build
compatible with your CUDA torch. This machine uses torch 2.14.0+cu130,
torchvision 0.29.0+cu130, torch-fidelity 0.3.0 and scipy 1.17.1. SciPy <1.18 is
required by torch-fidelity 0.3.0's matrix-square-root call.

```sh
.venv/bin/python experiments/train_cifar_ddgan.py --prepare-data
.venv/bin/python experiments/train_cifar_ddgan.py
```

No arguments loads `configs/cifar_ddgan/default.yaml`: the measured 10k-update
U-Net32 / pretrained-feature-D recipe, FID50k31.555 in9.22 training minutes.
It uses exact bcap every fourth D update at4×weight, batch-local frozen xt
feature caching and fused Adam. FD remains an optional experimental method. Joint timestep/class UCD, learned particles, Gaussian step noise,
constant LR and toy optimizer rates remain. See [moonshot/READOUT.md](moonshot/READOUT.md)
for the four-way architecture comparison, timing and limitations. Each run needs
a fresh out_dir; existing training is protected against accidental overwrite.

Full configs/manifests are in configs/cifar_ddgan/moonshot and moonshot_followup.
Both architecture rounds are completed; reruns require fresh output directories.
Speed-round configs are under configs/cifar_ddgan/speed_*. Historical combined log:

```sh
tail -F results/cifar_ddgan/speed.live.log
```

The architecture scouts used10,000 updates, batch 64 and seed 24002; no seed sweep. D and G use
independent real batches sampled with replacement. Training images receive
horizontal flips; the FID reference is unaugmented.

## Architecture and objective

```
clean = G(xt, z, t, class)
xt-1  = A[t]*clean + B[t]*xt + sqrt(posterior_var[t])*GaussianNoise
```

Four steps use `alpha_bar: [1, .9, .5, .05, .0001]`. G is a residual U-Net at
32/16/8/4 resolution, with nearest-neighbor upsampling, skip concatenation,
GroupNorm, and affine modulation from latent/time/class embeddings. Output is
RGB in [-1,1] via tanh. The latent dimension is 128; 20,000 learned rows are
sampled uniformly and independently at each reverse step. Step noise and the
initial noisy image are fresh image-shaped Gaussians.

D receives concatenated candidate/xt images and uses residual
convolutions with per-image GroupNorm (`d_norm: group`). Joint UCD emits40 logits
without a timestep embedding; `(t-1)*10+c` selects the score and CE target.
`ucd_target: class` retains explicit time embeddings and ten class logits.
`d_norm: none` retains the original unnormalized architecture. D conditioning
is additive when present, and normalization uses no batch statistics.
The promoted `d_backbone: pretrained_resnet18` adds frozen ImageNet ResNet18
features (layers 1/2/3 at 64px input) and trainable joint-UCD heads to this pixel
branch. Combined logits receive the same losses and bcap. Features use frozen
batch-normalization statistics and preserve candidate derivatives. Select
`d_backbone: pixel` with `cache_condition: false` for the original critic; class-only/concat require that mode.
`d_backbone: pretrained_resnet34` optionally uses deeper frozen ImageNet V1
features with the same stage widths, input size, trainable heads and caching.
The controlled G-width and D-backbone comparison is recorded in
[capacity/PLAN.md](capacity/PLAN.md).
`architecture: flat_hybrid` enables the tested constant-token generator; its
width/depth/head count/spatial particle channels are config controlled.
`architecture: ncsnpp` selects the official DDGAN generator architecture with our
class/particle interface. Its full width128 config is in
`configs/cifar_ddgan/scale_50k/ncsnpp128.yaml`; source/adaptations are recorded in
`lib/ddgan_ncsnpp/UPSTREAM.md`. No pretrained generator weights are used.
`g_attn_resolutions: [8, 16]` optionally adds residual spatial attention after
encoder and decoder blocks at those U-Net resolutions, using `g_heads` heads.
The default empty list preserves the plain U-Net. Attention starts as an
identity and retains skip paths and particle conditioning. The config option
requires `architecture: unet`; see the completed attention comparison above.

Real/fake cross-entropies have coefficient .02. The bcap
penalty differentiates only the candidate image, with class/time/xt fixed.
Shared repo loss/regularizer implementations preserve the toy Rp logistic,
bcap coefficient/kappa 1, unique-row latent VICReg 1, Adam (0,.999), LR .0006,
D multiplier1.5, prior multiplier10 and EMA .995. `lr_floor: 1` keeps all rates
constant; setting it below1 enables delayed cosine decay. No reconstruction/diffusion MSE was
added. Training uses float32 with TF32 enabled for throughput.

## Evaluation and artifacts

Every 10,000 updates: EMA5k-sample diagnostic FID, sample grid, full checkpoint.
At completion: EMA 50k-sample FID. Generation is balanced over requested classes.
Grid rows: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
Each grid reuses fixed evaluation RNG seeds. Evaluation uses separate generators
and cannot consume training RNG streams. FID extraction disables TF32 regardless
of training precision. Real 50k CIFAR train statistics are cached by protocol.
Generated images are clamped and rounded to uint8 before torch-fidelity's
TensorFlow-compatible Inception preprocessing. Small-sample FID is a progress
indicator and must not be compared numerically with a 50k benchmark as if equal.

Each output contains `config.yaml`, `source.zip`, source hashes, package/GPU
versions, `fid_protocol.json`, `metrics.jsonl`, `samples.png`, numbered grids,
`checkpoint.pt`, and final `summary.json`. The grid runner adds a completion
certificate. Checkpoints store G/D/prior, EMA G/prior, both Adam states, all
named RNG streams, global CPU/CUDA RNG states, step and training time. Data
sampling has no hidden loader state. Resume requires identical source and config. The completed checkpoints precede
default promotions: restore their matching `source.zip` and use the saved
`config.yaml` before
using the resume command below. The runner will reject changed source rather
than silently claim exact continuation:

```sh
.venv/bin/python experiments/train_cifar_ddgan.py \
  --config results/cifar_ddgan/normalized_d/toy_lr/config.yaml \
  --resume results/cifar_ddgan/normalized_d/toy_lr/checkpoint.pt
```

CUDNN benchmarking, TF32 and CUDA resize gradients permit numerical nondeterminism; this is not a
bitwise determinism claim. Resume restores the planned schedule; changing the
training horizon is not an exact continuation and is currently rejected.

The original DDGAN [paper](https://arxiv.org/abs/2112.07804) reports 3.75 FID for
unconditional CIFAR-10. This experiment uses labels and a different architecture
and training recipe. Its FID protocol is explicit, but equivalence to that
published evaluation has not been established. Global FID also does not by
itself verify requested-class correctness. Inspect class grids; the training
critic is not an independent classifier.
