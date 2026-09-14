# CIFAR-10 U-Net DDGAN baseline

Run from the repository root. Install the `images` extra with a torchvision build
compatible with your CUDA torch. This machine uses torch 2.14.0+cu130,
torchvision 0.29.0+cu130, torch-fidelity 0.3.0 and scipy 1.17.1. SciPy <1.18 is
required by torch-fidelity 0.3.0's matrix-square-root call.

```sh
.venv/bin/python experiments/train_cifar_ddgan.py --prepare-data
.venv/bin/python experiments/train_cifar_ddgan.py
```

No arguments loads `configs/cifar_ddgan/default.yaml`: width32/UCD with the
selected GroupNorm discriminator and toy rates (G .0006, D .0009, particles .006).
It now uses30k updates, constant LR, joint timestep/class UCD and FID every10k.
Joint UCD is a promoted candidate from the toy scout; full CIFAR quality is not
yet measured. The best measured FID43.678 used class-only UCD at30k constant LR.
See [READOUT.md](READOUT.md) for completed results; these are starting defaults,
not a claim of strong CIFAR quality. Each run needs a fresh
`out_dir`; existing training is protected against accidental overwrite.

To run the prepared class-only versus joint-UCD comparison on both GPUs:

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python -u experiments/follow_grid.py \
  --root results/cifar_ddgan/joint_ucd \
  --log results/cifar_ddgan/live.log \
  --runner-log results/cifar_ddgan/joint_ucd.runner.log -- \
  --config_manifest configs/cifar_ddgan/joint_ucd/manifest.json \
  --trainer experiments/train_cifar_ddgan.py --gpus 0,1 --workers_per_gpu 1
```

Follow training in another terminal:

```sh
tail -F results/cifar_ddgan/live.log
```

The two configs hold seed, width32, update count, batch size and objectives fixed,
and compare class-only versus joint timestep/class UCD. No seed sweep.
Each receives30,000 updates at batch64;
D and G each use independently drawn real batches. Sampling is with replacement.
Training data gets random horizontal flips; FID reference images are unaugmented.

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
Real/fake cross-entropies have coefficient .02. The bcap
penalty differentiates only the candidate image, with class/time/xt fixed.
Shared repo loss/regularizer implementations preserve the toy Rp logistic,
bcap coefficient/kappa 1, unique-row latent VICReg 1, Adam (0,.999), LR .0006,
D multiplier1.5, prior multiplier10 and EMA.995. `lr_floor: 1` keeps all rates
constant; setting it below1 enables delayed cosine decay. No reconstruction/diffusion MSE was
added. Training uses float32 with TF32 enabled for throughput.

## Evaluation and artifacts

Every10,000 updates: EMA5k-sample diagnostic FID, sample grid, full checkpoint.
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

CUDNN benchmarking/TF32 permit small numerical nondeterminism; this is not a
bitwise determinism claim. Resume restores the planned schedule; changing the
training horizon is not an exact continuation and is currently rejected.

The original DDGAN [paper](https://arxiv.org/abs/2112.07804) reports 3.75 FID for
unconditional CIFAR-10. This experiment uses labels and a different architecture
and training recipe. Its FID protocol is explicit, but equivalence to that
published evaluation has not been established. Global FID also does not by
itself verify requested-class correctness. Inspect class grids; the training
critic is not an independent classifier.
