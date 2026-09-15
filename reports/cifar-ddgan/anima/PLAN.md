# Anima block transplant, constant learning rates

Branch: `experiment/anima-transplant`. The user requested an experimental
generator transplant and explicitly rejected adding learning-rate decay.

## Question

Can a small frozen slice of a pretrained image denoiser improve our existing
pixel-space ParticleGAN DDGAN? Compare pretrained and random **frozen** donor
weights with the same architecture, initialization of all trainable layers,
seed, batch, sample budget, optimizer recipe, and evaluation protocol.

Anima-Base v1.0 source:
https://huggingface.co/circlestone-labs/Anima/tree/f973fc41ec7545364ac9776c2440285f43ff2a30

The inspected checkpoint contains 28 width-2048 transformer blocks. Each has
69,206,528 parameters, including its timestep modulation. We select the first
two contiguous blocks, retaining their complete self-attention, cross-attention,
MLP, Q/K normalization, and timestep modulation. The common timestep embedder
is also loaded. Only 310.4 MB of tensor data is fetched, without the remaining
blocks, text adapter/encoder, VAE, or original output head. Downloading and
hashing is implemented in `experiments/prepare_anima_transplant.py`.

The donor's weights carry the upstream model's non-commercial license; this
branch is a research experiment. No donor weights are committed to Git.

## Architecture

The plain U-Net's last encoder feature map is 128×8×8. The new branch is:

```
h = existing_UNet_encoder(xt, particle, class, time)
tokens = trainable_projection(h)               # 64 tokens × 2048 channels
context = trainable_context(particle, class)   # four × 1024 channels
features = frozen_Anima_blocks(tokens, context, donor_time)
h = h + trainable_output(LayerNorm(features))  # output starts exactly zero
clean = existing_UNet_decoder(h, skips, conditioning)
xt_minus_1 = existing_DDGAN_reverse(clean, xt, time, Gaussian_noise)
```

The existing image path and skip connections remain. The side branch preserves
the 8×8 grid and expands channels. Learned particles feed both the existing
U-Net modulation and the new cross-attention context. The original text
conditioning path is replaced; this is not an Anima inference pipeline or a
flow-matching finetune. All donor parameters stay frozen, but gradients pass
through donor operations into trainable input/context adapters.

Donor timestep inputs use `sqrt(1-alpha_bar)/(sqrt(alpha_bar)+sqrt(1-alpha_bar))`
to match the signal/noise ratio with the donor's flow-time convention. This is
only an embedding lookup: the existing DDGAN corruption and posterior sampler
are unchanged. Frozen timestep modulation is precomputed for the discrete
timesteps. Rotary positions use the Cosmos single-frame split-half convention
on the 8×8 grid. No resolution stretching or learned positional interpolation.

Frozen donor matrix operations use bfloat16; donor residuals, layer normalization,
trainable U-Net/adapters, D, and losses use float32. No global mixed-precision
change to the discriminator's double backward. Optimizers exclude frozen
parameters. EMA skips immutable parameters instead of repeatedly interpolating
two identical 155M-parameter donor copies.

## Runs and promotion rule

1. Both GPUs: 128-update runtime profiles. These are correctness/throughput
   checks; ten-sample FID is not a quality result.
2. Both GPUs: 1k-update scouts, pretrained versus frozen random, batch64,
   FID5k. One worker per GPU, no seed-only experiments.
3. Promote a viable promising result to 10k updates with final FID50k. Run
   both arms if their cost is practical, to preserve the transfer comparison.
   Compare quality **and wall time** against the historical plain/attention
   baselines; a larger transplant is not automatically a useful improvement.

All runs retain four DDGAN steps, learned 20k×128 latent particles, Gaussian
step noise, joint time/class UCD, Rp logistic, VICReg, exact lazy-4 bcap,
cached frozen ResNet18 D features, fused Adam, constant learning rates, EMA .995.
The no-argument generator remains the plain U-Net.

```
tail -F results/cifar_ddgan/anima.live.log
```

Full configurations: `configs/cifar_ddgan/anima_{profile,1k,10k}/*.yaml`.
Sources/configs/environments and completion certificates are captured by the
existing grid runner. No source changes while runs are active. Future promotion
is conditional on completed results, not queued automatically by this document.

## Completed screen and promotion decision

Both 128-update profiles completed successfully. Both real-weight GPU smoke
checks passed: exact initial output match with the original U-Net, active
image/particle/input/context-adapter gradients after two updates, and immutable
frozen donor parameters. The CPU tests passed (50 initially plus the corrected
default-YAML equality check; 13 subtests). The only initial failure was that the
default YAML had not yet been synchronized with the optional config keys.

The paired 1k scouts completed: pretrained FID5k67.514 in 2.00 training minutes;
random FID5k74.227 in 2.06 minutes. Historical plain U-Net at this budget was
68.455. Steady throughput: 541.3 / 525.1 samples/s. Both used 2.63 GiB peak
training allocation. The small early gain over plain U-Net is not sufficient
to justify doubled cost, but the matched pretrained-versus-random improvement
supports the planned 10k comparison. Both 10k configs were launched unchanged
after the completed scouts; final evaluation uses 50k generated images.
