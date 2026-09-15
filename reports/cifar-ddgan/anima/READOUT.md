# Frozen Anima weights help the transplant, but attention remains better value

Six runs completed successfully on both GPUs: two runtime profiles, two 1k
scouts, and two 10k validations. The pretrained transplant reaches **FID50k
30.263** at 10k, versus **32.550** for the same architecture with frozen random
weights. This supports useful transfer in this particular configuration. It
does not beat the historical attention U-Net's 29.327, and costs almost twice
as much training time. Keep the plain U-Net defaults and retain this experiment
on `experiment/anima-transplant`. No further runs are active or queued.

## Leaderboard at 10k updates

All rows use batch64, 640,000 samples per optimizer, and final FID with 50,000
generated images against the same CIFAR-train reference. Historical rows have
different source provenance; the two transplant rows are the controlled pair.

| Generator | FID50k ↓ | Training min | Total min | Steady samples/s |
|---|---:|---:|---:|---:|
| Attention U-Net, historical | **29.327** | **10.73** | 12.44 | 998.2 |
| U-Net + frozen pretrained Anima | 30.263 | 19.98 | 22.62 | 534.8 |
| Plain U-Net, historical | 31.555 | **9.22** | 10.75 | 1161.8 |
| U-Net + frozen random donor | 32.550 | 20.82 | 23.54 | 512.7 |

The pretrained-vs-random difference is 2.286 FID points. Compared with plain
U-Net, the transplant improves FID by 1.292 while requiring 2.17× the training
time. Attention is both better and faster at this budget. Neither the early
screen nor this pair establishes statistical significance; no seed repeats were
run. The faster pretrained arm used GPU0; GPU1 also serves the desktop, so do
not attribute the small throughput difference between the pair to its weights.

For longer-budget context, the established plain U-Net reaches FID 25.397 at
50k updates in 45.73 training minutes. No Anima 50k training run was attempted.

## Screening and runtime

| Frozen donor | 1k FID5k ↓ | Training min | Steady samples/s |
|---|---:|---:|---:|
| Pretrained Anima | **67.514** | 2.00 | 541.3 |
| Random weights | 74.227 | 2.06 | 525.1 |

Historical plain U-Net 1k FID5k was 68.455, attention 69.311. These 5k-sample
scores are not numerically comparable with final FID50k. The 10k diagnostic
FID5k was 33.862 pretrained and 36.842 random; the final larger evaluation
preserved the ordering. Both 128-update profiles passed; their ten-sample FID
was only a smoke check and is excluded from quality comparisons.

The transplant has 157,339,107 total G parameters, of which 2,146,787 train.
155,192,320 donor parameters remain frozen, including the common timestep
embedder and block modulation weights. Baseline G has 1,044,835 trainable
parameters. D and the 20k×128 learned particle table are unchanged. Peak
training allocation was 2.63 GiB for each transplant; sampling/FID allocation is
excluded from this figure. Optimizers and EMA avoid updates to frozen weights.

## What was transplanted

Anima-Base v1.0 blocks 0 and 1 form a side branch at the existing U-Net's 8×8
encoder feature map. Trainable projections expand 128 features to 2048 and
project back. The output projection starts at zero, preserving the original
G output exactly at initialization. Four trainable cross-attention context
tokens come from the particle and class; particles also retain their original
U-Net conditioning path. Existing image skips and spatial resolutions remain.

The donor includes self-attention, cross-attention, MLP, Q/K RMS normalization,
rotary positions, and pretrained timestep modulation. Its parameters stay
fixed, with gradients through its operations into the adapters. Only donor
matrix operations use bfloat16; its residuals and layer normalization are
float32, as are the trainable model, discriminator, and losses. The original
VAE, text network/adapter, output head, and sampler are absent.

The DDGAN generator still predicts a clean RGB image, and the existing reverse
transition adds Gaussian step noise. Four steps, joint time/class UCD,
ParticleGAN prior/VICReg, Rp logistic, exact lazy-4 bcap, optimizer rates,
EMA .995 and **constant learning rates** remain intact. No flow-matching loss,
additional diffusion objective, alternative bcap, or learning-rate decay.

The weak assumption being tested is that frozen blocks can usefully process
features learned by our new input adapter despite the donor's latent-space,
resolution, and training-domain mismatch. This pair supports that narrow
possibility. It does not establish that Anima's whole network, different block
slices, or trainable donor weights would work, or identify G/D as the limiter.

The final grids contain recognizable class structure with persistent animal
shape/detail errors. Visual inspection and global FID do not establish mode
coverage or requested-class accuracy.

## Validation and reproducibility

- 51 CPU tests and 13 subtests passed across the targeted suite and the fixed
  default-config equality check. The initial equality failure was corrected
  before any GPU run; optional keys now appear in both DEFAULTS and YAML.
- Actual full-size bfloat16 checks passed on both GPUs: exact initial output
  match, finite nonzero particle/image/input/context-adapter gradients after
  two updates, and unchanged donor parameter version counters.
- Offline tests cover strict weight loading, matched trainable initialization,
  frozen gradients, EMA, state restoration, batch independence, rotary
  positions, hash rejection, and invalid configurations.
- All six completion certificates verified. Sources remained fixed across
  profiles, scouts, and promotions. The 1k and10k pretrained checkpoints were
  audited tensor-by-tensor: **every donor parameter in G and EMA still matches
  its original loaded weight**, and the output adapter has learned nonzero
  weights.
- The donor revision, selected blocks, tensor hashes, and bundle hash are
  recorded in each environment/summary and the full config. Downloads only
  parse safetensors bytes; the local bundle loads with `weights_only=True`.

See [design and sources](PLAN.md), [1k exports](scouts/TABLE.md),
[10k exports](promotions/TABLE.md), [throughput](promotion_speed/TABLE.md),
[pretrained samples](promotions/pretrained/samples.png), and
[random-control samples](promotions/random/samples.png).

Full configs: `configs/cifar_ddgan/anima_{profile,1k,10k}/*.yaml`.
Implementation: `lib/image_anima.py`; shared trainer and loss loop are reused.
Recreate the pinned donor bundle with
`.venv/bin/python experiments/prepare_anima_transplant.py` when absent.
Run from the repository root, use fresh output directories, and pass
`--workers_per_gpu 1` to the existing grid runner. Checkpoints/source archives
and donor weights stay in ignored results/data directories.

```
tail -F results/cifar_ddgan/anima.live.log
```

## Recommendation

Do not promote this two-block transplant or scale to the whole 2B model on the
strength of these results. If continuing the transplant idea, the next useful
question is whether **one frozen block retains the benefit at lower cost**,
again with a matched random control. That is a proposal, not a queued run.
Constant learning rates remain a user constraint; the earlier decay proposal
is withdrawn. The default baseline remains unchanged.
