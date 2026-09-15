# CIFAR capacity comparison: keep the fast baseline

Four runs completed: two 1k screens and two 10k validations. No defaults promoted.
Both 10k jobs used separate A6000 GPUs. The deeper frozen discriminator and
wider generator each cost more and neither improved final FID at this budget.

| Setup | Updates | Final FID (50k samples) ↓ | Training min | Total min |
|---|---:|---:|---:|---:|
| Current baseline (historical) | 10,000 | 31.555 | 9.22 | 10.75 |
| U-Net32 + ResNet34 | 10,000 | 32.506 | 11.11 | 12.68 |
| U-Net64 + ResNet18 | 10,000 | 33.332 | 10.93 | 12.63 |

## What this tells us

- **G capacity:** width 32 → 64 increases G from 1.045M to 4.129M parameters.
  FID 33.332 versus 31.555 gives no evidence that this simple capacity increase
  improves learning at 10k. It does not establish a long-run capacity ceiling.
- **Pretrained D:** ResNet18 → ResNet34 preserves G, input 64px, feature stages
  and trainable head count (1,000,256). FID 32.506 gives no improvement from
  this deeper representation. This does not rule out a different pretraining
  task or a better way of using the features.
- **Images:** all three final grids show recognizable vehicles and broad class
  structure, with malformed or blurry animal examples. No obvious global
  collapse is visible in these 100-image grids. This is qualitative inspection,
  not a mode-coverage or class-fidelity measurement.
- **Diagnosis:** D/G loss magnitudes alone do not locate the limiting network.
  Controlled interventions are more useful. These two interventions did not
  help; neither demonstrates that G or D is universally the bottleneck.

## Next step

Run the unchanged fast baseline for 50k updates before another architecture jump.
The older every-step baseline improved from about 31.7 at 10k to 26.68 at 50k.
The exact-lazy4 recipe has only been validated at 10k. Its projected 50k cost is
46.1 training minutes plus evaluation and I/O; this is an estimate.
A full config is prepared at configs/cifar_ddgan/capacity_50k/baseline.yaml.
It uses constant LR, batch 64 and diagnostics only every 10k. It is **not queued**.
50k updates equals 3.2M samples per optimizer (64 dataset passes); separate
D/G real batches mean 6.4M actual real-image draws. No 1200-epoch run is proposed.

## Protocol and limits

ParticleGAN DDGAN, learned 20k×128 latent particles, Gaussian step noise, four
diffusion steps, joint time/class UCD, Rp logistic, exact lazy-4 bcap, batch 64,
optimizer rates and constant learning rate were held fixed. No FD changes.
All configs use seed 24002; there are no seed sweeps. Architecture changes
also change initialization and optimization trajectories; these are practical
configuration comparisons rather than an isolated mathematical capacity test.

The baseline is a certified historical run. The new optional backbone changes
source provenance; it does not change the existing ResNet18 execution branch.
Small FID differences do not establish significance or equivalence. FID uses
the same 50k CIFAR-train reference, 50k final generated samples and existing
TF-compatible Inception protocol. Training time excludes evaluation and I/O.

The 1k screens reached FID5k 78.684 (G64) and 79.293 (ResNet34), versus historical
baseline 68.455. Their launcher accidentally used the runner default of five
workers per GPU, putting both jobs on GPU 0 concurrently. Their throughput is
contended and should not be compared to isolated baseline throughput. The 10k
launch corrected this with --workers_per_gpu 1: G64 on GPU 0, ResNet34 on GPU 1.

## Validation and artifacts

- 14 targeted tests passed, including cached versus uncached logits, candidate
  gradients and second-order bcap parameter gradients for both ResNet backbones.
  Frozen feature weights and evaluation mode are checked. An older pixel-D
  test now explicitly disables the pretrained-only cache.
- All four runs have verified completion certificates and exported summaries.
- [1k results](scouts/TABLE.md), [10k results](promotions/TABLE.md),
  [isolated 10k throughput](promotion_speed/TABLE.md).
- Configs: configs/cifar_ddgan/capacity_1k and capacity_10k.
- Historical logs: results/cifar_ddgan/capacity_1k.live.log and
  results/cifar_ddgan/capacity_10k.live.log.
- Source archives and checkpoints remain under ignored results directories.

Reproduce the 10k grid from the repository root (completed outputs are verified
for reuse; use fresh output paths for an intentional rerun):

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -u experiments/follow_grid.py \
  --root results/cifar_ddgan/capacity_10k --log results/cifar_ddgan/capacity_10k.live.log \
  -- --configs 'configs/cifar_ddgan/capacity_10k/*.yaml' \
  --trainer experiments/train_cifar_ddgan.py --gpus 0,1 --workers_per_gpu 1
```
