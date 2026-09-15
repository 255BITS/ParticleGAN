# Frozen Anima Turbo versus Base transplant

Question: does Anima-Turbo v1.1 improve the frozen two-block generator transplant
relative to Anima-Base v1.0 at identical CIFAR training exposure?

Both arms run 10,000 updates × batch64, final FID50k. This is a matched donor
checkpoint comparison, not a seed experiment. Base is rerun under the current
source alongside Turbo; historical frozen Base scored 30.263 FID50k. The earlier
1k screens did not reliably predict 10k ranking, so use the established 10k
budget directly after real-weight GPU correctness checks. Cost estimate is
about 25 minutes wall time with one worker on each of two A6000 GPUs.

Both retain blocks 0 and 1 plus the common timestep layers, BF16 frozen donor
parameters, trainable U-Net/adapters/context, cached donor time modulation, and
all original image skips. Preserve learned latent particles, T=4 DDGAN,
Gaussian step noise, joint time/class UCD, Rp logistic, VICReg, exact lazy-4
bcap, cached ResNet18 discriminator, constant LR and EMA. No sampler, loss,
architecture or optimizer changes. No default promotion before results.

Upstream source: https://huggingface.co/circlestone-labs/Anima
Pinned revision: f973fc41ec7545364ac9776c2440285f43ff2a30
Files: split_files/diffusion_models/anima-{base-v1.0,turbo-v1.1}.safetensors

The model card describes Turbo as distilled for 8–12 steps/CFG1 with reduced
diversity. We only transplant the same two-block slice; the donor's original
sampling speed does not imply lower cost in our unchanged four-step sampler.
Base exports keys under `net.`; Turbo uses `model.diffusion_model.`. The
preparation script strictly selects one complete namespace and removes that
prefix without modifying weights. Tensor shapes are checked by strict model
loading. Both downloads pin the revision and record per-tensor and bundle
hashes. Only the selected 310.4MB is downloaded; donor weights remain ignored.
The upstream model license applies as in the earlier Anima experiments.

Configs: configs/cifar_ddgan/anima_turbo_10k/{base,turbo}.yaml

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -u experiments/prepare_anima_transplant.py \
  --filename split_files/diffusion_models/anima-turbo-v1.1.safetensors \
  --revision f973fc41ec7545364ac9776c2440285f43ff2a30 \
  --out data/anima/turbo_v1_1_blocks_0_1.pt
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -u experiments/follow_grid.py \
  --root results/cifar_ddgan/anima_turbo_10k \
  --log results/cifar_ddgan/anima_turbo.live.log -- \
  --configs 'configs/cifar_ddgan/anima_turbo_10k/*.yaml' --gpus 0,1 \
  --workers_per_gpu 1 --trainer experiments/train_cifar_ddgan.py
```

```sh
tail -F results/cifar_ddgan/anima_turbo.live.log
```

Wait for both runs to complete before inspecting training results. Afterward,
verify certificates, audit both checkpoint and EMA donors against source
weights, inspect samples, and report matched FID50k plus wall time. Compare
historical from-scratch attention (29.327) and plain U-Net (31.555) separately;
these have different source provenance and roughly half the training cost.
