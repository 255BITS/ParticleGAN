# Next experiment after compact: CIFAR joint UCD confirmation

The30k schedule round and56k toy joint-UCD scout are complete. User requested
constantLR as the CIFAR default and prefers joint(t,c) if competitive. The scout
was competitive with small metric tradeoffs, so the port is implemented and
smoke-tested. **No full CIFAR joint-UCD run has been launched.**

## Prepared matched pair, both GPUs

Full configs: `configs/cifar_ddgan/joint_ucd/class_only.yaml` and
`time_class.yaml`; manifest in the same directory. The pair differs only in
ucd_target and out_dir. Both30k updates,constantLR,seed24002,one run perGPU,
FID every10k (5k generated),final50k FID. No seed sweep.

Keep the width32 U-Net/GroupNorm D, learned20k x128 latent prior,Gaussian step
noise,T4,shared posterior,Rp logistic,candidate-only bcap,CE.02,VICReg and toy
optimizer rates. Class-only:10 heads with D timestep input. Joint:40 heads,
no D timestep input,select `(t-1)*10+c` for adversarial score and CE. Both keep
xt conditioning and the same G. D head parameter counts differ; don't attribute
results exclusively to removal of conditioning. Joint does not change G loss.

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python -u experiments/follow_grid.py \
  --root results/cifar_ddgan/joint_ucd \
  --log results/cifar_ddgan/live.log \
  --runner-log results/cifar_ddgan/joint_ucd.runner.log -- \
  --config_manifest configs/cifar_ddgan/joint_ucd/manifest.json \
  --trainer experiments/train_cifar_ddgan.py --gpus 0,1 --workers_per_gpu 1
```

Tail: `tail -F results/cifar_ddgan/live.log`. Estimate35–37min pair wall time.
Compare finalFID50k, matched-budget curves and class-row samples. No independent
class classifier yet. The43.678 best measured CIFAR score belongs to class-only.
The no-argument default selects the joint candidate; don't describe its quality
as measured. Future CIFAR configs should retain the10k FID cadence.

## Then: spatial particle injection

Currently z enters G through global scale/shift modulation. Test an added
z->Linear->[channels,4,4] map concatenated at the U-Net bottleneck, retaining
encoder skips and existing modulation. Keep128D first to isolate injection.
Then test smaller particles separately; dimension also changes VICReg covariance
pressure. xt and Gaussian noise provide stochasticity. Try the simple map before
StyleGAN3 Fourier spatial coordinates. Keep the posterior and adversarial recipe.

## Later: pretrained D features

Consider frozen candidate-image features with trainable xt conditioning and
appropriate UCD heads. Preserve input gradients through the frozen extractor
for bcap. Noisy transitions may mismatch clean-image features. No pretrained
backbone or spatial injection has been implemented. Width scaling/transformers
remain later options. Full prior findings in READOUT.md and ROUND2.md.
