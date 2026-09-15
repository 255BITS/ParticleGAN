# CIFAR throughput round

User authorizes both GPUs, 1k-update scouts then a 10k promotion scout. Batch
changes must preserve samples seen. Same seed 24002; no seed-only repeats.
Retain joint UCD, learned particles, Gaussian four-step DDGAN, Rp logistic,
VICReg and model/optimizer defaults. User explicitly permits lazy/finite-
difference bcap experiments; these are approximations, not exact equivalents.

Baseline profiles: U-Net32 on GPU0 and NCSN++128 on GPU1, 1,000 updates at
batch64, 5k-sample diagnostic FID once at the end. Profile steps101–108.
Steady throughput will use steps201–1000 to exclude startup/profiler overhead.
Samples/s = batch * updates/s per optimizer; total real draws are twice this
because G and D each receive a separately drawn real batch. Evaluation and
checkpoint time are excluded from train_seconds; reported total includes them.
Peak allocated memory includes the resident dataset and Inception evaluator.

Candidates: frozen xt feature reuse (within each optimizer batch), avoiding an
unused penalty scalar synchronization, channels-last, fused Adam, lazy bcap,
gradient-aligned central finite differences, batch128 with half as many steps.
EMA decay will be squared for batch128 to preserve smoothing in sample units;
optimizer hyperparameters stay fixed, so this remains a batch-size experiment.
No final ranking from 1k FID alone; it is an early stability screen.

Lazy: every Nth step, multiply bcap by N, add to the ordinary D loss. No extra
optimizer step or StyleGAN optimizer rescaling: their implementation uses a
separate regularizer phase. Source:
https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/training_loop.py

FD: compute v = grad_x D / ||grad_x D|| with ordinary backward, detach v;
use [D(x+h*v)-D(x-h*v)]/(2h) as the norm in the same bcap. h=0.05 is L2 image
displacement (~0.0009 RMS per CIFAR component), not Gaussian perturbation.
This avoids double backward but needs extra forwards. Nonlinear activation
crossings and TF32 cancellation can make it inaccurate; measure rather than
assuming speed or quality. Smooth-critic norm/parameter-gradient unit test.

37 focused tests passed before launch, including cached logits, candidate
input gradients and bcap parameter gradients, FD analytic gradients and lazy
scaling. Existing shared toy regularizer/training modules remain unchanged.

Tail: `tail -F results/cifar_ddgan/speed.live.log`

## Adaptive follow-up

Cache plus suppressed unused stats synchronization gave a 22% steady U-Net
throughput gain. Channels-last was slower for this G; fused Adam only ~1% faster.
Regularizer scouts therefore use cached NCHW + fused Adam at batch64. FD+lazy4
won the initial screen (1,255 samples/s; 1k FID64.20); exact lazy4 reached1,164
withFID68.46. Batch128/256 matched sample exposure but hadFID77.70/91.18.

NCSN++ bundle unexpectedly achieved1kFID59.33 vsbaseline306.67, with unchanged
speed182 vs183samples/s. User explicitly asked to investigate. Run four more
1k NCSN++ scouts across both GPUs: cache-only (also suppress unused scalar
sync), channels-last-only, fused-Adam-only, fullbundle+exactlazy4. Save their
checkpoints for inspection. No altered seed, architecture, learning rates,
training budget, model capacity or evaluation protocol in these ablations.

The bundle's image grid has recognizable class templates but limited within-
class variety. Its lower FID is not proof of diversity or a robust numerical
fix. Isolate the change and validate longer; do not attribute causality to the
bundle or promote it on one early grid alone.
