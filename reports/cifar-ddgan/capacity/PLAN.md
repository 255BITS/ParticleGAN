# CIFAR generator versus discriminator capacity

Keep ParticleGAN DDGAN, joint time/class UCD, Gaussian step noise, exact
lazy-4 bcap, batch64, seed24002, rates and constant schedule fixed.
No seed sweeps. This is a budget-limited intervention, not a proof of an
intrinsic generator/discriminator bottleneck. D/G losses alone cannot diagnose it.

- GPU0: U-Net width64, frozen ResNet18 D; compare with width32 baseline.
- GPU1: U-Net width32, frozen ResNet34 D; compare with ResNet18 baseline.
  Same ImageNet V1 weight family, stages1–3, 64px input and trainable heads.
  This changes pretrained representation/depth, not the adversarial objective.

Screen both at1k (FID5k), promote finite, viable runs to10k (final FID50k).
Evaluate at the endpoint only; no expensive intermediate FID checks.
Use the certified historical exact-lazy4 baseline: 10k FID50k31.555,
9.22 training minutes. Source changed to add an optional backbone; old results
are historical controls, not reruns with identical source. Small differences
are inconclusive. Compare matched updates/samples and training-time cost.
A materially promising10k result can justify longer validation; avoid treating
an attractive1k number as proof (the NCSN++ round failed that test).

Configs: configs/cifar_ddgan/capacity_{1k,10k}/{g64,r34}.yaml.
Logs: tail -F results/cifar_ddgan/capacity_1k.live.log results/cifar_ddgan/capacity_10k.live.log

At batch64, 10k updates = 640k examples per optimizer =12.8 passes through
50k images; 50k updates =64 such passes. Separate real D/G batches double
actual real-image draws. Report updates and samples to avoid ambiguous epochs.

## Scout scheduling correction

The1k launcher omitted --workers_per_gpu1 (runner default5), placing both
jobs on GPU0 concurrently. These completed scout speed measurements are
contended and must not be compared with isolated baseline throughput.
The10k launch explicitly uses --workers_per_gpu1 on GPUs0,1. No seed repeats
or retries of the completed1k scouts are needed for the quality screen.
