# CIFAR AE-GAN GPU 1 performance analysis

2026-09-17, `feat/cifar-ae-gan-pretrained-encoder`. The best measured candidate is a **bundle of batched real/fake discriminator forwards, reused bcap logits, and foreach EMA**. It delivered approximately **9% higher training throughput** in a warmed, bracketed confirmation. Recommend a short matched quality scout before adopting it. No production training code or running experiment was changed.

## Leaderboard

These are training microbenchmarks, **not FID measurements**. Same N=8 scratch-encoder configuration, batch 64, seed 24002, TF32 enabled, fused Adam, and actual CIFAR-10 data/augmentation as the running 30k experiment.

| Candidate | Measured steps/s | Bracketed speedup | Assessment |
|---|---:|---:|---|
| Batched D + bcap-logit reuse + foreach EMA | 24.98 / 23.63 | **1.108× / 1.086×** | Best candidate; confirm quality |
| Bcap-logit reuse + restricted D backward | 23.27 / 22.89 | 1.032× / 1.052× | Smaller preliminary gain |
| D backward restricted to trainable parameters | 22.44 / 21.81 | 0.995× / 1.002× | No meaningful gain |
| Baseline brackets | 23.43 / 21.74 / 21.79 | 1.000× | Timing drift; cause not isolated |

“Restricted D backward” means `dl.backward(inputs=trainable_D_parameters)`, avoiding requests for unused gradients on the regularizer's detached input leaves. This did not produce a material end-to-end benefit by itself.

A follow-up phase check measured regularized D backward at 49.18 ms baseline, 45.63 ms restricted, and 38.63 ms reuse+restricted. Restricting backward alone saves only ~0.44 ms/update after averaging over N=8, consistent with an effect too small to establish in the throughput drift. Full phase results are in [pruned_phases.json](pruned_phases.json).

Each confirmation block measured 160 updates after 16 warmup updates. The same harness/model/optimizers continued through: baseline → batched bundle → restricted backward → reuse+restricted → baseline → reuse+restricted → restricted → batched bundle → baseline. Initial warmup was 160 updates. Bracketed speedup divides the arithmetic mean of the surrounding baseline milliseconds/update by candidate milliseconds/update; it is not a confidence interval.

The first sweep tested bcap-logit reuse alone, foreach EMA alone, their combination, and the batched bundle. Baseline drifted from 23.75 to 21.86 steps/s, so that sweep **cannot establish small improvements**. Foreach EMA alone did not show a convincing overall gain. Its phase is only about 0.14 ms/update, making it a small cleanup rather than a major accelerator.

## Where the time goes

CUDA-event spans from 32 warmed baseline updates (28 ordinary, 4 regularized):

| Phase | Ordinary update, ms | Bcap update, ms |
|---|---:|---:|
| D preparation, prior sample, G forward for D | 1.46 | 1.44 |
| Bcap D forwards + input-gradient construction | 0.00 | 15.10 |
| D adversarial forwards | 6.53 | 6.54 |
| D backward, including bcap when present | 7.38 | 48.51 |
| D optimizer | 0.05 | 0.05 |
| G update: real D forward | 3.27 | 3.22 |
| G generated sample + fake D forwards | 4.61 | 4.56 |
| Encoder + particle routing forward | 0.27 | 0.26 |
| G reconstruction forward | 1.35 | 1.34 |
| Prior regularizer | 0.10 | 0.10 |
| G/E/prior joint backward | 9.87 | 9.69 |
| G preparation + optimizer | 0.23 | 0.23 |
| EMA | 0.14 | 0.14 |
| **Sum of measured spans** | **35.28** | **91.17** |

At N=8, the weighted sum is 42.26 ms/update. D-related phases are about 53% of that sum, G/E/prior phases 47%, and EMA 0.3%. The additional work on the regularized update averages approximately 6.99 ms/update, or **16.5%**. G/E/prior backward is intentionally measured jointly because the real training graph shares G and the particle table across its losses; splitting backward passes would change the workload.

These event spans include device gaps within phases and are **not additive CPU+GPU timings**. Host enqueue work overlaps GPU execution. The separate uninstrumented benchmarks determine throughput. One 8-update `torch.profiler` trace confirms substantial convolution forward/backward and normalization work. Do not sum its nested annotation/operator/kernel rows; they overlap.

**No hidden per-update scalar synchronization was found.** The profiled loop had no `aten::item` or `_local_scalar_dense`; its single `cudaDeviceSynchronize` was at profiler completion. Production explicitly synchronizes at logging/evaluation boundaries, which the short benchmark excludes. Data is already resident on the GPU; sampled index/flip augmentation plus G-side prior preparation costs only ~0.14 ms/update of device spans. Adding DataLoader workers is unlikely to help this trainer.

## Correctness and limits

- D uses fixed pretrained batch-normalization statistics and per-image GroupNorm. Concatenating real/fake batches preserves the mathematical per-example function. It changes floating-point kernel/reduction choices.
- Reusing regularized logits preserves the exact double-backprop bcap objective and N=8 multiplier. It avoids redundant forwards; it does **not** replace bcap with finite differences or change its schedule.
- Fixed-batch D checks used TF32 disabled, deterministic cuDNN, and compared raw parameter gradients before Adam. At the production cap of 1, the initial test D's penalty was zero. An **explicitly diagnostic cap of 0** exercised nonzero penalty derivatives (penalty 0.0631493); this cap was never used in throughput benchmarks or production.
- For that active diagnostic, baseline repeat had relative gradient L2 error 1.09e-7; restricted backward 1.09e-7; reuse 6.14e-6; batched D 7.61e-6; reuse+restricted 6.13e-6. All gradients were finite and below the test threshold of 1e-4. The isolated foreach EMA check was bitwise equal.
- Initial full training-step comparisons with production TF32 settings showed particle-table update differences up to ~0.0086 after two updates, including ~0.0058 for foreach-only. These checks do **not** prove identical trajectories. Small numerical differences can be amplified by Adam and hard routing; quality must be measured. The stronger raw-gradient checks validate the local calculation, not long-run FID equivalence.
- Short warmed training on GPU 1 is not a long training/FID run. GPU 1 also drives the desktop. Both GPUs shared host resources; the harness used two intra-op CPU threads and one inter-op thread, while production used four. GPU 1 ran around 1770–1830 MHz in sampled observations versus GPU 0 around 1590–1665 MHz, so compare candidate/baseline ratios **on GPU 1**, not absolute speeds across GPUs. Baseline drift is reported rather than hidden.
- No mixed precision, channels-last, finite-difference penalty, seed sweep, or full additional training was run.

## Recommended next work

1. **Scout the batched bundle against the existing baseline**, preserving seed, N=8, losses, sample count, and FID protocol. It is the strongest measured candidate. Confirm throughput on GPU 0 after the current run finishes. The observed speedup corresponds to roughly 8–10% less training time; applying it to the production ~22.2 steps/s would save about two minutes over 30k updates, excluding evaluation. This is an estimate, not a measured production improvement.
2. **Consider real-image frozen-feature caching next.** Two ordinary real-image critic passes occur per update. CIFAR augmentation here is only identity/horizontal flip, and the pretrained extractor is frozen. Caching all three feature stages for 50k images and both flips in FP32 would require about **10.68 GiB**, before allocator/temporary overhead. Keep the full image-dependent extractor on bcap input-gradient passes and generated images, and keep trainable critic heads live. This is a source-inspection opportunity, not a benchmarked gain. Each complete real-D forward costs about 3.2 ms; caching could remove only its frozen-feature portion.
3. **Treat EMA and prior/data plumbing as low priority.** Their measured costs are small; context features are already cached, pretrained D parameters are frozen, and optimizers already use fused Adam. Removing the real D forward from the generator loss would change the current relativistic-pairing objective and is not an equivalent optimization.

## Reproduction and artifacts

```sh
tail -F runs/cifar_particle_ae/performance_gpu1/PROFILE.log

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -u experiments/profile_cifar_ae_gpu1.py \
  > runs/cifar_particle_ae/performance_gpu1/PROFILE.log 2>&1

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -u experiments/verify_cifar_ae_gpu1.py \
  >> runs/cifar_particle_ae/performance_gpu1/PROFILE.log 2>&1

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -u experiments/confirm_cifar_ae_gpu1.py \
  >> runs/cifar_particle_ae/performance_gpu1/PROFILE.log 2>&1
```

- [Confirmation measurements](confirmation.json)
- [Initial sweep and phase measurements](initial_benchmarks.json)
- [Raw gradient and EMA checks](gradient_checks.json)
- Runtime folder: `runs/cifar_particle_ae/performance_gpu1/` contains `PROFILE.log`, `profiler.txt`, `trace.json` (~65 MiB), and the raw JSON outputs.
- Standalone scripts: `experiments/profile_cifar_ae_gpu1.py`, `verify_cifar_ae_gpu1.py`, `confirm_cifar_ae_gpu1.py`, and `phases_cifar_ae_gpu1.py`. All reject execution unless `CUDA_VISIBLE_DEVICES=1`.

The current trainer, grid runner, config module, and all `lib/` and `particlegan/` source remained untouched, preserving the running experiment's source fingerprint.
