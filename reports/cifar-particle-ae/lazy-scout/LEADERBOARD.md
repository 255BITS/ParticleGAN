# CIFAR-10 AE-GAN lazy bcap scouts

3/3 certified runs; identical initialization, one shared seed, one GPU at a time.

| Rank | N | FID50k ↓ | Test MSE ↓ | Steps/s | Train min | Total min | Speedup vs N=4 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 21.993 | 0.08558 | 22.02 | 3.78 | 5.28 | 1.13x |
| 2 | 4 | 22.276 | 0.08708 | 19.42 | 4.29 | 5.78 | 1.00x |
| 3 | 16 | 23.930 | 0.08395 | 23.89 | 3.49 | 4.99 | 1.23x |

Exact double-backprop bcap is applied every N steps with coefficient multiplied by N. Optimizer rates/betas are fixed. Only N varies. The scratch encoder won the previous scout; the discriminator still uses frozen ImageNet features.

## Runtime projections

| N | Measured final FID50k eval min | 30k updates total min | 50k updates total min | 100k updates total min |
|---:|---:|---:|---:|---:|
| 8 | 1.18 | 25.4 | 41.8 | 82.7 |
| 4 | 1.17 | 28.5 | 46.9 | 92.8 |
| 16 | 1.18 | 23.7 | 38.8 | 76.8 |

Projections assume unchanged hardware/architecture, constant measured training throughput, FID5k every 5k steps, final FID50k and 10k test reconstructions at each evaluation. Reconstruction cost is scaled linearly from 1k images; setup/checkpoint overhead uses the scout residual. These estimate compute cost, not the time required to attain a target FID.

## BigGAN comparison

The original [BigGAN paper, appendix C.2](https://arxiv.org/html/1809.11096#A3.SS2) reports CIFAR-10 FID 14.73 and IS 9.22 without truncation. This is a historical context value, not a matched baseline in this grid. BigGAN uses class conditioning; our model is unconditional and uses an ImageNet-pretrained discriminator. We measure FID50k against CIFAR train50k using torch-fidelity TF-compatible Inception. We have not established an exact match to every detail of the published evaluation. The [author implementation](https://github.com/ajbrock/BigGAN-PyTorch#an-important-note-on-inception-metrics) explicitly distinguishes its PyTorch monitoring scores from official TF scores. A strict comparison should re-evaluate a specified CIFAR BigGAN checkpoint through our evaluator. Do not compare intermediate FID5k with final FID50k as a learning trend.

## Recommendation

Promote N=8 by lowest final FID50k. Inspect the speed/quality tradeoff and samples if gaps are small. Prepared winner_long.yaml for 30k updates from the same initial seed; longer training is not launched. The scout alone cannot predict whether the configuration will reach BigGAN quality.
