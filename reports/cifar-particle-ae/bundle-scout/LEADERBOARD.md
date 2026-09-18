# AE-GAN optimization bundle: FID scout

2/2 certified runs complete. Same initialization and seed; GPU 1, sequential execution.

| Rank | Implementation | FID50k ↓ | Test MSE ↓ | Steps/s | Train min | Total min |
|---:|---|---:|---:|---:|---:|---:|
| 1 | baseline | 22.799 | 0.08577 | 21.85 | 3.81 | 5.31 |
| 2 | bundle | 23.203 | 0.08585 | 23.53 | 3.54 | 5.04 |

5k updates, N=8 exact double-backprop bcap, same optimizer settings, same scratch encoder, and frozen pretrained discriminator. The bundle batches D real/fake forwards, reuses their logits for bcap, and uses foreach EMA. Both arms use two CPU threads. Final FID50k uses the CIFAR train50k reference and TF-compatible Inception; reconstruction uses 1k test images. GPU 0 concurrently trains the existing long run, so shared-host variation can affect timing.

## Recommendation

Bundle minus baseline FID: **+0.405** (negative is better). Training throughput: **1.077x** baseline.

Keep the current baseline for the ongoing long run. The bundle had worse final FID; weigh the size of that gap against its speed gain before further validation.

This is one configuration comparison, not a seed sweep or proof of equivalent long-run quality. Hard routing and Adam can amplify floating-point differences. No implementation is automatically installed in the production trainer.
