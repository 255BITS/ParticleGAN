# 50k scaling round: completed small G, interrupted large G

The U-Net32 finished 50k updates at **26.680 final FID (50k samples)**, taking
99.34 training minutes (101.75 minutes including evaluation/startup). The prior
10k pretrained-D winner scored31.741 in20.05 training minutes. Five times the
updates improved final FID by15.9%; this is useful but far from the sub-8 target.
The no-argument default remains the fast10k winner; no new default was promoted.

NCSN++128 was interrupted by a host restart after the last logged update27,200.
Its last evaluation and recoverable checkpoint are at20,000 updates:
**65.209 diagnostic FID (5k samples)**, improving from240.594 at10k. It never
finished50k and has no final50k-sample FID or completion certificate. Do not rank
its diagnostic result against completed50k-sample scores as equivalent.

| G | 10k diagnostic | 20k | 30k | 40k | 50k | Final 50k-sample FID |
|---|---:|---:|---:|---:|---:|---:|
| U-Net32 | 38.341 | 33.540 | 31.605 | 32.192 | 30.755 | **26.680** |
| NCSN++128 | 240.594 | 65.209 | — | — | — | incomplete |

All diagnostic entries use5k samples. NCSN++ initially produced repetitive blobs,
then recovered substantially. The evidence supports slow/problematic early
learning under our recipe, not a proven failure of the architecture. The small
G improved slowly; neither its near-plateau nor the unfinished large-G run
establishes a discriminator capacity ceiling.

Both runs use the same pretrained D, joint UCD, learned particles, Gaussian step
noise and shared DDGAN losses. NCSN++ adopts the official generator architecture,
not the official optimization setup. See PLAN.md and lib/ddgan_ncsnpp/UPSTREAM.md.
Its native PyTorch FIR fallback is a speed hypothesis to profile, not a measured
bottleneck. Observed average throughput: U-Net8.389 updates/s, NCSN++2.932 updates/s.

50k updates correspond to64 epoch-equivalents per optimizer. At measured large-G
speed, matching1,200/1,800 epoch-equivalents would take about3.7/5.5 GPU-days,
excluding evaluation, with our batch64 and one GPU. This is an exposure-based
extrapolation, not a reproduction of NVIDIA's distributed training setup.
D/G draw independent real batches, giving twice as many total real-image draws.

## State and next decision

Nothing is running or queued after restart. The user chose **training speed** as
the next round's focus; do not automatically restart the large50k run. No extra
architecture or pretrained encoder trial is queued. Preserve current artifacts.

The interrupted checkpoint remains in results/cifar_ddgan/scale_50k/ncsnpp128.
It stores all optimizers, particles, EMA and RNG state at20k. Strict resume needs
its saved config and identical source; source.zip is retained. Following a speed
implementation change, use fresh benchmark outputs and do not label them exact
continuations. The next session should profile before optimizing.

A user-reported optimization subagent was not recoverable after restart: no live
agent, standalone profiling report, or identified new speed patch was found in
this workspace. Do not assume optimization has already been completed.

Validation before launch:34 focused tests and4 real-CIFAR CUDA checkpoint replay
tests passed; both full-size100-update GPU preflights passed. See validation.txt.
Exported completed evidence is in unet32; interrupted evidence is explicitly in
ncsnpp128_interrupted. Checkpoints/source archives remain ignored under results.
