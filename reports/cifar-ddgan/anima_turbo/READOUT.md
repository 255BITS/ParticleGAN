# Frozen Turbo did not improve the Anima transplant

Both matched 10k runs completed on separate A6000 GPUs with final FID50k.
Turbo v1.1 scored **37.330**, versus **31.306** for Base v1.0: 6.024 FID points
worse at essentially the same cost. Keep existing defaults; do not promote
Turbo or extend this setting automatically.

| Generator | FID50k ↓ | Training min | Total min | Steady samples/s |
|---|---:|---:|---:|---:|
| Frozen Base v1.0, matched control | **31.306** | **20.29** | 22.98 | 526.3 |
| Frozen Turbo v1.1 | 37.330 | 20.88 | 23.60 | 511.4 |

Both: 10,000 updates × batch64, 640,000 examples per optimizer phase and
1,280,000 total real draws under the existing two-phase loop. G has 157,339,107
total parameters, of which 2,146,787 train; all 155,192,320 donor parameters
remain frozen. Peak training allocation is 2.63 GiB in both runs.
Base ran on GPU0; Turbo on GPU1, which also serves the desktop. Do not attribute
the small speed difference to donor weights. Their architecture and work are
identical. Total paired wall time was about 24 minutes.

For context, historical 10k FID50k results are attention U-Net **29.327** in
10.73 training minutes, plain U-Net **31.555** in 9.22 minutes, and the original
frozen Base transplant **30.263** in 19.98 minutes. These runs have different
source provenance; they are not part of the new matched pair. The new Base
control is 1.043 points worse than its historical run. CUDA execution is not
bitwise guaranteed; one pair does not establish statistical significance.
No seed-only repeats were run.

## What changed and what did not

Only the frozen donor checkpoint differs between the paired configs, aside
from its path/hash and output directory. Both use two contiguous blocks (0,1)
and common timestep layers, the same initialized trainable U-Net/adapters,
seed24002, batch64, learned 20k×128 latent particles, T=4 DDGAN, Gaussian step
noise, joint time/class UCD, Rp logistic, VICReg, exact lazy-4 bcap, frozen
ResNet18 D features, constant G LR0.0006 and EMA0.995. The model/loss/training
code was unchanged this round; only the donor downloader gained revision/file
arguments and strict support for Turbo's exported tensor namespace.

Both checkpoints have the same 43 tensor keys/shapes in the selected slice;
32 tensors differ. All eight Q/K norm tensors and the three common time
embedding/norm tensors are exactly equal in the original bundles. Thus the
experiment did load distinct donor weights, while retaining the same tensor
layout and donor time embedding. See checkpoint_audit.json.

[The upstream model card](https://huggingface.co/circlestone-labs/Anima) describes
Turbo as distilled for 8–12 sampling steps at CFG1, with increased stability
and reduced diversity. Our result tests a frozen two-block transplant into a
new learned interface and four-step DDGAN. It does not test the complete Turbo
model, its original sampler, or its native image quality. Distilled donor
weights provided neither a quality improvement nor an architectural speedup
here. The metric alone does not tell us whether diversity caused the regression.

## Validation and visual check

- 18 existing Anima tests and three new namespace-selection tests passed.
- Both full-size GPU smoke checks passed: exact initial U-Net output identity,
  loaded source weights, finite active image/particle/adapter gradients after
  two updates, and frozen donor parameters. Script/results are included.
- Both completion certificates validate config, source and summary digests;
  the pair has matching source provenance.
- At step10k every donor tensor in both G and EMA equals its respective source
  exactly after dtype conversion. All G/EMA tensors are finite and the zero-
  initialized output adapters have learned nonzero weights.
- Both final class-row grids contain varied outputs with recognizable vehicle
  classes and weaker animal detail. Neither shows the prior trainable-random
  solid-yellow collapse. These grids do not prove mode coverage or explain
  all of the FID gap.
- No startup/runtime warnings or errors appeared in the combined log. Both
  GPUs are free; no training remains active or queued.

## Recommendation and artifacts

Park the frozen Turbo transplant. The cheaper attention U-Net remains the
stronger historical 10k choice; plain U-Net remains the no-argument default
and has the best longer-run result (25.397 at50k). No baseline defaults changed.

- [Base config](../../../configs/cifar_ddgan/anima_turbo_10k/base.yaml)
- [Turbo config](../../../configs/cifar_ddgan/anima_turbo_10k/turbo.yaml)
- [Base samples](promotions/base/samples.png), [Turbo samples](promotions/turbo/samples.png)
- [Certified comparison](promotions/TABLE.md), [speed](speed/TABLE.md)
- [Protocol and reproduction](PLAN.md)
- Raw results/checkpoints/source archives: results/cifar_ddgan/anima_turbo_10k/
- Combined log: results/cifar_ddgan/anima_turbo.live.log

The experiment is retained on experiment/anima-transplant. The user requested
committing and pushing this branch, then returning to master for future work.
No experimental changes are promoted to master.
