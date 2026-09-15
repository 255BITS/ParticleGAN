# Next round after the Anima transplant

The [Anima round](anima/READOUT.md) completed all six runs on both GPUs.
Pretrained two-block transplant: FID50k 30.263 at 10k,19.98 training minutes.
Matched frozen random control:32.550,20.82 minutes.
Historical attention U-Net:29.327,10.73 minutes; plain U-Net:31.555,9.22 minutes.
The weights help this transplant, but the architecture has not earned promotion.

The user rejected learning-rate decay because of its tuning burden. Keep
constant rates; the previous decay proposal is withdrawn. Preserve the current
four-step ParticleGAN DDGAN, joint time/class UCD, Gaussian step noise, exact
lazy-4 bcap, VICReg and optimizer recipe. No seed-only repeats.

If continuing the transplant direction, test one frozen donor block against
its matched frozen random control to see whether the gain survives at lower
cost. Do not infer that the whole 2B model should be trained or loaded into G.
This is only a proposal: no further runs are active or queued.

The established cheap baseline remains plain U-Net; optional attention is still
our best10k scout. Plain50k FID 25.397 is the best certified longer-budget score.
The overwritten attention30k checkpoint is still unavailable; no checkpoint
retention change was made in this round.

Current work is on experiment/anima-transplant, based on edcdb14. See
[runbook](RUNBOOK.md), [transplant design](anima/PLAN.md), and the full configs
in configs/cifar_ddgan/anima_{profile,1k,10k}. Use fresh out_dirs and explicitly
pass --workers_per_gpu 1. The user wants run updates only after completion.

Historical tail: tail -F results/cifar_ddgan/anima.live.log
