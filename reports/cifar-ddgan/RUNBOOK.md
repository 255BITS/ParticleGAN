# Latest: frozen Anima Turbo comparison completed

[Turbo readout](anima_turbo/READOUT.md): matched frozen Base FID50k31.306 versus
frozen Turbo v1.1 37.330, each at10k updates, batch64. Training took20.29/20.88
minutes on GPU0/1. Both certified; all donor tensors stayed frozen in G/EMA.
No default promotion. Both GPUs free; no active or queued jobs. Work remains
on experiment/anima-transplant. The user requested committing/pushing this
branch and returning to master for future experiments, without merging.
Configs: configs/cifar_ddgan/anima_turbo_10k/{base,turbo}.yaml.
Historical tail: tail -F results/cifar_ddgan/anima_turbo.live.log

# CIFAR handoff: trainable Anima experiment completed

Read [latest results](anima_trainable/READOUT.md),
[10k leaderboard](anima_trainable/promotions/TABLE.md), and
[design](anima_trainable/PLAN.md). Work remains on experiment/anima-transplant;
the preceding frozen-transplant commit is 6e53731. No merge or push requested.

All six new processes completed and certified: two 128-update profiles, two
1k scouts, and two 10k validations. No active or queued training; both GPUs
are free. The user wants experiment updates only after completion.

At 10k, trainable pretrained donor FID50k 36.558 in 24.89 training minutes;
trainable random collapsed to solid-yellow images, FID 496.351 in 25.62 minutes.
The preceding frozen pretrained result was 30.263 in 19.98 minutes. Keep the
existing defaults. No more training at this setting is proposed.

Config flag anima_trainable:true unfreezes every donated parameter, including
time/AdaLN. The G rate stays constant at 0.0006; no added LR multiplier or
schedule. Learned timestep conditioning is recomputed from current G/EMA
weights. Trainable parameters/Adam/EMA stay float32, with BF16 donor matrix
operations. Frozen-mode compatibility remains; old configs default to frozen.

Checks: 56 tests plus 13 subtests passed, followed by all 18 Anima tests after
a dtype-warning fix. Both full-size GPU checks confirmed actual updates and
finite nonzero gradients for every donor parameter tensor. Checkpoint audits
at 1k/10k confirmed all donor tensors changed in G and EMA. The final random
EMA probe produced one quantized image for 100 inputs and strong output-head
saturation. Pretrained produced 100 distinct images, without that saturation.

The pretrained 10k run had an early loss excursion (600–2100) before recovering;
random's persisted from about 3300 onward. The 1k and 10k prefixes were not
bitwise identical. Their configs differ only in steps/final_samples/out_dir;
TF32/cuDNN benchmark execution is not guaranteed bitwise deterministic. Do
not combine these separate launches into a single trajectory.

Full configs: configs/cifar_ddgan/anima_trainable_{profile,1k,10k}/*.yaml.
Historical tail: tail -F results/cifar_ddgan/anima_trainable.live.log
Use fresh output directories and explicitly pass --workers_per_gpu 1.
No seed-only repeats; preserve sample exposure if batch changes.
The user rejected learning-rate decay: keep constant rates.

## Current result and defaults

Plain U-Net32 + frozen ResNet18 D reaches FID 25.397 at 50k in 45.73 training
minutes (48.27 total). The older every-step50k baseline was 26.680 in 99.34
training minutes. The fast recipe is now validated at the longer budget.

Attention improves the 10k scout to29.327 (historical plain baseline31.555),
but its50k endpoint is26.334 in 53.09 training minutes. Attention's best5k-sample
diagnostic is27.799 at 30k, followed by regression. Plain U-Net recovers from
a30k setback and finishes with its best diagnostic29.379. These are5k-sample
diagnostics, not interchangeable with final50k-sample scores.

Keep no-argument defaults: plain U-Net,10k steps, batch64, exact bcap every
fourth D update at 4x weight, cached frozen conditioning features, fused Adam,
NCHW, no unused regularizer scalar synchronization, constant LR. Preserve the
four-step ParticleGAN DDGAN, joint time/class UCD, learned particles, Gaussian
step noise, Rp logistic, VICReg and optimizer rates.

Optional g_attn_resolutions:[8,16] inserts four residual spatial attention
blocks into the shared U-Net. Default is[]; g_heads remains4. Tests establish
identity initialization, unchanged initial convolution/D weights and CPU RNG,
and subsequent attention/particle gradients. No alternate training loop.

## Artifacts and next proposal

Full configs: configs/cifar_ddgan/duration_50k/baseline.yaml and
configs/cifar_ddgan/attention_{1k,10k,50k}/attention.yaml.
Results use corresponding fresh directories under results/cifar_ddgan.
Reports export configs, metrics, sample grids, environment/provenance,
completion certificates and FID curves. Raw checkpoints/source.zip stay ignored.
All sources were held fixed while training. Saved source/config are authoritative
for strict reproduction; do not overwrite completed outputs or alter checkpoints
and claim an exact resume.

The next optional proposal, if continuing trainable transplants, is one
smaller constant donor rate, motivated by large weight movement at the
inherited G rate. This is not proof of a fix and no sweep/run is queued.
See [NEXT_ROUND.md](NEXT_ROUND.md). The earlier one-block transplant idea
is also untested. The trainer still overwrites checkpoint.pt; old attention
30k weights remain unavailable.

## Earlier evidence and validation

The [capacity round](capacity/READOUT.md) tested G64 and frozen ResNet34 at 1k/10k;
neither earned promotion. The [speed/toy round](speed/READOUT.md) completed40
runs and selected exact lazy-4 over FD. FD retains mode locations but damages
one-shot toy cores; it is not the default. Shared toy defaults remain unchanged.
No alternate bcap objectives or broad FD search is planned.
NCSN++ full-bundle10k failed with FID 161.587; do not resume the old interrupted
50k run automatically. Its early1k benefit did not validate.

The frozen Anima round completed six runs and retained all original donor
weights. The new trainable round is described at the top of this handoff.
Keep unrelated .claude/ and sparse-ucd.log untouched. Do not merge or push
the experimental branch automatically.
