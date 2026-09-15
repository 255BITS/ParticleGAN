# CIFAR handoff: frozen Anima transplant completed

Read [latest results](anima/READOUT.md), [10k leaderboard](anima/promotions/TABLE.md),
and [design](anima/PLAN.md). Branch: experiment/anima-transplant, based on edcdb14.
Six new runs completed and certified: two 128-update profiles, two 1k scouts,
two 10k validations. No active or queued jobs; both GPUs are free.

At 10k, pretrained donor FID50k 30.263 versus frozen random 32.550. Training time
19.98/20.82min. This supports transfer within the matched pair, but historical
attention remains better/faster (29.327,10.73min). No default promotion.
Two frozen Anima-Base blocks attach to the existing U-Net 8x8 grid; only U-Net
and new image/particle/class adapters train. No VAE/text pipeline or new loss.
Donor weights are pinned/hashed in data/anima/blocks_0_1.pt, not committed.

The user explicitly rejected learning-rate decay. Keep constant rates; the
previous decay experiment proposal is withdrawn. Run updates only after
completion, no seed repeats, one worker per GPU.

## Current result and defaults

Plain U-Net32 + frozen ResNet18 D reaches FID 25.397 at 50k in 45.73 training
minutes (48.27 total). The older every-step50k baseline was26.680 in 99.34
training minutes. The fast recipe is now validated at the longer budget.

Attention improves the10k scout to29.327 (historical plain baseline31.555),
but its50k endpoint is26.334 in 53.09 training minutes. Attention's best5k-sample
diagnostic is27.799 at 30k, followed by regression. Plain U-Net recovers from
a30k setback and finishes with its best diagnostic29.379. These are5k-sample
diagnostics, not interchangeable with final50k-sample scores.

Keep no-argument defaults: plain U-Net,10k steps, batch64, exact bcap every
fourth D update at4x weight, cached frozen conditioning features, fused Adam,
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

The next optional proposal is a one-block transplant with a matched random
control to test whether the modest transfer gain can be retained more cheaply.
See [NEXT_ROUND.md](NEXT_ROUND.md). Nothing is queued. The trainer still
overwrites checkpoint.pt; the old attention30k weights remain unavailable.

Full Anima configs: configs/cifar_ddgan/anima_{profile,1k,10k}/*.yaml.
Historical tail: tail -F results/cifar_ddgan/anima.live.log
Always pass --workers_per_gpu 1. Preserve exposure if changing batch size.

## Earlier evidence and validation

The [capacity round](capacity/READOUT.md) tested G64 and frozen ResNet34 at1k/10k;
neither earned promotion. The [speed/toy round](speed/READOUT.md) completed40
runs and selected exact lazy-4 over FD. FD retains mode locations but damages
one-shot toy cores; it is not the default. Shared toy defaults remain unchanged.
No alternate bcap objectives or broad FD search is planned.
NCSN++ full-bundle10k failed with FID161.587; do not resume the old interrupted
50k run automatically. Its early1k benefit did not validate.

This round: 51 CPU tests plus 13 subtests passed, with both full-size GPU
adapter/gradient/frozen-weight checks passing before scouts. All six actual
runs certified. Final checkpoint audits confirmed every pretrained donor
parameter in G and EMA unchanged from the source. No generator default
promotion. Optional Anima keys are included in DEFAULTS/default.yaml.

The implementation and reports live on the experimental branch. Do not merge
or push automatically. Preserve unrelated .claude/ and sparse-ucd.log.
