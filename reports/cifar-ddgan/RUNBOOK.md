# CIFAR handoff: back on master after Anima experiments

<!-- superseded-banner: generated -->
> **Superseded — historical record.** Current status: [reports/forge/technique-inventory.md](../forge/technique-inventory.md).
> Results below are preserved as recorded; statements about "current" selections, defaults,
> leaderboards, queues or GPUs describe the time of writing.

Start with [the compaction handoff](ANIMA_HANDOFF.md). The feature branch is
fully committed through 5102d8c; its experimental code remains separate.
Attention from scratch still beats both pretrained transplants at 10k.
The user rejected learning-rate decay. All experiments are finished, both
GPUs are free, and no next experiment is queued. The following sections
describe the established baseline on master.

Read [latest results](attention_duration/READOUT.md) and
[50k leaderboard](attention_duration/finals/TABLE.md).
All four new runs completed and are certified; no jobs are running or queued.
Both GPUs are free. The user wants experiment updates only after completion.

## Current result and defaults

Plain U-Net32 + frozen ResNet18 D reaches FID25.397 at 50k in 45.73 training
minutes (48.27 total). The older every-step50k baseline was 26.680 in 99.34
training minutes. The fast recipe is now validated at the longer budget.

Attention improves the 10k scout to 29.327 (historical plain baseline 31.555),
but its 50k endpoint is 26.334 in 53.09 training minutes. Attention's best 5k-sample
diagnostic is 27.799 at 30k, followed by regression. Plain U-Net recovers from
a30k setback and finishes with its best diagnostic 29.379. These are 5k-sample
diagnostics, not interchangeable with final 50k-sample scores.

Keep no-argument defaults: plain U-Net,10k steps, batch 64, exact bcap every
fourth D update at4x weight, cached frozen conditioning features, fused Adam,
NCHW, no unused regularizer scalar synchronization, constant LR. Preserve the
four-step ParticleGAN DDGAN, joint time/class UCD, learned particles, Gaussian
step noise, Rp logistic, VICReg and optimizer rates.

Optional g_attn_resolutions:[8,16] inserts four residual spatial attention
blocks into the shared U-Net. Default is[]; g_heads remains 4. Tests establish
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

Current planning context is in [NEXT_ROUND.md](NEXT_ROUND.md). The earlier LR
tail proposal is withdrawn; keep constant learning rates. Periodic/best
checkpoint retention remains unimplemented. The trainer overwrites checkpoint.pt;
attention 30k weights are gone, so we cannot retrospectively certify that point.

Tail historical logs:
tail -F results/cifar_ddgan/duration.live.log results/cifar_ddgan/attention.live.log

Always pass --workers_per_gpu1 for independent CIFAR jobs. Use both GPUs when
useful, preserve sample exposure if batch changes, and do not run seed sweeps.

## Earlier evidence and validation

The [capacity round](capacity/READOUT.md) tested G64 and frozen ResNet34 at 1k/10k;
neither earned promotion. The [speed/toy round](speed/READOUT.md) completed 40
runs and selected exact lazy-4 over FD. FD retains mode locations but damages
one-shot toy cores; it is not the default. Shared toy defaults remain unchanged.
No alternate bcap objectives or broad FD search is planned.
NCSN++ full-bundle10k failed with FID161.587; do not resume the old interrupted
50k run automatically. Its early 1k benefit did not validate.

This round: 38 tests plus 13 subtests passed before launch, then all four actual
GPU runs completed. No architecture default promotion. The optional empty
attention field is now included in DEFAULTS and default.yaml.

The preceding speed-round commit is d42abb2. This commit records the completed
capacity and attention rounds. Preserve unrelated .claude/ and sparse-ucd.log.
The user requested a local commit before compaction; no push requested.
