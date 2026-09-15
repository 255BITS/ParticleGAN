# Speed round complete

All 40 runs completed successfully. No active training jobs or queued experiments.
Old tool sessions/cells in prior conversation are finished and must not be reused.

Read [READOUT.md](READOUT.md) for results, methodology and recommendations.
The exported runs contain 40 verified completion certificates, configurations,
source provenance, metrics and sample figures. Raw artifacts/checkpoints/source
archives remain in ignored results directories. Combined historical log:
`tail -F results/cifar_ddgan/speed.live.log`.

Promoted CIFAR no-argument defaults: exact bcap every fourth D update ×4,
cached frozen xt features, fused Adam, no unused scalar-stat synchronization,
NCHW. Batch64 and all DDGAN/ParticleGAN/joint-UCD equations remain as before.
Toy and shared regularizer no-argument defaults remain every-step exact.
FD is optional and not promoted. Do not explore alternate bcap objectives.

10k CIFAR final FID50k / train minutes:
- U-Net exact cached/fused: 32.821 / 16.59.
- U-Net exact lazy4 cached/fused: 31.555 / 9.22 (promoted).
- U-Net FD lazy4 cached/fused: 37.003 / 8.56.
- NCSN++ full bundle: 161.587 / 58.48, unstable; early recovery did not validate.

One-shot10k: all variants100 modes, but FD jointHQ85.87% and65/100 narrow
mode cores versus exact99.01%HQ, lazy4 98.84%HQ, neither with narrow cores.
Denoising56k: all100 modes. JointHQ exact92.35%, fused91.98%, lazy4 91.46%,
FDlazy4 90.17%. FD catches up on coverage but has no speed advantage over
exactlazy4 on the MLP. Do not claim universal FD collapse.

Validation after default promotion:53 tests+13 subtests and4 CUDA resume tests
passed. Earlier image-model/prior-controls tests and numerical gradient probes
also passed; numerical probes document meaningful FD approximation error.

Source changed after initial scouts for shared FD refactor and again after all
runs for default promotion. Exact resume/reproduction requires the matching
saved source.zip and config.yaml; do not silently rerun over completed paths.

User requested a local commit before compaction; no push requested. Preserve unrelated .claude/ and
sparse-ucd.log. The speed changes, configs, scripts and reports are included in the
local speed-round commit. A future50k promoted-CIFAR run (~46 training min, extrapolated) is
recommended but not launched. No seed-only experiments.
