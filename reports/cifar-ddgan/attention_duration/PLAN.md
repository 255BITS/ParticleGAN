# Longer baseline versus attention U-Net

User authorized both GPUs, with result updates only after experiments complete.

GPU0: baseline U-Net32 + frozen ResNet18 D, exact lazy-4 bcap, 50k updates.
GPU1: same recipe with residual attention at encoder and decoder 8px/16px
resolutions; 1k screen, then 10k if the screen completes with finite metrics.
The attention screen is not a quality-ranking decision. No seed experiments.

All use batch64, seed24002, constant LR, learned particles, Gaussian reverse
noise and joint time/class UCD. No loss, bcap, schedule or optimizer change.
Optional g_attn_resolutions defaults to []; g_heads remains4. Four attention
blocks use GroupNorm, 1x1 QKV projections, spatial multihead attention, and a
zero-initialized output projection. They start as identity mappings. Existing
convolution weights and D initialization are preserved by isolating the CPU
RNG while constructing the added modules. No token pooling or skip removal.

Configs: configs/cifar_ddgan/{duration_50k,attention_1k,attention_10k}.
No-argument architecture defaults remain unchanged.

Diagnostics every10k, final50k samples (1k screen uses5k). Compare attention
against the historical 10k FID50k31.555 baseline. Compare50k baseline against
the older every-step50k FID26.680 run, keeping source and regularizer differences
explicit. Do not compare FID5k with FID50k as equal estimators. Training time
excludes evaluation/I/O; batch64 means50k updates=3.2M samples per optimizer,
with twice that many real-image draws across D and G.

Tail:
```sh
tail -F results/cifar_ddgan/duration.live.log results/cifar_ddgan/attention.live.log
```

Training source stays fixed while jobs run. Checkpoints, metrics, source
archives, full YAMLs and completion certificates use fresh output directories.
The 50k config from the preceding capacity round stays unused; this round uses
duration_50k/baseline to keep its provenance and outputs together.

## Adaptive promotion after the completed 10k scout

Attention reached final FID50k29.327 at10k in10.73 training minutes, versus
the historical baseline31.555 in9.22 minutes. Its1k screen had FID5k69.311
(the historical baseline was68.455). The early screen alone did not predict
the10k improvement. GPU1 now runs a fresh50k attention config for a matched
long-budget comparison, configs/cifar_ddgan/attention_50k/attention.yaml.
No source changes or altered-checkpoint resumes. Defaults remain unchanged.
Estimated attention training cost54min plus evaluation/I/O; no other runs queued.
