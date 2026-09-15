# Next round: validate the faster baseline over a longer budget

The speed round is complete: [readout](speed/READOUT.md).
Use the promoted U-Net recipe in configs/cifar_ddgan/default.yaml:
exact bcap every fourth update ×4, frozen-condition feature caching, fused
Adam, NCHW, batch64. Same DDGAN/ParticleGAN/joint-UCD formulation and rates.
No training jobs are running or queued.

The useful next experiment is50k updates with these settings, preserving the
constant learning rate. At10k it reaches finalFID50k31.555 in9.22 training
minutes. Extrapolated50k training cost is46min plus evaluation/I/O. Compare
against the completed prior every-step50k result,FID26.680 in99.34min.
Use final50k-sample FID and diagnostic5k-sample FID only every10k updates.
Use a fresh output directory and full YAML; do not treat an altered horizon
as exact resume. No seed-only repeats.

FD is not the default: the tested FDlazy4 givesFID37.003 in8.56min and causes
one-shot toy shape failures. On the56k denoising toy it recovers100 modes,
but exactlazy4 is faster and higher quality. No broad FD hyperparameter hunt
or alternate bcap objectives this session.

The NCSN++ optimization bundle failed10k validation,FID161.587 with substantial
oscillation. Its attractive1k number was insufficient evidence. Do not resume
the old interrupted NCSN++50k job automatically. Future architecture questions
remain separate from this speed task and are not queued.

Retain repeatable YAMLs, source provenance, combined tail logs, and both-GPU
utilization for independent experiments when useful. If increasing batch,
match total sample exposure and report fewer optimizer updates explicitly.
