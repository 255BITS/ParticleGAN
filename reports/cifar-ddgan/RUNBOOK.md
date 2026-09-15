# CIFAR speed round handoff

All 40 speed and transfer experiments completed with no failures. No jobs are
running or queued. Read [results](speed/READOUT.md),
[CIFAR finals](speed/promotions/TABLE.md), and [toy confirmations](speed/toys_56k/TABLE.md).

The no-argument CIFAR recipe now uses exact bcap every fourth D update at4×
weight, cached frozen conditioning features, fused Adam, NCHW and no unused
regularizer scalar synchronization. It reaches FID31.555 after10k updates in
9.22 training minutes, versus32.821 in16.59 for cached/fused every-step exact.
Batch64, four-step DDGAN, joint UCD, learned particles, Gaussian step noise,
Rp logistic, VICReg and optimizer rates remain unchanged.

FD is shared in lib/grad_regularizers.py and available by config in CIFAR,
train_denoising.py and the actual100gaussians.py loop. It is not promoted:
CIFAR FID37.003 saves only40 additional seconds; one-shot toy cores fail.
Denoising56k FD recovers100 modes, but exactlazy4 is faster and higher quality.
Toy no-argument defaults remain unchanged. The user excludes alternate cap
objectives or reinventing bcap this session; keep that scope.

NCSN++ full-bundle10k ended atFID161.587 after58.48 training minutes. Its early
1k gain did not validate as a reliable fix. Do not automatically resume the
older restart-interrupted NCSN++50k job or queue architecture changes.

Run from the repo root with a fresh out_dir. Default config:
configs/cifar_ddgan/default.yaml. For pixel-only D set cache_condition:false.
Full tested configs/manifests live under configs/cifar_ddgan/speed_* and
configs/speed_{100gaussians,denoising}_*. All completed outputs are protected
against accidental overwrite. Default promotion changed source hashes, so
restore matching source.zip before exact resume/reproduction of saved runs.

Historical combined log: tail -F results/cifar_ddgan/speed.live.log.
Use experiments/follow_grid.py for tagged combined logs in the next round.
Both GPUs are authorized; no seed-only repeats. If batch changes, preserve
sample exposure. Distinguish training time from evaluation and FID5k screens
from finalFID50k. Report mode mass and within-mode shape alongside coverage.

Final validation:53 tests+13 subtests, and4 CUDA resume tests with promoted
settings passed. See speed/STATUS.md for artifact and provenance details.
User requested a local commit before compaction; no push requested. Preserve unrelated .claude/ and
sparse-ucd.log. Next suggested validation is a50k promoted U-Net run; it is
not queued. Expected training~46min is an extrapolation, not a measurement.
