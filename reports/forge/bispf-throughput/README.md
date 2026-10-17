# BISPF throughput study

This study starts at `origin/develop` commit `de43f22f0`. The control is the
selected `bispf-optimized-quintic-v1`, rather than the older native-SVD BISPF.
The target is at least twice the complete-update throughput on the existing
256×256 DCGAN task, with all six DCGAN workloads reported and no quality-pass
regressions. Small-image training should also improve; no uniform speedup is
assumed. The current technique inventory remains the single leaderboard.

The candidate `bispf-throughput-v1` changes two global recipe fields:

- `optimizer_svd_backend="batched"`: native smoothed, rank-masked SVD is batched
  for CUDA per-offset channel matrices of at most 32 channels. Larger quintic
  matrices replay the same FP32 tensor calculation in optimizer-local CUDA
  graphs. Dense, bias, sampled-prior and CPU updates retain their original laws.
- `training_precision="bf16_channels_last"`: CUDA G/D networks containing a
  convolution with a channel-matrix dimension above 32 train under BF16
  autocast with channels-last storage. Outputs and losses return to FP32;
  master parameters, optimizer arithmetic, prior learning and SBNP stay FP32.
  Sampling/evaluation and smaller/nonconvolutional networks remain FP32.

These numerical policies require their own evidence. The learning rates,
momentum, Born–Infeld penalty, prior update policy and regularizer are unchanged.
All arms use protocol seed 0, public deterministic initialization, isolated
checkpointed RNG streams, and unchanged task architectures, data/seen batches,
initial prior distributions/capacities, sampling laws, update budgets and cadence.
No seed study, duration tuning or gate changes are authorized by this protocol.

The initial component profile reserves 480 seconds for current-source 64/256
controls, with plain and instrumented replay equality. Three feasibility probes
reserve 240 seconds each: BF16, then channels-last, then graph replay. These
are nonqualifying development diagnostics, and their original outcomes remain
separate. The first BF16 prototype misses the timing target and is stopped.

The four wide-image control/candidate arms reserve 1,200 seconds total, at most
300 seconds each, using the unchanged `bispf_fastpath_quality` tasks. Require
both complete candidate gates to pass before the full ordinary quality study.
The ordinary study reserves 46,620 seconds through Tier 3 and retains all held
failures. The final same-source performance comparison reserves 2,400 seconds
for two whole recipes × six unchanged workloads, run serially on one physical
GPU. All original per-task wall caps remain enforced; no paid retries are planned.
Finite synthetic updates establish execution/throughput, not natural-image
quality. Contention receipts delimit observed ratios; point-in-time process
queries cannot establish exclusive occupancy throughout a run.

Bulk artifacts and tail-friendly logs live outside Git at:
`/mnt/ml7tb/ParticleGAN-forge/bispf-throughput-20261017/`.
Only compact metrics, provenance, reports, reproduction sources and
actual-training GIFs will be published here.
