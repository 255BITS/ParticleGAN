# Archived and current ring16 binding audit

The archived 1,600-update ring PASS and the current selected BCAP ring FAIL
have the same recipe, prior, initialization, numerical gates and scoring
cadence. Their execution histories differ: the archive restores and extends a
400-update checkpoint; the current run executes 1,600 updates uninterrupted.
Saved samples and every scored metric match exactly through update 400, then
first diverge at 417. This audit does not establish the cause of that divergence.
Both original outcomes remain under their actual source and execution identity.

The supplied `tier1-prior-smoke-v1/mog100-n256/ring16_acquisition` directory
contains the original **400-update FAIL**. Its separate
`tier1-prior-duration-v1/mog100-n256-ring16_acquisition` continuation completed
1,600 updates and recorded **PASS**, with six passing terminal checks. The
current attempt is `2e6abacb6ffe4423be611480f4bb6496`, selected configuration
`bcap-dualnorm--7beb7378d81dc3be2c648438661e0376fe2805298232f5c2398be835ddaad6f9`.
It completed all 1,600 updates and recorded numerical **FAIL**; its raw execution
status is `completed`.

The full resolved recipe dictionaries match exactly: DualNorm BCAP, G rate
0.012, D multiplier 1.5, prior multiplier 2.5, batch 128, schedule horizon 400,
LR floors 1, no input/output noise or latent damping. Both use a learned,
unstandardized 256-particle MoG with sigma 0.1; z is 4, hidden width 64, depth 2,
and the critic has two Fourier frequencies. Initial G, D and prior state hashes,
parameter initialization seeds and the complete named RNG manifests match.
The training data stream is `data/target/training/cpu`; evaluation uses
`eval/live/samples/cuda:0`, with 4,096 clean samples at each of the same 96
checkpoints. All 91 checked public trainer/library/host/scorer source-file hashes
match. Ring target geometry and full distribution thresholds are unchanged.

All 24 saved sample tensors and metrics through update 400 are byte-identical.
At the first differing checkpoint, 417, median point displacement is 0.023211;
at 1,600 it is 0.058058, with maximum displacement 3.035741. The final numerical
comparison is:

| Metric | Archived continuation | Current uninterrupted run |
|---|---:|---:|
| Modes | 16 | 16 |
| Quality fraction | 0.937744 | 0.936035 |
| Mass TV | 0.058594 | 0.057129 |
| Normalized sliced W1 | 0.024723 | 0.028891 |
| Component covariance error, required ≤ 0.85 | 0.514315 | 2.220268 |
| Component core covariance error, diagnostic | 0.384515 | 0.387413 |
| Minimum component eigenvalue ratio, required ≥ 0.15 | 0.383696 | 0.337657 |

The current endpoint fails the untrimmed component covariance bound. Component
11's covariance error is 29.243107, versus 1.833634 in the archive, and dominates
the average despite similar core covariance and overall quality. The current
complete-curve reducer records no full passing observation and a zero passing
terminal suffix. None of these gates is changed by this report.

The archived continuation rebuilds the public host, restores its full serialized
context and named streams, verifies the restored state digest, then changes only
the external execution cap from 400 to 1,600. Its retained resume proof states
that the serialized state was restored exactly; the parent and restored state
digest is `208e2d7b241ffeac11915972dbb90979dd686ad5285768282e2182d7da5ad2cd`.
No reset of serialized optimizer/history/RNG state is inferred. The current
adapter also gathers public update statistics and software diagnostics, whereas
the older direct caller does not; their first 400 updates still produce identical
saved observations. The initial illustration in the archive preserves every
named stream and adds no consumed scoring draw.

The current generic ring task retains scored arrays but produces no learned-model
checkpoint and records no training-data digest. Consequently, its post-400
learned-state and consumed-stream equality cannot be established against the
archived continuation using these receipts. Similar sample positions suggest
aligned draws, but they do not prove RNG or learned-state parity. This evidence
does not identify a restore defect, a trainer regression or an evaluation-only
cause. Retain the historical PASS and report the new covariance FAIL honestly.

[The compact audit receipt](ring-binding-audit.json) binds original file hashes,
saved tensor hashes, exact matching checkpoints and both source identities:
prefix `ec9be602b2b8d587b8c8c3f8bc10e98c96580dc1`, continuation
`b4d1f95a074ffa64ac8f64e12aa0a049f836c0af`, current
`a4caa21d1039684d23e57be8e133b51b3b120775`. The historical
[duration report](../tier1-prior-duration/README.md) retains the original PASS
and its actual-training GIF. This display-only audit adds zero updates,
observations or sampling draws and grants no qualification credit.

Recreate it from the original saved receipts and arrays:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /usr/bin/python \
  reports/forge/gaussian-smoke-inventory/audit_ring_binding.py \
  --prior-root /home/martyn/dev/ParticleGAN-tier1-prior-smoke \
  --inventory-root /home/martyn/dev/ParticleGAN-gaussian-smoke-inventory
```

[The audit source](audit_ring_binding.py) loads saved arrays on CPU and never
constructs, restores or evaluates a model. CPU is used only for receipt and
saved-output comparison; no neural execution occurs.
