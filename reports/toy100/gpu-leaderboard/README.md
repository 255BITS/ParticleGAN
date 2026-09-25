# GPU leaderboard rebuild

This evaluates eight continuous-learning variants on CUDA. The original
[22/22 CPU winner and its 16/22 GPU control](../gpu-known-winner-control/README.md)
are a separate preserved-recipe comparison. See the [leaderboard and
complete 22-toy matrix](LEADERBOARD.md). A toy PASS always comes from that
candidate's own GPU run. Unsupported cells receive no credit. The separate
convergence gate measures stability after acquisition, rather than rejecting
every dip during learning.

## Protocol

- `cuda_fp32_v1`: NVIDIA RTX A6000, PyTorch 2.13.0+cu126, CUDA 12.6,
  cuDNN 91002, driver 580.173.02, Python 3.12.13.
- FP32 model training; inherited curvature reductions retain float64.
  TF32 disabled; deterministic algorithms enabled;
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`; one CPU thread; unfused, non-foreach Adam.
  Three independent jobs share physical GPU 1. Every scored optimizer call
  asserts CUDA parameters and gradients; first updates also check CUDA moments.
- Existing seeds, targets, architectures, thresholds, and budgets remain fixed.
  The three native 100-mode toys each run 7,000 updates and require both the
  coverage and accuracy gates. Their existing affine-generator model card is
  retained; older toys retain their own declared architectures. All policies
  remain adversarial GAN policies with learned latent particles.
- All supported toys run even after a quality failure. A harness error pauses
  dispatch; the failed attempt is retained and retried after the device fix.
  There are no seed sweeps or CPU-result substitutions.
- Convergence uses a fresh cold run, the first 200 consecutive full-ring checks
  with HQ >= .90 after update 1,200, and then a disjoint 1,200-update hold.
  Confirmation must finish by update 6,000. Stop at the first hold failure;
  never select a later passing window. Measurements are dense after 1,200.
  PR140/143 retain their existing quality-triggered arm only on the original
  10-update observation cadence; the new dense observer adds no control events.

Four policies support all 22 toys: epsilon, H, shared RMS, and shared column RMS.
The published averaging adapter supports only two-pole, ring, and unipolar.
The published PR107/140/143 adapters support only ring and trajectory. They are
limited comparators, not fully qualified 22-toy candidates. No different policy
is substituted to fill their missing cells. All eight get convergence runs.

The declared matrix therefore has 97 executable GPU toy runs, eight GPU
convergence runs, and 79 unsupported toy cells. Raw coverage counts are research
comparisons; promotion still requires the required toy gates and hold to pass.
Live-model scores determine verdicts; EMA does not rescue a failed live model.

## Reproduce

Use an environment matching [requirements.txt](requirements.txt) and the CUDA
profile above. The replay helper checks the archive and every extracted source
file, verifies the CUDA stack and GPU model, and refuses a CPU fallback.

```bash
python replay.py --gpu 0 --candidate shared_column_rms \
  --task mode_hold --workdir /tmp/particlegan-gpu-ring-new
python replay.py --gpu 0 --candidate shared_column_rms \
  --task convergence --workdir /tmp/particlegan-gpu-hold-new
```

Use a new work directory for every invocation. `--task` accepts any supported
name in [protocol.json](protocol.json). GPU identity and commands are saved for
each run. Other GPU families need a separate recorded profile and leaderboard;
this receipt does not establish cross-device bitwise agreement.

To recompute every verdict from the retained results and native sample evidence,
without retraining:

```bash
python audit.py
CUDA_VISIBLE_DEVICES=0 python audit_rng.py
```

The original unattended three-worker sweep is `batch.py`; its original absolute
worktree paths and physical GPU 1 selection are preserved for provenance. Use
`replay.py` for a portable run. Original logs can be tailed with
`tail -F /ml2/hypergan/gpu-leaderboard-20260924/batch.log`.

## Evidence and device migration

- [Candidate declarations](candidates.json), [protocol](protocol.json),
  [source archive checksums](source-archives.json), and `sources-*.json` bind the
  candidate options and CUDA adaptations. Archives include all executable
  benchmark, policy, public trainer, and helper sources.
- `runs/<candidate>/<task>/` contains the declaration, result, device proof,
  optimizer/policy trace, and status. Native runs additionally retain their
  coverage/accuracy evidence. Convergence checkpoints capture model, optimizer,
  and RNG tensors for inspection; replay starts cold and does not claim that
  these files serialize every adaptive policy's Python state.
- The [RNG validation](rng-validation-detailed.json) compares 72 tensor leaves
  and one generator state (73 state tensors total) after a
  PR143 prefix with sparse versus dense measurement. Model, Adam, data/noise,
  CPU, and CUDA states match exactly. Measurement cadence does not change that
  tested prefix's trajectory.
- A [portable archive replay](portable-replay-validation.json) of shared column
  RMS reproduces all 24 cold-ring observations and every non-timing result field
  exactly. [Negative replay checks](replay-guard-tests.json) reject unavailable
  CUDA and altered source archives before training.
- Device plumbing selects CUDA generators, preserves CUDA evaluation RNG, and
  includes CUDA state in curvature replay. The public training package is
  unchanged. Two older host assumptions required [repairs](repairs.json): a
  mid-scale CPU-only guard, and autoencoder CPU-only RNG restore. Both failed
  before their first optimizer update. The original error receipts remain;
  their successful retries determine the matrix.

Previous CPU receipts remain historical comparisons and are not pooled into
this leaderboard. This GPU run does not adopt the MKL vendor-dispatch shim.
The old PR143 115/120 sparse CPU stay and this dense post-convergence hold use
different windows and cadences; their counts are not directly comparable.

## Current priority

The user has prioritized [porting the original 22/22 CPU recipe to GPU](../cpu-recipe-gpu-port/README.md).
See the [active base declaration](../current-research-base.json). The old
continuous-rate reference below is historical; its search is secondary.

## Earlier continuous-rate experiment priorities

The earlier recommendation was to use shared column RMS as the full-coverage GPU research reference: it ties for
the most toy passes, passes cold ring acquisition, and has the longest measured
post-convergence hold among the four fully tested candidates. This is a starting
point, not promotion of a passing release. Keep H as an image-task comparator.

Gate new changes on the recorded failures before paying for another native
100-mode sweep. The shared failures include cover-leftover, unequal mass, AE
hold, and unused-token hold; column RMS additionally loses image bars, blobs,
intensity, and unequal width. The existing AE encoder has no adversarial path
under this strict policy, so changing its learning rate cannot fix that missing
signal. Any repair still needs a genuine adversarial training path.

Allow the bounded convergence diagnostic when testing late stability, even if
the earlier acquisition suffix fails. Judge the first confirmed hold, never a
later selected window. Keep seeds fixed and compare changed mechanisms on this
same CUDA profile. The completed all-toy audit is evidence to use, not a batch
to repeat unchanged.
