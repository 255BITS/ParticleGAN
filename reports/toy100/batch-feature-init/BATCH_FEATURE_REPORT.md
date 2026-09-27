# QR with an initially neutral batch-distance branch

The exact recipe, equations, and linear/conv/transformer/LoRA mapping are in
[the initialization guide](../../../docs/initialization.md). Transformer and LoRA
adaptations are proposals; they are not covered by this training result.

Completed 2026-09-26 (America/Denver). **The candidate passes 22/22 benchmark
tasks.** It uses the original optimizer-time QR initializer and zeros only the
coefficients connecting explicit batch-distance features to the critic output.
All 22 tasks were freshly executed, plus both hold/shift stress checks.

| Initializer | Benchmark gates | Historical priority convention |
|---|---:|---:|
| Existing K3P reference (published) | 22/22 | 8/8 |
| Original optimizer-time QR (archived matched controls) | 21/22 | 7/8 |
| QR + all discriminator outputs zero | 19/22 | 6/8 |
| **QR + batch-distance coefficients zero** | **22/22** | **8/8** |

Priority counts pre-shift STAY, not shifted-target recovery. Eight-offset
robustness was not rerun; no new percentage-of-K3P robustness score is assigned.
No seed, hash-salt, optimizer, loss, schedule, architecture, sampling, budget,
or acceptance-threshold search was performed in this follow-up.

## The identified mechanism

The failed QR unequal-mass critic concatenates 96 learned per-point features
and four batch-distance features before its scalar output. At initialization,
the four batch-feature coefficients produce about
25.79 times the input-gradient RMS of
the per-point contribution on the initialized generated cloud. Their summed
score induces a negative-definite covariance velocity: a contracting initial
preference unrelated to the target distribution.

The local coincident-cloud formula has coefficient sum(b_k/s_k^2) =
-4.349527. Its negative sign predicts contraction when
ascending the summed batch score. The autodifferentiated Hessian agrees with
the formula to maximum absolute error 0.
See [BATCH_FEATURE_MATH.md](BATCH_FEATURE_MATH.md) for assumptions and derivation.

Initializing those four coefficients to zero removes that initial coupling
while retaining the learned per-point network's original QR gradients. The
batch coefficients remain trainable. There is no task-name lookup or access to
target data. Critics without these explicit batch features retain original QR.

## Intervention results

| Screen | Unequal mass | Trajectory | Stripes | Ring |
|---|---|---|---|---|
| Zero only batch-feature coefficients | PASS | evaluated in full suite | evaluated in full suite | evaluated in full suite |
| Zero only per-point coefficients | FAIL: minimum covariance eigenratio 0.0343 | not run | not run | not run |
| Halve every critic output | PASS | FAIL: identity MSE 0.5244 | PASS | FAIL: 7 modes |

The selected candidate changes exactly four initial scalar values on unequal
mass. Its final minimum component covariance eigenratio is
**0.695370** (required >=0.15; original QR
0.02833), mass TV is 0.025488, normalized SW1 is
0.044885, and HQ is 1.000000. It ends with
7 consecutive passing
observations (required 5).

The previous zero-output regressions are checked directly: trajectory identity
MSE is 0.00074521 (limit 0.02), stripes PASS,
and grid100 holdout center RMS/sigma is 0.106728 (limit 0.20), with
7 stable coverage
observations. Full details: [batch-feature-leaderboard.md](batch-feature-leaderboard.md).

## Repeatability and validity

- All 30 follow-up runs completed: six screening runs and 24 full-suite/stress runs.
- All 27 runs with audited randomness match their original QR sample streams.
- All 19 small full-suite initializations match QR except the four intended
  unequal-mass coefficients. Training/configuration/driver hashes match controls.
- The 18 unchanged small tasks reproduce final learned
  parameters, optimizer states, and saved global RNG states bit for bit.
- Both stress runs exactly reproduce all recorded trajectories and full RNG
  digests. Their archived drivers do not save final parameter checkpoints, so
  final parameter/optimizer parity is not claimed for those two runs.
- All three native tasks reproduce original QR final and independent holdout
  arrays bit for bit (live, EMA, and target samples). Native drivers do not
  record the full RNG digest used by the smaller tasks.
- The unequal-mass candidate was independently rerun on the other A6000 and
  reproduces final learned parameters, optimizer states, and saved global RNG
  states bit for bit.
- Both frozen source manifests remain unchanged: 1,629 files checked per phase.

Receipts: [batch-feature-audit.json](batch-feature-audit.json) and
[batch-force-diagnostic.json](batch-force-diagnostic.json).

## Stress behavior and scope of the result

Long hold: **PASS**. Shifted-target recovery:
**FAIL**, with pre-shift STAY passing
120/120. Both stress trajectories are exactly
the original QR trajectories. This change addresses the unequal-mass failure
without changing the other hosts; it does not solve shifted-target recovery.
The minimum HQ in the required hold is 0.903564.
The extra observation window ends at step 2900 with
all eight modes and HQ 0.999023, above the 0.90 threshold.

The mathematical guarantee is precise: the batch branch contributes no input
gradient or cross-sample derivative at initialization. The finite training
gates and deterministic replays are experimental evidence of convergence over
their measured horizons. A convergence theorem for the complete nonlinear,
sampled Adam trainer remains open.

## Recommended starting point and reproduction

Use the tested original optimizer-time QR construction, with auxiliary
batch-distance readout coefficients initially zero. This is now the preferred
research starting point for these fixed toys; transfer beyond the evaluated
architectures/data remains unmeasured. Do not zero every discriminator output.

The tested initializer is named `batch_feature_zero` in `failure_init.py`.
The frozen source is under `/ml2/hypergan/pr194-init-search-20260926/batch-feature-full-suite/source`. `failure_worker.py` installs
`ortho_init.py`'s original F construction before models and Adam optimizers are
created. The exact research adapter is now available through the registry as
`batch_feature_zero`; the public recipe uses the direct `initialize_` API.
See [the integration report](README.md) for the difference and its validation. The PR's
existing `qr_pb_pq` registry entry resolves a different construction and should
not be substituted for this frozen implementation.

For an explicitly supplied `BatchDistanceDiscriminator` after the tested QR
initialization, the additional operation is simply:

```python
with torch.no_grad():
    critic.head.weight[:, -critic.scales.numel():].zero_()
```

The experiment commands, environment and output paths are retained in
`/ml2/hypergan/pr194-init-search-20260926/batch-feature-full-suite/events.jsonl`. `jobs.json` records every declared job. The same
initializer is used for all jobs. To reproduce, use a new output directory
with a recorded command; existing outputs are preserved.

```bash
python reports/toy100/pr194-init-diagnosis/collect_failure_search.py
/tmp/k3p-audit-20260925/venv/bin/python \
  reports/toy100/pr194-init-diagnosis/audit_failure_search.py
```
