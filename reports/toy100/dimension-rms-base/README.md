# Selected research base: dimension RMS hybrid

The user selected `dimension_rms_hybrid` as the starting formulation for the next
GPU search cycle. It is a research base, not a release-qualified default.

**6 PASS / 7 measured GPU toys; 15 NOT_RUN.** All six previous regressions pass:
trajectory, ring, unequal mass, intensity, bars and blobs. The next gate,
`two_pole`, fails: movement .104394905 < .30, with zero passing observations.
Its gradient median .066183 passes its <=1 bound. No full-22 score or own-state
post-convergence qualification exists for this candidate.

The GAN uses original Rp logistic and Adam with the retained decay/noise schedules
and original auxiliary AE/token host terms. Its critic penalty is

`lambda/2 * (mean(||grad D(real)||² / d) + mean(relu(||grad D(fake)|| / sqrt(d) - kappa)²))`

where d is the number of input elements per example, lambda=1, kappa=1. The fake
term is b-cap; the real term is dimension-normalized R1. No target centers,
labels, metric feedback or target fitting enters training.

**Load config.json together with mechanism.py through probe.py.** The legacy
`reg_arm: a_r1r2` config value selects the patched path; config.json alone runs a
different formulation. The exact executed worker and mechanism hashes are pinned
in original-declaration.json and the active current-research-base.json.

CPU initialization is retained; all model training, gradients and Adam state
(including counters) are CUDA FP32. Original non-capturable Adam arithmetic is
preserved; a 25-update check found exact parameter/moment agreement. Use
PyTorch2.13.0+cu126, deterministic algorithms, TF32 off, one CPU thread and the
same GPU profile. PyTorch2.14 did not fix the earlier blockers.

| Gate | Verdict | Final metrics | Terminal passing checks |
|---|---|---|---:|
| trajectory | PASS | MSE .000738294 | 22 |
| mode_hold | PASS | 8 modes, HQ1 | 15 |
| vector_unequal_mass | PASS | min eigen ratio .388526 | 10 |
| img_intensity2 | PASS | HQ1 | 7 |
| img_bars4 | PASS | HQ .96875 | 20 |
| img_blobs4 | PASS | HQ .96875 | 17 |
| two_pole | FAIL | movement .104395 < .30 | 0 |

[Independent seven-gate audit](audit.json) · [Exact raw records](results/) ·
[Original declaration](original-declaration.json) · [Previous round](round-summary.json)

The prior round tested nine proposals and 24 GPU training gates: 14 PASS,10 FAIL.
Two receipt-audit corrections reused saved results and are not extra training.
The other critic-scale candidates failed ring; the conditional-cap variant failed
intensity; the cap-dynamics variants failed either trajectory or both coverage
blockers. Do not rerun those unchanged proposals. Original reports and exact local
provenance are retained; this bundle includes the selected candidate's full code,
metrics, CUDA/source receipts and the extra two_pole initialization fixture.

Replay a recorded gate, requiring all non-timing metrics, initial parameters and
random draws to match the retained receipt:

```bash
/tmp/pr38-default-env/bin/python reports/toy100/dimension-rms-base/replay.py \
  --task two_pole --gpu 1 --workdir /tmp/dimension-rms-replay-new
```

The replay helper reconstructs checksum-verified baseline sources. It rejects
existing work directories and changed candidate bytes. The one-time promotion
replay is recorded in replay-check.json; it is verification, not a new proposal.

Next gate order: two_pole, trajectory, ring, unequal mass, intensity, bars, blobs.
Advance only measured successes; keep every failure. A candidate clearing these
seven needs its own remaining fifteen GPU toys, then own-state continuation.
Never add these six passes to the historical native 16/22 control.
