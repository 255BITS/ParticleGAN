# Previous research base: direct_particle_response

This historical research/launcher base has been superseded by
[K3P](../k3p-base/README.md). The measurements and sources below describe this
earlier formulation.
It remains an Rp logistic GAN with the parent dimension-RMS critic penalty.
Public training-package defaults are unchanged; release qualification is pending.

**15 PASS / 1 FAIL / 6 NOT_RUN across the 22 GPU toys.** All seven initial
gates pass. Unequal-width covariance is the remaining measured failure:
component covariance error .981321 exceeds .85, with zero terminal passing
checks. Its first component contributes error 2.72220; mass, HQ, SW1 and the
minimum-eigenvalue bounds pass. There were 11 passing observations out of 24,
so an earlier passing check would have hidden the failure.

The critic retains the parent's penalty:

`lambda/2 * (mean(||grad D(real)||² / d) + mean(relu(||grad D(fake)|| / sqrt(d) - kappa)²))`

Here d counts input elements per example; lambda=kappa=1. This is normalized
real R1 and fake RMS b-cap, with the original schedules and auxiliary host terms.
The new response applies only to direct sample-particle optimizer groups,
excluding registered ParticlePrior latent parameters and network parameters.
For their ordinary Adam update it uses betas (0,.9) and multiplies the scheduled
learning rate by `1 + clamp(cos(center(g_t), center(g_previous)), 0, 1)`.
The first gain is 1; all gains are between 1 and 2. Centering measures alignment;
the raw gradient is unchanged. The scheduled LR and betas are restored afterward.
No target statistics, mode identities, evaluation feedback, extra model calls
or extra optimizer steps enter this mechanism. The two optimizer changes have
not been causally isolated.

This response is active on two_pole among the sixteen measured hosts. Movement
improves from .104395 to .644163, with ten terminal passing checks. The six
previous regression hosts reproduce the parent's non-timing results in fresh
executions. Unequal width uses the unchanged latent path; boosting the direct
response further cannot fix that host.

**Use config.json + mechanism.py + response.py through the exact probe.py.**
The legacy config field `reg_arm: a_r1r2` selects an installed patch; config alone
runs a different formulation. All four hashes are pinned in the declaration.
Model training, gradients, response history and all Adam state tensors are CUDA
FP32. Retained CPU initialization performs zero optimizer updates. Keep the
PyTorch2.13.0+cu126 deterministic, TF32-off, one-thread environment and original
noncapturable Adam arithmetic.

**probe-fast.py measures the same numbers for less instrumentation cost.** It
caches the per-overload tag lookup in the random-op dispatch hook, clones
parameters and gradients only for recorded mobility steps, and stops recomputing
invariant optimizer receipts every step. The audited random-operation set, the
per-draw stream digest and every frozen spec are unchanged. Eight gates,
including the two_pole PASS and the unequal-width FAIL, reproduce the committed
results exactly under replay.py's non-timing comparison; probe-fast-check.json
holds the receipt and the measured times. A candidate adopting it declares that
hash in its own declaration.json; probe.py and the bundle hashes are unchanged.

| Measured toy | Verdict | Terminal passing checks |
|---|---|---:|
| ae_gan_hold | PASS | 22 |
| cover_leftover | PASS | 13 |
| img_bars4 | PASS | 20 |
| img_blobs4 | PASS | 17 |
| img_intensity2 | PASS | 7 |
| img_stripes2 | PASS | 9 |
| mid_scale_identity | PASS | 17 |
| mode_hold | PASS | 15 |
| residual_student | PASS | 19 |
| trajectory | PASS | 22 |
| two_pole | PASS | 10 |
| unipolar | PASS | 19 |
| unused_token_hold | PASS | 11 |
| vector_two_broad | PASS | 23 |
| vector_unequal_mass | PASS | 10 |
| vector_unequal_width | FAIL | 0 |

NOT_RUN: vector_anisotropic, vector_overlap, vector_spiral, grid100, rotated100,
staggered100. No full22 qualification or own-state post-convergence result exists.
The module-global response history must be saved/restored with model, optimizer
and RNG state before a checkpoint-resume stability claim can be made.

[Independent audit and metrics](audit.json) · [All sixteen raw records](results/)
· [Exact declaration](original-declaration.json) · [Previous round](round-summary.json)
· [Original detailed report](original-attempt-report.md)

The completed round tested nine proposals and 34 GPU training gates: 25 PASS,
9 FAIL. Real-gradient dead-zone and real-penalty warmup candidates failed either
movement or ring. Applying coherent response to latent priors too lost ring.
Do not rerun those unchanged proposals. The unipolar receipt auditor initially
assumed one regularizer call per step; its frozen host uses two scales. This
was corrected without training, and the original audit error is retained.

The bundle includes all sixteen zero-update initialization fixtures, exact code,
raw curves, CUDA receipts, immutable source manifest and original command logs.
Original attempt helpers are retained for provenance; replay.py is the portable
entry point and reconstructs the archived baseline sources without changing them:

```bash
/tmp/pr38-default-env/bin/python reports/toy100/direct-particle-base/replay.py \
  --task two_pole --gpu 1 --workdir /tmp/direct-particle-replay-new
```

The helper rejects existing work directories and verifies code, source and
fixture hashes. It compares every non-timing metric, verdict, RNG digest,
initial parameter, response/mobility receipt and CUDA update count against the
recorded gate. Promotion checks are in replay-checks.json. Reproducing the
unequal-width FAIL validates replay; it does not qualify the candidate.

Next: unequal width first, then protect all fifteen passes. Only a candidate
clearing those sixteen continues to its own six remaining GPU toys, then
own-state retention under its declared schedule. No inherited passes or seed
sweeps. Historical native GPU16/22 and CPU22/22 results belong to other recipes.
