The exact current-source default `baseline/movable` jobs pass both paired target
laws at their declared 6,000-update budget. Each now has a GIF with 13 real
validation checkpoints. This adds fresh training evidence for catalog IDs
`source-family-08` and `source-family-09`; the original catalog and historical
failure map stay unchanged.

| Data law | Original live / EMA validation NMSE | Original live / EMA p95 distance | Added live / EMA paired MSE relative to identity | Result | Actual training GIF |
| --- | --- | --- | --- | --- | --- |
| Affine2 | 0.00026991 / 0.000020391 | 0.039155 / 0.011897 | 0.0005683 / 0.00004293 | Both original gates and added gate **PASS** | [13 checkpoints](media/affine2-baseline-movable.gif) |
| Swirl2 | 0.0048225 / 0.0013840 | 0.15892 / 0.074438 | 0.0024738 / 0.00070995 | Both original gates and added gate **PASS** | [13 checkpoints](media/swirl2-baseline-movable.gif) |

Affine2 verifies the specified row-wise transport
`y = x @ [[.8,-.6],[.6,.8]] + [.2,-.3]`. Swirl2 verifies a nonlinear,
radius-dependent rotation with angle `1.7 * ||x/sqrt(3)||²`, preserving each
input point's radius. The source rows are a uniform square scaled by `sqrt(3)`.
The task is to predict **the target belonging to each input row**. Matching the
target cloud alone does not satisfy these paired errors. Red links in the GIFs
join the EMA prediction and target at the same fixed row indices; the curves
score every one of the 1,024 validation rows.

The source [task](../../../benchmarks/paired_error_2d/task.py) and
[runner](../../../benchmarks/paired_error_2d/run.py) remain unchanged. Each job
uses seed 0, a 128×4 movable bank, generator width 48, router width 16, batch 64,
1,024 training rows and 1,024 validation rows. Source initialization preserves
its exact draw order: generator seed 0 and critic seed 10,000. The baseline
uses generator/critic/particle rates 0.0006/0.0009/0.006, Adam `(0,.999)`, EMA
`.995`, cap threshold/coefficient 1/1 with every-fourth-step compensation, and
particle VIC weight 1. The paired Gaussian residual-noise schedule retains its
8,000-step horizon and hold rule; it is not rescaled to the 6,000-update budget.
The current source explicitly imports its retained legacy logistic/RpGAN loss
and gradient-cap implementations. This is source execution, with no transfer of
its result to native Atlas or a new production recipe.

Original `stable_live` and `stable_ema` each require the complete budget and
NMSE≤0.01 plus p95 distance≤0.2 at all three final original validation
observations. Both flags pass for both jobs. The separate added definition uses
clean paired MSE divided by the identity witness's MSE, requiring a ratio≤0.10
for both live and EMA and at least five terminal passing observations. Affine2
passes all 12 non-initial observations; swirl2 passes its final 11. These
validation metrics are reporting/selection measurements, not output MSE in
training. The unchanged training objective is paired-error RpGAN plus its
source cap and particle VIC.

The audit registers one scientific job per target law before execution. The
source CLI normally runs three recipes and two cloud controls per law, totaling
twelve jobs. Here `baseline/movable` is the source's default model control;
fixed-cloud and scheduled-candidate arms were not run or selected afterward.
The actual attempts cost 22.32 seconds for affine2 and 18.91 seconds for swirl2,
41.23 seconds total on a shared CPU with one thread per job, below each
120-second cap. No batch reduction, retry, seed study or failed-gate extension
was used. These runs do not establish that particle movement or the routed code
is necessary; those would require the omitted matched model controls.

The source `freeze_and_evaluate` requires all twelve jobs to finish before
freezing checkpoint choices and opening the historical test split. That rule is
retained: this audit did not call the test evaluator or open test rows. The
reported numbers are terminal **validation** results; the best validation EMA
observation is separately labeled in the compact receipt. Neither a full
twelve-arm test qualification nor Forge/default qualification is inferred.

Before each full attempt, four exact source updates compare observer on/off
state and loss hashes, including the first lazy cap. These short software
controls are not extra scientific arms. During each full job, the observer acts
only after the source saves its original `latest.pt` checkpoint. Independent
inference snapshots and private RNG preservation leave the full G/D/Adam/EMA,
sampler and global RNG state unchanged at all 13 captured boundaries. Source
updates, original validation checkpoint selection and checkpoint files retain
their original behavior. All GIF coordinates are captured states with no
interpolation.

[coverage.json](coverage.json) has a `records` list with catalog IDs, named jobs,
execution/original/added statuses, relative GIF paths, final and labeled best
metrics, exact source/runtime/recipe identities, prefix controls and external
archive hashes. Full source histories, observation arrays, stdout and all
checkpoints remain outside Git under
`/ml2/hypergan/toy-audit-artifacts-20261001/source-paired-transport-v1`.

Reproduce either named job in a new artifact directory:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_CBWR=AVX2 \
  python -u -m benchmarks.toy_audit.source_paired_transport_training affine2 \
  --artifacts /tmp/new-paired-source/affine2 --output /tmp/new-paired-source/affine2-receipt.json
MPLCONFIGDIR=/tmp/toy-audit-mpl python -m benchmarks.toy_audit.source_paired_transport_media \
  --receipt /tmp/new-paired-source/affine2-receipt.json \
  --output /tmp/new-paired-source/affine2.gif --media-receipt /tmp/new-paired-source/affine2-media.json
```
