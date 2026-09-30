# Geometry correction

The private patch changes `feature_cells.py` and the corresponding prior
argument in `training.py`. Apply `geometry.patch` to the shared candidate;
merge the training dispatch with the stability lane's actual-backend fallback.
The unchanged E22/reference path retains its original sampler.

## What failed

The fixed .05 latent norm cap gives RMS .004419 at z128. The final CUDA E22
MNIST training perturbations were .513–.535. Candidate row copies used the
same small cap. Its final median nearest latent distance is .269, versus
13.670 for E22, after 5182 ordinary copies versus E22's 415 total moves/83
isolation moves. Many candidate rows form dense clone families.

Saved-model inference on CPU separates the causes. Rows and Gaussian draws
are exactly matched across kernels, using seed314259, 4096 samples and the
original evaluator/normalization/k5 rules. These are diagnostics, with a
different CPU random stream from the original CUDA evaluator.

| Saved MNIST model | Kernel | Active recall | Active FD |
|---|---|---:|---:|
| E22 | Fixed .025/.05 | .7554 | .8392 |
| E22 | Exact DV12 | .8623 | .6924 |
| E22 | Proposed local kernel | .8955 | .6652 |
| CB64-RA | Fixed .025/.05 | .2925 | 1.4486 |
| CB64-RA | Exact DV12 | .3345 | 1.4494 |
| CB64-RA | Proposed local kernel | .3394 | 1.4479 |

The candidate's clean-center FD is 1.4363 versus E22's .6148. Equalizing
classifier mass reduces candidate clean-center FD only to 1.3511. Final
serving decisions are the same (fast iterate), so neither final sampling
alone nor a class-weight correction explains the trained-center regression.
The new law must be applied while training and copying rows.

## Correction

Retain the original controller's coordinate bandwidth. Build deterministic
sort orders on up to rank8 varying latent coordinates and query at most64
nearby distinct projected coordinates. Full-vector distances select a local
neighborhood; the norm cap is half the nearest nonidentical candidate distance.
Bound each coordinate's bandwidth by the RMS pair difference among the
nearest rank8 candidates divided by sqrt(2). This supplies local anisotropy
from the particle table and suppresses unused/narrow coordinates.

The first isotropic bounded proposal improved folded copies but retained
only .88235 rare mass after four turns, below the unchanged .90 gate. That
proposal and its results remain in `isotropic-proposal/` and `*-isotropic.json`.
The local width correction addresses this remaining narrow-coordinate error;
there was no seed sweep, scalar search, oracle input or gate change.

| Four actual folded2D turns | Support | Rare/target | TV | Original final gate |
|---|---:|---:|---:|---|
| Original fixed kernel | .97070 | .60784 | .03760 | FAIL |
| First bounded isotropic proposal | .99414 | .88235 | .00586 | FAIL |
| Proposed local kernel | 1.0 | 1.0 | 0.0 | PASS |

The final local run makes 85 isolation copies and zero ordinary moves. The
original fixed run makes 85 isolation and37 ordinary moves. Its detector
still has three supported false positives, including rare row1745. Thus the
sampling/copy gate passes while complete family qualification remains open
for the support lane. On the saved CPU native20k tight-density state, local
kernel RMS is .00119 versus exact DV12 .00075 and fixed kernel .02356.
This does not certify native CUDA quality.

The neighborhood radius is approximate and may exceed the exact DV12 radius.
There is no nearest-neighbor equivalence claim. Setup is O(rank*N*logN), query
degree≤64, and query blocks≤256. Live/EMA prior selection is explicit. Derived
orders use prior identity/tensor version; copies, optimizer motion, serving
swaps and loads trigger rebuilds. No derived cache enters checkpoints.
Backend schema2 and kernel settings reject old fixed-kernel continuation.
New-law checkpoints replay exactly.

## Verification and CUDA command

Five focused CPU tests pass: shared training/fake-pool/copy kernel and latent
gradient; optimizer/EMA/history row copies; corresponding EMA geometry and RNG
ownership; analytical cache/duplicate cases; exact checkpoint continuation
after nonzero ordinary moves. Package test hashes match current private bytes.

`GPU-INPUTS.json` and `gpu-inputs.pt` contain six fixed saved-input cases.
Root schedules this single command through its GPU slot coordinator:

```
/tmp/pr38-default-env/bin/python -u -B /ml2/hypergan/gan-attempts/feature-cells-fixes-20260929/geometry/gpu_kernel_check.py
```

Use `--package-root /ml2/hypergan/gan-attempts/feature-cells-fixes-20260929/pkg-CB64-RA2`
to check the integrated candidate. The command verifies physicalGPU0 identity,
memory/thread/determinism policy, frozen input hashes, bounded work, cached
repeat equality and actual CUDA row-copy/EMA/optimizer/history/private-RNG/load
contracts. It runs no training or original quality acceptance. CUDA retraining
and the unchanged full acceptance suite remain root's next validation step.

Artifacts: `saved-baseline-results.json`, `saved-proposed-results.json`,
`folded-results.json`, `copies-density-results.json`, `test-results.json`,
`geometry.patch`, and `READY.json`. All baseline audit/study/source bytes
remain unchanged.
