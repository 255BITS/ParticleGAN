# RA9 native affine and conditional prior means — fixed saved-state diagnostic

One corrected completing CPU pass used only final7000 tensors and original
paired20k/100k arrays. No fitted chart, forward module, draw, emission, action,
quality scorer or gate was run. The original grid remains canonicalVALID/FAIL.

## Actual trained maps and compensation

The native map was identity initialized. Final trained FAST and EMA matrices
have diagonals about0.9768, off-diagonals about0.002, and biases about
(0.0217,−0.0172). Every decomposition uses those saved A,b values. This native
initialization reference is not a generic identity-generator assumption.

For fixed diagnostic component annotations of actual x=A z+b:

`mean(x)−t = A(mean(z)−t) + (A t+b−t)`.

| Raw checkpoint role | Transformed table RMS/σ | Affine RMS/σ | Twice cross/σ² | Residual centroid RMS/σ |
|---|---:|---:|---:|---:|
| FAST | 3.272010 | 3.276292 | -21.372945 | 0.259210 |
| EMA | 3.264072 | 3.277923 | -21.358986 | 0.199888 |

EMA table/map centroid-vector cosine is−0.99814: the large movements cancel.
The residual is small compensation error expressed in the current map.
Replacing G with its initial identity would undo that compensation. Algebraic
inverse-map target means are interpretations only, not latent proposals.

FAST-to-EMA paired row displacement is0.375252σ from table coordinates versus
0.004190σ from the map difference; all rows have the same diagnostic output
component. EMA prior averaging therefore explains most of the final saved
FAST/EMA view difference. The saved clean holdout centroid vectors closely
track EMA anchors (difference0.019122σ, cosine0.99542). Live/EMA saved arrays
are byte-identical under the eligible paired-average lease19888/19000.

Relative to saved holdout real-target means, EMA anchor residual RMS is
0.204126σ; relative to the training FIFO it is0.233748σ. Those references
have finite-sample and training-adaptation uncertainty. No new confidence
claim or correction is derived. Oracle components annotate diagnostic points
only; there is no production fit to oracle centers.

The final-state decomposition is not a historical causal ablation. It supports
examining real-only conditional mean compensation in the current G/D chart.
Any generic repair must account for current G and EMA_G independently,
support/component validity, uncertainty, shared action accounting and noise;
it cannot assume that raw latent coordinates are output coordinates.

## Retained technical failure and immutable evidence

The original helper/preparation/failed log remain unchanged. Its supplementary
raw-z annotation wrongly required3σ output coverage in every group; the
trained map makes that requirement invalid. The corrected helper computes
raw prior conditional means using the already fixed actual-output row groups.
Only that supplementary block and the preparation filename changed; reversing
both text substitutions restores the complete original helper bytes. There
was no parameter, input, selector, affine/target equation or quality change.

The first partial pass failed before producing any result; one corrected pass
completed. `CORRECTION.json` and the corrected prefreeze bind this distinction.
All57 corrected source/input maps and the original53 map subset remained exact.
CPU and NumPy global RNG remained exact; CUDA was never initialized.

`receipt.json` is the diagnostic result, `PREPARATION-CORRECTION-FROZEN.json`
is the pre-execution seal, and `FROZEN.json` is the authoritative post-exit seal.
Float64 algebra on saved float32 coefficients is not historical GPU replay.
