# High-dimensional support diagnosis

## Status

The original GPU baseline is valid and fails quality acceptance. The saved-only
audit passed 660/660 checks: four learned runs completed 2000 updates, four
checkpoint replays passed, and all 16 screens completed with nine PASS/seven
FAIL. [CHECKS.json](../../../feature-cells-cuda-retest-20260929/audit/CHECKS.json)
records that distinction.

The integrated geometry kernel contract is independently valid: six fixed
input cases, all actual row-copy checks, physical GPU0 identity and unchanged
selected-package hashes passed. [GPU-CONTRACT-REVIEW.json](GPU-CONTRACT-REVIEW.json)
is a review of that diagnostic, not corrected-model quality acceptance.

No universal support correction is qualified yet. The new real-only Fisher
metric fixes the high-dimensional detector failure but leaves folded controls
that fail. No shared package, original study or prior READY source was edited.

The original frozen static E22 reference also fails these gates: zero recall
in all four highdim cases, trained d128 folded recall 0/.829/.585 at
N1024/2048/4096, and two rare FP in trained N2048. Only one of ten archived
d128 geometry detector controls passes. Full jitter qualifications also fail.
[REFERENCE-CONTEXT.json](REFERENCE-CONTEXT.json) preserves the original CPU
receipts and hashes. A universally passing E22 fallback has not been shown;
these failures do not change any acceptance gate.

## Cause and fixed-case receipts

The original high-dimensional critic captures 128 varying dimensions. Planted
unsupported rows have narrower nuisance coordinates than real rows, making
their aggregate orthogonal residual *smaller*. The original rank-eight radial
metric, full standardized actual-real anchor distances and aggregate residual
norm all have zero recall. This is not absence of captured critic information.

Between-cell versus within-cell variation can isolate the missing direction.
Fit a dictionary of existing even-real cell means with width at most 64, whiten
its within-cell covariance, and retain at most the existing eight directions.
The original PCA cell partition/count law remains unchanged. Odd rows still
calibrate the original finite upper-tail p-value and BH Q=.05; labels are read
only when computing the final diagnostic metrics.

| Fixed highdim size | Original recall | Bounded Fisher recall | Bulk/rare FP |
| --- | ---: | ---: | ---: |
| 1024 | 0 | 1 | 0/0 |
| 2048 | 0 | 1 | 0/0 |
| 4096 | 0 | 1 | 0/0 |
| 8192 | 0 | 1 | 0/0 |

All four original detector gates pass in this CPU diagnostic. Both the full
covariance reference and bounded dictionary results are preserved in
[diagnosis.json](diagnosis.json). Replacing arbitrary QR completion with a
numerical SVD dictionary rank retains the result;
[diagnosis-rank.json](diagnosis-rank.json) also records the failing folded
controls. This is not a GPU acceptance verdict or a seed sweep.

## Anchor normalization was tested and rejected

The initial full-feature anchor test used raw distance. A focused second test
used the existing 64 actual-real representatives in Fisher geometry and each
representative's original reference k-th even-real leave-one-out radius.
Selecting the nearest actual anchor before dividing by its radius preserves
locality. It passes all four highdim gates (recall .891/.957/.886/1, zero FP)
but fails trained folded 1024/2048 with zero recall and trained folded 4096
with ten bulk FP. Taking the minimum ratio over all anchors also fails.

The trained folded 1024 centroid score has bad minimum 57.20 below null maximum
82.13. Raw actual-anchor distance changes this to bad minimum 508.55 above null
maximum 360.72 and detects all 41 planted rows with zero FP. Local-radius
normalization reintroduces calibration tail overlap. The trained folded 2048
raw-anchor control still flags supported rare row 1745, while larger folded
fixtures expose insufficient bulk coverage. These are concrete limitations,
not grounds to select a different score by fixture.

Full margins and failed variants are preserved in
[diagnosis-local-anchors.json](diagnosis-local-anchors.json) and
[diagnosis-nearest-local.json](diagnosis-nearest-local.json). The support lane
completed the actual-anchor posterior uncertainty test using the same four-row
prior. It passes 14/18 detector gates but fails trained folded 2048 (rare row
1745), frozen folded 4096 (11 bulk FP), and nominal/highdim 1024 (zero recall).
Rare-hole 2048 also has two supported rare FP, despite passing that fixture's
detector gate. [Its receipt](https://github.com/255BITS/ParticleGAN/blob/bdf05d1be0f68cfdb0c71e81e7e0d3cce477572f/reports/toy100/lrfree-search/feature-cells-cb64-ra/fixes/support/diagnosis-anchor-posterior.json)
is therefore rejected as a universal correction. The corrected main package
keeps the original support law while the verified sampling, mass, routing and
performance fixes receive full GPU validation.

Projection/nuisance loss and centroid placement are demonstrated score defects
that can be corrected without oracle labels. Remaining rare tail errors occur
with sparse even-reference cells and finite held-out support. In the actual
anchor posterior test, supported rare row 1745 chooses a cell with one even
row and no odd calibration rows; its score 17.72 exceeds the global null max
15.81. This is evidence of finite local calibration coverage, not a proof
that no real-only score could resolve the row. Split calibration controls
population errors under its assumptions, not a deterministic zero-FP promise
for every named component of one finite fixture. A universal correction
remains unestablished; failed variants are not selected by fixture.

## Integrity and bounded work

[metric-contract.json](metric-contract.json) passes ten CPU contracts:

- Replacing odd rows leaves even-fit mean/scale/PCA basis/centers/reference
  counts/representatives and Fisher fitting bit-exact, changing only held-out
  calibration.
- Neither global nor private RNG advances during new metric fitting/scoring.
  Original partition tensors are unchanged.
- Query permutation/chunking, the exact original finite p-value formula,
  invalid-input rejection, constant features and duplicate guards hold.
- The new covariance is at most 64 by 64. Retained arrays are H by at most
  eight, K by at most eight, and K scales. N8192 and H1024 storage checks pass;
  no N by N or H by H array is retained.

Metric-only fitting/scoring on the deterministic N8192/H128 storage case took
.0125/.0037 CPU seconds; N1024/H1024 took .0080/.0020 seconds. These are small
diagnostic timings, not a GPU throughput claim.

Strict finite-eigenvalue tests caught hidden NaN generalized eigenvalues when
within covariance was exactly zero. The failed source and receipt are retained
in [initial-numerics/strict-contract.json](initial-numerics/strict-contract.json).
The correction uses an observed reference variance roundoff bound:
`eps*D*max(lambda_max_within, eps*trace_between)`.
It has no absolute feature unit or quality threshold. The positive within
covariance remains dominant on the fixed learned fixtures. Constant heads
still yield rank zero; duplicate guards and BH are unchanged.
[FLOOR-DELTA.json](FLOOR-DELTA.json) bounds the added floor's contribution from
the saved prior eigenspectra and confirms it is inactive in all eight earlier
ranked fixed cases. Their original helper bytes are preserved in
[initial-numerics/fisher_rank.py](initial-numerics/fisher_rank.py).

All diagnostics in this directory used CPU, recorded unchanged inputs/sources
and assert that no CUDA context was initialized. GPU numerical parity,
post-correction training quality and universal folded/highdim support gates
remain unestablished.
