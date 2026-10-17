# CB64-RA GPU baseline

**Evidence VALID; quality acceptance FAIL.** All 22 prescribed jobs finished on
physical GPU 0 (RTX A6000), with no runtime errors. The independent saved-artifact
audit passed 660/660 checks. This retest uses the unchanged archived CB64-RA
package and config. It is the baseline for the separate corrected CB64-RA2 work.

## Results

| Acceptance | Result |
|---|---|
| Canonical portability | 9/13 PASS |
| Canonical native accuracy | 0/3 PASS |
| Learned runs | 4/4 complete |
| Exact checkpoint continuation | 4/4 PASS |
| Source, fixture, stream and artifact checks | 660/660 PASS |

Portability failures: mode_hold, img_blobs4, vector_unequal_mass and ring_shift.
Native failures: grid100, rotated100 and staggered100. All native runs used the
original 7000-update budget, 34 observations, five terminal 20k clouds and 100k
holdout. Live noisy output decides acceptance; clean/EMA metrics are diagnostics.
The clean ring_shift branch passes while the required noisy branch fails.

| Learned problem / metric, 2000 updates | E22 | CB64-RA |
|---|---:|---:|
| Toy precision | 0.7155 | 0.6233 |
| Toy supported modes /25 | 25 | 24 |
| Toy mass TV | 0.2845 | 0.3780 |
| Toy CUDA training seconds | 72.34 | 80.79 |
| MNIST active embedding Frechet distance | 0.5445 | 1.4888 |
| MNIST active precision | 0.8691 | 0.8525 |
| MNIST active recall | 0.8472 | 0.3228 |
| MNIST class TV | 0.0364 | 0.1244 |
| MNIST CUDA training seconds | 90.96 | 103.26 |

Both nonlinear toy runs fail the declared independent gate (precision >= .9,
25 supported modes, mass TV <= .1). No absolute MNIST quality gate was declared;
the candidate has a clear recall/distance regression against the matched control.
MNIST embeddings use a real-only classifier and 39 active standardized dimensions;
raw 64-dimensional diagnostics are retained in the learned report.

## Diagnosis and recommendation

The GPU baseline confirms the regressions first observed on CPU. The original
fixed latent perturbation also changes with dimension: its norm cap makes the
per-coordinate width collapse at z=128. Saved checkpoints show dense clone
families and degraded particle centers, so changing only serving cannot repair
the trained model. Separate focused diagnostics address sampling geometry,
small-population feasibility, parent reuse and mass allocation, and CUDA scalar
synchronization. Their fixes and quality retest are separate from this baseline.

Keep the existing reference/default. CB64-RA is an experimental archive and does
not pass this CUDA suite. CB64-RA2 requires its own frozen quality acceptance.

## Scope and reproduction

Learned runs use identical saved real streams, evaluator and initial G/D/prior
hashes, seed 314159, N=1024, z=128, batch 128 and 2000 updates. All checkpoint
continuations preserve loss bits and semantic state after every update. Only the
observational birth_death.last.eval_seconds field is excluded from comparison.
CPU RNG buffers remain on CPU. The retest pins deterministic algorithms and
no TF32 for both variants; it does not claim bitwise trajectory identity with
older runs that did not enable those settings.

Training timers synchronize complete updates and exclude evaluator/checkpoint
I/O. Screen elapsed times include diagnostics. GPU 0 is shared, so timings are
descriptive; launcher wall time also includes reservations for later diagnostic
jobs and is not a throughput measure. GPU 1 was never used.

- [Canonical screens](screens/REPORT.md) and [leaderboard](screens/leaderboard.json)
- [Learned comparison and replay](learned/REPORT.md)
- [Independent audit](audit/CHECKS.json)
- [Protocol](PROTOCOL.md), [frozen sources](source-freeze.json) and [job results](execution-results.json)
- [Copied-file manifest](ARCHIVE.json)

Raw checkpoints, clouds, datasets and logs remain at the original study path
recorded by the manifest. The preserved source/receipt hashes support verifying
those artifacts; this Git archive includes the runners and compact receipts.
