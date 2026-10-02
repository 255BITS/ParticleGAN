# Completed API reform and full-budget readout

All **110 retained questions** have runnable ParticleGAN API variants, frozen
numeric PASS/FAIL bounds and goal illustrations: **176 variants**, **171 completed
protocols**, **51 PASS**, **120 completed FAIL**, and **5 ERROR/FAIL attempts**.
There is an actual-observation GIF for every variant, including explicitly
unqualified failure supplements. A model FAIL does not make a test useless.

[Sorted results, goals and every rejected bound](LEADERBOARD.md) ·
[All goal GIFs](GALLERY.md) · [Exact definitions and gates](cases.json) ·
[Completed receipts](runs.json) · [Failed-attempt supplements](failed-runs.json) ·
[Independent source/media audit](binding-review.json).

| Family | Variants | Default PASS | Completed FAIL | ERROR/FAIL |
|---|---:|---:|---:|---:|
| Images and five-word inverse/generation | 80 | 31 | 49 | 0 |
| Vector distributions and continuation | 54 | 4 | 48 | 2 |
| Conditional, paired and temporal questions | 23 | 7 | 16 | 0 |
| Native mechanism and behavior diagnostics | 19 | 9 | 7 | 3 |
| Total | 176 | 51 | 120 | 5 |

## What changed in the tests

Every variant executes a public `Recipe` and actual public trainer, prior,
loss/optimizer or native routed policy. Unsupported combinations are rejected.
The resolved recipe, initialization, sampling law, resource counts, seed,
source hashes, runtime, budget and evaluation cadence belong to each receipt.
The variants retain useful restricted-architecture and negative controls.
Three exact image aliases are named explicitly; separate retained executions
do not provide three independent scientific confirmations.

The image gates now reject correct-looking templates with wrong mixture mass.
The vector gates additionally check fixed projected CDFs against the declared
observable law. Conditional variants feed the real context to both sides of
the game and test held-out correspondence, so an unconditional marginal fit
cannot pass a paired query. PR61, PR63 and PR65 gain actual paired completion,
ambiguous masked completion and RGB assignment variants. The two-pole test
checks the specified finite offsets rather than only travel from the origin.
PR227 checks that removing code worsens the game with the correct sign.
PR231 uses a recipient-relative quality denominator that rejects zero output.

Scope changes remain explicit. PR196 variants are oracle-labelled conditional
imputer baselines; they do not establish learning from incomplete data alone.
The sprite variant learns fully observed six-state dynamics with the source
renderer. Sparse and posterior variants test direct conditional kernels, rather
than an iterative DDGAN chain. None of these narrower passes would validate the
original broader application claim. The family documents explain each limit.

All mathematical scorers have positive exact-law/exact-pair references and
relevant destructive controls. Software tests execute every fixture and check
that inserting an observation preserves training state, named random streams
and the next actual update. These controls validate the test machinery; their
PASS is distinct from a trained model PASS.

## What the strongest questions verify

The three stationary **100-mode problems** verify coverage, mixture mass,
centers and within-mode width in ordinary, rotated and staggered geometry.
Their 7,000-update noisy served-law runs pass, with final center RMS errors
0.15577, 0.11862 and 0.13082 target sigma. The same runs' output-noise-disabled
diagnostics fail: covariance trace bias is approximately -0.94 and radial KS
approximately 0.79. A noisy PASS grants no clean-distribution credit.

**Unequal mass** asks whether the rare 2% component has the right probability;
**unequal width** and **anisotropy** ask whether local density shape survives.
These remain informative FAIL tests. Unequal mass has final minimum mass ratio
0.97612 and mass TV 0.01618, but projected CDF KS 0.06339 exceeds 0.06. This is
evidence of distribution mismatch, not evidence that the rare mode vanished.
Likewise, a projected-CDF rejection alone does not identify a training mechanism.

The **annulus** verifies uniform radial area mass and continuous rotational
support; it passes the full 1,600-update protocol. The Gaussian example verifies
mean, covariance and radial/projected distribution shape. At update 1,000 all
its metrics pass, but update 959 has minimum covariance eigenratio 0.849324,
below 0.85: its declared final-five test correctly remains FAIL.

The **stiff-release causal unit** retains its native instability as an expected
model FAIL while cancellation and safe-geometry controls pass. It isolates a
specific constructed controller release; it does not justify a general policy
repair. **Routed moving** passes all three actual orientation endpoints at 500,
1,000 and 1,500 updates, with maximum held-out RMSE 0.002062. **Routed replay**
passes both learned fit and exact native state/activation replay. Replay alone
would be software evidence without the learned-fit gate.

The **PR227 neutral acquisition** arm passes its fresh 6,400-update signed gate:
relative edit MSE 0.02350 and zero-code-minus-live game +0.57338. Its original
initialization arm fails. The **five-word joint autoencoder** passes the full
20,001 updates: all five words, quality fraction 1.0, mass TV 0.02891, exact
paired reconstruction and minimum reconstruction token probability 0.99975.
Decoded strings and full token confidence are both shown and scored.

## What fails, and what the evidence establishes

The sorted table names every rejected endpoint bound and each failed terminal
check. **14 completed runs pass their final instantaneous metric but fail the
required final window**. Most learned tasks require all five final post-update
metric observations to pass; scoring cadence is independent of GIF frame
selection. A nicer endpoint cannot change that grade.

Healthy residual bars illustrate why sharpness and coverage are insufficient:
all four modes and HQ 1.0 still fail with mass TV 0.1533. Restricted tiny,
spatially uniform and mean-only hosts remain capacity/information controls.
All 26 published/control vector-proposal arms fail their composite gates;
their jointly changed critic/initialization comparisons do not isolate either
factor's causal effect. No failing model was repaired, rerun with another seed,
extended, or made to pass by weakening its bounds.

The two **ring continuation** attempts stop after the actual 1,200-update
acquisition prerequisite fails. Their planned 2,400-update hold and 3,600-update
shift tests are not attempted. The prefix has all eight modes and mass TV
0.0730, but fails HQ and within-mode width/radial shape. The twelve-row control
has four of eight modes and remains a distinct resource-stress FAIL; the public
DV12 latent perturbation prevents interpreting it as an exact finite-atom law.

The three **critic-lag** attempts complete 800 updates and pass all five numeric
terminal checks, then fail the original exporter on two feature channels.
Original ERROR/FAIL receipts and missing checkpoint artifacts remain unchanged.
Separate lossless retained-array GIFs label the export error and numeric PASS;
they confer no completed-execution PASS. Future fixtures explicitly pack the
two feature blocks as grayscale rows, and the runner saves bound observations
and checkpoint before export. Focused tests cover that future failure path.

## Frozen execution, media review and reproduction

Training source: `5d75e6556bb65726e71a602dbce708986a5b49f5`.
Reviewed renderer: `f8ecb7d64648d00d0a22fa56b518e2079b1bb0c2`.
One protocol seed, **24002**, CPU, one Torch thread per worker; Python 3.12.13,
Torch 2.13.0+cu126. Eight independent workers execute 371,829 actual updates.
The sum of per-case wall durations is 15,121.35 seconds; concurrent execution
and unequal host cost make this unsuitable as a training-speed leaderboard.

The renderer uses retained numeric arrays without training, sampling or
rescoring. All source, raw artifact, observation-step, frame and grade bindings
passed independent review. Default and instantaneous verdicts remain separate
in every GIF. Long captions have dedicated space; physical coordinates have
equal units; feature heatmaps retain their actual rows and columns. Original
ratings, numeric receipts, target banks and historical GIFs retain their bytes.
Package, trainer, legacy library and production config trees are unchanged.

The integrated API software suite passes **578 tests** in 49.79 seconds on the
recorded Python/Torch runtime. Its nine test files cover factories, exact-law
controls, observer isolation, run grading, media layout, immutable reframing and
publication tamper rejection. The local log remains outside Git at
`/ml2/hypergan/toy-api-publication-focused-validation-20261002.log`.

Raw arrays, checkpoints, progress logs and first rendering attempts remain
outside Git in `/ml2/hypergan/toy-api-full-protocols-20261002` and the separate
media archives. Compact receipts bind those artifacts by SHA-256. Only the
final reviewed goal GIFs, definitions, readout and provenance are published.

```sh
python -m benchmarks.toy_audit.api_run --list
python -m benchmarks.toy_audit.api_run --case api-grid100 --device cpu \
  --recipe auto --output /tmp/toy-api-grid100

# A fresh complete archive; command returns nonzero if any test fails.
python -m benchmarks.toy_audit.api_run --all --device cpu --recipe auto \
  --jobs 8 --output /tmp/toy-api-full

# Read retained evidence only; never train or rescore during publication.
python -m benchmarks.toy_audit.api_reframe --runs ORIGINAL_ARCHIVE \
  --output NEW_MEDIA_ARCHIVE --jobs 8 --include-failure-views
python -m benchmarks.toy_audit.api_publish --runs ORIGINAL_ARCHIVE \
  --media-review NEW_MEDIA_ARCHIVE --failure-media-review NEW_MEDIA_ARCHIVE \
  --output reports/toy_audit/api_contract
python -m benchmarks.toy_audit.api_readout --runs ORIGINAL_ARCHIVE \
  --output reports/toy_audit/api_contract
```

`--include-failure-views` and `--failure-media-review` recognize only these five
exact frozen failed attempts. They are not a general route for qualifying an
ERROR receipt. The default publisher continues to reject unsuccessful execution.
For new cases, use the [contract](README.md), declare the default bounds and
budget before training, and retain explicit narrowed scope where necessary.
