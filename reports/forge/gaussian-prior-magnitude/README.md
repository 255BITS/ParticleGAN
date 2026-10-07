# Magnitude-sensitive learned-prior diagnostic

**This fixed-scale prior rule does not solve the 1D Gaussian continuous learner.**
All three Gaussian timing arms fail acquisition, strict retention and shifted
reacquisition. Extrapolation raises stationary passing checks from the parent's
3/72 to 10/72, but its longest passing streak remains 2. Alternating ends with a
good Gaussian snapshot (KS `.03028`) while retaining only 2/72 checks. The three
ring arms also fail the combined gates; retain the adopted normalized-prior
alternating ring recipe, which already retains 144/144.

This investigation tests a fixed-scale prior response while keeping BCAP's G/D
dualnorm updates unchanged. It is a separate task-only diagnostic, with no
ordinary qualification credit. PR #317 supplies the shared joint-update host.

The public settings are `prior_update="row_capped"` and
`prior_gradient_scale=.001`. Only actual generator-sampled rows use
`g / max(norm(g), .001)`. Large row gradients retain the existing unit direction
cap; small gradients yield proportionally small motion. Unsampled rows stay
fixed, including when a regularizer creates dense gradients. The scale remains
fixed throughout training and target shifts. G/D directions and nominal rates
remain `.012`, `.018`, and `.03` for G, D, and the learned prior respectively.

The row helper is shared between applied optimizer steps and extrapolation's
cached preview. The fixed scale is an optimizer-group checkpoint field;
incompatible scales fail validation before loading. New Recipe defaults are
omitted from serialized packets so historical default recipes retain their
identity. Forge registers the response rule as a technique field and the scale
as a hyperparameter. Unsupported current caller-owned task hosts fail preflight.

The scale `.001` was declared before the separate initialization-only gradient
probe. It was not selected from a successful checkpoint. At initialization,
Gaussian's 102 selected rows have median gradient norm `8.30e-5`, maximum
`3.92e-4`, and 0% saturation. Their mean implied prior displacement is `.002976`,
versus nearly `.03` under the parent unit-row response. Ring's median is
`4.70e-5`, maximum `2.12e-4`, 0% saturation, and mean displacement `.001574`.
This probe consumes and checkpoints its own named streams but applies zero
training updates and leaves all initial model tensors unchanged.

## Frozen comparison

Each task uses seed 0, the public deterministic initializer, batch 128, 256
learned uniform MoG locations, sigma `.1`, `init_std=1`, and no standardization.
Gaussian targets `N(2,.5²)` with z=2 and width32/depth2 G/D; its critic has the
same Fourier2 features. Ring uses its original z=4, width64/depth2 architecture
and 16-component target. Initial tensors, constructor streams, data batches,
sampling law, constant rates and all evaluation schedules match the source-bound
parent. CUDA worker ordinal is explicitly normalized only in the initial proof;
named RNG seeds do not depend on physical device ordinal.

Three timing arms compare alternating D-then-G, simultaneous joint evaluation,
and extrapolation from the past. All six stationary trials train 4,000 updates.
Gaussian must acquire five terminal full passes by 1,000 and pass every 72
remaining checks. Ring must acquire by 1,600 and pass every 144 remaining checks.
There are 24 checks per original 1,000/400-update block, each using 4,096 clean
live GPU samples. Original full bounds, including Gaussian KS≤.05, stay fixed.

All three Gaussian arms then continue from their own 4,000-update checkpoint,
changing only the target mean from 2 to 3. They must reacquire five terminal
passes by 5,000 and retain every 24 check through 6,000. An independent frozen
copy of each pre-shift checkpoint receives exactly the same public evaluation
draws. No history is reset. Stationary failures already preclude a continuous
learner pass, even if a subsequent shift arm fits.

The finite reservation is nine new trials, 30,000 updates, 4,500 seconds and
zero scientific retries. All declared peers execute after a scientific failure.
The nine parent results in
[PR #317's readout](../bcap-past-extrapolation/results.json) are reused under
their original evidence identities at zero new cost; no parent training repeats.

## Reproduction

The frozen scientific source is commit `7c5d66e6`. Protocol SHA256:
`0d89553efe7ffe10638e9d4eeb578b30f7b584fe72631eacb39796bfb1478b39`.
Public-API training is performed exclusively by `GANTrainer`; the diagnostic
adapter temporarily binds its protocol to the shared host and restores the
host's globals in `finally`. The archived scientific driver and protocol retain
their original bytes. Replaying an older source-bound study requires its pinned
source; software tests separately build current code from the original metadata.

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /usr/bin/python -u -m benchmarks.toy_audit.gaussian_prior_magnitude run \
  --output runs/api/gaussian-prior-magnitude-v1 --device cuda:1
tail -f runs/api/gaussian-prior-magnitude-v1.run.log
```

Per-cell logs are named `runs/api/gaussian-prior-magnitude-v1/<arm>-<task>-<phase>.log`.
Bulk checkpoints, observations, JSONL/curves and software logs remain ignored;
the committed report contains compact final metrics, source receipts, and
actual-training GIFs. Saved output scoring and deterministic target-reference
rendering use CPU numerical tools; every neural fixture, training update,
model-sampling draw and restore uses CUDA.

## Numerical results

These are diagnostic readouts, not a second qualification leaderboard. The
repository's current qualification table remains
[technique-inventory.md](../technique-inventory.md). Compact source-bound metrics
are in [results.json](results.json); full curves and samples remain in the archive.

| Stationary task / timing | Acquisition | Full hold passes | Longest full streak | Final KS / component covariance error | Final width ratio / minimum eigen ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| Gaussian / alternating | FAIL | 2/72 | 1 | .03028 | 1.05109 |
| Gaussian / simultaneous | FAIL | 0/72 | 0 | .20625 | .83725 |
| Gaussian / past extrapolation | FAIL | 10/72 | 2 | .19872 | .79545 |
| Ring / alternating | PASS | 142/144 | 178 | .48971 | .14670 |
| Ring / simultaneous | PASS | 46/144 | 42 | .58380 | .15350 |
| Ring / past extrapolation | FAIL | 0/144 | 0 | 1.60823 | .51718 |

Each combined verdict is FAIL. Alternating ring acquires earlier (first five
full passes at 984 versus the parent's 1,584), but fails mode coverage at 3,884
and minimum-eigenvalue spread at 4,000. Its long mid-run success is not retained to the
end. Simultaneous ring's first five-pass window is at 934, yet it fails 98 hold
checks. Past ring passes none of its 240 scheduled full checks. Prior magnitude
and timing interact strongly; the previously useful normalized-field ring
extrapolation cannot be assumed useful with a different prior field.

| Gaussian mean2→3 continuation | Reacquisition | Full hold passes | Full passes /48 | Longest streak | Final KS | Final width ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Alternating | FAIL | 1/24 | 1/48 | 1 | .08544 | .97987 |
| Simultaneous | FAIL | 0/24 | 0/48 | 0 | .35390 | 1.46553 |
| Past extrapolation | FAIL | 3/24 | 4/48 | 3 | .11994 | .97768 |

Every frozen counterpart passes 0/48 shift checks. Active models respond to the
shift, but none establishes a five-check acquisition window. Mean/width alone
would miss the remaining distribution error: alternating's final shifted mean
error is `.03396` sigma and width ratio `.97987`, yet KS `.08544` exceeds `.05`.
Past's improved shift KS versus its normalized parent's `.13998` also does not
meet the full gate.

## Explanation and recommendation

The intended response change is real. The final past-extrapolation Gaussian
cache has 97 nonzero prior rows, with direction norms `.00365–1.0`, mean `.67134`,
and 37.11% at the unit cap. Its implied mean prior displacement is `.02014`,
versus the parent's nearly `.03`. After the shift, 103 rows have mean direction
norm `.53876`, 13.59% at the cap, and implied mean displacement `.01616`. Thus
small gradients can produce small motion, while many current gradients remain
large enough to drive substantial motion. The initialization reduction was
stronger than the terminal reduction. These are saved-state explanations,
not additional pass criteria or full-trajectory gradient measurements.

Changing only the prior's response is insufficient at this scale. G/D retain
the original normalized field, which still produces substantial motion for
small nonzero gradients. This comparison does not isolate network normalization
as the sole cause, and it rejects only this frozen `.001` prior-scale revision.
It does not justify concluding that every magnitude-sensitive prior must fail.

Stop this exact revision as a Gaussian repair. Compare the separately authorized
frozen-prior and network-magnitude investigations before selecting another
bounded mechanism or combining changes. Retain the adopted ring recipe. Do not
promote isolated good snapshots, average serving, shorten retention checks or
introduce elapsed-step annealing as a continuous-learner pass.

## Evidence and verification

The nine new CUDA trials execute exactly **30,000 updates**, **3,840,000 real
training examples**, and **368.727114 measured loop seconds**, within the
4,500-second reservation, with zero scientific retries. Timing includes scheduled
evaluation and excludes construction, source capture, restoration, checkpoint
serialization, rendering and separate software fixtures.

All **69 software checks pass**: 11 new CUDA/contract checks, 14 existing CUDA
mechanism/current-host checks, and 44 metadata-only checks. The five archived host
tests use their original task metadata with current software, without altering
the immutable archived scientific declaration. A separate refusal test proves
that the archived scientific driver still rejects changed source; the scoped
adapter restores its global bindings even after exceptions. Earlier preflight
test failures (list-versus-dictionary checkpoint fixtures and an error-message
assertion) were repaired before scientific spend and retained in raw logs.

All nine final CUDA contexts restore exactly, with constant rates and optimizer
counts checked. The three frozen controls' models, optimizer states and cached
fields remain unchanged. Publication recomputes all **1,308** live/frozen saved
sample metric packets and all numerical verdicts. Source capture verifies a
consistent **1,186-file** scientific source union across the nine cells. The
post-scientific search-activity metadata fix declares the prior scale inactive
unless learned capped rows are used; it changes no executed training direction
and required no science rerun.

- [Frozen protocol](protocol.json), [provenance](provenance.json), [restore proof](restore-proof.json).
- [Saved field audit](bounded-field-audit.json), [initialization-only probe](initialization-probe.json), [publication verification](verification.json).
- [Stationary metrics plot](stationary.svg), [shift metrics plot](shift.svg).
- Actual-training Gaussian stationary GIFs: [alternating](alternating-gaussian1d_acquisition-stationary.gif), [simultaneous](simultaneous-gaussian1d_acquisition-stationary.gif), [past](extrapolation_from_past-gaussian1d_acquisition-stationary.gif).
- Actual-training ring GIFs: [alternating](alternating-ring16_acquisition-stationary.gif), [simultaneous](simultaneous-ring16_acquisition-stationary.gif), [past](extrapolation_from_past-ring16_acquisition-stationary.gif).
- Actual-training shifted Gaussian GIFs: [alternating](alternating-gaussian1d_acquisition-shift.gif), [simultaneous](simultaneous-gaussian1d_acquisition-shift.gif), [past](extrapolation_from_past-gaussian1d_acquisition-shift.gif).

Saved-output publication and restore verification are reproducible with:

```sh
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/gaussian-prior-magnitude/publish.py \
  --raw runs/api/gaussian-prior-magnitude-v1
CUBLAS_WORKSPACE_CONFIG=:4096:8 /usr/bin/python reports/forge/gaussian-prior-magnitude/verify.py \
  --raw runs/api/gaussian-prior-magnitude-v1 --device cuda:1
```
