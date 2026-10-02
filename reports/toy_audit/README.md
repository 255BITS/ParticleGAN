# Toy-test review on develop

This review ranks test definitions and explains their claims. It adds training
visualizations and reproducible observation tools; it makes **no production
config or trainer repairs**. Base: `4b16312e56328a679b92da69a287c0c9490259d9`.
Open-PR snapshot: 2026-10-01, 134 PRs, exact heads in
[OPEN_PRS.md](OPEN_PRS.md). The [PR226/227 addendum](PR226_PR227.md) reviews
two later proposals on pinned develop `6ec7e578`, using their retained evidence.

Start with the [current open-PR decisions](OPEN_PR_DECISIONS.md), then the
[sorted results and improvements](IMPROVEMENTS.md). The refreshed complete
starting inventory contained 138 open PRs and 49 problem proposals. The review
also includes newly opened PR231: **50 reviewed toy PRs, four merged, 46 kept
open and none closed**. Each has an explicit disposition and reason to keep its
scientific question; the final open inventory confirms every retained head.
The [canonical problem selection and keep reasons](CANONICAL_SELECTION.md)
folds 12 proved same-law entries into shared problems: 97 problem entries
retain all 109 named cases. Architecture, capacity, optimizer and negative
controls stay linked when they verify a distinct property. Similar appearance
alone is insufficient to remove a case.
All 64 original low-rated entries have stronger definitions, controls or
comparison audits. Seven explicitly scoped questions receive separate
follow-up definition scores; original ratings and training receipts are
preserved. The follow-up score is **3.49/5**: seven entries at 5/5, 42 at
4/5, 57 at 3/5 and three at 2/5. The [image](IMAGE_QUALITY_V2.md), [non-image](NON_IMAGE_QUALITY.md)
and [per-failure diagnosis](FAILURE_DIAGNOSIS.md) reports explain the changes.

All 17 formerly source-only entries now have separately bound execution
attempts. The fresh [sign/landing/native](source_families/README.md),
[routed/ring](SOURCE_FAMILY_TRAINING.md), [word/Gaussian](SOURCE_DEMOS.md),
[sparse/denoising](conditional_sources/README.md),
[trajectory/transition](route_sources/README.md) and
[paired affine/swirl](paired_sources/README.md) readouts include actual
checkpoint GIFs where training starts, including failed and incomplete runs.
Source hashes, declared budgets and actual serving laws remain separate.
These source fixtures use their original APIs; they do not all execute Atlas.
The original-entry ledger counts **107 actual-state GIFs across 103/109 entries**.
The expanded PR227 has [two supplemental training GIFs](pr227_current_media/README.md),
with guided support and the rotated-teacher failure shown separately.
PR231 adds [eight actual-metric training GIFs](pr231_media/README.md) for its
whole-FiLM diagnostic and counterexamples: **117 GIFs in total**. These added
cohorts do not change the original 103/109-entry coverage denominator.
The [new CUDA receipts](cuda_sources/README.md) preserve the original CPU blockers
and document the later reproducibility prerequisite failure with no full campaign.

The frozen audit runner now avoids one identical residual-bars reference
execution and writes an explicit shared-evidence alias. All shipped task
definitions remain useful and are retained. The source/runtime/recipe-bound
[alias proof](frozen-image-alias.json) disables reuse when the conditions change;
`--separate-execution` preserves separate historical runs.
The subsequent PR229 native changes correctly disable that frozen alias on
current develop. The software tests exercise dispatch using temporary current-byte
fixtures; they do not authorize reuse of the historical training capture.

[Current merge decisions](OPEN_PR_DECISIONS.md) supersede the earlier
[offline preparation](merge_readiness/README.md). GitHub access is restored:
PR224, PR226, PR227 and PR231 merged into develop after their current checks passed.
PR226 has 25 focused checks on the refreshed base; PR224 retains its strict
expected failure alongside 9 passing targeted checks and a green CI matrix.
PR227's [signed repair PR230](https://github.com/255BITS/ParticleGAN/pull/230)
is merged into its proposal. Its refreshed head passes 177 focused tests, with
one optional learned run skipped and the separate controller strict XFAIL retained.
Frozen qualification cards still reject the subsequent native-source changes;
only unauthorized temporary software fixtures bind current source for guard tests.
PR231 passes its 96 adjacent software checks and current CI as a bounded
conditioning diagnostic; its learned quality has no frozen convergence gate.
Useful low-tier and blocked proposals stay open with their reasons documented.

For the frozen original review, read the [sorted problem list](PROBLEMS.md), then
[what the highly rated tests verify](HIGH_RATED.md). The
[follow-up findings](FOLLOW_UP.md) identify broken imports, stale contrasts
and unsupported application interpretations. The
[additional shipped families](EXISTING_FAMILIES.md) distinguish source-reviewed
tests from freshly executed evidence.

## Findings

- **Good:** native 100-Gaussian density fidelity; unequal mass, unequal width
  and anisotropic mixtures; PR224's precise controller unit counterexample.
  MisGAN's analytic conditional oracle, circle closed-loop control and sprite
  dynamics also ask valuable questions, with the limitations below. PR226's
  asymmetric-critic mechanism and PR227's teacher-aligned initialization control
  add useful bounded convergence diagnostics.
- **Well defined but narrow/redundant:** most two-template image PRs and
  wide-gap polygon variants. Their exact sampler is clear; their application
  names and number of variants add little independent scientific coverage.
- **Bad as a broader claim/gate:** unconditional grayscale fixtures described
  as conditional inpainting/colorization/translation; travel-only `two_pole`
  interpreted as density recovery; deliberately impossible generator or
  nonidentifiable critic diagnostics used as qualification blockers.

The **original 90 actual training GIFs**, across **109 ranked entries**, cover 43 original two-arm counterexample proposals,
36 declared transfer hosts, three stationary Atlas native targets, one moving
Atlas target, four MisGAN observation laws, PR224's native/control game and the
two PR226/227 diagnostics.
Circle and sprite are the **two original API execution blockers**. The four
denoising/trajectory CPU attempts were blocked by unavailable CUDA. Their new
CUDA attempts all stopped at the source-prefix reproducibility prerequisite,
before any full training attempt. These six entries still lack training GIFs.
Historical endpoint rollout media is labelled
separately. No failed run is illustrated as converging.

For the 43 counterexample proposals, 39 still show **FAIL/PASS** under their
declared hosts/recipes. PR76 b/d and PR70 corner-ramp now show **PASS/PASS**;
PR154 chirp and PR63 mask-inpaint show **FAIL/FAIL**. The 13 vector scripts
first fail on removed public historical recipe fields; the visualizations use
an explicit archived GAN-v3 factory adapter. They do not certify current
KA2/Atlas. Current parameter identities and final live/EMA results remain
separate in [catalog.json](catalog.json).

PR226's even critic lowers reporting RMSE 19.96%. PR227's initial H/b control
closes its learned-game gap while measured particle-code removal worsens every
judge's score. Its original regression gate uses an unsigned ablation effect,
which also accepts harmful particle use; an executed synthetic control exposes
that weakness. The [addendum](PR226_PR227.md) explains the signed check needed
and the limits of both convergence claims. The new [signed-gate repair](merge_readiness/README.md)
rejects all four harmful-judge controls while preserving the saved positive
endpoint measurements and original training provenance.

All three stationary Atlas runs pass with learned output noise and fail the
clean native spread gates. The moving run passes its original reacquisition
criterion but fails the stricter final density gate. MisGAN can cover all
projected modes while missing the conditional posterior or eight-dimensional
plane. Model success and test quality are separate columns throughout.

## Recommended compact test set

Use native density fidelity, rare mass, unequal width and anisotropy for
unconditional quality; analytic conditional-posterior tests for ambiguous
observations; one basic image orientation/location/intensity fixture; intended
edit plus unused-content preservation; and PR224 as a controller unit fixture.
Use PR226 for paired critic-lag diagnosis and PR227 for teacher-aligned adapter
initialization regression after strengthening its beneficial-contribution gate.
Keep a small representative set of scale and multiscale-gap stresses.

Treat undercapacity, mean-only critic, impossible uniform generator and
unresolved narrow modes as diagnostics. Preserve redundant image variants as
an archived gallery rather than giving each an equal vote. Before adding the
circle/sprite positives to a gate, require a current compatible host and
training-checkpoint evidence. Before adding MisGAN to a binary gate, freeze
posterior/diversity tolerances relative to its analytic finite-draw reference.

## Evidence and reproduction

[PROVENANCE.md](PROVENANCE.md) records source/runtime/sampling identities,
instrumentation parity checks, the media format, local raw artifact location
and exact reproduction commands. The [follow-up validation receipt](improvement-validation.json)
records 53 passing audit tests, original byte comparisons and complete media
hash/decoding checks. The [retention validation](retention-validation.json)
records 99 passing combined checks and real-capture alias verification without
training. GIFs use actual observed checkpoints with
step labels, numerical curves and full-budget outcomes. They do not interpolate
clouds or substitute a best checkpoint for a failed final window.

The [current integration validation](current-validation.json) records 313 passing
checks, three explicit skips and the strict controller XFAIL, plus nine independently
executed CUDA software controls. All six original receipts and the original 90
media hashes are unchanged. The ten added GIFs pass source/receipt/frame checks;
the separately labelled historical sprite rollout is excluded from the 117
actual-state visualization count.

This is a scientific test review, not a Forge calibration, default-promotion
decision or algorithm review of every open PR.
