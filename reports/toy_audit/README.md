# Toy-test review on develop

This review ranks test definitions and explains their claims. It adds training
visualizations and reproducible observation tools; it makes **no production
config or trainer repairs**. Base: `4b16312e56328a679b92da69a287c0c9490259d9`.
Open-PR snapshot: 2026-10-01, 134 PRs, exact heads in
[OPEN_PRS.md](OPEN_PRS.md). The [PR226/227 addendum](PR226_PR227.md) reviews
two later proposals on pinned develop `6ec7e578`, using their retained evidence.

Start with the [latest sorted results and improvements](IMPROVEMENTS.md).
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
The complete ledger counts **107 actual-state GIFs across 103/109 entries**.

[Merge readiness](merge_readiness/README.md) identifies PR224/226 and repaired
PR227 as local diagnostic candidates. The follow-up repair is committed and
the prospective integration is verified, but **zero remote PRs have merged**:
authenticated GitHub access is unavailable and fresh remote checks are unknown.

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
Circle and sprite are the **two original API execution blockers**. Four newly
attempted denoising/trajectory entries require unavailable CUDA. These six
entries lack current training GIFs. Historical endpoint rollout media is labelled
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
hash/decoding checks. GIFs use actual observed checkpoints with
step labels, numerical curves and full-budget outcomes. They do not interpolate
clouds or substitute a best checkpoint for a failed final window.

This is a scientific test review, not a Forge calibration, default-promotion
decision or algorithm review of every open PR.
