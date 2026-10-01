# Toy-test review on develop

This review ranks test definitions and explains their claims. It adds training
visualizations and reproducible observation tools; it makes **no production
config or trainer repairs**. Base: `4b16312e56328a679b92da69a287c0c9490259d9`.
Open-PR snapshot: 2026-10-01, 134 PRs, exact heads in
[OPEN_PRS.md](OPEN_PRS.md).

Start with the [sorted problem list](PROBLEMS.md), then
[what the highly rated tests verify](HIGH_RATED.md). The
[follow-up findings](FOLLOW_UP.md) identify broken imports, stale contrasts
and unsupported application interpretations. The
[additional shipped families](EXISTING_FAMILIES.md) distinguish source-reviewed
tests from freshly executed evidence.

## Findings

- **Good:** native 100-Gaussian density fidelity; unequal mass, unequal width
  and anisotropic mixtures; PR224's precise controller unit counterexample.
  MisGAN's analytic conditional oracle, circle closed-loop control and sprite
  dynamics also ask valuable questions, with the limitations below.
- **Well defined but narrow/redundant:** most two-template image PRs and
  wide-gap polygon variants. Their exact sampler is clear; their application
  names and number of variants add little independent scientific coverage.
- **Bad as a broader claim/gate:** unconditional grayscale fixtures described
  as conditional inpainting/colorization/translation; travel-only `two_pole`
  interpreted as density recovery; deliberately impossible generator or
  nonidentifiable critic diagnostics used as qualification blockers.

**88 actual training GIFs** cover 43 original two-arm counterexample proposals,
36 declared transfer hosts, three stationary Atlas native targets, one moving
Atlas target, four MisGAN observation laws and PR224's native/control game.
Circle and sprite are the **two explicit execution blockers**, so their current
training GIFs remain missing. Historical endpoint rollout media is labelled
separately. No failed run is illustrated as converging.

For the 43 counterexample proposals, 39 still show **FAIL/PASS** under their
declared hosts/recipes. PR76 b/d and PR70 corner-ramp now show **PASS/PASS**;
PR154 chirp and PR63 mask-inpaint show **FAIL/FAIL**. The 13 vector scripts
first fail on removed public historical recipe fields; the visualizations use
an explicit archived GAN-v3 factory adapter. They do not certify current
KA2/Atlas. Current parameter identities and final live/EMA results remain
separate in [catalog.json](catalog.json).

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
and exact reproduction commands. GIFs use actual observed checkpoints with
step labels, numerical curves and full-budget outcomes. They do not interpolate
clouds or substitute a best checkpoint for a failed final window.

This is a scientific test review, not a Forge calibration, default-promotion
decision or algorithm review of every open PR.
