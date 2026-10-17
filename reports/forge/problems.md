<!-- Generated Forge problem list -->
# Unresolved training problems

[Full-suite leaderboard](technique-inventory.md) · [All test/protocol results](test-performance.md)

This page explains the outstanding tests, including the held failures. **Held means excluded from qualification, not solved:** these failures are included in the leaderboard's full-suite comparison. Passing one problem does not establish that the same configuration passes the whole suite.

Selected results below use one recorded recipe/source/runtime cohort per family. Research, diagnostic and historical configurations are named separately, with links to their original evidence. An oracle or hand-built solution demonstrates feasibility, not successful training. Not measured, blocked, timeout and error are never relabelled FAIL.

Passer summaries reflect the evidence audit dated **2026-10-15**. Selected-result tables are regenerated from the registered publication.

| Problem | What it tests | Has any configuration passed? |
| --- | --- | --- |
| [Keep a bell curve stable, then learn a moved target](problems.md#gaussian1d_stability) | After learning a simple bell curve, keep its centre, width and distribution correct while training continues; then track a shifted target and stay there. | Yes, individual research configurations; no selected recipe passes the complete test. |
| [Preserve rare clusters](problems.md#vector_unequal_mass) | Learn several blobs with different probabilities, including a rare 2% blob, without losing its population or collapsing its shape. | Yes, transport and cell-moment research configurations; BCAP and BISPF fail. |
| [Learn narrow and broad clusters together](problems.md#vector_unequal_width) | Give four equally likely blobs different widths while preserving both their shapes and their equal probabilities. | Yes, balanced assignment and cell-moment research configurations. |
| [Learn stretched and tilted clusters](problems.md#vector_anisotropic) | Match the populations and shapes of three equally likely blobs stretched horizontally, vertically and diagonally. | Yes: K3P, KA2, R1/R2 in separate diagnostics, plus research configurations. |
| [Settle on an overlapping distribution](problems.md#vector_overlap) | Match the observable centre, spread and distribution of two overlapping blobs; no hidden component labels are required. | Yes: K3P in separate diagnostics and multiple research recipes. |
| [Generate all four bar images](problems.md#img_bars4) | Generate each of four simple bar pictures, rather than producing only a few convincing examples. | Yes: R1/R2 and GAN v3 in separate diagnostic measurements. |
| [Generate all four image corners](problems.md#img_blobs4) | Generate a small bright square at each of four corners, with enough clean examples of every corner. | Yes: R1/R2 and kinetic research configurations. |
| [Generate both brightness levels accurately](problems.md#img_intensity2) | Generate a square at either of two target brightnesses, with both levels represented and little brightness error. | Yes: GAN v3, R1/R2 and kinetic transport in separate measurements. |
| [A grid of 100 clusters](problems.md#grid100) | Learn all 100 small blobs with correct locations, populations and widths, scoring the current model's clean outputs without adding evaluation noise. | No full PASS under the current clean-sampling declaration in the audited evidence. |
| [A rotated grid of 100 clusters](problems.md#rotated100) | Learn all 100 small blobs with correct locations, populations and widths, scoring the current model's clean outputs without adding evaluation noise. | No full PASS under the current clean-sampling declaration in the audited evidence. |
| [A staggered grid of 100 clusters](problems.md#staggered100) | Learn all 100 small blobs with correct locations, populations and widths, scoring the current model's clean outputs without adding evaluation noise. | No full PASS under the current clean-sampling declaration in the audited evidence. |
| [Keep the 100-cluster grid correct for longer](problems.md#grid100_14k) | Continue a model that passed the 7,000-update 100-cluster test to 14,000 updates without losing quality. | No qualifying PASS; no passing current-declaration parent is available. |
| [Keep the 100-cluster rotated grid correct for longer](problems.md#rotated100_14k) | Continue a model that passed the 7,000-update 100-cluster test to 14,000 updates without losing quality. | No qualifying PASS; no passing current-declaration parent is available. |
| [Keep the 100-cluster staggered grid correct for longer](problems.md#staggered100_14k) | Continue a model that passed the 7,000-update 100-cluster test to 14,000 updates without losing quality. | No qualifying PASS; no passing current-declaration parent is available. |
| [Recover quickly after the target moves](problems.md#target_shift_recovery) | Learn and hold a ring of blobs, then track a sudden sideways shift within 400 updates and stay correct; a frozen model must fail to show the change is real. | No complete PASS under the current declaration in the audited evidence. |
| [Keep every learned mode alive](problems.md#mode_hold) | Keep generating every target mode during continued training, rather than losing modes after initially finding them. | Yes: the selected BISPF recipe and some research configurations. |
| [Keep a learned ring stable for a long run](problems.md#ring_extension) | Acquire all eight ring blobs, hold them for 1,200 consecutive checks, then survive 300 additional checks on the same uninterrupted run. | Yes: BISPF in a saved separate-runtime diagnostic; no selected ordinary PASS. |
| [Learn sixteen distinct clusters from scratch](problems.md#ring16_acquisition) | Learn all sixteen blobs on a ring with balanced populations and real within-blob spread, rather than sixteen collapsed points. | Yes: selected BCAP and BISPF recipes. |
| [Learning must not depend on external clock labels](problems.md#clockfree_audit_measurement_v1) | Check that the same learning state and inputs produce the same update despite changed step labels, declared horizon, evaluation cadence or a save/restart. | Yes: selected BCAP and BISPF recipes. |

<a name="gaussian1d_stability"></a>

## Keep a bell curve stable, then learn a moved target

Test: `gaussian1d_stability` · Held for qualification; included in full-suite results.

**What it tests.** After learning a simple bell curve, keep its centre, width and distribution correct while training continues; then track a shifted target and stay there.

**The problem.** BCAP and BISPF keep oscillating instead of settling. BISPF passes only 5 of 72 stationary checks, misses deadline re-acquisition and passes 0 of 24 shifted-hold checks. A one-time smoke pass does not establish stability.

**Has any configuration beaten it?** Yes, individual research configurations; no selected recipe passes the complete test.

- `kinetic-local-nsgda-cross-v1`, `kinetic-local-dualnorm-g-history-v1` and `kinetic-local-dualnorm-g-history-bank8-v-v1` pass, but fail ring16 and mode hold.
- Cell-gradient-signal configurations and `cell-quota-minimum-change-forward-kl-g-v1` also pass. The latter passes mode hold too, but its later ring16 probe fails and it cannot run two-pole or word smoke.
- A hand-built network can draw the target; that is a capacity witness, not a trained full-protocol PASS.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) |
| [BISPF](families/bispf.md#cohort-cuda-3faddec347a0) | [FAIL](families/bispf.md#cohort-cuda-3faddec347a0-experiment-gaussian1d_stability) |
| [Atlas](families/atlas.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/atlas.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) |

**What remains to solve.** Find an update rule that settles when correct but responds after a shift, then verify the same recipe on ring16 and mode hold. Existing trade-offs do not prove the problem impossible.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#gaussian1d_stability).


<a name="vector_unequal_mass"></a>

## Preserve rare clusters

Test: `vector_unequal_mass` · Held for qualification; included in full-suite results.

**What it tests.** Learn several blobs with different probabilities, including a rare 2% blob, without losing its population or collapsing its shape.

**The problem.** The rare blob receives little training signal. BISPF preserves its share but flattens its shape: minimum eigen ratio 0.004 against the required 0.15. BCAP also collapses it.

**Has any configuration beaten it?** Yes, transport and cell-moment research configurations; BCAP and BISPF fail.

- `kinetic_transport_local_v2` and the two kinetic g-history reference8 variants pass; they fail unequal-width and mode hold.
- Round-5 direction blend plus local-v2 passes, but fails Gaussian stability and unequal-width and was not measured on most of the required suite.
- Cell-moment aggregate recipes pass all four vector tests; they lose mode hold or ring16. True-target sampling passes 200/200 reference runs, which validates the gate but is not a training win.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) |
| [BISPF](families/bispf.md#cohort-cuda-3faddec347a0) | [FAIL](families/bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_mass) |
| [Atlas](families/atlas.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) |

**What remains to solve.** Preserve rare-cluster coverage and shape while checking unequal-width, ring16 and mode hold with one unchanged recipe.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#vector_unequal_mass).


<a name="vector_unequal_width"></a>

## Learn narrow and broad clusters together

Test: `vector_unequal_width` · Held for qualification; included in full-suite results.

**What it tests.** Give four equally likely blobs different widths while preserving both their shapes and their equal probabilities.

**The problem.** BISPF gets the shares wrong (mass TV 0.25) even where shapes pass. BCAP has both share and spread errors. A few strays can dominate a narrow blob.

**Has any configuration beaten it?** Yes, balanced assignment and cell-moment research configurations.

- Round-5 balanced coordinate assignment passes with 14 terminal checks and spread error 0.286, but loses unequal-mass, Gaussian stability and native 100-blob quality; much of the required suite is untried.
- Cell-moment aggregate and per-kernel recipes pass but fail other required tests. G-angle has a historical-source pass, not a current selected pass.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) |
| [BISPF](families/bispf.md#cohort-cuda-3faddec347a0) | [FAIL](families/bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_width) |
| [Atlas](families/atlas.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) |

**What remains to solve.** Test the balanced-assignment reference on the missing suite and solve its rare-cluster regression before claiming a global improvement.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#vector_unequal_width).


<a name="vector_anisotropic"></a>

## Learn stretched and tilted clusters

Test: `vector_anisotropic` · Held for qualification; included in full-suite results.

**What it tests.** Match the populations and shapes of three equally likely blobs stretched horizontally, vertically and diagonally.

**The problem.** Both leading recipes allocate the wrong amounts to the blobs. BISPF has mass TV 0.191 against 0.15 and covariance error 2.0.

**Has any configuration beaten it?** Yes: K3P, KA2, R1/R2 in separate diagnostics, plus research configurations.

- K3P, KA2 and R1/R2 pass in the attempt index, separately from qualifying selected-cohort results. They fail mode hold, ring16 and the clock measurement in the recorded comparison.
- Nine additional indexed research recipes and four cell-moment aggregates pass. Recorded passers with mode-hold measurements fail it.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) |
| [BISPF](families/bispf.md#cohort-cuda-3faddec347a0) | [FAIL](families/bispf.md#cohort-cuda-3faddec347a0-experiment-vector_anisotropic) |
| [Atlas](families/atlas.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) |

**What remains to solve.** Retain the successful mass allocation while preserving mode hold and basic acquisition.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#vector_anisotropic).


<a name="vector_overlap"></a>

## Settle on an overlapping distribution

Test: `vector_overlap` · Held for qualification; included in full-suite results.

**What it tests.** Match the observable centre, spread and distribution of two overlapping blobs; no hidden component labels are required.

**The problem.** BCAP reaches the target temporarily and drifts away. BISPF ends with centre error 0.28, covariance error 0.53 and distribution distance 0.19; it misses the sustained ending.

**Has any configuration beaten it?** Yes: K3P in separate diagnostics and multiple research recipes.

- K3P and ten other indexed configurations pass under the recorded declaration; all fail mode hold.
- Four cell-moment aggregates pass; three lose mode hold and the Bures variant loses ring16.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) |
| [BISPF](families/bispf.md#cohort-cuda-3faddec347a0) | [FAIL](families/bispf.md#cohort-cuda-3faddec347a0-experiment-vector_overlap) |
| [Atlas](families/atlas.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) |

**What remains to solve.** Add stability without losing multimodal acquisition or mode hold. A good endpoint alone does not meet the sustained gate.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#vector_overlap).


<a name="img_bars4"></a>

## Generate all four bar images

Test: `img_bars4` · Held for qualification; included in full-suite results.

**What it tests.** Generate each of four simple bar pictures, rather than producing only a few convincing examples.

**The problem.** BCAP and BISPF produce clean pictures but only 2 of 4 bars. Some transport recipes reach more bars but produce too many blurry pictures.

**Has any configuration beaten it?** Yes: R1/R2 and GAN v3 in separate diagnostic measurements.

- R1/R2 and GAN v3 pass this declaration, but fail ring16, the clock measurement and the required stripes test.
- Older historical recipes pass bars and stripes together under different contracts; they cannot fill today's cells.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) |
| [BISPF](families/bispf.md#cohort-cuda-3faddec347a0) | [FAIL](families/bispf.md#cohort-cuda-3faddec347a0-experiment-img_bars4) |
| [Atlas](families/atlas.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) |

**What remains to solve.** Preserve all four clean bar modes while passing stripes and the acquisition tests with the same recipe.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#img_bars4).


<a name="img_blobs4"></a>

## Generate all four image corners

Test: `img_blobs4` · Held for qualification; included in full-suite results.

**What it tests.** Generate a small bright square at each of four corners, with enough clean examples of every corner.

**The problem.** BISPF reaches 3 corners and 28/32 clean pictures; the gate requires all 4 and at least 29/32. BCAP reaches 2 corners. A near miss or four passing checks is not a five-check sustained pass.

**Has any configuration beaten it?** Yes: R1/R2 and kinetic research configurations.

- R1/R2 passes in separate diagnostic measurements with an 18-check streak, but loses ring16 and the clock measurement.
- `kinetic_transport_local_v2` passes but fails two-pole and unused-token hold. Four kinetic history variants pass but are blocked on basic behavioural tests.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) |
| [BISPF](families/bispf.md#cohort-cuda-3faddec347a0) | [FAIL](families/bispf.md#cohort-cuda-3faddec347a0-experiment-img_blobs4) |
| [Atlas](families/atlas.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) |

**What remains to solve.** Keep all four corners clean and sustained without breaking basic learning and identity preservation.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#img_blobs4).


<a name="img_intensity2"></a>

## Generate both brightness levels accurately

Test: `img_intensity2` · Held for qualification; included in full-suite results.

**What it tests.** Generate a square at either of two target brightnesses, with both levels represented and little brightness error.

**The problem.** BISPF collapses to one brightness level. BCAP produces both but 10/32 pictures have inaccurate brightness.

**Has any configuration beaten it?** Yes: GAN v3, R1/R2 and kinetic transport in separate measurements.

- GAN v3, R1/R2 and `kinetic_transport_local_v2` pass the recorded declaration, but fail required Tier 1 tests.
- A historical wider-network pass belongs to `img_intensity2_residual16`, a different task variant.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) |
| [BISPF](families/bispf.md#cohort-cuda-3faddec347a0) | [FAIL](families/bispf.md#cohort-cuda-3faddec347a0-experiment-img_intensity2) |
| [Atlas](families/atlas.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) |

**What remains to solve.** Learn accurate, balanced brightness modes while retaining the other image and acquisition passes.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#img_intensity2).


<a name="grid100"></a>

## A grid of 100 clusters

Test: `grid100` · Held for qualification; included in full-suite results.

**What it tests.** Learn all 100 small blobs with correct locations, populations and widths, scoring the current model's clean outputs without adding evaluation noise.

**The problem.** BCAP/BISPF put only 24–37% of samples on target, against 97% required. Some blobs are thin or nearly empty; the best audited 7,000-update result reaches 54%, still below the gate.

**Has any configuration beaten it?** No full PASS under the current clean-sampling declaration in the audited evidence.

- Historical K3P and other recipes have PASS receipts under earlier contracts, including different priors, averaged weights or evaluation noise. Those are separate problems and do not beat this declaration.
- True-target samplers demonstrate feasibility. Longer BISPF training improves precision initially but plateaus below the gate.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-grid100) |
| [BISPF](families/bispf.md#cohort-cuda-3faddec347a0) | [FAIL](families/bispf.md#cohort-cuda-3faddec347a0-experiment-grid100) |
| [Atlas](families/atlas.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/atlas.md#cohort-cuda-1bf9d7d34422-experiment-grid100) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-grid100) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-grid100) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-grid100) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-grid100) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-grid100) |

**What remains to solve.** Improve both precise placement and within-cluster shape, then meet the sustained checks and independent holdout. Longer training alone has not solved it.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#grid100).


<a name="rotated100"></a>

## A rotated grid of 100 clusters

Test: `rotated100` · Held for qualification; included in full-suite results.

**What it tests.** Learn all 100 small blobs with correct locations, populations and widths, scoring the current model's clean outputs without adding evaluation noise.

**The problem.** BCAP/BISPF put only 26–27% of samples on target, against 97% required. Some blobs are thin or nearly empty; the best audited 7,000-update result reaches 51%, still below the gate.

**Has any configuration beaten it?** No full PASS under the current clean-sampling declaration in the audited evidence.

- Historical K3P and other recipes have PASS receipts under earlier contracts, including different priors, averaged weights or evaluation noise. Those are separate problems and do not beat this declaration.
- True-target samplers demonstrate feasibility. Longer BISPF training improves precision initially but plateaus below the gate.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) |
| [BISPF](families/bispf.md#cohort-cuda-3faddec347a0) | [FAIL](families/bispf.md#cohort-cuda-3faddec347a0-experiment-rotated100) |
| [Atlas](families/atlas.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/atlas.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) |

**What remains to solve.** Improve both precise placement and within-cluster shape, then meet the sustained checks and independent holdout. Longer training alone has not solved it.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#rotated100).


<a name="staggered100"></a>

## A staggered grid of 100 clusters

Test: `staggered100` · Held for qualification; included in full-suite results.

**What it tests.** Learn all 100 small blobs with correct locations, populations and widths, scoring the current model's clean outputs without adding evaluation noise.

**The problem.** BCAP/BISPF put only 30–34% of samples on target, against 97% required. Some blobs are thin or nearly empty; the best audited 7,000-update result reaches 56%, still below the gate.

**Has any configuration beaten it?** No full PASS under the current clean-sampling declaration in the audited evidence.

- Historical K3P and other recipes have PASS receipts under earlier contracts, including different priors, averaged weights or evaluation noise. Those are separate problems and do not beat this declaration.
- True-target samplers demonstrate feasibility. Longer BISPF training improves precision initially but plateaus below the gate.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) |
| [BISPF](families/bispf.md#cohort-cuda-3faddec347a0) | [FAIL](families/bispf.md#cohort-cuda-3faddec347a0-experiment-staggered100) |
| [Atlas](families/atlas.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/atlas.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) |

**What remains to solve.** Improve both precise placement and within-cluster shape, then meet the sustained checks and independent holdout. Longer training alone has not solved it.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#staggered100).


<a name="grid100_14k"></a>

## Keep the 100-cluster grid correct for longer

Test: `grid100_14k` · Held for qualification; included in full-suite results.

**What it tests.** Continue a model that passed the 7,000-update 100-cluster test to 14,000 updates without losing quality.

**The problem.** The required passing parent checkpoint does not exist. These are blocked continuations, not completed numerical failures. Continuing failed BISPF parents diagnostically still did not pass.

**Has any configuration beaten it?** No qualifying PASS; no passing current-declaration parent is available.

- The BISPF rescue continuations from failed parents remain below the precision gate and do not establish a valid endurance pass.
- The rotated rescue has an INVALID cross-device coverage audit near a scoring boundary; it is not a PASS or a scientific FAIL.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-grid100_14k) |
| [BISPF](families/bispf.md#cohort-cuda-3faddec347a0) | [NOT MEASURED](families/bispf.md#cohort-cuda-3faddec347a0-experiment-grid100_14k) |
| [Atlas](families/atlas.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/atlas.md#cohort-cuda-1bf9d7d34422-experiment-grid100_14k) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-grid100_14k) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-grid100_14k) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-grid100_14k) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-grid100_14k) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-grid100_14k) |

**What remains to solve.** First obtain a genuine passing parent under the unchanged clean-sampling task, then continue that exact checkpoint. Keep rescue experiments separate.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#grid100_14k).


<a name="rotated100_14k"></a>

## Keep the 100-cluster rotated grid correct for longer

Test: `rotated100_14k` · Held for qualification; included in full-suite results.

**What it tests.** Continue a model that passed the 7,000-update 100-cluster test to 14,000 updates without losing quality.

**The problem.** The required passing parent checkpoint does not exist. These are blocked continuations, not completed numerical failures. Continuing failed BISPF parents diagnostically still did not pass.

**Has any configuration beaten it?** No qualifying PASS; no passing current-declaration parent is available.

- The BISPF rescue continuations from failed parents remain below the precision gate and do not establish a valid endurance pass.
- The rotated rescue has an INVALID cross-device coverage audit near a scoring boundary; it is not a PASS or a scientific FAIL.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_14k) |
| [BISPF](families/bispf.md#cohort-cuda-3faddec347a0) | [NOT MEASURED](families/bispf.md#cohort-cuda-3faddec347a0-experiment-rotated100_14k) |
| [Atlas](families/atlas.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/atlas.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_14k) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_14k) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_14k) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_14k) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_14k) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_14k) |

**What remains to solve.** First obtain a genuine passing parent under the unchanged clean-sampling task, then continue that exact checkpoint. Keep rescue experiments separate.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#rotated100_14k).


<a name="staggered100_14k"></a>

## Keep the 100-cluster staggered grid correct for longer

Test: `staggered100_14k` · Held for qualification; included in full-suite results.

**What it tests.** Continue a model that passed the 7,000-update 100-cluster test to 14,000 updates without losing quality.

**The problem.** The required passing parent checkpoint does not exist. These are blocked continuations, not completed numerical failures. Continuing failed BISPF parents diagnostically still did not pass.

**Has any configuration beaten it?** No qualifying PASS; no passing current-declaration parent is available.

- The BISPF rescue continuations from failed parents remain below the precision gate and do not establish a valid endurance pass.
- The rotated rescue has an INVALID cross-device coverage audit near a scoring boundary; it is not a PASS or a scientific FAIL.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_14k) |
| [BISPF](families/bispf.md#cohort-cuda-3faddec347a0) | [NOT MEASURED](families/bispf.md#cohort-cuda-3faddec347a0-experiment-staggered100_14k) |
| [Atlas](families/atlas.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/atlas.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_14k) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_14k) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_14k) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_14k) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_14k) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_14k) |

**What remains to solve.** First obtain a genuine passing parent under the unchanged clean-sampling task, then continue that exact checkpoint. Keep rescue experiments separate.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#staggered100_14k).


<a name="target_shift_recovery"></a>

## Recover quickly after the target moves

Test: `target_shift_recovery` · Held for qualification; included in full-suite results.

**What it tests.** Learn and hold a ring of blobs, then track a sudden sideways shift within 400 updates and stay correct; a frozen model must fail to show the change is real.

**The problem.** BISPF learns and holds the original ring, then recovers too late: it becomes steady 620 updates after the shift, missing the 400-update deadline. This is an adaptation-speed failure, not an inability to recover at all.

**Has any configuration beaten it?** No complete PASS under the current declaration in the audited evidence.

- BISPF meets three of four parts but misses the recovery deadline.
- Older PR215/PR217 recipes pass a different ring-shift question under earlier contracts, not this test.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) |
| [BISPF](families/bispf.md#cohort-cuda-3faddec347a0) | [NOT MEASURED](families/bispf.md#cohort-cuda-3faddec347a0-experiment-target_shift_recovery) |
| [Atlas](families/atlas.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/atlas.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) |

**What remains to solve.** Improve adaptation speed while keeping stationary stability. If the deadline itself is reconsidered, use an explicitly versioned task and preserve the old failure.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#target_shift_recovery).


<a name="mode_hold"></a>

## Keep every learned mode alive

Test: `mode_hold` · Required for qualification and included in full-suite results.

**What it tests.** Keep generating every target mode during continued training, rather than losing modes after initially finding them.

**The problem.** BCAP falls one sustained window short in the reviewed run (3 terminal passing checks; 5 required). This single test accounts for BISPF's six-point lead when repeated across six views.

**Has any configuration beaten it?** Yes: the selected BISPF recipe and some research configurations.

- `bispf-v1` has a recorded PASS with a 12-check terminal streak. Its source is stale; this is not a fresh current-source qualification.
- The cell-quota minimum-change recipe also passes mode hold but fails ring16. The reviewed tether-only BCAP ablation does not reproduce the BISPF win.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) |
| [BISPF](families/bispf.md#cohort-cuda-3faddec347a0) | [PASS](families/bispf.md#cohort-cuda-3faddec347a0-experiment-mode_hold) |
| [Atlas](families/atlas.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/atlas.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) |

**What remains to solve.** Preserve the BISPF pass on current source and combine it with repairs to Gaussian, vector and image failures.

Evidence: [BISPF lead and ablation evidence](WHAT_WORKED_REVIEW_20261015.md).


<a name="ring_extension"></a>

## Keep a learned ring stable for a long run

Test: `ring_extension` · Required for qualification and included in full-suite results.

**What it tests.** Acquire all eight ring blobs, hold them for 1,200 consecutive checks, then survive 300 additional checks on the same uninterrupted run.

**The problem.** BISPF lacks an ordinary qualifying run in the selected cohort. BCAP's diagnostic run loses quality during the hold phase; the extension cannot rescue a failed hold.

**Has any configuration beaten it?** Yes: BISPF in a saved separate-runtime diagnostic; no selected ordinary PASS.

- BISPF producer `bf5d13ff` passes the full diagnostic run: 1,200/1,200 hold checks and 300/300 extension checks, with minimum quality above 0.99.
- That diagnostic result is evidence the problem is solvable. It does not fill the selected cohort's unmeasured cell. `ring_hold` remains a visible phase, not a second independent test.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) |
| [BISPF](families/bispf.md#cohort-cuda-3faddec347a0) | [PASS](families/bispf.md#cohort-cuda-3faddec347a0-experiment-ring_extension) |
| [Atlas](families/atlas.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/atlas.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) |

**What remains to solve.** Measure the unchanged BISPF recipe in its ordinary cohort on current source and preserve the whole acquisition/hold/extension protocol.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#ring_extension).


<a name="ring16_acquisition"></a>

## Learn sixteen distinct clusters from scratch

Test: `ring16_acquisition` · Required for qualification and included in full-suite results.

**What it tests.** Learn all sixteen blobs on a ring with balanced populations and real within-blob spread, rather than sixteen collapsed points.

**The problem.** The selected K3P, KA2, R1/R2 and GAN v3 measurements fail this Tier 1 gate, preventing later qualification. Their results on other questions do not erase this acquisition failure.

**Has any configuration beaten it?** Yes: selected BCAP and BISPF recipes.

- The selected BCAP DualNorm and `bispf-v1` records pass. Their recorded results remain bound to their original budgets and sources.
- Some research recipes pass ring16 but lose mode hold; per-test winners do not establish a single winner for the entire suite.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [PASS](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) |
| [BISPF](families/bispf.md#cohort-cuda-3faddec347a0) | [PASS](families/bispf.md#cohort-cuda-3faddec347a0-experiment-ring16_acquisition) |
| [Atlas](families/atlas.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/atlas.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) |

**What remains to solve.** Retain the passing ring baseline while repairing the held tests. Longer-budget or changed-schedule results must remain explicitly separate until adopted and measured.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#gaussian1d_stability) · [Task declaration](../../configs/forge/tasks/ring16_acquisition.json).


<a name="clockfree_audit_measurement_v1"></a>

## Learning must not depend on external clock labels

Test: `clockfree_audit_measurement_v1` · Required for qualification and included in full-suite results.

**What it tests.** Check that the same learning state and inputs produce the same update despite changed step labels, declared horizon, evaluation cadence or a save/restart.

**The problem.** The recorded scheduled K3P, KA2, R1/R2 and GAN v3 recipes depend on step or horizon through annealing. Good numerical fits do not satisfy this separate invariance requirement.

**Has any configuration beaten it?** Yes: selected BCAP and BISPF recipes.

- BCAP and BISPF pass the Tier 1 measurement with constant-rate configurations.
- `clockfree_audit` is the duplicate longer-form gate and remains visible as a diagnostic; it is excluded from the distinct full-suite total to avoid counting the same question twice.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [PASS](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) |
| [BISPF](families/bispf.md#cohort-cuda-3faddec347a0) | [PASS](families/bispf.md#cohort-cuda-3faddec347a0-experiment-clockfree_audit_measurement_v1) |
| [Atlas](families/atlas.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/atlas.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) |

**What remains to solve.** Use saved internal state rather than external labels for adaptation, or explicitly version the desired policy. Do not silently treat an annealed recipe as clock-free.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#clockfree_audit).
