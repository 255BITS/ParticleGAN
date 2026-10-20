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
| [Escape a tanh saturation rail after the target shifts to zero](problems.md#zero_tanh_rail_escape) | First converge near the tanh saturation rail at (+0.90, -0.90) where tanh' is attenuated (~0.19), then escape the rail within 400 updates when the target shifts to (0.0, 0.0) at update 1,001 and stay settled there. | Yes in standalone public-API seed-0 execution (K3P); not yet measured across the full selected-family roster. |
| [Survive a 16-layer unnormalized MLP generator on the 100-cluster grid](problems.md#grid100_mlp16) | Learn all 100 equal-weight Gaussian modes on `grid100` using a 16-hidden-layer 128-wide LeakyReLU(0.2) MLP generator without residual skips or normalization layers. | No full PASS under the current clean-sampling declaration; constructive capacity witness passes. |
| [Learn the 100-cluster grid with a 16-layer residual MLP and bounded tanh head](problems.md#grid100_res16_tanh) | Recover all 100 Gaussian modes on `grid100` using a 16-layer identity-skip residual MLP generator and a bounded `S=5.25` scaled `tanh` output head without saturating the `\|x\| > 0.99 * 5.25` rails. | No full PASS under the current declaration; constructive residual-tanh witness passes with zero saturation. |
| [Recover an 8D lifted 100-mode manifold from incomplete block-masked data (MisGAN + ppost)](problems.md#misgan_block_recovery) | Train a 2-pair incomplete-data MisGAN (`(G_x, D_x)` + `(G_m, D_m)` with detached mask gradient in `L_x`) observing only masked 8D data `(x * m, m)` under structured block dropout, then impute held-out rows via native Bayesian particle-posterior (`ppost`) resampling from `G_x`. | Solvable via detached-mask MisGAN + `ppost` (`acc_lo = 0.759` in `origin/misgan-toy` and `tests/test_solvability_misgan.py`); default `ka2` public-API baseline fails. |
| [Learn a multimodal neural imputer on ambiguous single-sensor block masks without ppost](problems.md#misgan_block_neural_imputer) | Train a 3-pair MisGAN (`(G_x, D_x)` + `(G_m, D_m)` + `(G_i, D_i)`) and grade the neural imputer `G_i([x * m, m, z_i])` directly on the 16 held-out ambiguous single-sensor queries without `ppost` particle resampling. | No adversarial recipe PASS yet; trained supervised neural witness in `tests/test_solvability_misgan.py` passes all gates. |
| [Learn a continuous 16-mode ring joint Encoder-Generator-Critic BiGAN](problems.md#ring16_bigan_continuous) | Train a continuous 2D multimodal joint BiGAN `(G(z), E(x), D(x, z))` on the 16-Gaussian ring (`MoGParticlePrior(sigma=0.1)`), simultaneously satisfying forward 16-mode generation and paired round-trip reconstruction `x -> G(E(x))`. | No full adversarial PASS yet (passes 2 of 5 gates out of the box under `bcap`); trained `(G, E)` witness in `tests/test_solvability_bigan.py` passes all gates. |

<a name="gaussian1d_stability"></a>

## Keep a bell curve stable, then learn a moved target

Test: `gaussian1d_stability` · Required for qualification and included in full-suite results.

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
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) |
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
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) |
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
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) |
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
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) |
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
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) |
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
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) |
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
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) |
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
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) |
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

Test: `grid100` · Required for qualification and included in full-suite results.

**What it tests.** Learn all 100 small blobs with correct locations, populations and widths, scoring the current model's clean outputs without adding evaluation noise.

**The problem.** BCAP/BISPF put only 24–37% of samples on target, against 97% required. Some blobs are thin or nearly empty; the best audited 7,000-update result reaches 54%, still below the gate.

**Has any configuration beaten it?** No full PASS under the current clean-sampling declaration in the audited evidence.

- Historical K3P and other recipes have PASS receipts under earlier contracts, including different priors, averaged weights or evaluation noise. Those are separate problems and do not beat this declaration.
- True-target samplers demonstrate feasibility. Longer BISPF training improves precision initially but plateaus below the gate.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-grid100) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100) |
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

Test: `rotated100` · Required for qualification and included in full-suite results.

**What it tests.** Learn all 100 small blobs with correct locations, populations and widths, scoring the current model's clean outputs without adding evaluation noise.

**The problem.** BCAP/BISPF put only 26–27% of samples on target, against 97% required. Some blobs are thin or nearly empty; the best audited 7,000-update result reaches 51%, still below the gate.

**Has any configuration beaten it?** No full PASS under the current clean-sampling declaration in the audited evidence.

- Historical K3P and other recipes have PASS receipts under earlier contracts, including different priors, averaged weights or evaluation noise. Those are separate problems and do not beat this declaration.
- True-target samplers demonstrate feasibility. Longer BISPF training improves precision initially but plateaus below the gate.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) |
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

Test: `staggered100` · Required for qualification and included in full-suite results.

**What it tests.** Learn all 100 small blobs with correct locations, populations and widths, scoring the current model's clean outputs without adding evaluation noise.

**The problem.** BCAP/BISPF put only 30–34% of samples on target, against 97% required. Some blobs are thin or nearly empty; the best audited 7,000-update result reaches 56%, still below the gate.

**Has any configuration beaten it?** No full PASS under the current clean-sampling declaration in the audited evidence.

- Historical K3P and other recipes have PASS receipts under earlier contracts, including different priors, averaged weights or evaluation noise. Those are separate problems and do not beat this declaration.
- True-target samplers demonstrate feasibility. Longer BISPF training improves precision initially but plateaus below the gate.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) |
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

Test: `grid100_14k` · Required for qualification and included in full-suite results.

**What it tests.** Continue a model that passed the 7,000-update 100-cluster test to 14,000 updates without losing quality.

**The problem.** The required passing parent checkpoint does not exist. These are blocked continuations, not completed numerical failures. Continuing failed BISPF parents diagnostically still did not pass.

**Has any configuration beaten it?** No qualifying PASS; no passing current-declaration parent is available.

- The BISPF rescue continuations from failed parents remain below the precision gate and do not establish a valid endurance pass.
- The rotated rescue has an INVALID cross-device coverage audit near a scoring boundary; it is not a PASS or a scientific FAIL.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-grid100_14k) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100_14k) |
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

Test: `rotated100_14k` · Required for qualification and included in full-suite results.

**What it tests.** Continue a model that passed the 7,000-update 100-cluster test to 14,000 updates without losing quality.

**The problem.** The required passing parent checkpoint does not exist. These are blocked continuations, not completed numerical failures. Continuing failed BISPF parents diagnostically still did not pass.

**Has any configuration beaten it?** No qualifying PASS; no passing current-declaration parent is available.

- The BISPF rescue continuations from failed parents remain below the precision gate and do not establish a valid endurance pass.
- The rotated rescue has an INVALID cross-device coverage audit near a scoring boundary; it is not a PASS or a scientific FAIL.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_14k) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_14k) |
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

Test: `staggered100_14k` · Required for qualification and included in full-suite results.

**What it tests.** Continue a model that passed the 7,000-update 100-cluster test to 14,000 updates without losing quality.

**The problem.** The required passing parent checkpoint does not exist. These are blocked continuations, not completed numerical failures. Continuing failed BISPF parents diagnostically still did not pass.

**Has any configuration beaten it?** No qualifying PASS; no passing current-declaration parent is available.

- The BISPF rescue continuations from failed parents remain below the precision gate and do not establish a valid endurance pass.
- The rotated rescue has an INVALID cross-device coverage audit near a scoring boundary; it is not a PASS or a scientific FAIL.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_14k) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_14k) |
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

Test: `target_shift_recovery` · Required for qualification and included in full-suite results.

**What it tests.** Learn and hold a ring of blobs, then track a sudden sideways shift within 400 updates and stay correct; a frozen model must fail to show the change is real.

**The problem.** BISPF learns and holds the original ring, then recovers too late: it becomes steady 620 updates after the shift, missing the 400-update deadline. This is an adaptation-speed failure, not an inability to recover at all.

**Has any configuration beaten it?** No complete PASS under the current declaration in the audited evidence.

- BISPF meets three of four parts but misses the recovery deadline.
- Older PR215/PR217 recipes pass a different ring-shift question under earlier contracts, not this test.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) |
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
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [PASS](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) |
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
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [PASS](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) |
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
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [PASS](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) |
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
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [PASS](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) |
| [Atlas](families/atlas.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/atlas.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) |

**What remains to solve.** Use saved internal state rather than external labels for adaptation, or explicitly version the desired policy. Do not silently treat an annealed recipe as clock-free.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#clockfree_audit).


<a name="zero_tanh_rail_escape"></a>

## Escape a tanh saturation rail after the target shifts to zero

Test: `zero_tanh_rail_escape` · Held for qualification; included in full-suite results.

**What it tests.** First converge near the tanh saturation rail at (+0.90, -0.90) where tanh' is attenuated (~0.19), then escape the rail within 400 updates when the target shifts to (0.0, 0.0) at update 1,001 and stay settled there.

**The problem.** Bounded tanh heads attenuate pre-activation gradients near +/-1. Recipes that allow large pre-activation drift during the first 1,000 updates can lock into the rail or overshoot and oscillate around zero after the shift; no ordinary selected-cohort qualification run has been recorded on this newly added Tier 3 gate yet.

**Has any configuration beaten it?** Yes in standalone public-API seed-0 execution (K3P); not yet measured across the full selected-family roster.

- Standalone public-API execution of `k3p` at seed 0 passes all 5 terminal checks (`mean_abs_error = 0.00298 <= 0.005`, `rms_error = 0.00421 <= 0.01`, `p99_abs_error = 0.01416 <= 0.025`, `saturated_fraction = 0.0`).
- The 3-level solvability suite (`tests/test_solvability_tanh.py`) verifies 200/200 oracle passes and a constructive network witness.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-zero_tanh_rail_escape) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-zero_tanh_rail_escape) |
| [Atlas](families/atlas.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/atlas.md#cohort-cuda-1bf9d7d34422-experiment-zero_tanh_rail_escape) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-zero_tanh_rail_escape) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-zero_tanh_rail_escape) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-zero_tanh_rail_escape) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-zero_tanh_rail_escape) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-zero_tanh_rail_escape) |

**What remains to solve.** Run the full Tier 3 qualified family roster (`k3p`, `ka2`, `r1r2`, `bcap-pure`, `bcap-dualnorm`, `atlas`, `e22`, `release07-gan-v3`) in ordinary cohorts and promote from held to required once a global recipe passes it together with all counted Tier 1-3 requirements.

Evidence: [Public-API readout](../toy_audit/api_contract/api-zero-tanh-rail-escape/readout.md) · [Task declaration](../../configs/forge/tasks/zero_tanh_rail_escape.json).


<a name="grid100_mlp16"></a>

## Survive a 16-layer unnormalized MLP generator on the 100-cluster grid

Test: `grid100_mlp16` · Held for qualification; included in full-suite results.

**What it tests.** Learn all 100 equal-weight Gaussian modes on `grid100` using a 16-hidden-layer 128-wide LeakyReLU(0.2) MLP generator without residual skips or normalization layers.

**The problem.** Increasing generator depth from 3 to 16 hidden layers contracts initial output spread by ~156x and attenuates first-layer and particle-prior gradients by ~73x at step 1. In the standalone seed-0 `k3p` run, the 16-layer generator collapses to 0/100 modes (`output_sigma = 0.029`, `mass_tv = 0.99`).

**Has any configuration beaten it?** No full PASS under the current clean-sampling declaration; constructive capacity witness passes.

- Constructive 16-layer capacity witness in `tests/test_solvability_deep100g.py` recovers 100/100 modes and passes all coverage and 100k accuracy bounds, proving the 16-layer architecture can represent the target.
- Oracle sampler passes 200/200 draws with >= 2x noise-floor headroom.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-grid100_mlp16) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100_mlp16) |
| [Atlas](families/atlas.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/atlas.md#cohort-cuda-1bf9d7d34422-experiment-grid100_mlp16) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-grid100_mlp16) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-grid100_mlp16) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-grid100_mlp16) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-grid100_mlp16) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-grid100_mlp16) |

**What remains to solve.** Pair layer-wise gradient normalization (such as `bcap-nsgda-layer` or `bcap-dualnorm`) with balanced prior-to-generator step scaling so early updates do not collapse the 16-layer Jacobian before particles separate across the 10x10 grid.

Evidence: [Public-API readout](../toy_audit/api_contract/api-grid100-mlp16/readout.md) · [Task declaration](../../configs/forge/tasks/grid100_mlp16.json).


<a name="grid100_res16_tanh"></a>

## Learn the 100-cluster grid with a 16-layer residual MLP and bounded tanh head

Test: `grid100_res16_tanh` · Held for qualification; included in full-suite results.

**What it tests.** Recover all 100 Gaussian modes on `grid100` using a 16-layer identity-skip residual MLP generator and a bounded `S=5.25` scaled `tanh` output head without saturating the `|x| > 0.99 * 5.25` rails.

**The problem.** Identity skip connections prevent the 16-layer vanishing-gradient collapse of `grid100_mlp16`, but under unscaled critic updates (`k3p` at seed 0) 96.4% of generated samples are pushed into the `tanh` saturation rails (`saturated_fraction = 0.9639` vs. `<= 0.001` bound, `modes = 0/100`).

**Has any configuration beaten it?** No full PASS under the current declaration; constructive residual-tanh witness passes with zero saturation.

- Constructive 16-layer residual + `S=5.25` tanh witness in `tests/test_solvability_deep100g.py` passes all 100-mode coverage, 100k accuracy, and `saturated_fraction <= 0.001` bounds (`saturated_fraction = 0.0`).
- Oracle sampler passes 200/200 draws.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-grid100_res16_tanh) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100_res16_tanh) |
| [Atlas](families/atlas.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/atlas.md#cohort-cuda-1bf9d7d34422-experiment-grid100_res16_tanh) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-grid100_res16_tanh) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-grid100_res16_tanh) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-grid100_res16_tanh) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-grid100_res16_tanh) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-grid100_res16_tanh) |

**What remains to solve.** Control pre-tanh logit growth via bounded step norms (`bcap-ada-nsgda` / `bcap-dualnorm`) or critic-gradient damping near the `S=5.25` boundary so residual branches do not drive particles onto the rails.

Evidence: [Public-API readout](../toy_audit/api_contract/api-grid100-res16-tanh/readout.md) · [Task declaration](../../configs/forge/tasks/grid100_res16_tanh.json).


<a name="misgan_block_recovery"></a>

## Recover an 8D lifted 100-mode manifold from incomplete block-masked data (MisGAN + ppost)

Test: `misgan_block_recovery` · Held for qualification; included in full-suite results.

**What it tests.** Train a 2-pair incomplete-data MisGAN (`(G_x, D_x)` + `(G_m, D_m)` with detached mask gradient in `L_x`) observing only masked 8D data `(x * m, m)` under structured block dropout, then impute held-out rows via native Bayesian particle-posterior (`ppost`) resampling from `G_x`.

**The problem.** When 10% of rows observe only sensor `{7}`, naive conditional generators collapse the multimodal posterior. In the standalone seed-0 `ka2` public-API run, unbalanced multi-pair updates allow off-plane leakage (`off_plane = 3.55` vs. `<= 0.04`, `modes = 3/100`, `mask_tv = 0.956`).

**Has any configuration beaten it?** Solvable via detached-mask MisGAN + `ppost` (`acc_lo = 0.759` in `origin/misgan-toy` and `tests/test_solvability_misgan.py`); default `ka2` public-API baseline fails.

- Constructive witness and `origin/misgan-toy` detached-mask `G_x` + `ppost` achieve 100/100 modes, `hq >= 0.99`, and `acc_lo = 0.759 >= 0.70` (with 2.88x noise-floor headroom relative to the 0.777 Bayes ceiling).
- Oracle `bayes_posterior` + `ppost` passes 200/200 draws in `tests/test_solvability_misgan.py`.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_recovery) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_recovery) |
| [Atlas](families/atlas.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/atlas.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_recovery) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_recovery) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_recovery) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_recovery) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_recovery) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_recovery) |

**What remains to solve.** Balance data-pair `(G_x, D_x)` and mask-pair `(G_m, D_m)` critic regularization under a global recipe so `G_x` stays concentrated on the 2D subspace embedded in R^8 without off-plane drift.

Evidence: [Public-API readout](../toy_audit/api_contract/api-misgan-block-recovery/readout.md) · [Task declaration](../../configs/forge/tasks/misgan_block_recovery.json).


<a name="misgan_block_neural_imputer"></a>

## Learn a multimodal neural imputer on ambiguous single-sensor block masks without ppost

Test: `misgan_block_neural_imputer` · Held for qualification; included in full-suite results.

**What it tests.** Train a 3-pair MisGAN (`(G_x, D_x)` + `(G_m, D_m)` + `(G_i, D_i)`) and grade the neural imputer `G_i([x * m, m, z_i])` directly on the 16 held-out ambiguous single-sensor queries without `ppost` particle resampling.

**The problem.** Standard neural imputers `G_i([x * m, m, z_i])` satisfy the observed-sensor identity (`observed_max_error = 0.0`) and global missing-coordinate spread (`missing_std_ratio = 0.803` in `[0.7, 1.3]`), but suffer from conditionalGAN `z_i` noise-channel ignore on the 10% single-sensor rows (`acc_lo = 0.00019` vs. `>= 0.65`).

**Has any configuration beaten it?** No adversarial recipe PASS yet; trained supervised neural witness in `tests/test_solvability_misgan.py` passes all gates.

- Trained PyTorch neural network witness `G_i` in `tests/test_solvability_misgan.py` passes 100% of gates (`observed_max_error <= 1e-6`, `acc >= 0.95`, `acc_lo >= 0.65`, `missing_std_ratio` in `[0.70, 1.30]`), proving the neural imputer architecture has sufficient capacity.
- Oracle posterior sampler passes 200/200 draws with >= 2x noise headroom.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_neural_imputer) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_neural_imputer) |
| [Atlas](families/atlas.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/atlas.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_neural_imputer) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_neural_imputer) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_neural_imputer) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_neural_imputer) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_neural_imputer) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_neural_imputer) |

**What remains to solve.** Couple the imputer latent `z_i` to the particle prior or add an explicit conditional diversity / mutual-information pressure so `D_i` penalizes mode-averaging on single-sensor masks.

Evidence: [Public-API readout](../toy_audit/api_contract/api-misgan-block-neural-imputer/readout.md) · [Task declaration](../../configs/forge/tasks/misgan_block_neural_imputer.json).


<a name="ring16_bigan_continuous"></a>

## Learn a continuous 16-mode ring joint Encoder-Generator-Critic BiGAN

Test: `ring16_bigan_continuous` · Held for qualification; included in full-suite results.

**What it tests.** Train a continuous 2D multimodal joint BiGAN `(G(z), E(x), D(x, z))` on the 16-Gaussian ring (`MoGParticlePrior(sigma=0.1)`), simultaneously satisfying forward 16-mode generation and paired round-trip reconstruction `x -> G(E(x))`.

**The problem.** In the standalone seed-0 `bcap` public-API run, the joint game discovers all 16/16 modes (`modes = 16`) and achieves high round-trip mode consistency (`mode_consistency = 0.9951 >= 0.98`) and near-passing mass balance (`mass_tv = 0.1557` vs. `<= 0.15`), but within-mode variance remains too broad (`hq = 0.238` vs. `>= 0.85`, `recon_rmse = 0.220` vs. `<= 0.12`).

**Has any configuration beaten it?** No full adversarial PASS yet (passes 2 of 5 gates out of the box under `bcap`); trained `(G, E)` witness in `tests/test_solvability_bigan.py` passes all gates.

- Trained PyTorch `(G, E)` witness in `tests/test_solvability_bigan.py` passes all forward generation and round-trip reconstruction gates (`modes = 16`, `hq >= 0.85`, `mass_tv <= 0.15`, `recon_rmse <= 0.12`, `mode_consistency >= 0.98`).
- Oracle joint sampler passes 200/200 draws with >= 2x noise headroom.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-ring16_bigan_continuous) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-ring16_bigan_continuous) |
| [Atlas](families/atlas.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/atlas.md#cohort-cuda-1bf9d7d34422-experiment-ring16_bigan_continuous) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-ring16_bigan_continuous) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-ring16_bigan_continuous) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-ring16_bigan_continuous) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-ring16_bigan_continuous) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-ring16_bigan_continuous) |

**What remains to solve.** Tighten joint-critic `(x, z)` local curvature regularization (`k3p` or `bcap-dualnorm`) so `G(z)` and `E(x)` contract within-mode variance without losing 16-mode coverage.

Evidence: [Public-API readout](../toy_audit/api_contract/api-ring16-bigan-continuous/readout.md) · [Task declaration](../../configs/forge/tasks/ring16_bigan_continuous.json).
