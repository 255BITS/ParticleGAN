# Forge family leaderboard

A **family** can have several **recipes**. Each score belongs to one exact recipe and source. For example, BCAP's selected recipe uses DualNorm; Confidence LR, Finite critic cap and Positive evidence are separate BCAP research recipes, with their own scores.

Click a count for its gates, measurements and evidence. Required lower tiers must pass before advancing.

## Full-suite results — failures included

**33 distinct tests**, including held tests. Each test counts once. A held FAIL is still a failure in this table. Only duplicate questions and phases already included in another test are excluded. These are recorded results for the selected recipe and source, not new measurements or current-source qualification.

[Problem list: what each test asks, what fails, and which configurations have passed](problems.md).

| Family / selected recipe | Pass | Fail | Other outcomes | Total tests |
| --- | ---: | ---: | --- | ---: |
| **[BCAP](families/bcap-pure.md)** · [BCAP dualnorm (experimental starting point)](families/bcap-dualnorm.md)<br>Recorded source measurement · **stale**; no current qualification | 16 | **12** | 5 not measured | 33 |
| **[BISPF](families/bispf.md)** | 18 | **11** | 4 not measured | 33 |
| **[Atlas](families/atlas.md)** | 0 | **0** | 29 blocked; 4 not measured | 33 |
| **[E22](families/e22.md)** | 0 | **0** | 29 blocked; 4 not measured | 33 |
| **[GAN v3 release 0.7](families/release07-gan-v3.md)**<br>Recorded source measurement · **stale**; no current qualification | 5 | **2** | 26 not measured | 33 |
| **[K3P](families/k3p.md)**<br>Recorded source measurement · **stale**; no current qualification | 5 | **2** | 26 not measured | 33 |
| **[KA2](families/ka2.md)**<br>Recorded source measurement · **stale**; no current qualification | 5 | **2** | 26 not measured | 33 |
| **[R1/R2](families/r1r2.md)**<br>Recorded source measurement · **stale**; no current qualification | 5 | **2** | 26 not measured | 33 |

Other outcomes are not scientific failures and earn no pass credit. Separate diagnostic and historical passes remain on the problem and family pages; they do not replace a selected recipe's result.

<details>
<summary>Qualification subset — held failures excluded from these scores</summary>

## Selected recipes — qualification subset

**This subset excludes held tests. Use the full-suite results above to compare failures.**

Passes / required-view instances for one selected recipe per family. Shared tests count once in every view that requires them: **81 means 81 counted view requirements, including repeated tests**, from 18 distinct tests. 66 held and 5 removed view assignments (15 held tests and 2 removed tests) are still measured and listed under **Held and removed tests** below. These totals retain the selected recipe's recorded qualification; separate diagnostic runs appear below.

| Family / selected recipe | Tier 1 requirements | Tier 2 requirements | Tier 3 requirements | Total view requirements |
| --- | ---: | ---: | ---: | ---: |
| [BCAP](families/bcap-pure.md) · [BCAP dualnorm (experimental starting point)](families/bcap-dualnorm.md) | **[22/22](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-tier-1)** | **[49/55](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-tier-2)** | **[0/4 (4 blocked)](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-tier-3)** | **[71/81 (4 blocked)](families/bcap-pure.md#cohort-cuda-1bf9d7d34422)** |
| [BISPF](families/bispf.md) | **[22/22](families/bispf.md#cohort-cuda-3faddec347a0-tier-1)** | **[55/55](families/bispf.md#cohort-cuda-3faddec347a0-tier-2)** | **[4/4](families/bispf.md#cohort-cuda-3faddec347a0-tier-3)** | **[81/81](families/bispf.md#cohort-cuda-3faddec347a0)** |
| [Atlas](families/atlas.md) | **[0/22 (22 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-tier-1)** | **[0/55 (54 incompatible, 1 blocked)](families/atlas.md#cohort-cuda-1bf9d7d34422-tier-2)** | **[0/4 (4 blocked)](families/atlas.md#cohort-cuda-1bf9d7d34422-tier-3)** | **[0/81 (76 incompatible, 5 blocked)](families/atlas.md#cohort-cuda-1bf9d7d34422)** |
| [E22](families/e22.md) | **[0/22 (22 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-tier-1)** | **[0/55 (54 incompatible, 1 blocked)](families/e22.md#cohort-cuda-1bf9d7d34422-tier-2)** | **[0/4 (4 blocked)](families/e22.md#cohort-cuda-1bf9d7d34422-tier-3)** | **[0/81 (76 incompatible, 5 blocked)](families/e22.md#cohort-cuda-1bf9d7d34422)** |
| [GAN v3 release 0.7](families/release07-gan-v3.md) | **[20/22](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-tier-1)** | **[0/55 (55 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-tier-2)** | **[0/4 (4 blocked)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-tier-3)** | **[20/81 (4 blocked, 55 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422)** |
| [K3P](families/k3p.md) | **[20/22](families/k3p.md#cohort-cuda-1bf9d7d34422-tier-1)** | **[0/55 (55 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-tier-2)** | **[0/4 (4 blocked)](families/k3p.md#cohort-cuda-1bf9d7d34422-tier-3)** | **[20/81 (4 blocked, 55 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422)** |
| [KA2](families/ka2.md) | **[20/22](families/ka2.md#cohort-cuda-1bf9d7d34422-tier-1)** | **[0/55 (55 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-tier-2)** | **[0/4 (4 blocked)](families/ka2.md#cohort-cuda-1bf9d7d34422-tier-3)** | **[20/81 (4 blocked, 55 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422)** |
| [R1/R2](families/r1r2.md) | **[20/22](families/r1r2.md#cohort-cuda-1bf9d7d34422-tier-1)** | **[0/55 (55 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-tier-2)** | **[0/4 (4 blocked)](families/r1r2.md#cohort-cuda-1bf9d7d34422-tier-3)** | **[20/81 (4 blocked, 55 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422)** |

</details>

Counts follow the declared view revisions: adaptation 3, clockfree_continuous 4, discriminator_stability 11, formulation_comparison 2, host_profile_transfer 5, quality_coverage 3. The recorded qualification keeps discriminator_stability revision 9 until new ordinary evidence is registered; no recorded verdict was regraded.

Backfill coverage: **853/853 distinct recipe/test pairs attempted (selected and research)**; 0 pending and 0 running. [Outcomes, blocked attempts and setup](ATTEMPT_COVERAGE.md). This counts attempts, including start refusals, rather than passes or qualification.

## Held and removed tests

A **held** test is still declared, run and shown with its history, but it is not counted in the totals above or in qualification. It counts again once its re-entry rule is met. A test **removed from the count** stays in its views as a diagnostic, because another counted test already carries its question or it cannot run as declared. Rows come from the notes stored in the view declarations; each details link opens that test's explanation in [the decision record](test-dispositions-20261015/README.md), which also defines the words used.

Selected recipes' recorded results pass 4 of the held or removed view requirements; those passes stay visible here and on the family pages but no longer count. Separate diagnostic runs of 5 selected recipes pass some of these tests (named in the table); a diagnostic run never becomes a recorded result, so those passes count nowhere. BISPF's recorded measurement fails no counted test: 81/81 pass. Every test it fails is held. That is the one-global-recipe rule at work (one recipe must pass every counted test): BISPF passes every counted test that any other selected recipe passes. It is not a new measurement of BISPF.

While target_shift_recovery is held, adaptation counts the same tests as quality_coverage. Its highest tier, Tier 3, has only held tests, so it never shows PASS: a recipe that passes every counted tier gets HELD, and a counted failure still shows as a failure.

| Test | Role and reason | Views (tier) | Summary | Re-entry rule | Selected recipes passing (recorded result) | Details |
| --- | --- | --- | --- | --- | ---: | --- |
| `gaussian1d_stability` | held: no one recipe passes it together with the counted tests | discriminator_stability (Tier 2) | BCAP and BISPF neither keep a learned bell curve (2/72 and 5/72 checks) nor re-learn it after the target moves (0/24); K3P and KA2 keep 71/72 but miss re-learning. Its one candidate that also passes mode_hold fails ring16_acquisition (draft PR #427). | Count again when one global recipe, in one same-source, same-runtime cohort, passes this whole test (72/72, re-acquisition, 24/24) and every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold, plus the counted ring endurance test. | 0 of 8 | [explanation](test-dispositions-20261015/README.md#gaussian1d_stability) |
| `grid100` | held: no recipe has passed its current declaration | all 6 views (Tier 2) | 100 tiny blobs on a grid: BCAP puts 24% and BISPF 37% of samples on target; 97% is required (a perfect sampler scores 98.9%). The best 7,000-update result under this clean-sampling declaration is 54%, and none passes. | Count again when one global recipe has an ordinary PASS on this unchanged declaration (clean sampling, live weights, five final checks and the 100,000-sample holdout) and still passes every counted Tier 1 and Tier 2 test. A diagnostic PASS only triggers a review. | 0 of 8 | [explanation](test-dispositions-20261015/README.md#grid100) |
| `img_bars4` | held: no one recipe passes it together with the counted tests | all 6 views (Tier 2) | Four bar pictures: BCAP and BISPF draw clean bars but only 2 of the 4. R1/R2 and GAN v3 pass it but fail Tier 1 tests and img_stripes2; under the current declarations no recipe passes both bars4 and stripes2. | Count again when one global recipe has an ordinary PASS on this unchanged declaration and, with the same configuration, still passes img_stripes2 and every other counted Tier 1 and Tier 2 test. | 0 of 8; separate diagnostic runs only: GAN v3 release 0.7, R1/R2 | [explanation](test-dispositions-20261015/README.md#img_bars4) |
| `img_blobs4` | held: no one recipe passes it together with the counted tests | all 6 views (Tier 2) | Four corner-blob pictures: BCAP covers 2 corners (24/32 clean), BISPF 3 (28/32); the test needs all 4 corners and 29/32 clean. Every recipe passing its current declaration fails or cannot run Tier 1 tests. | Count again when one global recipe has an ordinary PASS on this unchanged declaration and, with the same configuration, still passes img_stripes2 and every other counted Tier 1 and Tier 2 test. | 0 of 8; separate diagnostic runs only: R1/R2 | [explanation](test-dispositions-20261015/README.md#img_blobs4) |
| `img_intensity2` | held: no one recipe passes it together with the counted tests | all 6 views (Tier 2) | Two brightness levels: BCAP draws both but 10 of 32 pictures are off in brightness; BISPF collapses to one level. The three recipes passing its current declaration fail Tier 1 tests that BCAP and BISPF pass. | Count again when one global recipe has an ordinary PASS on this unchanged declaration and, with the same configuration, still passes img_stripes2 and every other counted Tier 1 and Tier 2 test. | 0 of 8; separate diagnostic runs only: GAN v3 release 0.7, R1/R2 | [explanation](test-dispositions-20261015/README.md#img_intensity2) |
| `rotated100` | held: no recipe has passed its current declaration | all 6 views (Tier 2) | The same 100-blob test rotated by 25 degrees: BCAP 26% and BISPF 27% of samples on target against 97%. The best 7,000-update result under this clean-sampling declaration is 51%, and none passes. | Count again when one global recipe has an ordinary PASS on this unchanged declaration (clean sampling, live weights, five final checks and the 100,000-sample holdout) and still passes every counted Tier 1 and Tier 2 test. A diagnostic PASS only triggers a review. In practice it returns with grid100. | 0 of 8 | [explanation](test-dispositions-20261015/README.md#rotated100) |
| `staggered100` | held: no recipe has passed its current declaration | all 6 views (Tier 2) | The same 100-blob test with alternate rows shifted: BCAP 30% and BISPF 34% of samples on target against 97%. The best 7,000-update result under this clean-sampling declaration is 56%, and none passes. | Count again when one global recipe has an ordinary PASS on this unchanged declaration (clean sampling, live weights, five final checks and the 100,000-sample holdout) and still passes every counted Tier 1 and Tier 2 test. A diagnostic PASS only triggers a review. In practice it returns with grid100. | 0 of 8 | [explanation](test-dispositions-20261015/README.md#staggered100) |
| `vector_anisotropic` | held: no one recipe passes it together with the counted tests | all 6 views (Tier 2) | Three stretched, tilted blobs: BCAP and BISPF draw the shapes but the wrong amounts (mass TV 0.196 and 0.191; limit 0.15). Of the 16 recipes passing its current declaration, 12 fail Tier 1 ring16_acquisition, 1 timed out on it and 3 fail mode_hold. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. | 0 of 8; separate diagnostic runs only: K3P, KA2, R1/R2 | [explanation](test-dispositions-20261015/README.md#vector_anisotropic) |
| `vector_overlap` | held: no one recipe passes it together with the counted tests | all 6 views (Tier 2) | One broad cloud made of two overlapping blobs: BCAP and BISPF reach it early, then drift (centre error 0.39 and 0.28; limit 0.15). All 15 recipes passing its current declaration fail Tier 1 ring16_acquisition or mode_hold. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. | 0 of 8; separate diagnostic runs only: K3P | [explanation](test-dispositions-20261015/README.md#vector_overlap) |
| `vector_unequal_mass` | held: no one recipe passes it together with the counted tests | all 6 views (Tier 2) | The rare 2% blob loses its share or collapses (BCAP eigen ratio 0.009, BISPF 0.004; need 0.15). Perfect samples pass 200/200 runs. Its passers fail counted tests, except some round-5 research recipes that fail only held tests and never ran most of the counted suite. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. Round 5 shows it trades off against vector_unequal_width, so the two will likely return together. | 0 of 8 | [explanation](test-dispositions-20261015/README.md#vector_unequal_mass) |
| `vector_unequal_width` | held: no one recipe passes it together with the counted tests | all 6 views (Tier 2) | Equal shares, very different widths: BCAP and BISPF get the shares wrong (mass TV 0.29 and 0.25; limit 0.15). Perfect samples pass 200/200 runs. Cell-moment passers fail counted tests; round-5 balanced assignment fails only held tests but never ran most of the counted suite. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. Round 5 shows it trades off against vector_unequal_mass, so the two will likely return together. | 0 of 8 | [explanation](test-dispositions-20261015/README.md#vector_unequal_width) |
| `grid100_14k` | held: prerequisite grid100 is held | clockfree_continuous (Tier 3) | Continues a passing 7,000-update grid100 model to 14,000 updates. No recipe has a passing parent, so every start is refused; a BISPF rescue run from its failed parent also fails (precision 0.68). | Returns in the revision that restores grid100, once the global recipe has an ordinary grid100 PASS with its own saved state. That revision also restores clockfree_audit to the count as this test's gate. | 0 of 8 | [explanation](test-dispositions-20261015/README.md#grid100_14k) |
| `rotated100_14k` | held: prerequisite rotated100 is held | clockfree_continuous (Tier 3) | Continues a passing 7,000-update rotated100 model to 14,000 updates. No recipe has a passing parent; a BISPF rescue run from its failed parent also fails (precision 0.65). | Returns in the revision that restores rotated100, once the global recipe has an ordinary rotated100 PASS with its own saved state. That revision also restores clockfree_audit to the count as this test's gate. | 0 of 8 | [explanation](test-dispositions-20261015/README.md#rotated100_14k) |
| `staggered100_14k` | held: prerequisite staggered100 is held | clockfree_continuous (Tier 3) | Continues a passing 7,000-update staggered100 model to 14,000 updates. No recipe has a passing parent; a BISPF rescue run from its failed parent also fails (precision 0.70). | Returns in the revision that restores staggered100, once the global recipe has an ordinary staggered100 PASS with its own saved state. That revision also restores clockfree_audit to the count as its gate. | 0 of 8 | [explanation](test-dispositions-20261015/README.md#staggered100_14k) |
| `target_shift_recovery` | held: no recipe has passed its current declaration | adaptation (Tier 3), clockfree_continuous (Tier 3) | Learn an 8-blob ring, hold it, then recover within 400 updates after it shifts. BISPF recovers but 220 updates late; BCAP flickers before and after the shift. No recipe has passed its current declaration. | Count again in adaptation and clockfree_continuous when one global recipe passes all four parts in one cohort (5/5, 120/120, 81/81 after the deadline, frozen copy 0/81) and every counted Tier 1 and Tier 2 test plus the counted ring endurance test. | 0 of 8 | [explanation](test-dispositions-20261015/README.md#target_shift_recovery) |
| `clockfree_audit` | removed: duplicate of clockfree_audit_measurement_v1 (counted there) | clockfree_continuous (Tier 3) | Same clock probe and grader as clockfree_audit_measurement_v1, which this view counts at Tier 1. They run as separate jobs (this strict copy also checks learning-rate floors first) and pass for the same recipes. Kept as the 14k tests' gate. | — | 0 of 8; separate diagnostic runs only: BCAP | [explanation](test-dispositions-20261015/README.md#clockfree_audit) |
| `ring_hold` | removed: phase of ring_extension (counted there) | clockfree_continuous (Tier 3), discriminator_stability (Tier 3), formulation_comparison (Tier 3), host_profile_transfer (Tier 3) | Same uninterrupted ring run as ring_extension, whose counted verdict already replays this acquisition and 1,200-check hold. Kept as a visible phase and counted once, through ring_extension. | — | 1 of 8 | [explanation](test-dispositions-20261015/README.md#ring_hold) |

## Research recipes and separate diagnostics

Each row names its owning family and its own recipe or diagnostic scope. Counts are **passes / distinct required tests within that row**: 29 means 29 different tests. These scores are excluded from the selected-recipe totals above. Compare rows only when their test, source and sampling contracts match.

Research recipes use their recorded revision 9 assignments. Those assignments still count the 12 tests that discriminator_stability revision 11 holds or removes from the count (29 tests per row, against 17 counted in revision 11), so research rows are not on the same scale as the selected-recipe totals. Atlas selected-cloud has a separate seven-question Tier 1 policy scope. Same-source backfill results fill previously unmeasured research cells while retaining original verdicts and qualification. Historical seed-only studies are retained in the expandable section below and excluded from this backfill.

| Family | Recipe / diagnostic scope | Tier 1 tests | Tier 2 tests | Tier 3 tests | Total distinct tests |
| --- | --- | ---: | ---: | ---: | ---: |
| [BCAP](families/bcap-pure.md) | [Confidence LR; seed 0](families/bcap-pure.md#confidence-lr-correction-six-task-screen-and-own-retention) | [6/6](families/bcap-pure.md#confidence-lr-correction-six-task-screen-and-own-retention) | [9/21](families/bcap-pure.md#confidence-lr-correction-six-task-screen-and-own-retention) | [0/2 (1 blocked)](families/bcap-pure.md#confidence-lr-correction-six-task-screen-and-own-retention) | [15/29 (1 blocked)](families/bcap-pure.md#confidence-lr-correction-six-task-screen-and-own-retention) |
| [BCAP](families/bcap-pure.md) | [Finite critic cap; seed 0](families/bcap-pure.md#later-gaussian-corrections-one-failed-acquisition-one-failed-retention) | [4/6 (1 timeout)](families/bcap-pure.md#later-gaussian-corrections-one-failed-acquisition-one-failed-retention) | [7/21 (2 blocked)](families/bcap-pure.md#later-gaussian-corrections-one-failed-acquisition-one-failed-retention) | [0/2 (1 blocked)](families/bcap-pure.md#later-gaussian-corrections-one-failed-acquisition-one-failed-retention) | [11/29 (3 blocked, 1 timeout)](families/bcap-pure.md#later-gaussian-corrections-one-failed-acquisition-one-failed-retention) |
| [BCAP](families/bcap-pure.md) | [Critic orientation; seed 0](families/bcap-pure.md#later-gaussian-corrections-one-failed-acquisition-one-failed-retention) | [1/6 (4 incompatible)](families/bcap-pure.md#later-gaussian-corrections-one-failed-acquisition-one-failed-retention) | [1/21 (13 incompatible, 1 blocked)](families/bcap-pure.md#later-gaussian-corrections-one-failed-acquisition-one-failed-retention) | [0/2 (1 incompatible, 1 blocked)](families/bcap-pure.md#later-gaussian-corrections-one-failed-acquisition-one-failed-retention) | [2/29 (18 incompatible, 2 blocked)](families/bcap-pure.md#later-gaussian-corrections-one-failed-acquisition-one-failed-retention) |
| [BCAP](families/bcap-pure.md) | [Signed confidence; seed 0](families/bcap-pure.md#signed-and-positive-evidence-responses-retention-remains-unsolved) | [5/6 (1 timeout)](families/bcap-pure.md#signed-and-positive-evidence-responses-retention-remains-unsolved) | [8/21 (1 blocked)](families/bcap-pure.md#signed-and-positive-evidence-responses-retention-remains-unsolved) | [0/2 (1 blocked)](families/bcap-pure.md#signed-and-positive-evidence-responses-retention-remains-unsolved) | [13/29 (2 blocked, 1 timeout)](families/bcap-pure.md#signed-and-positive-evidence-responses-retention-remains-unsolved) |
| [BCAP](families/bcap-pure.md) | [Positive evidence; seed 0](families/bcap-pure.md#signed-and-positive-evidence-responses-retention-remains-unsolved) | [5/6 (1 timeout)](families/bcap-pure.md#signed-and-positive-evidence-responses-retention-remains-unsolved) | [8/21 (1 blocked)](families/bcap-pure.md#signed-and-positive-evidence-responses-retention-remains-unsolved) | [0/2 (1 blocked)](families/bcap-pure.md#signed-and-positive-evidence-responses-retention-remains-unsolved) | [13/29 (2 blocked, 1 timeout)](families/bcap-pure.md#signed-and-positive-evidence-responses-retention-remains-unsolved) |
| [BCAP](families/bcap-pure.md) | [G-only coherent mean (original seed0)](g-coherent-mean-seed0/README.md) | [1/6 (5 incompatible)](g-coherent-mean-seed0/README.md) | [0/21 (19 incompatible, 1 blocked)](g-coherent-mean-seed0/README.md) | [0/2 (1 incompatible, 1 blocked)](g-coherent-mean-seed0/README.md) | [1/29 (25 incompatible, 2 blocked)](g-coherent-mean-seed0/README.md) |
| [BCAP](families/bcap-pure.md) | [Fresh matched protected NS/transport control](g-coherent-mean-seed0/README.md) | [1/6 (5 incompatible)](g-coherent-mean-seed0/README.md) | [0/21 (19 incompatible, 1 blocked)](g-coherent-mean-seed0/README.md) | [0/2 (1 incompatible, 1 blocked)](g-coherent-mean-seed0/README.md) | [1/29 (25 incompatible, 2 blocked)](g-coherent-mean-seed0/README.md) |
| [BCAP](families/bcap-pure.md) | [kinetic_transport_local_v2; seed 0; b95b46ed](families/bcap-pure.md#pr399-retained-research) | [2/6 (1 timeout)](families/bcap-pure.md#pr399-retained-research) | [11/21 (1 blocked)](families/bcap-pure.md#pr399-retained-research) | [0/2 (1 blocked)](families/bcap-pure.md#pr399-retained-research) | [13/29 (2 blocked, 1 timeout)](families/bcap-pure.md#pr399-retained-research) |
| [BCAP](families/bcap-pure.md) | [kinetic-local-nsgda-cross-v1; seed 0; b95b46ed](families/bcap-pure.md#pr399-retained-research) | [1/6](families/bcap-pure.md#pr399-retained-research) | [7/21 (1 blocked)](families/bcap-pure.md#pr399-retained-research) | [0/2 (1 blocked)](families/bcap-pure.md#pr399-retained-research) | [8/29 (2 blocked)](families/bcap-pure.md#pr399-retained-research) |
| [BCAP](families/bcap-pure.md) | [cell-quota-dualnorm-cross-v1; seed 0; b95b46ed](families/bcap-pure.md#pr399-retained-research) | [1/6 (4 incompatible)](families/bcap-pure.md#pr399-retained-research) | [4/21 (5 incompatible, 1 blocked, 4 errors, 3 timeouts)](families/bcap-pure.md#pr399-retained-research) | [0/2 (1 blocked)](families/bcap-pure.md#pr399-retained-research) | [5/29 (9 incompatible, 2 blocked, 4 errors, 3 timeouts)](families/bcap-pure.md#pr399-retained-research) |
| [BCAP](families/bcap-pure.md) | [kinetic-local-dualnorm-g-history-v1; seed 0; 9bc9e6ae](families/bcap-pure.md#pr399-retained-research) | [2/6 (1 timeout)](families/bcap-pure.md#pr399-retained-research) | [9/21 (1 blocked)](families/bcap-pure.md#pr399-retained-research) | [0/2 (1 blocked)](families/bcap-pure.md#pr399-retained-research) | [11/29 (2 blocked, 1 timeout)](families/bcap-pure.md#pr399-retained-research) |
| [BCAP](families/bcap-pure.md) | [kinetic-local-dualnorm-g-history-reference8-v1; seed 0; 9d618ca1](families/bcap-pure.md#pr399-retained-research) | [0/6 (4 incompatible)](families/bcap-pure.md#pr399-retained-research) | [6/21 (5 incompatible, 2 blocked, 3 timeouts)](families/bcap-pure.md#pr399-retained-research) | [0/2 (1 blocked)](families/bcap-pure.md#pr399-retained-research) | [6/29 (9 incompatible, 3 blocked, 3 timeouts)](families/bcap-pure.md#pr399-retained-research) |
| [BCAP](families/bcap-pure.md) | [kinetic-local-dualnorm-g-history-reference8-current-v1; seed 0; 9d618ca1](families/bcap-pure.md#pr399-retained-research) | [0/6 (4 incompatible)](families/bcap-pure.md#pr399-retained-research) | [5/21 (5 incompatible, 2 blocked)](families/bcap-pure.md#pr399-retained-research) | [0/2 (1 blocked)](families/bcap-pure.md#pr399-retained-research) | [5/29 (9 incompatible, 3 blocked)](families/bcap-pure.md#pr399-retained-research) |
| [BCAP](families/bcap-pure.md) | [kinetic-local-dualnorm-g-history-reference8-u-v1; seed 0; 79315f58](families/bcap-pure.md#pr399-retained-research) | [0/6 (4 incompatible, 1 timeout)](families/bcap-pure.md#pr399-retained-research) | [7/21 (5 incompatible, 2 blocked, 3 timeouts)](families/bcap-pure.md#pr399-retained-research) | [0/2 (1 blocked)](families/bcap-pure.md#pr399-retained-research) | [7/29 (9 incompatible, 3 blocked, 4 timeouts)](families/bcap-pure.md#pr399-retained-research) |
| [BCAP](families/bcap-pure.md) | [kinetic-local-dualnorm-g-history-u-v1; seed 0; 79315f58](families/bcap-pure.md#pr399-retained-research) | [1/6 (4 incompatible)](families/bcap-pure.md#pr399-retained-research) | [6/21 (5 incompatible, 1 blocked)](families/bcap-pure.md#pr399-retained-research) | [0/2 (1 blocked)](families/bcap-pure.md#pr399-retained-research) | [7/29 (9 incompatible, 2 blocked)](families/bcap-pure.md#pr399-retained-research) |
| [BCAP](families/bcap-pure.md) | [kinetic-local-dualnorm-g-history-bank8-u-v1; seed 0; 89b0e03f](families/bcap-pure.md#pr399-retained-research) | [1/6 (4 incompatible)](families/bcap-pure.md#pr399-retained-research) | [6/21 (5 incompatible, 1 blocked, 2 timeouts)](families/bcap-pure.md#pr399-retained-research) | [0/2 (1 blocked)](families/bcap-pure.md#pr399-retained-research) | [7/29 (9 incompatible, 2 blocked, 2 timeouts)](families/bcap-pure.md#pr399-retained-research) |
| [BCAP](families/bcap-pure.md) | [kinetic-local-dualnorm-g-history-bank8-v-v1; seed 0; 89b0e03f](families/bcap-pure.md#pr399-retained-research) | [1/6 (4 incompatible)](families/bcap-pure.md#pr399-retained-research) | [7/21 (5 incompatible, 1 blocked, 2 timeouts)](families/bcap-pure.md#pr399-retained-research) | [0/2 (1 blocked)](families/bcap-pure.md#pr399-retained-research) | [8/29 (9 incompatible, 2 blocked, 2 timeouts)](families/bcap-pure.md#pr399-retained-research) |
| [BCAP](families/bcap-pure.md) | [generator-idle-ema-bcap-direction-v1; seed 0; 27ab4cf9](families/bcap-pure.md#pr399-retained-research) | [4/6 (2 incomplete)](families/bcap-pure.md#pr399-retained-research) | [8/21 (2 blocked)](families/bcap-pure.md#pr399-retained-research) | [0/2 (1 blocked)](families/bcap-pure.md#pr399-retained-research) | [12/29 (3 blocked, 2 incomplete)](families/bcap-pure.md#pr399-retained-research) |
| [BCAP](families/bcap-pure.md) | [bcap-default-baseline-direction-v1; seed 0; 27ab4cf9](families/bcap-pure.md#pr399-retained-research) | [5/6 (1 incomplete)](families/bcap-pure.md#pr399-retained-research) | [8/21 (1 blocked, 1 error)](families/bcap-pure.md#pr399-retained-research) | [0/2 (1 blocked)](families/bcap-pure.md#pr399-retained-research) | [13/29 (2 blocked, 1 error, 1 incomplete)](families/bcap-pure.md#pr399-retained-research) |
| [BCAP](families/bcap-pure.md) | [Strict clock-free audit; separate Torch 2.13 cohort](tier-coverage-20261014/README.md) | — | — | [1/1](tier-coverage-20261014/README.md) | [1/1](tier-coverage-20261014/README.md) |
| [BCAP](families/bcap-pure.md) | [Unchanged selected recipe: deeper diagnostics (Torch 2.13)](actual-missing-diagnostics-20261014/README.md) | — | — | [1/7 (4 blocked)](actual-missing-diagnostics-20261014/README.md) | [1/7 (4 blocked)](actual-missing-diagnostics-20261014/README.md) |
| [BISPF](families/bispf.md) | [BISPF rebuilt checkpoints + matched-prior 14k diagnostics](ATTEMPT_COVERAGE.md#checkpoint-recovery) | — | [0/3](ATTEMPT_COVERAGE.md#checkpoint-recovery) | [0/3](ATTEMPT_COVERAGE.md#checkpoint-recovery) | [0/6](ATTEMPT_COVERAGE.md#checkpoint-recovery) |
| [BISPF](families/bispf.md) | [Verified extension from the existing uninterrupted run; source fafd66154108](grouped-evidence/4ee70c2cf634f45fbef5072e8689ed2a366f695603ad175fa4905dd19d9e04e8.json) | — | — | [1/1](grouped-evidence/4ee70c2cf634f45fbef5072e8689ed2a366f695603ad175fa4905dd19d9e04e8.json) | [1/1](grouped-evidence/4ee70c2cf634f45fbef5072e8689ed2a366f695603ad175fa4905dd19d9e04e8.json) |
| [Atlas](families/atlas.md) | [Selected-cloud policy; seed 0 (separate scope)](atlas-compatible-coverage-20261013/README.md) | [0/7 (4 incompatible)](atlas-compatible-coverage-20261013/README.md) | — | — | [0/7 (4 incompatible)](atlas-compatible-coverage-20261013/README.md) |
| [Atlas](families/atlas.md) | [Unchanged public policy: image diagnostics](actual-missing-diagnostics-20261014/README.md) | — | [1/4](actual-missing-diagnostics-20261014/README.md) | — | [1/4](actual-missing-diagnostics-20261014/README.md) |
| [Atlas](families/atlas.md) | [Selected-cloud execution after adapter investigation (pre-repair source)](ATTEMPT_COVERAGE.md#remaining-attempts) | [0/7 (3 incompatible, 2 errors, 1 incomplete)](ATTEMPT_COVERAGE.md#remaining-attempts) | — | — | [0/7 (3 incompatible, 2 errors, 1 incomplete)](ATTEMPT_COVERAGE.md#remaining-attempts) |
| [Atlas](families/atlas.md) | [ATLAS checkpoint-export repair (same recipe)](ATTEMPT_COVERAGE.md#checkpoint-recovery) | [0/2](ATTEMPT_COVERAGE.md#checkpoint-recovery) | — | — | [0/2](ATTEMPT_COVERAGE.md#checkpoint-recovery) |
| [Atlas](families/atlas.md) | [Policy-cloud scoped tests, clean (learned output noise removed); seed 0](atlas-e22-policy-cloud-20261015/README.md#atlas) | [1/7 (3 incompatible)](atlas-e22-policy-cloud-20261015/README.md#atlas) | [5/21 (6 incompatible)](atlas-e22-policy-cloud-20261015/README.md#atlas) | [0/7 (1 blocked)](atlas-e22-policy-cloud-20261015/README.md#atlas) | [6/35 (9 incompatible, 1 blocked)](atlas-e22-policy-cloud-20261015/README.md#atlas) |
| [Atlas](families/atlas.md) | [Policy-cloud scoped tests, as served (learned output noise kept); seed 0 · 12 law-independent verdicts shared with the clean row](atlas-e22-policy-cloud-20261015/README.md#atlas) | [1/7 (3 incompatible)](atlas-e22-policy-cloud-20261015/README.md#atlas) | [7/21 (6 incompatible)](atlas-e22-policy-cloud-20261015/README.md#atlas) | [3/7 (1 blocked)](atlas-e22-policy-cloud-20261015/README.md#atlas) | [11/35 (9 incompatible, 1 blocked)](atlas-e22-policy-cloud-20261015/README.md#atlas) |
| [E22](families/e22.md) | [Unchanged public policy: image diagnostics](actual-missing-diagnostics-20261014/README.md) | — | [1/4](actual-missing-diagnostics-20261014/README.md) | — | [1/4](actual-missing-diagnostics-20261014/README.md) |
| [E22](families/e22.md) | [Selected-cloud execution after adapter investigation (pre-repair source)](actual-missing-diagnostics-20261014/README.md) | [0/7 (3 incompatible)](actual-missing-diagnostics-20261014/README.md) | — | — | [0/7 (3 incompatible)](actual-missing-diagnostics-20261014/README.md) |
| [E22](families/e22.md) | [E22 checkpoint-export repair (same recipe)](ATTEMPT_COVERAGE.md#checkpoint-recovery) | [0/2](ATTEMPT_COVERAGE.md#checkpoint-recovery) | — | — | [0/2](ATTEMPT_COVERAGE.md#checkpoint-recovery) |
| [E22](families/e22.md) | [Policy-cloud scoped tests, clean (learned output noise removed); seed 0](atlas-e22-policy-cloud-20261015/README.md#e22) | [1/7 (3 incompatible)](atlas-e22-policy-cloud-20261015/README.md#e22) | [5/21 (6 incompatible)](atlas-e22-policy-cloud-20261015/README.md#e22) | [0/7 (1 blocked)](atlas-e22-policy-cloud-20261015/README.md#e22) | [6/35 (9 incompatible, 1 blocked)](atlas-e22-policy-cloud-20261015/README.md#e22) |
| [E22](families/e22.md) | [Policy-cloud scoped tests, as served (learned output noise kept); seed 0 · 12 law-independent verdicts shared with the clean row](atlas-e22-policy-cloud-20261015/README.md#e22) | [1/7 (3 incompatible)](atlas-e22-policy-cloud-20261015/README.md#e22) | [5/21 (6 incompatible)](atlas-e22-policy-cloud-20261015/README.md#e22) | [1/7 (1 blocked)](atlas-e22-policy-cloud-20261015/README.md#e22) | [7/35 (9 incompatible, 1 blocked)](atlas-e22-policy-cloud-20261015/README.md#e22) |
| [GAN v3 release 0.7](families/release07-gan-v3.md) | [Selected recipe Tier 2 diagnostics; seed 0](tier-coverage-20261014/README.md) | — | [9/21](tier-coverage-20261014/README.md) | — | [9/21](tier-coverage-20261014/README.md) |
| [GAN v3 release 0.7](families/release07-gan-v3.md) | [Unchanged selected recipe: deeper diagnostics (Torch 2.13)](actual-missing-diagnostics-20261014/README.md) | — | — | [0/7 (4 blocked)](actual-missing-diagnostics-20261014/README.md) | [0/7 (4 blocked)](actual-missing-diagnostics-20261014/README.md) |
| [K3P](families/k3p.md) | [Selected recipe Tier 2 diagnostics; seed 0](tier-coverage-20261014/README.md) | — | [11/21](tier-coverage-20261014/README.md) | — | [11/21](tier-coverage-20261014/README.md) |
| [K3P](families/k3p.md) | [Unchanged selected recipe: deeper diagnostics (Torch 2.13)](actual-missing-diagnostics-20261014/README.md) | — | — | [0/7 (4 blocked)](actual-missing-diagnostics-20261014/README.md) | [0/7 (4 blocked)](actual-missing-diagnostics-20261014/README.md) |
| [KA2](families/ka2.md) | [Selected recipe Tier 2 diagnostics; seed 0](tier-coverage-20261014/README.md) | — | [10/21](tier-coverage-20261014/README.md) | — | [10/21](tier-coverage-20261014/README.md) |
| [KA2](families/ka2.md) | [Unchanged selected recipe: deeper diagnostics (Torch 2.13)](actual-missing-diagnostics-20261014/README.md) | — | — | [0/7 (4 blocked)](actual-missing-diagnostics-20261014/README.md) | [0/7 (4 blocked)](actual-missing-diagnostics-20261014/README.md) |
| [R1/R2](families/r1r2.md) | [Selected recipe Tier 2 diagnostics; seed 0](tier-coverage-20261014/README.md) | — | [9/21](tier-coverage-20261014/README.md) | — | [9/21](tier-coverage-20261014/README.md) |
| [R1/R2](families/r1r2.md) | [Unchanged selected recipe: deeper diagnostics (Torch 2.13)](actual-missing-diagnostics-20261014/README.md) | — | — | [0/7 (4 blocked)](actual-missing-diagnostics-20261014/README.md) | [0/7 (4 blocked)](actual-missing-diagnostics-20261014/README.md) |

<details>
<summary>Historical seed-only studies (excluded from backfill)</summary>

These 12 archived seed 1–6 rows retain **324 unrun recipe/test pairs**. They are excluded by the repository instruction against seed-only experiments. Their original counts and evidence remain visible here; they are outside the completed backfill scope.

| Family | Historical seed-only recipe | Tier 1 tests | Tier 2 tests | Tier 3 tests | Total distinct tests |
| --- | --- | ---: | ---: | ---: | ---: |
| [BCAP](families/bcap-pure.md) | [G-coherence candidate; research seed 1](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/6 (5 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/21 (20 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/2 (2 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/29 (27 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) |
| [BCAP](families/bcap-pure.md) | [G-coherence control; research seed 1](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/6 (5 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/21 (20 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/2 (2 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/29 (27 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) |
| [BCAP](families/bcap-pure.md) | [G-coherence candidate; research seed 2](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/6 (5 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/21 (20 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/2 (2 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/29 (27 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) |
| [BCAP](families/bcap-pure.md) | [G-coherence control; research seed 2](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/6 (5 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/21 (20 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/2 (2 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/29 (27 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) |
| [BCAP](families/bcap-pure.md) | [G-coherence candidate; research seed 3](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/6 (5 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/21 (20 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/2 (2 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/29 (27 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) |
| [BCAP](families/bcap-pure.md) | [G-coherence control; research seed 3](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/6 (5 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/21 (20 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/2 (2 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/29 (27 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) |
| [BCAP](families/bcap-pure.md) | [G-coherence candidate; research seed 4](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/6 (5 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/21 (20 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/2 (2 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/29 (27 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) |
| [BCAP](families/bcap-pure.md) | [G-coherence control; research seed 4](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/6 (5 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/21 (20 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/2 (2 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/29 (27 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) |
| [BCAP](families/bcap-pure.md) | [G-coherence candidate; research seed 5](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/6 (5 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/21 (20 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/2 (2 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/29 (27 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) |
| [BCAP](families/bcap-pure.md) | [G-coherence control; research seed 5](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/6 (5 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/21 (20 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/2 (2 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/29 (27 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) |
| [BCAP](families/bcap-pure.md) | [G-coherence candidate; research seed 6](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/6 (5 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/21 (20 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/2 (2 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/29 (27 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) |
| [BCAP](families/bcap-pure.md) | [G-coherence control; research seed 6](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/6 (5 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/21 (20 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/2 (2 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/29 (27 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) |

</details>

<details>
<summary>Selected recipe breakdown by view (shared tests repeat across views)</summary>

Each view uses the selected recipe above. View scores are the requirements summed in its selected-recipe total. Held cells count passes / held tests of that view; they are in no total.

| Family / selected recipe | View | Tier 1 | Tier 2 | Tier 3 | Total view requirements | Held (measured, not counted) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| [BCAP](families/bcap-pure.md) | ↳ [adaptation](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-adaptation) | [3/3](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) | [8/9](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) | [0/0](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) | [11/12](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-adaptation) | [0/11 (1 diagnostic attempt)](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-held) |
| [BCAP](families/bcap-pure.md) | ↳ [clockfree_continuous](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [4/4](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | [8/9](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) | [0/1 (1 blocked)](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | [12/14 (1 blocked)](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [0/14 (3 blocked, 1 diagnostic attempt)](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-held) |
| [BCAP](families/bcap-pure.md) | ↳ [discriminator_stability](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [6/6](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | [9/10](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | [0/1 (1 blocked)](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | [15/17 (1 blocked)](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [0/11](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-held) |
| [BCAP](families/bcap-pure.md) | ↳ [formulation_comparison](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [3/3](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) | [8/9](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) | [0/1 (1 blocked)](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) | [11/13 (1 blocked)](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [0/10](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-held) |
| [BCAP](families/bcap-pure.md) | ↳ [host_profile_transfer](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [3/3](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) | [8/9](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) | [0/1 (1 blocked)](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | [11/13 (1 blocked)](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [0/10](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-held) |
| [BCAP](families/bcap-pure.md) | ↳ [quality_coverage](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [3/3](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | [8/9](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | [0/0](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-3) | [11/12](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [0/10](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-held) |
| [BISPF](families/bispf.md) | ↳ [adaptation](families/bispf.md#cohort-cuda-3faddec347a0-adaptation) | [3/3](families/bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-1) | [9/9](families/bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2) | [0/0](families/bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-3) | [12/12](families/bispf.md#cohort-cuda-3faddec347a0-adaptation) | [0/11 (1 not run)](families/bispf.md#cohort-cuda-3faddec347a0-held) |
| [BISPF](families/bispf.md) | ↳ [clockfree_continuous](families/bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous) | [4/4](families/bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-1) | [9/9](families/bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2) | [1/1](families/bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-3) | [14/14](families/bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous) | [0/14 (4 not run)](families/bispf.md#cohort-cuda-3faddec347a0-held) |
| [BISPF](families/bispf.md) | ↳ [discriminator_stability](families/bispf.md#cohort-cuda-3faddec347a0-discriminator_stability) | [6/6](families/bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-1) | [10/10](families/bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2) | [1/1](families/bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-3) | [17/17](families/bispf.md#cohort-cuda-3faddec347a0-discriminator_stability) | [0/11](families/bispf.md#cohort-cuda-3faddec347a0-held) |
| [BISPF](families/bispf.md) | ↳ [formulation_comparison](families/bispf.md#cohort-cuda-3faddec347a0-formulation_comparison) | [3/3](families/bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-1) | [9/9](families/bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) | [1/1](families/bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-3) | [13/13](families/bispf.md#cohort-cuda-3faddec347a0-formulation_comparison) | [0/10](families/bispf.md#cohort-cuda-3faddec347a0-held) |
| [BISPF](families/bispf.md) | ↳ [host_profile_transfer](families/bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer) | [3/3](families/bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-1) | [9/9](families/bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2) | [1/1](families/bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-3) | [13/13](families/bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer) | [0/10](families/bispf.md#cohort-cuda-3faddec347a0-held) |
| [BISPF](families/bispf.md) | ↳ [quality_coverage](families/bispf.md#cohort-cuda-3faddec347a0-quality_coverage) | [3/3](families/bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-1) | [9/9](families/bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) | [0/0](families/bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-3) | [12/12](families/bispf.md#cohort-cuda-3faddec347a0-quality_coverage) | [0/10](families/bispf.md#cohort-cuda-3faddec347a0-held) |
| [Atlas](families/atlas.md) | ↳ [adaptation](families/atlas.md#cohort-cuda-1bf9d7d34422-adaptation) | [0/3 (3 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) | [0/9 (9 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) | [0/0](families/atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) | [0/12 (12 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-adaptation) | [0/11 (11 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-held) |
| [Atlas](families/atlas.md) | ↳ [clockfree_continuous](families/atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [0/4 (4 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | [0/9 (9 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) | [0/1 (1 blocked)](families/atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | [0/14 (13 incompatible, 1 blocked)](families/atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [0/14 (11 incompatible, 3 blocked)](families/atlas.md#cohort-cuda-1bf9d7d34422-held) |
| [Atlas](families/atlas.md) | ↳ [discriminator_stability](families/atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [0/6 (6 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | [0/10 (9 incompatible, 1 blocked)](families/atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | [0/1 (1 blocked)](families/atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | [0/17 (15 incompatible, 2 blocked)](families/atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [0/11 (10 incompatible, 1 blocked)](families/atlas.md#cohort-cuda-1bf9d7d34422-held) |
| [Atlas](families/atlas.md) | ↳ [formulation_comparison](families/atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [0/3 (3 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) | [0/9 (9 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) | [0/1 (1 blocked)](families/atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) | [0/13 (12 incompatible, 1 blocked)](families/atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [0/10 (10 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-held) |
| [Atlas](families/atlas.md) | ↳ [host_profile_transfer](families/atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [0/3 (3 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) | [0/9 (9 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) | [0/1 (1 blocked)](families/atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | [0/13 (12 incompatible, 1 blocked)](families/atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [0/10 (10 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-held) |
| [Atlas](families/atlas.md) | ↳ [quality_coverage](families/atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [0/3 (3 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | [0/9 (9 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | [0/0](families/atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-3) | [0/12 (12 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [0/10 (10 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-held) |
| [E22](families/e22.md) | ↳ [adaptation](families/e22.md#cohort-cuda-1bf9d7d34422-adaptation) | [0/3 (3 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) | [0/9 (9 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) | [0/0](families/e22.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) | [0/12 (12 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-adaptation) | [0/11 (11 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-held) |
| [E22](families/e22.md) | ↳ [clockfree_continuous](families/e22.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [0/4 (4 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | [0/9 (9 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) | [0/1 (1 blocked)](families/e22.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | [0/14 (13 incompatible, 1 blocked)](families/e22.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [0/14 (11 incompatible, 3 blocked)](families/e22.md#cohort-cuda-1bf9d7d34422-held) |
| [E22](families/e22.md) | ↳ [discriminator_stability](families/e22.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [0/6 (6 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | [0/10 (9 incompatible, 1 blocked)](families/e22.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | [0/1 (1 blocked)](families/e22.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | [0/17 (15 incompatible, 2 blocked)](families/e22.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [0/11 (10 incompatible, 1 blocked)](families/e22.md#cohort-cuda-1bf9d7d34422-held) |
| [E22](families/e22.md) | ↳ [formulation_comparison](families/e22.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [0/3 (3 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) | [0/9 (9 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) | [0/1 (1 blocked)](families/e22.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) | [0/13 (12 incompatible, 1 blocked)](families/e22.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [0/10 (10 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-held) |
| [E22](families/e22.md) | ↳ [host_profile_transfer](families/e22.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [0/3 (3 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) | [0/9 (9 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) | [0/1 (1 blocked)](families/e22.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | [0/13 (12 incompatible, 1 blocked)](families/e22.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [0/10 (10 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-held) |
| [E22](families/e22.md) | ↳ [quality_coverage](families/e22.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [0/3 (3 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | [0/9 (9 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | [0/0](families/e22.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-3) | [0/12 (12 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [0/10 (10 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-held) |
| [GAN v3 release 0.7](families/release07-gan-v3.md) | ↳ [adaptation](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-adaptation) | [3/3](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) | [0/9 (9 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) | [0/0](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) | [3/12 (9 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-adaptation) | [0/11 (11 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-held) |
| [GAN v3 release 0.7](families/release07-gan-v3.md) | ↳ [clockfree_continuous](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [3/4](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | [0/9 (9 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) | [0/1 (1 blocked)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | [3/14 (1 blocked, 9 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [0/14 (3 blocked, 11 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-held) |
| [GAN v3 release 0.7](families/release07-gan-v3.md) | ↳ [discriminator_stability](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [5/6](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | [0/10 (10 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | [0/1 (1 blocked)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | [5/17 (1 blocked, 10 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [0/11 (11 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-held) |
| [GAN v3 release 0.7](families/release07-gan-v3.md) | ↳ [formulation_comparison](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [3/3](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) | [0/9 (9 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) | [0/1 (1 blocked)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) | [3/13 (1 blocked, 9 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [0/10 (10 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-held) |
| [GAN v3 release 0.7](families/release07-gan-v3.md) | ↳ [host_profile_transfer](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [3/3](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) | [0/9 (9 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) | [0/1 (1 blocked)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | [3/13 (1 blocked, 9 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [0/10 (10 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-held) |
| [GAN v3 release 0.7](families/release07-gan-v3.md) | ↳ [quality_coverage](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [3/3](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | [0/9 (9 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | [0/0](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-3) | [3/12 (9 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [0/10 (10 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-held) |
| [K3P](families/k3p.md) | ↳ [adaptation](families/k3p.md#cohort-cuda-1bf9d7d34422-adaptation) | [3/3](families/k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) | [0/9 (9 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) | [0/0](families/k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) | [3/12 (9 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-adaptation) | [0/11 (11 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-held) |
| [K3P](families/k3p.md) | ↳ [clockfree_continuous](families/k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [3/4](families/k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | [0/9 (9 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) | [0/1 (1 blocked)](families/k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | [3/14 (1 blocked, 9 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [0/14 (3 blocked, 11 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-held) |
| [K3P](families/k3p.md) | ↳ [discriminator_stability](families/k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [5/6](families/k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | [0/10 (10 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | [0/1 (1 blocked)](families/k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | [5/17 (1 blocked, 10 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [0/11 (11 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-held) |
| [K3P](families/k3p.md) | ↳ [formulation_comparison](families/k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [3/3](families/k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) | [0/9 (9 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) | [0/1 (1 blocked)](families/k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) | [3/13 (1 blocked, 9 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [0/10 (10 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-held) |
| [K3P](families/k3p.md) | ↳ [host_profile_transfer](families/k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [3/3](families/k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) | [0/9 (9 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) | [0/1 (1 blocked)](families/k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | [3/13 (1 blocked, 9 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [0/10 (10 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-held) |
| [K3P](families/k3p.md) | ↳ [quality_coverage](families/k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [3/3](families/k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | [0/9 (9 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | [0/0](families/k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-3) | [3/12 (9 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [0/10 (10 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-held) |
| [KA2](families/ka2.md) | ↳ [adaptation](families/ka2.md#cohort-cuda-1bf9d7d34422-adaptation) | [3/3](families/ka2.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) | [0/9 (9 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) | [0/0](families/ka2.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) | [3/12 (9 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-adaptation) | [0/11 (11 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-held) |
| [KA2](families/ka2.md) | ↳ [clockfree_continuous](families/ka2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [3/4](families/ka2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | [0/9 (9 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) | [0/1 (1 blocked)](families/ka2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | [3/14 (1 blocked, 9 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [0/14 (3 blocked, 11 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-held) |
| [KA2](families/ka2.md) | ↳ [discriminator_stability](families/ka2.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [5/6](families/ka2.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | [0/10 (10 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | [0/1 (1 blocked)](families/ka2.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | [5/17 (1 blocked, 10 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [0/11 (11 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-held) |
| [KA2](families/ka2.md) | ↳ [formulation_comparison](families/ka2.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [3/3](families/ka2.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) | [0/9 (9 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) | [0/1 (1 blocked)](families/ka2.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) | [3/13 (1 blocked, 9 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [0/10 (10 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-held) |
| [KA2](families/ka2.md) | ↳ [host_profile_transfer](families/ka2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [3/3](families/ka2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) | [0/9 (9 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) | [0/1 (1 blocked)](families/ka2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | [3/13 (1 blocked, 9 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [0/10 (10 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-held) |
| [KA2](families/ka2.md) | ↳ [quality_coverage](families/ka2.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [3/3](families/ka2.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | [0/9 (9 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | [0/0](families/ka2.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-3) | [3/12 (9 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [0/10 (10 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-held) |
| [R1/R2](families/r1r2.md) | ↳ [adaptation](families/r1r2.md#cohort-cuda-1bf9d7d34422-adaptation) | [3/3](families/r1r2.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) | [0/9 (9 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) | [0/0](families/r1r2.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) | [3/12 (9 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-adaptation) | [0/11 (11 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-held) |
| [R1/R2](families/r1r2.md) | ↳ [clockfree_continuous](families/r1r2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [3/4](families/r1r2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | [0/9 (9 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) | [0/1 (1 blocked)](families/r1r2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | [3/14 (1 blocked, 9 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [0/14 (3 blocked, 11 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-held) |
| [R1/R2](families/r1r2.md) | ↳ [discriminator_stability](families/r1r2.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [5/6](families/r1r2.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | [0/10 (10 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | [0/1 (1 blocked)](families/r1r2.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | [5/17 (1 blocked, 10 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [0/11 (11 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-held) |
| [R1/R2](families/r1r2.md) | ↳ [formulation_comparison](families/r1r2.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [3/3](families/r1r2.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) | [0/9 (9 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) | [0/1 (1 blocked)](families/r1r2.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) | [3/13 (1 blocked, 9 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [0/10 (10 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-held) |
| [R1/R2](families/r1r2.md) | ↳ [host_profile_transfer](families/r1r2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [3/3](families/r1r2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) | [0/9 (9 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) | [0/1 (1 blocked)](families/r1r2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | [3/13 (1 blocked, 9 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [0/10 (10 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-held) |
| [R1/R2](families/r1r2.md) | ↳ [quality_coverage](families/r1r2.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [3/3](families/r1r2.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | [0/9 (9 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | [0/0](families/r1r2.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-3) | [3/12 (9 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [0/10 (10 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-held) |

</details>

## Reading the results

Counts show passes / required tasks; selected-row pass counts remain unchanged. For rows with run-attempt coverage, the suffix counts genuinely unattempted tasks as not run, constructor refusals as incompatible, missing required artifacts as blocked, and other separate-runtime attempts as diagnostic attempts. A start refusal is not a numerical failure. Rows without that registry retain their recorded missing/preflight counts. No formulation is changed to fit a test. Recorded PASS and FAIL are both executed results. Attempted ERROR, INVALID or INCOMPLETE results show their cause and execution evidence on the family page and earn no pass credit. Zero recorded passes always displays as 0, including unrun families.

Counts retain each original recipe, prior, initialization, budget, serving law and source. Family pages retain exact metrics, phase checks, failures, blockers, historical comparisons and calibration requirements. Observation checks and catalog attempt counts add no benchmark passes; untiered historical studies remain on those detail pages.

[Actual attempts of unrun and blocked tests](actual-missing-diagnostics-20261014/README.md). The new diagnostic rows show executions beyond failed tier gates and Atlas/E22 public-policy tests. Atlas's original 19/19 CUDA passes remain valid under their original task and sampling contracts.

[Diagnostic completion, new strict audit and every skip reason](tier-coverage-20261014/README.md). The selected-recipe Tier 2 diagnostic rows retain their original 21-task scope. The new strict BCAP audit has a separate runtime and changes no qualification.

[Prior-search measurements and gates](bcap-prior-search/README.md) · [Prior-search findings](bcap-prior-search/findings.md)

## Browse by technique tag

Tags describe properties shared by a family's supported variants, not performance or qualification. Configuration-dependent behavior is explained on each family page. Historical pages retain their own scope.

<a name="tag-adaptive-training-policy"></a>

### `adaptive-training-policy`

Uses an observation-driven training policy rather than a fixed update schedule.

- [Atlas](families/atlas.md)
- [E22](families/e22.md)

<a name="tag-adversarial-training"></a>

### `adversarial-training`

Trains a generator against a critic; task-owned auxiliary objectives may also apply.

- [Atlas](families/atlas.md)
- [BCAP](families/bcap-pure.md)
- [BCAP](families/bcap-pure-configuration.md)
- [BCAP ada_nsgda](families/bcap-ada-nsgda.md) (historical cohort)
- [BCAP dualnorm (experimental starting point)](families/bcap-dualnorm.md)
- [BCAP dualnorm_D_only](families/bcap-dualnorm-d-only.md) (historical cohort)
- [BCAP nsgda_global](families/bcap-nsgda-global.md) (historical cohort)
- [BCAP nsgda_layer](families/bcap-nsgda-layer.md) (historical cohort)
- [BCAP particle_rownorm_only](families/bcap-particle-rownorm-only.md) (historical cohort)
- [BCAP sgda](families/bcap-sgda.md) (historical cohort)
- [BCAP with K3P](families/bcap.md) (historical cohort)
- [BISPF](families/bispf.md)
- [E22](families/e22.md)
- [GAN v3 release 0.7](families/release07-gan-v3.md)
- [GAN v3 release 0.7 (MoG)](families/release07-gan-v3-mog.md) (historical cohort)
- [GAN v3 release 0.7 (cloud)](families/release07-gan-v3-cloud.md) (historical cohort)
- [Halloween optimizer/loss transfer](families/halloween.md) (historical cohort)
- [K3P](families/k3p.md)
- [K3P without A2](families/k3p-no-a2.md) (historical cohort)
- [K3P without critic anchor](families/k3p-no-anchor.md) (historical cohort)
- [K3P without critic penalty](families/k3p-no-penalty.md) (historical cohort)
- [K3P without training output noise](families/k3p-no-training-noise.md) (historical cohort)
- [KA2](families/ka2.md)
- [R1/R2](families/r1r2.md)

<a name="tag-capped-input-gradients"></a>

### `capped-input-gradients`

Penalizes critic input-gradient norms above a threshold on real and generated inputs; this is a soft loss penalty.

- [BCAP](families/bcap-pure.md)
- [BCAP](families/bcap-pure-configuration.md)
- [BCAP ada_nsgda](families/bcap-ada-nsgda.md) (historical cohort)
- [BCAP dualnorm (experimental starting point)](families/bcap-dualnorm.md)
- [BCAP dualnorm_D_only](families/bcap-dualnorm-d-only.md) (historical cohort)
- [BCAP nsgda_global](families/bcap-nsgda-global.md) (historical cohort)
- [BCAP nsgda_layer](families/bcap-nsgda-layer.md) (historical cohort)
- [BCAP particle_rownorm_only](families/bcap-particle-rownorm-only.md) (historical cohort)
- [BCAP sgda](families/bcap-sgda.md) (historical cohort)
- [BCAP with K3P](families/bcap.md) (historical cohort)
- [BISPF](families/bispf.md)
- [GAN v3 release 0.7](families/release07-gan-v3.md)
- [GAN v3 release 0.7 (MoG)](families/release07-gan-v3-mog.md) (historical cohort)
- [GAN v3 release 0.7 (cloud)](families/release07-gan-v3-cloud.md) (historical cohort)

<a name="tag-constant-learning-rate"></a>

### `constant-learning-rate`

Declared baseline uses constant learning rates; resolved per-role rates remain configuration-specific.

- [BISPF](families/bispf.md)
- [Halloween optimizer/loss transfer](families/halloween.md) (historical cohort)

<a name="tag-critic-gradient-penalty"></a>

### `critic-gradient-penalty`

Uses a penalty on critic input gradients; its formula and strength are configuration-specific.

- [Atlas](families/atlas.md)
- [BCAP](families/bcap-pure.md)
- [BCAP](families/bcap-pure-configuration.md)
- [BCAP ada_nsgda](families/bcap-ada-nsgda.md) (historical cohort)
- [BCAP dualnorm (experimental starting point)](families/bcap-dualnorm.md)
- [BCAP dualnorm_D_only](families/bcap-dualnorm-d-only.md) (historical cohort)
- [BCAP nsgda_global](families/bcap-nsgda-global.md) (historical cohort)
- [BCAP nsgda_layer](families/bcap-nsgda-layer.md) (historical cohort)
- [BCAP particle_rownorm_only](families/bcap-particle-rownorm-only.md) (historical cohort)
- [BCAP sgda](families/bcap-sgda.md) (historical cohort)
- [BCAP with K3P](families/bcap.md) (historical cohort)
- [BISPF](families/bispf.md)
- [E22](families/e22.md)
- [GAN v3 release 0.7](families/release07-gan-v3.md)
- [GAN v3 release 0.7 (MoG)](families/release07-gan-v3-mog.md) (historical cohort)
- [GAN v3 release 0.7 (cloud)](families/release07-gan-v3-cloud.md) (historical cohort)
- [K3P](families/k3p.md)
- [K3P without A2](families/k3p-no-a2.md) (historical cohort)
- [K3P without critic anchor](families/k3p-no-anchor.md) (historical cohort)
- [K3P without training output noise](families/k3p-no-training-noise.md) (historical cohort)
- [KA2](families/ka2.md)
- [R1/R2](families/r1r2.md)

<a name="tag-historical-cohort"></a>

### `historical-cohort`

A retained historical family with its original source, prior and scoring contracts.

- [GAN v3 release 0.7 (MoG)](families/release07-gan-v3-mog.md) (historical cohort)
- [GAN v3 release 0.7 (cloud)](families/release07-gan-v3-cloud.md) (historical cohort)

<a name="tag-learning-rate-annealing"></a>

### `learning-rate-annealing`

Declared baseline decreases learning rates during training; horizons and floors are configuration-specific.

- [GAN v3 release 0.7](families/release07-gan-v3.md)
- [GAN v3 release 0.7 (MoG)](families/release07-gan-v3-mog.md) (historical cohort)
- [GAN v3 release 0.7 (cloud)](families/release07-gan-v3-cloud.md) (historical cohort)

<a name="tag-optimizer-interventions"></a>

### `optimizer-interventions`

Includes formulation-specific interventions around optimizer steps; availability depends on the host and resolved recipe.

- [Atlas](families/atlas.md)
- [BCAP ada_nsgda](families/bcap-ada-nsgda.md) (historical cohort)
- [BCAP dualnorm (experimental starting point)](families/bcap-dualnorm.md)
- [BCAP dualnorm_D_only](families/bcap-dualnorm-d-only.md) (historical cohort)
- [BCAP nsgda_global](families/bcap-nsgda-global.md) (historical cohort)
- [BCAP nsgda_layer](families/bcap-nsgda-layer.md) (historical cohort)
- [BCAP particle_rownorm_only](families/bcap-particle-rownorm-only.md) (historical cohort)
- [BCAP sgda](families/bcap-sgda.md) (historical cohort)
- [BCAP with K3P](families/bcap.md) (historical cohort)
- [BISPF](families/bispf.md)
- [E22](families/e22.md)
- [K3P](families/k3p.md)
- [K3P without A2](families/k3p-no-a2.md) (historical cohort)
- [K3P without critic anchor](families/k3p-no-anchor.md) (historical cohort)
- [K3P without critic penalty](families/k3p-no-penalty.md) (historical cohort)
- [K3P without training output noise](families/k3p-no-training-noise.md) (historical cohort)
- [KA2](families/ka2.md)

<a name="tag-selectable-loss"></a>

### `selectable-loss`

The family explicitly supports multiple adversarial loss formulations.

- [BCAP](families/bcap-pure.md)
- [BCAP](families/bcap-pure-configuration.md)
- [BISPF](families/bispf.md)

<a name="tag-structural-ablation"></a>

### `structural-ablation`

Removes a named mechanism from a parent technique and retains a separate evidence identity.

- [K3P without A2](families/k3p-no-a2.md) (historical cohort)
- [K3P without critic anchor](families/k3p-no-anchor.md) (historical cohort)
- [K3P without critic penalty](families/k3p-no-penalty.md) (historical cohort)
- [K3P without training output noise](families/k3p-no-training-noise.md) (historical cohort)

<a name="tag-tensorflow-v1-adam"></a>

### `tensorflow-v1-adam`

Uses the declared dense TensorFlow-v1 Adam update law with checkpointed application clocks and epsilon before second-moment bias correction.

- [Halloween optimizer/loss transfer](families/halloween.md) (historical cohort)

## Refresh

```sh
python reports/forge/regenerate_technique_inventory.py
```

This regenerates the leaderboard, family pages and experiments-by-tier report from committed evidence and declarations. Register newly measured evidence with `--source-commit <executed-commit>`; advancing the recorded view policy also requires `--advance-policy`.

After any view revision (for example a held or removed test), always add `--refresh-publication`. It rebuilds the counts and pages from the declared views and keeps every recorded verdict and the recorded qualification. Without the flag, the command switches to that mode only when discriminator_stability itself changed; any other view change would take the ordinary publishing path, which re-reads earlier sources from Git.

[Experiments, criteria and tier assignments](EXPERIMENTS_BY_TIER.md) · [Complete numerical publication and provenance](technique-inventory.json)

Publication input digest `3efa61d00aac87a1be36d803816b514854022220a4c0dd6ea47859f9add32c4d`.
[Performance by test and complete recipe](test-performance.md): compatible passes, failures, execution errors and missing witnesses, with sortable evidence drill-down.
