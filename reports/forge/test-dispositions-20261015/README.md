# Forge test dispositions (2026-10-15)

Decided 2026-10-15 against the evidence on `develop` at `224958a315a81317b8943de5c4e4f3e1c1127385`. This is the
final record of the decision. No training, GPU job or new experiment was run for it, and no recorded verdict was
regraded. The machine-readable record is [dispositions.json](dispositions.json). The revised views are in
[configs/forge/views/](../../../configs/forge/views/), and each previous revision is archived byte for byte in
[configs/forge/view-history/](../../../configs/forge/view-history/).

## Summary

The request was to fix, hold with an explanation, or remove every required leaderboard test that no selected
recipe passes, because one training recipe may not be able to pass all of them. 18 required tests had zero selected
passes. All 18 ask valid questions and no recorded verdict is wrong; one evaluator tolerance question remains for a
diagnostic run (see [Follow-ups](#follow-ups)).

- **15 tests are held.** A held test stays declared, keeps running and keeps its full history on every page. It
  stops counting until one global recipe passes it together with every counted test. The 15 fall into four groups,
  and only the first group is a demonstrated conflict between tests:
  - **(a) Conflict shown (5):** `vector_anisotropic`, `vector_overlap`, `img_bars4`, `img_blobs4`,
    `img_intensity2`. Every recipe with a [recorded pass](#recorded-passers) of one of them fails a counted test
    that BISPF passes. The vector family as a whole also conflicts: the cell-moment recipes pass all four vector
    tests but fail `mode_hold` or `ring16_acquisition`.
  - **(b) Never passed under their current declaration (4):** `grid100`, `rotated100`, `staggered100`,
    `target_shift_recovery`. These are unsolved, not shown to conflict with anything. Older historical receipts
    pass the three 100-blob tests only under earlier contracts, including noise added at evaluation (see
    [Recorded passers](#recorded-passers)).
  - **(c) Blocked behind (b) (3):** `grid100_14k`, `rotated100_14k`, `staggered100_14k`. They continue a passing
    7,000-update model, and none exists.
  - **(d) Partly untried (3):** `gaussian1d_stability`, `vector_unequal_mass`, `vector_unequal_width`. Some research
    recipes pass them and fail only held tests, but were never run on the rest of the counted suite. Holding them is
    the cautious choice until that check is run; each explanation names the cheapest probe.
- **2 are removed from the count, not from the board.** `clockfree_audit` runs the same clock probe and grader that
  the same view already counts at Tier 1. `ring_hold` is the first two phases of the same uninterrupted run that
  `ring_extension` grades. Both stay assigned as diagnostics with a removal note.
- **`ring_extension` stays counted; it is now counted once instead of twice.** It is the single ring endurance
  verdict: acquisition, a 1,200-check hold and a 300-check extension of one run. No defect lies behind its zero:
  BISPF's zero is a tier-prerequisite zero (its ordinary cohort never reached Tier 3) and BCAP's is a real failure.
  The zero will not clear by itself.
- **The 7 Atlas/E22 policy questions** (3 held, 4 removed) are decided, but take effect in a follow-up change. Their
  view, `tier1_policy_coverage`, stays at revision 1 here and never fed the leaderboard total.
- **17 extra variants:** already uncounted, so no change.
- **Pre-existing pytest failures** were fixed where a production bug or a stale expectation with a cited cause was
  found; the rest are recorded with their cause in the [pytest ledger](#pytest).

**Effect on the leaderboard.** The six ordinary views go from **152 to 81 counted view requirements**, which is 35
to 18 distinct tests. No pass count changes and nothing is regraded; only failing or unreached requirements leave
the total.

- **BISPF:** 77/81. Its recorded measurement passes every counted Tier 1 and Tier 2 requirement in all six views.
  The other 4 are ring endurance in four views, which needs one ordinary run.
- **BCAP:** 71/81; it still fails `mode_hold` in every view.
- **K3P, KA2, R1/R2 and GAN v3:** 20/81; they still fail two Tier 1 tests.
- **Atlas and E22:** 0/81; they remain incompatible with the ordinary suite.

**Read this before reading those numbers.** After the holds, every counted test except ring endurance is one that
BISPF's recorded measurement passes, and every test it fails is held; ring endurance (`ring_extension`) has no
qualifying BISPF result yet. That follows from the one-global-recipe rule (one recipe must pass every counted test)
and from BISPF passing every counted test that any other selected recipe's recorded result passes. It is not a new
measurement of BISPF. No new test was passed: the totals shrank because the held and removed requirements left the
count. BISPF's measurement is also stale against the current source.

## Words used here

- **Test (task)**: one Forge question with numerical pass/fail gates, declared in `configs/forge/tasks/<task>.json`.
- **View**: a list of tests, each with a tier and a role, in `configs/forge/views/<view>.json`. The leaderboard adds
  up six ordinary views. A test that appears in several views is counted once per view (a **view requirement**).
- **Tier**: 1 smoke (can it learn the basics), 2 quality, 3 endurance. A counted lower tier must pass before the next
  tier counts toward qualification.
- **Counted requirement** (role `required`): counts toward the leaderboard total and toward qualification.
- **Held test** (role `held`, new): still declared, run, graded and shown with its history, but not counted, and it
  never blocks progress. The view stores a `hold` note: the reason, a short summary, a link to its explanation below
  and the rule for counting it again (the **re-entry criterion**).
- **Hold reasons**: `no_global_recipe` (recipes pass it, but none passes it together with every counted test),
  `unsolved` (no recorded recipe has ever passed its current declaration) and `held_prerequisite` (it continues a
  held test, named in `blocked_by`).
- **Removed from the count** (role `diagnostic` plus a `removal` note): the test stays in the view and on every page,
  but it is not counted. Reasons: `phase` (part of the same run that a counted test grades), `duplicate` (the same
  question is counted elsewhere in the view), `superseded` (a retired definition) and `ill_posed` (it cannot run as
  declared). A phase or duplicate names the counted test in `counted_by`.
- **HELD (status)**: a tier with only held tests reports HELD; it neither passes nor stops later tiers. A view reports
  PASS only when every counted tier passes and its highest tier is counted. If every counted tier passes but its
  highest tier holds only held tests, or nothing is counted, the view reports HELD, which is never eligible for
  promotion. A counted failure still shows as FAIL (or BLOCKED or INCOMPLETE).
- **Global recipe**: one training configuration used unchanged for every test. "Across tests" means one global
  recipe must pass every counted test.
- **Selected recipe** and **cohort**: the recipe each family puts forward, and the exact source code and runtime its
  results were recorded with.
- **Ordinary run** vs **diagnostic run**: an ordinary run in the recipe's own cohort can qualify. A diagnostic run
  (another runtime, or a separate investigation) is shown, but it never fills a qualifying cell.
- **Recorded pass** (recorded passer): a PASS under the test's current declaration, in a selected cohort, the attempt
  index or a research packet the leaderboard displays. Older historical receipts do not count; see
  [Recorded passers](#recorded-passers) below.
- **Archived view revision**: every change to a view gets a new revision number, and the previous file is first
  copied byte for byte to `configs/forge/view-history/<view>-v<N>.json`.
- **Statuses**: **FAIL** means the test ran to its end and missed a gate. **TIMEOUT** means the run hit its time
  limit before the final checks, so it has no verdict. **ERROR** means execution failed, for example before training.
  **BLOCKED** means it was refused before starting, for example because its parent failed. **UNKNOWN** or **not
  run** means no compatible run exists. TIMEOUT, ERROR and BLOCKED are never counted as FAIL in this record.
- **Recorded qualification** vs **declared view**: the leaderboard's frozen evidence is registered against
  `discriminator_stability` revision 9. The new revisions change what is counted and shown; the recorded
  qualification stays at revision 9 until a new ordinary measurement is registered.
- **Attempt index**: [attempt-coverage.json](../attempt-coverage.json), 853 recorded recipe/test attempts from the
  selected and research recipes.
- **Cell-moment aggregate**: research recipes that build nearest-neighbour cells around real samples in data space
  and match the cells' statistics (moments) as a whole. **Cell-quota** and **cell-gradient-signal** recipes are
  related cell-based research objectives.
- **Kinetic / transport recipes**: BCAP research recipes (retained from PR #399) that add a data-matching transport
  force, pulling generated samples toward real samples directly, on top of the critic's push. The kinetic variants
  also carry a running history of that force.
- **Direction blend plus local-v2**: a round-5 BCAP research recipe ([round 5](../bcap-physics/round5/README.md))
  that blends update directions and adds the local-v2 transport force.
- **Balanced assignment**: a round-5 research recipe that replaces local-v2's sliced transport with a balanced
  pairing of real and generated samples in blocks of 128.
- **DualNorm**: the optimizer that BCAP's selected configuration uses; it scales each update step by a norm of the
  weight matrices.
- **Finite-atom exemption**: an upstream scoring rule that skips the shape gates for a blob served by fewer than 32
  particles, because a generator that outputs one fixed point per particle cannot draw such a blob's shape reliably.
  Forge does not use it (`finite_atom_exemptions: false` in the task), because its prior has nonzero width.
- **Job key**: the hash that identifies one planned training job, so identical work runs once and is shared. It
  covers the task's execution and the source code digest, but not a test's role in a view.
- **Oracle witness**: a reference result from the true target law, registered next to a test to show that a perfect
  sampler passes it.
- **Routed rows**: prior points (rows of the particle table) that a host's encoder or token map assigns to specific
  inputs, so they must keep their identity.
- **Clean sampling**: outputs are scored as the generator makes them, with no noise added at evaluation.
- **Policy mode**: Atlas/E22 run with an automatic controller that chooses step sizes and which saved state to
  report, and with a fixed cloud of input points; only the policy questions use it.
- **Live weights**: scoring the generator's current weights, not an average of its past weights (an EMA,
  exponential moving average).
- **Public initializer**: the repository's deterministic network initializer (`deterministic_orthogonal`), which
  every compared recipe must use. **Recipe hash**: the hash of a recipe's complete settings; "one recipe hash" means
  the same settings for every test.
- **Scheduled recipes**: recipes whose step size follows a timetable of update counts, such as a learning rate that
  decays to a floor after a fixed number of updates (the **learning-rate-floor** runs, for example the historical
  k3, k3g and rg5 runs). K3P, KA2, R1/R2 and GAN v3 are scheduled; a timetable is a clock dependency.
- **Finite acceptance**: the update-acceptance layer of the round-5 BCAP recipes, which checks each proposed update
  and can shrink it (backtracking) before applying it ([round 5](../bcap-physics/round5/README.md)).
- **Fourier features**: an input encoding that feeds sines and cosines of the input to the generator.
- **Bures variant**: the cell-moment aggregate that compares cell covariances with the Bures-Wasserstein distance
  between Gaussians. **Per-kernel control**: a cell-moment control that matches each kernel's (particle's)
  statistics separately instead of in aggregate.
- **G-angle**: a historical research recipe (`G-angle-with-native-prior-affine-ball-v1`); its unequal-width pass was
  recorded on an older source ([test-performance.md](../test-performance.md)).
- **Native task**: the 100-blob coverage task family (`benchmarks/toy100`), whose "good modes" count the blobs drawn
  well.
- **Learned-MoG suite**: the ordinary tasks, whose starting prior is a learned mixture of Gaussians. **Particle
  birth/death formulation**: Atlas/E22's formulation, which adds and removes particles during training.
- **Receipt**: the saved record of one task result (inputs, metrics, verdict and hashes). **Producer**: the attempt
  whose saved run another result continues or certifies.
- **Mass TV**: the share of samples that fall in the wrong blobs (total variation between sample and target
  shares). **Eigen ratio**: a blob's thinnest direction relative to the target's. **Precision** (100-blob tests):
  the share of points within three blob widths of some blob centre. **Frozen copy**: a copy of the model that stops
  learning at the target shift, used to prove the shift is noticeable.
- **The handoff**: the 2026-10-15 failing-tests handoff note that asked for this decision; it is a working note kept
  outside the repository. It summarised the earlier test-by-test zero-pass audit
  ([pull request #421](https://github.com/255BITS/ParticleGAN/pull/421)), whose recommendations for the 18 tests
  were 11 keep, 6 modify and 1 remove as a separate count, and it named three failing pytest checks. Where that
  audit recommended keeping a test, this record keeps its declaration, gate and measurement unchanged: holding
  changes only whether the test is counted.

<a name="recorded-passers"></a>

**Recorded passers and older historical receipts.** In this record, "recorded passer", "has passed" and "never
passed" refer to results under each test's current declaration: the selected cohorts, the attempt index and the
research packets the leaderboard displays. The repository also keeps older receipts in `reports/forge/records`
(evidence scope `historical`) that pass several held test ids under earlier contracts, such as a fixed particle
prior of zero width instead of the learned prior, EMA 0.995 evaluation weights, scheduled learning rates or noise
added to model samples at evaluation. Examples: the September 2026 gap-fill K3P record
(`history-k3p-84363f69cb3a`) passes 22 tests, including the three image tests, all four vector tests and the three
100-blob tests; across the historical records there are 5 to 10 PASS receipts for each vector test, 8 for each image
test and 36 to 39 for each 100-blob test, and five records pass both `img_bars4` and `img_stripes2`. Forge never
reuses historical receipts as current evidence, because their recipe, prior, initialization and sampling contracts
differ from today's declarations, so they do not show that one current global recipe passes these tests. None of
them passes `gaussian1d_stability`, `target_shift_recovery`, the three 14k tests or the ring tests.

## Decision rules applied

1. **One global recipe.** A test is held only if no single recipe passes it together with every counted test.
   Different winners are never picked for different tests.
2. **HOLD needs a real numerical failure, or a held parent.** Every hold names the failing numbers against their
   thresholds, why the problem is hard, what was tried, who passes it alone and what those recipes break, and a
   concrete re-entry criterion. Where the conflict is untried rather than shown, the explanation says so.
3. **REMOVE needs redundancy, or a definition that is retired or cannot run for anyone.** Removed tests stay
   assigned and visible.
4. **KEEP when the zero has no defect behind it.** A counted test whose zero comes from tier order or a real failure
   stays counted, and the record names what the zero is.
5. **FIX needs a concrete defect and needs no training.** No scientific gate is weakened; a gate change would need a
   new task version, with old grades kept. In this change only pytest items are fixes.
6. **The same disposition applies in every view that counts the test.** A counted test may not depend on an
   uncounted one, with one narrow exception for a phase of the same uninterrupted run (see `ring_hold`).
7. **Nothing is regraded.** Every receipt, earlier view revision and verdict stays.
8. **No training, no GPU, no new recipe experiments.** The only computations were CPU-only: a declared calibration
   of the vector gates drawing from the true target law, closed-form noise bounds, read-only recomputation from saved
   results, and an in-memory prototype of the new view rules.

## All dispositions

"Count change" is the number of view requirements that leave the leaderboard total (152 before this change).

| # | Test | Counted before in (tier) | Disposition | Count change | Reason in one line |
| --- | --- | --- | --- | ---: | --- |
| 1 | [`clockfree_audit`](#clockfree_audit) | clockfree_continuous (3) | REMOVE (duplicate) | -1 | Same clock probe and grader as `clockfree_audit_measurement_v1`, counted at Tier 1 of the same view; kept as the 14k gate. |
| 2 | [`gaussian1d_stability`](#gaussian1d_stability) | discriminator_stability (2) | HOLD (no_global_recipe) | -1 | Neither BCAP nor BISPF (the only selected recipes with recorded Tier 2 results) keeps or re-learns the bell curve; the one passer that also passes `mode_hold` was never run on the ring tests. |
| 3 | [`vector_unequal_mass`](#vector_unequal_mass) | six views (2) | HOLD (no_global_recipe) | -6 | The rare 2% blob loses its share or collapses; the gate is sound; round-5 passers never ran most of the counted suite. |
| 4 | [`vector_unequal_width`](#vector_unequal_width) | six views (2) | HOLD (no_global_recipe) | -6 | Shares wrong; balanced assignment passes but fails held tests and never ran most of the counted suite. |
| 5 | [`vector_anisotropic`](#vector_anisotropic) | six views (2) | HOLD (no_global_recipe) | -6 | Wrong amounts per blob; of 16 passers, 12 fail Tier 1 `ring16_acquisition`, 1 timed out on it and 3 fail `mode_hold`. |
| 6 | [`vector_overlap`](#vector_overlap) | six views (2) | HOLD (no_global_recipe) | -6 | Acquired then drifts; all 15 passers fail `ring16_acquisition` or `mode_hold`. |
| 7 | [`img_bars4`](#img_bars4) | six views (2) | HOLD (no_global_recipe) | -6 | Mode dropping (2 of 4 bars); no recorded passer passes `img_bars4` and `img_stripes2` together. |
| 8 | [`img_blobs4`](#img_blobs4) | six views (2) | HOLD (no_global_recipe) | -6 | 2-3 of 4 corners; passers fail or cannot run Tier 1. |
| 9 | [`img_intensity2`](#img_intensity2) | six views (2) | HOLD (no_global_recipe) | -6 | Brightness off or one level only; passers fail Tier 1. |
| 10 | [`grid100`](#grid100) | six views (2) | HOLD (unsolved) | -6 | 24-37% of samples on target against 97%; never passed under the current (clean) declaration. |
| 11 | [`rotated100`](#rotated100) | six views (2) | HOLD (unsolved) | -6 | 26-27% against 97%; never passed under the current declaration. |
| 12 | [`staggered100`](#staggered100) | six views (2) | HOLD (unsolved) | -6 | 30-34% against 97%; never passed under the current declaration. |
| 13 | [`grid100_14k`](#grid100_14k) | clockfree_continuous (3) | HOLD (held_prerequisite: `grid100`) | -1 | Needs a passing `grid100` state; the rescue diagnostic also fails. |
| 14 | [`rotated100_14k`](#rotated100_14k) | clockfree_continuous (3) | HOLD (held_prerequisite: `rotated100`) | -1 | Same, behind `rotated100`. |
| 15 | [`staggered100_14k`](#staggered100_14k) | clockfree_continuous (3) | HOLD (held_prerequisite: `staggered100`) | -1 | Same, behind `staggered100`. |
| 16 | [`ring_hold`](#ring_hold) | four views (3) | REMOVE (phase of `ring_extension`) | -4 | Its verdict is replayed inside `ring_extension`'s grade of the same run. |
| 17 | [`ring_extension`](#ring_extension) | four views (3) | KEEP (stays counted) | 0 | The single ring endurance verdict, counted once instead of twice; no defect behind its zero. |
| 18 | [`target_shift_recovery`](#target_shift_recovery) | adaptation (3), clockfree_continuous (3) | HOLD (unsolved) | -2 | Never passed; BISPF recovers 220 updates late. |
| 19 | [`gaussian1d_acquisition_tier1_policy_selected_cloud_v1`](#gaussian1d_acquisition_tier1_policy_selected_cloud_v1) | tier1_policy_coverage (1) | REMOVE (superseded); follow-up change | 0 | Copies a retired Gaussian definition that the main suite split into smoke and stability. |
| 20 | [`ring16_acquisition_tier1_policy_selected_cloud_v1`](#ring16_acquisition_tier1_policy_selected_cloud_v1) | tier1_policy_coverage (1) | HOLD (unsolved); follow-up change | 0 | Covariance error 1.14 and 1.81 against 0.85; 0/96 checks. |
| 21 | [`two_pole_tier1_policy_selected_cloud_v1`](#two_pole_tier1_policy_selected_cloud_v1) | tier1_policy_coverage (1) | HOLD (unsolved); follow-up change | 0 | Points move 0.0026 against 0.3; 0/24 checks. |
| 22 | [`clockfree_audit_tier1_policy_selected_cloud_v1`](#clockfree_audit_tier1_policy_selected_cloud_v1) | tier1_policy_coverage (1) | HOLD (unsolved); follow-up change | 0 | The step counter and a save and restore change the learned state. |
| 23 | [`unused_token_hold_tier1_policy_selected_cloud_v1`](#unused_token_hold_tier1_policy_selected_cloud_v1) | tier1_policy_coverage (1) | REMOVE (ill_posed); follow-up change | 0 | Refused before training for every recipe by construction. |
| 24 | [`ae_gan_hold_tier1_policy_selected_cloud_v1`](#ae_gan_hold_tier1_policy_selected_cloud_v1) | tier1_policy_coverage (1) | REMOVE (ill_posed); follow-up change | 0 | Same. |
| 25 | [`five_word_joint_acquisition_tier1_policy_selected_cloud_v1`](#five_word_joint_acquisition_tier1_policy_selected_cloud_v1) | tier1_policy_coverage (1) | REMOVE (ill_posed); follow-up change | 0 | Same, and its parent was retired. |
| 26 | [17 indexed variants](#indexed-variants) | none (already diagnostic) | NO_CHANGE | 0 | Already uncounted; empty cells mean "not run". |
| 27-29 | The three handoff pytest failures and further pre-existing failures | pytest | FIX where a cause was found | n/a | See the [pytest ledger](#pytest). |

**Total: -71 view requirements, 152 to 81.** The policy view is a separate scope that never fed the 152, so its
decisions change no total.

## What changes on the leaderboard

The leaderboard ([technique-inventory.md](../technique-inventory.md)) is generated from the views and the
committed evidence; it is never edited by hand. Counted view requirements per view (Tier 1, Tier 2, Tier 3). Held
tests are still measured and shown.

| View | Revision | Before | After | Held (measured, not counted) | Removed from the count |
| --- | --- | --- | --- | --- | --- |
| adaptation | 2 to 3 | [3, 19, 1] = 23 | [3, 9, 0] = 12 | 10 at Tier 2, 1 at Tier 3 | none |
| clockfree_continuous | 3 to 4 | [4, 19, 7] = 30 | [4, 9, 1] = 14 | 10 at Tier 2, 4 at Tier 3 | `ring_hold`, `clockfree_audit` |
| discriminator_stability | 9 to 11 | [6, 21, 2] = 29 | [6, 10, 1] = 17 | 11 at Tier 2 | `ring_hold` |
| formulation_comparison | 1 to 2 | [3, 19, 2] = 24 | [3, 9, 1] = 13 | 10 at Tier 2 | `ring_hold` |
| host_profile_transfer | 4 to 5 | [3, 19, 2] = 24 | [3, 9, 1] = 13 | 10 at Tier 2 | `ring_hold` |
| quality_coverage | 2 to 3 | [3, 19, 0] = 22 | [3, 9, 0] = 12 | 10 at Tier 2 | none |
| **Six ordinary views** | | **152** | **81** | 66 view assignments | 5 view assignments |
| tier1_policy_coverage (separate scope) | 1, unchanged here | [7, 0, 0], not in the 152 | unchanged here | decided: 3 at Tier 1 | decided: 4 |

`discriminator_stability` skips revision 10. Committed research artifacts already cite an unadopted "revision10"
draft that moved Gaussian retention to Tier 3, and revision 11 is not that draft:

- `reports/forge/g-coherent-mean-seed0/candidate-declarations.json`
- `reports/forge/g-coherent-mean-paired-research/preregistration.json`
- `experiments/forge/source_bound_board.py` (and the family pages it renders)

Per family (selected recipe, counted requirements; computed read-only from the committed publication):

| Family | Before | After | Tier 1 | Tier 2 | Tier 3 | Can Tier 3 count now? |
| --- | --- | --- | --- | --- | --- | --- |
| BISPF (recorded measurement, stale) | 77/152 | 77/81 | 22/22 | 55/55 | 0/4 | **Yes, in four views.** Ring endurance (`ring_extension`) needs an ordinary run in clockfree_continuous, discriminator_stability, formulation_comparison and host_profile_transfer. That is one job: the four views plan the same ring group job. BISPF's saved diagnostic run already passes it, but a diagnostic run cannot fill an ordinary cell. quality_coverage has no Tier 3. adaptation's only Tier 3 test is held, so adaptation reports HELD (qualified Tier 2), never PASS. |
| BCAP | 71/152 | 71/81 | 22/22 | 49/55 | 0/4 | No: `mode_hold` fails at Tier 2 in all six views. |
| K3P, KA2, R1/R2, GAN v3 | 20/152 each | 20/81 each | 20/22 | 0/55 | 0/4 | No: they fail Tier 1 `ring16_acquisition` (discriminator_stability) and `clockfree_audit_measurement_v1` (clockfree_continuous), and have no ordinary Tier 2 runs. |
| Atlas, E22 | 0/152 | 0/81 | 0/22 | 0/55 | 0/4 | No: the ordinary learned-MoG suite is incompatible with their particle birth/death formulation. |

Caveats, also printed on the board:

- **No new test was passed.** The totals shrank because the held and removed requirements left the count; every
  pass count is unchanged. Every counted test except ring endurance is one that BISPF's recorded measurement passes;
  that follows from the one-global-recipe rule and is not a new measurement.
- **BISPF's measurement is stale.** It is a recorded measurement of an older source, so current qualification needs
  a fresh ordinary run of the counted suite at the current source anyway.
- **Two denominators share the page.** Selected-recipe totals follow the declared revisions. Research rows keep their
  recorded revision 9 denominators (29 tests, which still include the 12 tests that discriminator_stability revision
  11 holds or removes from the count), and recorded qualification stays at revision 9.
- **adaptation counts the same tests as quality_coverage** while `target_shift_recovery` is held, and it never shows
  PASS: BISPF, which passes every counted tier, gets HELD, and BCAP's counted `mode_hold` failure still shows as FAIL.
- **The board's "Held and removed tests" section** is built from the notes stored in the views, so it cannot drift
  from them. It lists each test's reason, views, summary, re-entry rule, how many selected recipes' recorded results
  pass it (0 of 8 for every held or removed test), which selected recipes pass it only in a separate diagnostic run
  (for example R1/R2 and GAN v3 on `img_bars4`, and BCAP and BISPF on `clockfree_audit`), and a link to its
  explanation below. A diagnostic run never becomes a recorded result, so those passes count nowhere.

<a name="rules-in-the-views"></a>

## Rules in the views

The view rules are in [experiments/forge/views.py](../../../experiments/forge/views.py); the archive-first apply
command is [experiments/forge/dispositions.py](../../../experiments/forge/dispositions.py) (`python -m
experiments.forge test-dispositions check|apply --registry <file>`), driven by [dispositions.json](dispositions.json).

- A held assignment carries a `hold` note with exactly: `schema` (`forge_hold_v1`), `previous_importance`,
  `reason`, `blocked_by`, `summary`, `explanation` and `reentry`. A removed assignment stays `diagnostic` and carries
  a `removal` note with `schema` (`forge_count_removal_v1`), `previous_importance`, `reason`, `counted_by`, `summary`
  and `explanation`.
- Tiers holding a counted or held test must run contiguously from Tier 1. A counted test may not depend on a held or
  uncounted test, directly or through a removed phase. `blocked_by` lists exactly the held parents.
- **Phase exception:** a counted test may depend on a removed test only when the removed test is a phase of it: its
  `counted_by` names the counted test, both sit in the same uninterrupted execution group and tier, and the counted
  test continues the phase's own saved state (a checkpoint dependency). The phase's own prerequisites must be
  counted too (for `ring_hold`, `mode_hold` is), so a held or diagnostic test can never decide a counted verdict
  through a phase. The counted test still needs the phase to pass, so no requirement is lost.
- **Duplicate rule:** a `duplicate` removal must name a counted test of the same view, at the same or an earlier tier,
  with the same adapter and the same evaluation declaration (thresholds, conditions and source pins). The counted
  copy may also pin the duplicate's own task file as provenance, as `clockfree_audit_measurement_v1` pins
  `clockfree_audit.json`.
- **Registry base pin:** each view entry of [dispositions.json](dispositions.json) stores the policy fingerprint of
  its base revision (`from_fingerprint`). `apply` refuses a live view that was edited without a new revision number,
  and `check` reports drift when an archive no longer matches that fingerprint.
- **Job identity:** a test's role is not part of job identity. The planning science that determines a job key has no
  role field; `tests/test_forge_dispositions.py` checks, for K3P on discriminator_stability and
  clockfree_continuous, that every planned job's science is identical with and without the role changes, apart from
  `candidate_revision`, which follows the source code digest (any code change gives new keys). The verification of
  this change repeated that comparison for the eight selected families and one extra BCAP configuration on each of
  the six revised views (54 pairs), with identical science in every pair. Existing evidence stays attached through
  unchanged task execution and evaluation fingerprints and the grouped-evidence certificate.

Every dependency edge that touches a decided test:

| Parent to child | Edge kind | Roles after the change | Result |
| --- | --- | --- | --- |
| `mode_hold` to `ring_hold` | gate | counted to removed phase | allowed |
| `ring_hold` to `ring_extension` | checkpoint | removed phase to counted | allowed only by the phase exception |
| `mode_hold` to `target_shift_recovery` | gate | counted to held | allowed |
| `gaussian1d_smoke` to `gaussian1d_stability` | checkpoint | counted to held | allowed |
| `grid100`, `rotated100`, `staggered100` to their 14k tests | checkpoint | held to held | allowed; `blocked_by` names the held parent |
| `clockfree_audit` to the three 14k tests | gate | removed duplicate to held | allowed |

**Hold and removal notes are view policy.** The summary, re-entry rule and explanation link are stored inside each
view, and they are part of the view's fingerprint, the hash that saved evidence, studies and searches pin. Changing a
word, a re-entry rule or a link is therefore a policy change: it needs a new archived view revision, never an
in-place edit. The anchors below are the targets of those links; do not rename them.

## How a held test returns

1. Someone shows that its re-entry criterion is met: one global recipe has an ordinary PASS on the unchanged task
   declaration and passes every counted test, in one same-source, same-runtime cohort. A diagnostic PASS only
   triggers a review.
2. A new view revision restores the test's role to `required` and drops its hold note, in every view that held it,
   after archiving the previous revision byte for byte (the archive-first rule that `retier` and `test-dispositions
   apply` follow). The 14k tests return in the revision that restores their 7k parents; that revision also restores
   `clockfree_audit` to the count as their gate, because a counted test may not depend on an uncounted one.
3. The leaderboard is regenerated with `--refresh-publication`, which updates what is counted and shown without
   regrading any recorded result.

## Per-test explanations: the 18 required tests

<a name="clockfree_audit"></a>

### 1. `clockfree_audit`: REMOVE from the count (duplicate), keep as the 14k gate

**What it asks.** Does training depend on a clock (the step number) instead of only on the model and the data? It
replays 4 updates of a 24-update run four ways:

- with a fake step counter (+8,192);
- with a different planned training length;
- with extra evaluation in between;
- after a save and restore.

The learned state must come out bit-for-bit identical, and a source check must find no step-based rule.

**Why the board shows zero.** It is not a failure. Tier 3 was never reached because BCAP had 11 and BISPF 10 Tier 2
failures in clockfree_continuous revision 3 (BISPF's 10 are all held now; BCAP still fails `mode_hold`). Both
recipes pass all four comparisons:

- in Torch 2.13 diagnostics (`a61d4c1b`, `6e14efde`);
- in their own selected cohorts, as the Tier 1 `clockfree_audit_measurement_v1` (receipts `00dd0a06`, `9fb83c34`).

The scheduled recipes (K3P, KA2, R1/R2, GAN v3) really fail, and they already fail the Tier 1 copy.

**Why it is a duplicate.** `clockfree_audit` and `clockfree_audit_measurement_v1` run the same probe and use the same
grader (`views._clockfree`, adapter `clockfree_audit`, evaluation kind `clockfree_parity`, the same four
conditions). They are separate jobs: the measurement copy's execution adds `clock_audit_scope` and
`clock_audit_parent`, and the strict copy refuses to start (BLOCKED) a recipe whose learning-rate floor (`lr_floor`)
is an unexplained clock dependency, and grades an unexplained dependency found by the source audit as BLOCKED where
the measurement copy records FAIL. Where the strict copy did run, its recorded diagnostic results for K3P, KA2,
R1/R2, GAN v3 and E22 are FAIL (the step counter changed the learned state;
[actual-missing-diagnostics-20261014](../actual-missing-diagnostics-20261014/README.md)). The two pass for the same
recipes, so counting both counts one claim twice in one view.

**Decision.** The clock-free claim stays counted once, at Tier 1. The Tier 3 copy stays assigned as a diagnostic in
clockfree_continuous with `removal.reason = duplicate` and `counted_by = clockfree_audit_measurement_v1`, because the
three 14k continuations declare it as their gate. Count change: -1.

**When the 14k tests return to the count**, that same revision restores `clockfree_audit` to `required` as their
gate.

<a name="gaussian1d_stability"></a>

### 2. `gaussian1d_stability`: HOLD (no_global_recipe; partly untried; confidence medium)

**What it asks.** The Tier 1 smoke run learns one bell curve (centre 2, width 0.5). This Tier 2 test keeps training
that same model:

- **Stay matched:** all 72 checks between updates 1,042 and 4,000 must match the target. KS (the largest gap between
  the model's and the target's cumulative curves) must be at most 0.05, the centre within 0.2 widths and the width
  within 20%.
- **Re-acquire:** the target then moves to centre 3. The model must match again by update 5,000 (the last 5 checks).
- **Hold the new target:** all 24 checks up to update 6,000 must match.

**What fails.**

- BCAP passes 2 of the 72 stay-matched checks, misses re-acquisition, and passes 0 of 24 afterwards. Final KS is
  0.32, the width ratio 0.66 and the centre error 0.26 widths.
- BISPF passes 5/72, misses re-acquisition and passes 0/24. Final KS is 0.22, the width ratio 0.63 and the centre
  error 0.33.
- Even the state handed over at update 1,000 is already off target (BCAP KS 0.072, BISPF 0.112). Their smoke pass was
  a brief hit, not a stable fit.

**Is the test wrong? No.**

- KS is computed exactly against the true curve on 4,096 clean samples, with no histogram bins.
- A sampler drawing from the true target would wrongly fail one check with probability about 2.6 in a billion, and
  any of the 96 gated checks with probability below 2.5 in ten million (closed-form bounds, not simulation).
- The grader's focused tests pass.

**Why it is hard.** The network can draw the target exactly: a hand-built version passes 24/24. The trouble is how
training moves it.

- **A constant step size wobbles.** GAN training is a tug-of-war, so even at the right answer the generator, the
  critic and the latent particles keep getting pushed. With a constant step size the output wobbles in and out of
  tolerance; in the saved curves KS crosses 0.05 every 1-4 checks.
- **Shrinking steps on a timer calms it.** K3P and KA2 do this and keep 71/72, but then they react too slowly when the
  target moves, and they break the clock-free rule.
- **So one rule must do both.** It has to stand still when right and move quickly after a shift.

**What has been tried (all failed this test):**

- direction blend plus local-v2: final KS 0.034, but only 6/72 and 2/24;
- removing finite acceptance: 8/72;
- confidence, positive-evidence, critic-orientation and finite-cap learning-rate responses;
- removing the Fourier features (3/72), a shallow generator (4/72) and smoothing (1/72);
- batch-size and prior-grid studies;
- a 204-run cell-based sweep.

**Who passes it alone, and what they break.**

- **Three kinetic research recipes pass** (`kinetic-local-nsgda-cross-v1`, `kinetic-local-dualnorm-g-history-v1`
  and its `-bank8-v-v1` variant). All three fail Tier 1 `ring16_acquisition`, `mode_hold` and `ring_hold`, and most
  also fail density, image and 100-blob tests.
- **Five distinct cell-gradient-signal recipes pass** (forward-KL g and gp, reverse-KL g and gp, moment gp). Each
  fails `mode_hold`, `ring16_acquisition`, `vector_unequal_mass` and `vector_unequal_width`.
- **`cell-quota-minimum-change-forward-kl-g-v1` passes this test and `mode_hold`**
  ([astra-cumulative-24.json](../test-performance-evidence/astra-cumulative-24.json)). Its only recorded failure is
  `vector_unequal_mass` (component covariance 1.069 against 0.85), which is itself held. It has never been run on
  `ring16_acquisition`, the rest of the counted suite or ring endurance. For this recipe "across tests" is untried,
  not shown to conflict.
- **The reverse holds too:** BISPF and BCAP hold the ring but not the Gaussian.

The view's summary for this test begins "Neither selected recipe": it describes BCAP and BISPF, the only selected
recipes with recorded Tier 2 results. In diagnostic runs K3P and KA2 keep 71 of the 72 stay-matched checks, miss
re-acquisition and hold 22/24 and 16/24 afterwards; R1/R2 and GAN v3 keep 19/72 and 11/72. All eight still fail the
test. Correcting the summary's wording needs a new view revision (see [Follow-ups](#follow-ups)).

No recorded recipe has shown this test, `ring16_acquisition`, `mode_hold` and ring endurance together; for the one
candidate that passes both this test and `mode_hold`, the ring tests were never run.

**Cheapest re-entry probe.** Run `cell-quota-minimum-change-forward-kl-g-v1` through the counted suite and ring
endurance. Whether a cell-based objective is allowed in a global recipe under the portability rule is an open
question (see [Unresolved](#unresolved)).

**Re-entry criterion.** Count it again when one global recipe, in one same-source and same-runtime cohort, passes
this whole test (72/72, re-acquisition, 24/24). The same recipe must also pass every counted Tier 1 and Tier 2 test,
including `ring16_acquisition` and `mode_hold`, and the counted ring endurance test.

<a name="vector-group"></a>

### 3-6. The four vector density tests: HOLD, each on its own re-entry rule

All four use the same scorer. Every 50 updates it draws 4,096 samples and assigns each sample to the nearest true
blob. A test passes when its last 5 of 24 checks pass every gate.

Each vector test re-enters on its own: per-test re-entry already requires one global recipe to pass that test and
every counted test, so it cannot produce a different winner per test. The round-5 study shows that `unequal_mass`
and `unequal_width` trade off against each other (one transport change flips both results), so those two will
likely return together. The cell-moment aggregate recipes pass all four vector tests under the current declarations
([nonatlas-cell-moment-full-runs.json](../test-performance-evidence/nonatlas-cell-moment-full-runs.json)); three of
them fail `mode_hold` (sample quality 0.47-0.66 against 0.9) and the fourth (Bures) fails `ring16_acquisition`, and
most of the counted suite is unmeasured for them. Running one
of them through the counted suite is the cheapest probe for the whole family; whether that objective is allowed in a
global recipe is listed under [Unresolved](#unresolved).

**Re-entry criterion (each).** Count it again when one global recipe passes it under its unchanged declaration.
"One global recipe" means one recipe hash and source, seed 0, the public initializer and one runtime. The same recipe
must also pass every counted Tier 1 and Tier 2 test, including `ring16_acquisition` and `mode_hold`.

<a name="vector_unequal_mass"></a>

#### `vector_unequal_mass` (no_global_recipe; partly untried)

**What it asks.** Four round blobs hold 55%, 30%, 13% and 2% of the samples. The checks are:

- overall distance to the real data (SW1, normalised): at most 0.18;
- share of samples in the wrong blobs (mass TV): at most 0.15;
- share of samples close to some blob (HQ): at least 0.85;
- the average spread error across the blobs (covariance error): at most 0.85; one narrow blob's own error can be far
  larger while the average passes;
- no squashed blob: the thinnest direction keeps at least 15% of the target variance (eigen ratio at least 0.15);
- every blob gets at least a quarter of its share (mass ratio at least 0.25).

**What fails.**

- **BCAP:** 0 of 24 checks pass. The 2% corner holds 17 samples where about 82 belong, forming a thin streak (eigen
  ratio 0.009). The mass ratio is 0.21 and the covariance error 3.69.
- **BISPF:** 0 of 24. The shares are fine, but none of the samples near the rare corner lie within three standard
  deviations of its centre (eigen ratio 0.004).

**Is the test wrong? No.** We checked it with a declared CPU calibration using true-target draws and the unchanged
scorer and grader ([calibration](#calibration)).

- **Perfect samples pass every time:** 4,800 of 4,800 checks and 200 of 200 runs. The worst eigen ratio was 0.51
  against 0.15, and the worst mass ratio 0.60 against 0.25.
- **The failure controls fail every time:** squashing the rare blob to a line or a point, deleting it, or giving the
  blobs uniform shares failed 200/200 runs.
- **The share floor sits between one and two particles' worth:** one particle's worth fails every run, and two pass
  92.5%.
- **Narrow kernels are a model-class effect, not noise:** the shape gate turns sensitive to particle placement only
  when the rare blob is built from about five very narrow kernels (65-72% pass). The trained failures are not in that
  regime.

No gate changes, and the finite-atom exemption is not imported.

**Why it is hard.** The generator turns 256 equally weighted starting points (particles) into samples, so a 2% blob
needs about five of them. The critic sees about 2.5 real samples from that corner per batch of 128, so its signal
there is weak and noisy. The few particles that stay get pushed to one spot and flatten into a line.

**Who passes it alone, and what they break.**

- **Three kinetic recipes in the attempt index** (`kinetic_transport_local_v2` and the two `reference8` g-history
  recipes). All three fail `vector_unequal_width` and `mode_hold`. Two fail Tier 1 `ring16_acquisition`; the third
  (`kinetic-local-dualnorm-g-history-reference8-u-v1`) timed out on it (TIMEOUT, not FAIL). Both `reference8`
  recipes also fail `gaussian1d_smoke`.
- **Some round-5 research recipes** ([round 5](../bcap-physics/round5/README.md)), for example direction blend plus
  local-v2, pass it and fail only held tests (`gaussian1d_stability`, `vector_unequal_width`), but were never run on
  `ring16_acquisition`, `mode_hold` or most other counted tests. Untried, not shown to conflict.
- **Balanced assignment** fixes `vector_unequal_width` but breaks this test.
- **Cell-moment aggregates** pass all four vector tests and fail `mode_hold` (or, for the Bures variant,
  `ring16_acquisition`); the two per-kernel controls also fail counted tests.

<a name="vector_unequal_width"></a>

#### `vector_unequal_width` (no_global_recipe; partly untried)

**What it asks.** Four equal-share blobs with standard deviations 0.07, 0.12, 0.20 and 0.30. The gates are the same,
without the share floor.

**What fails.**

- **BCAP:** gets the shares wrong. The 0.12 blob holds 197 of 4,096 samples and its own spread error is 23.9, so the
  average covariance error is 6.29 against 0.85; mass TV is 0.29 and SW1 0.30.
- **BISPF:** shapes pass, but mass TV is 0.25 and SW1 0.27.
- **No PASS in the attempt index:** none of its 27 indexed records passes.

**Is the test wrong? No.**

- Perfect samples pass 4,800/4,800 checks and 200/200 runs; the worst spread error was 0.10 against 0.85.
- One shared width for all blobs fails every check.
- The gate is strict about strays around tiny blobs (5% halo samples fail), but trained recipes can satisfy it:
  balanced assignment reaches 0.286.

**Why it is hard.** One smooth generator, fed the same small jitter everywhere, must squeeze some regions and stretch
others while the critic pushes on all blobs alike. A few stray samples dominate the spread of a blob with variance
0.0049. Transport-based research recipes pass for stretches, then a burst of strays breaks the last five checks (best
runs: 14-16 of 24 checks, but only 0-3 at the end).

**Who passes it alone, and what they break.**

- **Round-5 balanced assignment** (14 final checks in a row, spread error 0.286) fails `vector_unequal_mass`,
  `grid100` and `gaussian1d_stability`, all held tests, and halves the native task's good modes, from 17 to 9. It was
  never run on the rest of the counted suite. Untried, not shown to conflict.
- **Cell-moment recipes** (the four aggregates and the per-kernel control) pass, and each fails a counted test
  (`mode_hold`, `ring16_acquisition` or `gaussian1d_smoke`). The published fixture `vector_unequal_width_published`
  is also passed 5/5 by this family.
- **G-angle** passed only on a historical source.

<a name="vector_anisotropic"></a>

#### `vector_anisotropic` (no_global_recipe; conflict shown)

**What it asks.** Three equal-share blobs, stretched and tilted: long horizontally, long vertically, and at 45
degrees.

**What fails.**

- **BCAP:** draws the shapes but the wrong amounts: 1,276, 2,168 and 652 samples instead of about 1,365 each. Mass TV
  is 0.196 against 0.15 and SW1 0.198 against 0.18.
- **BISPF:** mass TV 0.191, SW1 0.215 and covariance error 2.0.

**Is the test wrong? No.** Each blob is at least 5.8 standard deviations from the dividing line, so a perfect sampler
is never confused. Its passers clear the gates with wide margins (shares off by under 2%).

**Why it is hard.** Equal shares across differently shaped blobs need the trainer to balance mass between modes, not
just place them. The recipes that manage it use a different critic (K3P, KA2, R1/R2) or data-matching transport
forces (the kinetic research recipes).

**Who passes it alone, and what they break.** 16 recipes have a recorded pass under the current declaration:

- **12 in the attempt index** (K3P, KA2, R1/R2 and nine research recipes): 11 fail Tier 1 `ring16_acquisition` and
  one (`kinetic-local-dualnorm-g-history-reference8-u-v1`) timed out on it; all 12 fail `mode_hold`. K3P, KA2 and
  R1/R2 also fail the clock measurement.
- **4 cell-moment aggregates:** 3 pass `ring16_acquisition` but fail `mode_hold`; the Bures variant fails
  `ring16_acquisition`.

All 15 passers with a `mode_hold` result fail `mode_hold`.

<a name="vector_overlap"></a>

#### `vector_overlap` (no_global_recipe; conflict shown)

**What it asks.** Two heavily overlapping blobs that look like one broad cloud. Which blob a sample came from cannot
be identified, so only visible statistics are scored: centre error at most 0.15, covariance error at most 0.45 and
SW1 at most 0.18.

**What fails.**

- **BCAP:** reaches the target early (first pass at update 100, 5 passing checks), then the cloud slides around. Its
  centre error swings between 0.03 and 0.56 and ends at 0.39, with SW1 0.26.
- **BISPF:** ends with centre error 0.28, covariance error 0.53 and SW1 0.19.
- **The last five checks:** neither recipe has a passing run at the end.

**Is the test wrong? No.** Sampling noise is about a tenth of each limit, and 15 recipes hold the gates.

**Why it is hard.** An adversarial game aimed at a broad target tends to circle the answer instead of settling on it,
so staying put needs an extra steadying force.

**Who passes it alone, and what they break.** 15 recipes have a recorded pass under the current declaration:

- **11 in the attempt index** (K3P and ten research recipes): all fail `mode_hold`; 10 fail Tier 1
  `ring16_acquisition` and one timed out on it. On `grid100` 8 fail and 3 timed out; on `img_bars4` 10 fail and 1
  ended in an ERROR.
- **4 cell-moment aggregates:** 3 fail `mode_hold`; the Bures variant fails `ring16_acquisition`.

Every passer fails `ring16_acquisition` or `mode_hold`.

<a name="image-group"></a>

### 7-9. The three image tests: HOLD (no_global_recipe; conflict shown)

All three use a small generator that turns 32 learned starting codes (particles) into 8x8 grey pictures over 600
updates. A picture is "clean" when its pixel error to the nearest real picture is under the limit. "Modes" counts how
many real pictures are each drawn cleanly by enough particles. Both conditions must hold on each of the last 5 of 24
checks. The scorer looks at all 32 particles exactly, so there is no sampling noise. Real pictures score about 0.01
error, and blurry blends of two pictures score far above every limit, so the scorer cannot be fooled by blur.

**Re-entry criterion (each).** Count it again when one global recipe has an ordinary PASS on its unchanged
declaration. With the same configuration, that recipe must still pass `img_stripes2` and every other counted Tier 1
and Tier 2 test.

<a name="img_bars4"></a>

#### `img_bars4`

**What fails.** Four bar pictures; each bar must be drawn cleanly by at least 4 particles, and at least 29 of 32
pictures must be clean.

- **BCAP:** 31/32 clean pictures, but only 2 of the 4 bars.
- **BISPF:** 32/32 clean, also 2 of 4. A closely matching BCAP-derived run puts 8, 21, 2 and 1 particles on the four
  bars.

**Why it is hard.** This is mode dropping. To add a missing bar, some particles must leave a bar they already draw
well, and every picture in between is a blur that the critic rejects. Recipes that do spread over all four bars pay
in clean pictures: `kinetic_transport_local_v2` has 27/32 clean, short of 29.

**Tried and failed.** BCAP convolution support (2 of 4 bars); the confidence-LR, finite-cap and positive-evidence
variants (2 of 4); kinetic transport and history variants (3-4 bars, 26-27 clean).

**Who passes it alone, and what they break.** R1/R2 and GAN v3 pass. Both fail Tier 1 `ring16_acquisition` and the
clock measurement, which BCAP and BISPF pass, and both fail the counted `img_stripes2`. Of 21 recipes with results on
both tests under the current declarations, none passes `img_bars4` and `img_stripes2` together; five older historical
receipts pass both under earlier contracts (see [Recorded passers](#recorded-passers)).

<a name="img_blobs4"></a>

#### `img_blobs4`

**What fails.** Four small corner squares; the test needs all 4 corners and 29/32 clean pictures.

- **BCAP:** 2 corners and 24/32 clean.
- **BISPF:** 3 corners and 28/32 clean.
- **Every check:** neither recipe passes any of the 24.

**Why it is hard.** A 2x2 blob differs from black in only 4 of 64 pixels, so the push toward a missing corner is
faint. Several recipes miss by one step: three kinetic history variants end with all four corners and 32/32 clean,
but hold it for 4 of the 5 required checks.

**Who passes it alone, and what they break.**

- **R1/R2** (18 checks in a row) fails Tier 1 `ring16_acquisition`, the clock measurement and more.
- **`kinetic_transport_local_v2`** fails Tier 1 `unused_token_hold` and `two_pole`.
- **Four kinetic history variants** are refused (BLOCKED) on the four Tier 1 behaviour tests (`two_pole`,
  `unused_token_hold`, `ae_gan_hold`, `five_word_joint_smoke`). Three of them pass `gaussian1d_smoke` and fail
  `ring16_acquisition`; the fourth fails `gaussian1d_smoke` and timed out on `ring16_acquisition`.

<a name="img_intensity2"></a>

#### `img_intensity2`

**What fails.** One grey square at brightness 0.35 or 0.85. Each level needs at least 8 particles, and 29/32 pictures
must be within the tighter 0.06 error, roughly 0.12 either side of the brightness.

- **BCAP:** draws both levels, but 10 of 32 pictures are off in brightness (HQ 0.69).
- **BISPF:** collapses to one level (HQ 0.5).

**Why it is hard.** Brightness is continuous, and in-between greys look plausible to the critic, so the push toward
the exact level is weak. Atlas/E22 and the BCAP convolution run reach a passing picture set but hold it for only 4 of
the 5 required checks.

**Who passes it alone, and what they break.** GAN v3, `kinetic_transport_local_v2` and R1/R2 pass, and each fails
Tier 1 tests that BCAP and BISPF pass. The historical published pass used a wider network, which is the separate
uncounted variant `img_intensity2_residual16`.

<a name="native-group"></a>

### 10-12. The three 100-blob tests: HOLD (unsolved)

<a name="grid100"></a>
<a name="rotated100"></a>
<a name="staggered100"></a>

The target is 100 tiny round blobs (spread 0.03), arranged as a grid (`grid100`), a grid rotated 25 degrees
(`rotated100`) or alternating shifted rows (`staggered100`). Neighbouring blobs are about 30 blob-widths apart.

- **On target (precision):** a point counts if it lands within 0.09 (three widths) of a blob centre. At least 97% of
  points must be on target; a perfect generator scores 98.9%.
- **Shape and placement:** every blob must hold about 1% of the points, sit in the right place, have the right size
  and have the right round shape.
- **When it is measured:** on the last 5 checks of 7,000 updates, and on a separate final sample of 100,000 points
  (the holdout).
- **Clean sampling:** no noise is added to the outputs.

**Is the test wrong? No.** The true target passes every limit easily (holdout precision 0.98914, centre error 0.042
widths, radial-shape error 0.0015). The evaluator refuses to grade a run if its saved true-target reference fails.
BISPF's rebuilt run reproduces its recorded result digit for digit.

**What fails.**

| Test | BCAP on target | BISPF on target | Best 7,000-update result, current declaration (any recipe) |
| --- | ---: | ---: | ---: |
| `grid100` | 24% | 37% | 54% |
| `rotated100` | 26% | 27% | 51% |
| `staggered100` | 30% | 34% | 56% |

- **BCAP and BISPF:** centres are off by 1.1-1.5 blob-widths, many blobs are thin or nearly empty, and 22-34% of
  points land far out in the gaps.
- **The best recipe under the current declaration:** places the centres correctly but makes every blob about twice too
  wide.
- **Attempts:** 84 in the attempt index (28 per test): 56 FAIL, 15 BLOCKED and 13 TIMEOUT. Every completed run passes
  0 of the final 5 checks; the 13 timed-out runs stopped before reaching them.

**Why it is hard.** A smooth network must send 1% of its inputs to each of 100 islands, leave the sea between them
empty, and draw each island at exactly the right size. Smooth networks naturally smear inputs across the gaps. The
critic gives a clear signal about where blobs are, but only a weak one about whether a blob is 0.03 or 0.06 wide. The
older Atlas line passed historically only because the evaluator then added 0.029 of noise to every output point; its
clean outputs fail 0 of 5 checks.

**Tried and failed.**

- Doubling BISPF's training lifts precision once, to about 0.68, then it stays flat.
- The closest clean result is K3P on an uncounted straight-line generator variant: 99.5% on target, but blobs 27% too
  small and shrinking.

**Who passes it alone.** Nobody under the current declaration: no PASS exists in the attempt index or in any
published research result. The recipes that reach 49-56% are refused on the Tier 1 smoke tests, or fail `two_pole`
and `unused_token_hold`. The 36 to 39 historical PASS receipts per test used the earlier noisy evaluation and
fixed-particle contracts (see [Recorded passers](#recorded-passers)). The view summaries' "best recipe ever" means
the best 7,000-update result under the current declaration; the 14k rescue continuations (precision 0.65-0.70) and
the uncounted straight-line variant above are separate questions.

**Re-entry criterion (each).** Count it again when one global recipe has an ordinary PASS on the unchanged
declaration (clean sampling, live weights, 5 final checks and the 100,000-point holdout). The recipe must still pass
every counted Tier 1 and Tier 2 test. A diagnostic PASS by any recipe only triggers a review. `rotated100` and
`staggered100` in practice return with `grid100`; their redundancy cannot be judged while all three fail together.

<a name="native-14k"></a>

### 13-15. `grid100_14k`, `rotated100_14k`, `staggered100_14k`: HOLD (held_prerequisite)

<a name="grid100_14k"></a>
<a name="rotated100_14k"></a>
<a name="staggered100_14k"></a>

**What they ask.** Take a recipe that already passed the 7,000-update test, keep training that very model to 14,000
updates, and check that it still passes. The task also requires the clock audit (`clockfree_audit`) as a gate.

**Why they cannot count now.** No recipe has a passing 7,000-update parent, so every ordinary start is refused before
training (10, 10 and 11 refused starts). That is a dependency, not a separate measured failure. A separate "rescue"
run continued BISPF's failed 7k states anyway. Precision jumped once and then stayed flat: 0.68, 0.65 and 0.70
against 0.97, with the blobs still far too spread out (relative covariance trace bias 0.822, 0.788 and 0.715
against a limit of 0.10). More training alone does not rescue a failed model.

**Mechanics.** A counted test may not depend on an uncounted one, so each 14k test is held with `blocked_by` naming
its held parent.

**Re-entry criterion (each).** It returns in the revision that restores its 7k parent, once the global recipe has an
ordinary parent PASS with its own saved state. That revision also restores `clockfree_audit` to the count as the
gate.

<a name="ring_hold"></a>
<a name="ring_extension"></a>

### 16-17. Ring endurance: `ring_extension` stays counted (KEEP), `ring_hold` REMOVE (counted inside it)

**What the run is.** An 8-blob ring is checked after every update from update 1,201 onward. A check is good when all
8 blobs are present and at least 90% of samples lie close to a blob centre (HQ at least 0.90). The run has three
parts:

1. **Acquisition:** by update 6,000, find a first window of 200 good checks in a row.
2. **Hold:** stay good for the next 1,200 checks, with no bad check allowed. `ring_hold` grades parts 1-2.
3. **Extension:** stay good for the next 300 checks. `ring_extension` grades parts 1-3 on the same uninterrupted run,
   because its grader replays the acquisition and hold first.

**Is the test sound? Yes.**

- **No false failures:** a perfect ring sampler scores HQ 0.989, 54 standard errors above the bar, so false failures
  over 1,500 checks are negligible.
- **BCAP's dips are real:** they are about nine times larger than counting noise alone.
- **It is solvable:** BISPF passes the whole run in a saved Torch 2.13 diagnostic (producer `bf5d13ff`). It is good
  from the first check, holds 1,200/1,200 (lowest HQ 0.993) and passes 300/300 extension checks (lowest HQ 0.992).
  The extension was certified from saved data in commit `224958a3`, with no retraining.

**Why the selected board shows zero. No defect lies behind it, and it will not clear by itself.**

- **BISPF:** Tier 3 is never reached in its ordinary cohort, because of Tier 2 failures that are now held. This is a
  tier-prerequisite zero.
- **BCAP:** the failure is real. It fails `mode_hold` (the gate) and loses the hold at update 3,760 (HQ 0.897).
- **K3P, KA2, R1/R2 and GAN v3:** never show all 8 blobs.
- **Duplicate records:** five research FAIL records have identical saved curves to BCAP's run (curve hash
  `248878f828c5`; first hold failure at update 3,760), so they are not independent evidence.

**What changes: counted once instead of twice, every requirement kept.** One run used to count twice in four views,
and a failed hold counted as two non-passes (the hold's FAIL plus a BLOCKED extension).

- **`ring_extension` stays counted** as the single ring endurance verdict. This keeps all 300 extension checks in
  the count. The handoff proposed the opposite assignment (stop counting `ring_extension` separately and keep
  `ring_hold`), with the goal of folding the 300-check requirement into one endurance score. This decision keeps that
  goal, one endurance score that keeps all 300 checks, but counts `ring_extension` instead, for the reasons below.
- **`ring_hold` leaves the count** (`removal.reason = phase`, `counted_by = ring_extension`). It stays assigned in the
  same tier and the same uninterrupted group, so the planner still runs one ring job. A `ring_extension` PASS implies
  a `ring_hold` PASS on the same run, so no gate is weakened.

**Why not the other options.**

- **Counting `ring_hold` and removing `ring_extension`** would drop the 300 extension checks from the count. A recipe
  that held but failed the extension would then pass Tier 3, which weakens the gate.
- **A new combined task** would change the run's identity and orphan BISPF's saved evidence.

Count change: -4 (`ring_hold` in four views). Gate change: none, so no new task version. The narrow phase exception
(see [Rules in the views](#rules-in-the-views)) lets the counted `ring_extension` keep its checkpoint dependency on
the removed `ring_hold`.

**Who passes it, and what they break elsewhere.** BISPF passes, but only in a diagnostic run. Elsewhere BISPF fails
`gaussian1d_stability`, `target_shift_recovery` and the ten density, image and 100-blob tests (all now held), and
passes every other counted Tier 1 and Tier 2 test. This is not a HOLD: after the other holds, the leading candidate
already passes it, so nothing conflicts with counting it.

This change does not yet show the ring phases on the board, nor label a `ring_extension` cell whose hold phase failed
as "FAIL (hold phase)"; both are [follow-ups](#follow-ups).

<a name="target_shift_recovery"></a>

### 18. `target_shift_recovery`: HOLD (unsolved)

**What it asks.** Learn the 8-blob ring and keep it: 5 checks over updates 1,000-1,200, then 120 checks up to 2,400.
At update 2,400 the ring slides 1 unit sideways. From update 2,800, 400 updates later, all 81 checks up to 3,600 must
be good. A frozen copy that stops learning at the slide must fail all 81, which proves the slide is big enough to
notice.

**What fails.**

- **BISPF:** passes the first two parts (5/5 and 120/120) and does relearn the moved ring, but late. Its first good
  check comes 180 updates after the slide, and it is steady only from update 3,020 (66/81; the deadline is 2,800).
- **BCAP:** already wobbles before the slide (4/5 and 116/120), and keeps dipping until 3,550 (66/81).
- **K3P, KA2 and R1/R2:** never learn the ring.
- **GAN v3:** learns only the moved ring (47/81).
- **The frozen copy:** fails 81/81 in every run, so the test works.

**Why it is hard.** The 400-update deadline is shorter than these recipes' own learning time: BISPF needs 720 updates
to learn the ring from scratch and 620 to recover after the slide. Recipes that adapt fast also tend to wobble when
they should hold still, which is the same conflict as in the Gaussian test.

**Tried and failed.** The selected recipes of BCAP, BISPF, K3P, KA2, R1/R2 and GAN v3, plus historical scheduled K3P
and k3/k3g/rg5 learning-rate-floor runs. No recipe, current or historical, has ever passed it.

**Who passes it alone.** Nobody. BISPF meets 3 of the 4 parts.

**Mechanics.** In adaptation it stays at Tier 3 as a held test; a tier may hold only held tests. While it is held,
adaptation counts exactly the same Tier 1 and Tier 2 tests as quality_coverage and reports HELD, and the board says
so.

**Re-entry criterion.** Count it again (in adaptation and clockfree_continuous) when one global recipe passes all four
parts in one paired run in one cohort: 5/5, 120/120, 81/81 after the deadline, and a frozen copy at 0/81. The same
recipe must also pass every counted Tier 1 and Tier 2 test and ring endurance.

No gate change. Whether 400 updates is the intended adaptation speed is listed under [Unresolved](#unresolved).

## The 7 Atlas/E22 policy questions: decided; applied in a follow-up change

These questions live in `tier1_policy_coverage`, a separate scope (`reporting.family_totals: false`) that never fed
the 152. Only Atlas and E22, run in policy mode, may run them. This change records the decisions in
[dispositions.json](dispositions.json) with `"applied": false` and leaves the view at revision 1, because revising it
first needs updates to three historical-replay test areas (the private-checkout helper that rewrites the policy
declarations, the tier-1 completion drift lookup and the tier-1 completion round's refusal message) and to the scoped
publication code. The follow-up revision will keep all seven assignments, so their receipts stay visible, and will
have no counted requirement.

**Stale declarations.** All seven committed v1 declarations are refused when validated from develop: their pinned
parent or source changed after they were written. The October runs used regenerated declarations under the same ids.
The receipts carry exact hashes, so no grade is wrong. Any future counted rerun must use a new v2 id.

The three holds rest on the recorded numerical failures of the policy recipes: no policy recipe has passed them.

<a name="ring16_acquisition_tier1_policy_selected_cloud_v1"></a>

### `ring16_acquisition_tier1_policy_selected_cloud_v1`: HOLD (unsolved)

**What fails.** Atlas/E22 find all 16 ring clusters with balanced mass, but the samples leak into the gaps.

- **Cluster-shape error:** 1.14 (E22) and 1.81 (Atlas) against 0.85.
- **Sample quality:** Atlas also has HQ 0.81 against 0.85.
- **Checks:** 0 of 96 pass in the 1,600-update version. In the original 400-update version the error is 4.39.

**Why it is hard.** Each cluster is built from about 16 fixed points that must be spread exactly right while staying
out of the gaps.

**Who passes it alone.** No policy recipe. The ordinary `ring16_acquisition` is passed by eight BCAP/BISPF recipes,
which cannot run in policy mode.

**Re-entry criterion.** Under a new v2 declaration bound to the current 1,600-update parent, one policy recipe passes
it on 5 final checks of all six bounds. The same recipe must be counted for every policy question.

<a name="two_pole_tier1_policy_selected_cloud_v1"></a>

### `two_pole_tier1_policy_selected_cloud_v1`: HOLD (unsolved)

**What fails.** 12 points start at zero and must move toward the data at -1 and +1 (mean distance at least 0.3)
within 80 updates. They reach 0.0026; 0 of 24 checks pass. The critic's slope at the points is tiny (0.0105).

**Why it is hard.** The policy's own critic smoothing and step-size controller appear to damp exactly this push.

**Who passes it alone.** No policy recipe. Twelve ordinary recipes pass `two_pole`, and none can run in policy mode.

**Re-entry criterion.** Under a v2 declaration with a complete receipt (including the prior-policy proof), one policy
recipe reaches 0.3 on 5 final checks, and the same recipe is counted for every policy question.

<a name="clockfree_audit_tier1_policy_selected_cloud_v1"></a>

### `clockfree_audit_tier1_policy_selected_cloud_v1`: HOLD (unsolved)

**What fails.** Changing the planned length or the evaluation cadence leaves the learned state identical. Changing
only the step counter changes it, and so does a save and restore. The source audit names the clocks:

- a penalty switch at call 800;
- a critic guard released after a fixed number of updates;
- the policy controller itself, which was never clock-audited.

**Why it is hard.** Passing means removing those counters from the recipe. It is a property of the recipe, not a
measurement problem.

**Who passes it alone.** No policy recipe. BCAP and BISPF pass the ordinary clock audit, but they cannot run in policy
mode.

**Re-entry criterion.** One policy recipe keeps an identical state under all four conditions with no unexplained
clock dependency, under a v2 declaration placed at Tier 3, as in the main suite.

<a name="gaussian1d_acquisition_tier1_policy_selected_cloud_v1"></a>

### `gaussian1d_acquisition_tier1_policy_selected_cloud_v1`: REMOVE (superseded)

**What happened.** Atlas/E22 reach the bell curve fast (first good check at update 84) and are good on 15 of 24
checks, including the last 3. They never manage 5 in a row; their longest run is 4.

**Why remove it.** The main suite deliberately retired this combined "reach it and keep it" Tier 1 definition. It
split the question into `gaussian1d_smoke` (Tier 1: reach it) and `gaussian1d_stability` (Tier 2: keep it).

- Its parent definition is in no current view.
- Its committed declaration is refused from develop.
- As a counted Tier 1 requirement, it would grade the policy recipes by a stricter Tier 1 rule than any other recipe
  faces.

All receipts and the archived revision 1 stay. If a policy-mode Gaussian question is wanted again, declare new policy
copies of the current smoke and stability tests under new ids.

<a name="unused_token_hold_tier1_policy_selected_cloud_v1"></a>
<a name="ae_gan_hold_tier1_policy_selected_cloud_v1"></a>
<a name="five_word_joint_acquisition_tier1_policy_selected_cloud_v1"></a>

### `unused_token_hold_...`, `ae_gan_hold_...`, `five_word_joint_acquisition_...`: REMOVE (ill_posed)

The code refuses all three before training, for every recipe (`tier1_policy.BLOCKED_HOSTS`), so as counted tests they
can only ever score zero. Policy mode may copy, reset or re-create rows, which breaks each host's law:

- **unused-token:** a shared vector moves every slot;
- **AE/GAN:** the encoder routes inputs to specific prior points (routed rows);
- **five-word:** exactly five fixed word codes.

A runnable version would change row ownership, serving or capacity, which makes it a new test with a new id.
`five_word`'s parent was also retired in favour of `five_word_joint_smoke` and `five_word_joint_hold`. The ordinary
`unused_token_hold`, `ae_gan_hold` and `five_word_joint_smoke` stay counted in the main views; they are passed by 12,
14 and 7 recipes respectively.

<a name="indexed-variants"></a>

## The 17 indexed variants: NO_CHANGE

All 17 appear only with the role `diagnostic`, and none has ever been counted. None has a selected-cohort or
attempt-index result; 25 research results exist for three published variants, all research scope:
`vector_anisotropic_published` (4 of 10 PASS), `vector_overlap_published` (9 of 10) and
`vector_unequal_width_published` (5 of 5), listed in [test-performance.md](../test-performance.md). An empty cell
means "not measured", not "failed". They sit in:

- formulation_comparison, Tier 2 (15 of them);
- host_profile_transfer, Tier 2 (13 of them);
- k3p_two_pole_horizon (the 2 two-pole schedule tasks).

The variants are:

- `grid100_affine_paired_laws_v1`, `grid100_affine_square_named_v1`, `grid100_release07_cloud_named_v1`,
  `rotated100_affine_square_named_v1`, `staggered100_affine_square_named_v1`;
- `img_bars4_residual16`, `img_blobs4_residual16`, `img_intensity2_residual16`, `img_stripes2_residual16`;
- `two_pole_800_schedule80_diagnostic_v1`, `two_pole_800_schedule800_diagnostic_v1`;
- `vector_anisotropic_published`, `vector_overlap_published`, `vector_spiral_published`,
  `vector_two_broad_published`, `vector_unequal_mass_published`, `vector_unequal_width_published`.

<a name="pytest"></a>

## Pre-existing pytest failures

All failures below already happened on the unmodified `develop` checkout (`224958a3`). The fixes are in commit
`579573b0` (`test: repair pre-existing Forge test failures`). Baselines are exact test ids: the CI suite (`tests/`,
without the late-phase renderer file) had 5,719 passed, 143 failed, 235 errors and 355 skipped; the report-local files
(`reports/forge/test_prepare_inventory_family_pins.py`,
`reports/forge/recipe-prior-refactor/{test_workflow,test_publication_migration,test_publication_roster}.py`,
`reports/forge/test_collect_gaussian_smoke_inventory.py`) had 79 passed, 12 failed and 9 errors. After that commit,
in a shallow clone (only the newest commits): CI suite 5,955 passed, 24 failed, 115 errors and 358 skipped (240 tests
newly pass or skip with a named reason); report-local files 81 passed, 10 failed and 9 errors, none newly failing.
The history-dependent tests changed in that commit were also run with those commits supplied read-only, and passed;
other history-only tests that still fail with the old commits present are listed below. In the final state of this
change, run in the same checkout after the old commits had become readable in it: CI suite 6,138 passed, 44 failed,
34 errors and 346 skipped, with none newly failing against `develop`; report-local files 97 passed and 4 failed.
Every remaining failure belongs to a group in the table below (`test_coulomb_prior` failed in that run and passes
when rerun alone).

- **Shallow clone**: a copy of the repository that holds only the newest commits. Tests that read an older commit
  cannot run in it. CI checks out the full history (`fetch-depth: 0`).
- **Fixture**: a small synthetic repository a test builds in a temporary folder.
- **Display index**: `reports/forge/test-performance-evidence/index.json`, the list of compact research packets that
  the leaderboard pages may show. It is optional for the page generators.
- **Backfill**: research-row cells that were never run are filled from later attempts of the same recipe and source
  (`reports/forge/attempt-coverage.json`); recorded verdicts are never overwritten.
- **Stale expectation**: a test value that was correct when written but no longer matches a deliberate, committed
  change. Such a test is updated only to the value that committed change intends, with the evidence named.

Dispositions: FIXED, SKIP_WITH_REASON (skips with a message naming the missing commit), OUT_OF_SCOPE (cause recorded
only) and LEFT_FAILING (cause and proposed fix recorded).

| Failing group (baseline count) | Disposition | Reason, in plain language |
| --- | --- | --- |
| Experiment-memory compiler: `test_forge_promotion` (40 errors), `test_forge_calibration_lane` (36), `test_forge_calibration_imports` (23), `test_forge_calibration_feasibility` (12), `test_forge_calibration_diagnostics` (9), `test_forge_knowledge` (5 failed), `test_forge_lifecycle` (1), `test_forge_research_diagnostics` (1): 127 in all | FIXED (production bug) | `experiments/forge/knowledge.py` installed caching helpers that used `deepcopy` without importing it and passed one argument too many to the attempt reader, so every full memory compile crashed (readout, promotion and calibration commands). Added the import and fixed the call. All 208 tests in these files pass. The leaderboard's lighter `--summaries-only` compile never used this code. |
| Missing display index in fixtures: 99 of the 107 `test_forge_current_technique_inventory` failures and both `test_forge_tier1_publication` failures | FIXED (production guard) | `experiments/forge/pr399_scores.py` `publications()` crashed when the display index was absent, unlike the other optional display inputs (`test_performance._add_supplemental`, `test_performance_render.render_discovery_navigation`, `research_tiers._confidence_rows`), which show nothing. It now returns no packets in that case. A present index that lacks a pinned packet still raises "missing from display index" (`tests/test_pr399_scores.py` checks this). This uncovered the masked groups below. |
| `_current_family_roster` check: `test_committed_atlas_progress_display_matches_its_generator_without_raw_evidence`, `test_one_visible_table_keeps_solution_families_with_view_breakdowns_and_separate_evidence`, `test_original_pr223_score_uses_verified_full_original_law_not_changed_c6_cells` (3) | FIXED (stale expectation) | The check required every declaration-only family to have a current row. The Halloween family is hidden (`inventory_visible: false` in `configs/forge/trainer-families.json`) and is now only in `historical_selections` of `configs/forge/selections/family-current-v1.json`; `trainer_families.current_family_candidates` skips hidden families by design, and the publication keeps Halloween in `historical_family_rows`. The check now exempts hidden families and requires them to keep their historical row instead. One of the three still fails for separate reasons (row below). |
| Old selected-table header: `test_completed_studies_cannot_replace_selected_family_configs_or_history`, `test_atlas_progress_is_source_bound_additive_display_and_preserves_every_ordinary_row` (masked), `test_forge_passive_publications` (1), `test_source_bound_board` (1) | FIXED (stale expectation) | The leaderboard's family table header changed from "Family / view, Tier 1, Tier 2, Tier 3, Total" to "Family / selected recipe, Tier 1 requirements, ..., Total view requirements" (`family_reports._table_header`, present at `17df086e`). The tests still require exactly one such table. `test_forge_passive_publications` also counted visible families from the pins only; BCAP is now shown through its DualNorm configuration pin (`current_configuration_family: bcap-dualnorm`), so the count follows that rule (8 rows). |
| "(*)" incomplete marker and research-row layout: `test_default_cli_refreshes_later_view_without_advancing_recorded_qualification`, `test_new_card_is_discovered_with_truthful_unknown_or_blocked_status[True]` (masked), `test_source_bound_board`, `test_confidence_coherence_scores` (1, StopIteration), `test_pr399_scores` (1), `test_forge_scored_research_integration` (2) | FIXED (stale expectation) | Counts no longer use a bare "(*)"; they name what is missing, for example "0/4 (3 not run)" or "1/6 (5 incompatible)" (`family_reports.score`; at `5737ade4` it printed "(*)"). Research rows moved from "↳ †" lines under the family table to their own "Research recipes and separate diagnostics" table, and seed-only studies moved to `research_tiers.historical_seed_rows`. Same-source backfill filled unrun research cells (published with the attempt-coverage pull requests #420, #422 and #423), so for example Confidence LR is now 15/29 instead of 7(*)/29 and the PR399 candidate 12/29 instead of 4(*)/29. Each new value equals the committed page, and for the PR399 rows it equals the 29 attempts recorded per row in `reports/forge/attempt-coverage.json`. The caveat "not distinct training runs" is now "including repeated tests". |
| "19/19" on the main board: `test_main_policy_rows_link_separate_history_and_failed_hold_without_credit` (1), and `test_original_pr223_score_uses_verified_full_original_law_not_changed_c6_cells` once its roster check passed | FIXED (stale expectation; judgment call) | `family_reports.render_leaderboard` deliberately adds one note when `reports/forge/actual-missing-diagnostics-20261014/` exists: "Atlas's original 19/19 CUDA passes remain valid under their original task and sampling contracts." The tests forbid "19/19" anywhere on the board. They now allow exactly that scoped sentence and still forbid any other "19/19", so a historical score can still never appear as a board score. If the owners prefer the stricter rule, the alternative is to drop that sentence from the renderer. |
| `test_cli_has_one_default_destination` (masked) | FIXED (stale expectation) | The command line deliberately gained `--output-prefix` for independent reports (help text "write an independent source report; do not register or change current selections"; guarded by `reports/forge/recipe-prior-refactor/test_publication_roster.py::test_independent_report_cli_cannot_mutate_current_policy`). The test now checks that an output prefix cannot refresh the current board and leaves its files unchanged, instead of expecting the option to be unknown. |
| `test_original_completion_cohorts_rebuild_every_scientific_row_without_raw_logs` (1, hash mismatch) | FIXED (stale replay; needs full history) | The test rebuilds the registry of scoped publications as recorded by the tier-1 completion round. A later cohort (`atlas-compatible-coverage-20261013`) was registered afterwards, so the rebuilt registry now keeps only the round's own cohort; its hash then equals the recorded `3bd16ed0...` exactly. The rest of the test reads commits `9c4fcd86` and `928b485f`, so in a shallow clone it still errors at `git rev-parse`; with those commits supplied read-only it passes. CI has full history. |
| Registration path renders before writing its new snapshot: `test_explicit_frozen_regrade_registers_evidence_and_updates_canonical`, `test_source_registration_preserves_interrupted_receipt_with_missing_worker_time`, `test_explicit_policy_advance_preserves_original_cohorts_without_new_credit`, `test_policy_advance_rebinds_same_selected_recipes_and_archives_exact_card[True/False]`, `test_archived_policy_corruption_cannot_overwrite_current_publication`, `test_same_source_cpu_registration_retains_cuda_evidence`, `test_unmeasured_other_backend_cannot_overwrite_measured_registration` (8; 3 visible at baseline, 5 masked) | LEFT_FAILING (production bug, not in the decided scope) | When `publish_current` registers new evidence (`--source-commit`), it validates and renders all pages before writing the new content-addressed snapshot (`reports/forge/technique-evidence/<sha256>.json`), by design (see the comment in `family_reports.render_family`). The test-performance page builder (`test_performance.build_report`, called from `_family_pages`) reads that snapshot from disk, so registration fails with "file not found" before writing anything. `reports/forge/pure-bcap/publish.py` has the same order. Proposed fix: pass the pending snapshot bytes from `publish_current` (and the pure-BCAP composer) through `_family_pages` and `generated_reports` into `build_report`, which preloads them into its `Inputs` cache; existing callers are unchanged. After that, `test_explicit_policy_advance_preserves_original_cohorts_without_new_credit` also needs its stale "0(*)/5" updated. |
| `test_one_visible_table_keeps_solution_families_with_view_breakdowns_and_separate_evidence` (after the roster fix) | LEFT_FAILING (needs an owner decision) | It now stops at the `"tensorflow"` clause of its board assertion, and then at `"halloween"`: the hidden Halloween family and its `tag-tensorflow-v1-adam` tag appear on the main board as a "(historical cohort)" in the tag directory, because `_tag_directory` lists every historical family regardless of `inventory_visible`. Either historical listings should honour the hidden flag (renderer change) or the test should accept historical listings. The test also still assumes the old one-table layout ("\| ↳ [" view rows and "families/" in every table line); those parts should be rewritten together with that decision. |
| `test_forge_views.py::test_frozen_thresholds_reference_the_declared_gate_policy_and_current_scorer` (1) | FIXED (stale expectation) plus SKIP_WITH_REASON for its history part | PR #399 deliberately rebound the source pins of 7 of the 30 frozen legacy control cards (`evaluator_revision` `pr399-host-device-replay-source-only-v1`), and `tests/test_legacy_host_cuda_harness.py` requires exactly 27 such rebindings. The test now accepts exactly that one declared source-only layer over the frozen card and still checks everything else. Its comparison with commit `03466efa` skips with a message naming that commit when it is absent; with the commit supplied it passes. |
| `test_forge_family_reports.py`: `test_current_clock_measurement_retains_original_evidence_alongside_policy_advance`, `test_current_measurement_families_complete_only_their_declared_view_scope` (2) | SKIP_WITH_REASON | Both rebuild the board published at commit `5737ade4`, absent from a shallow clone; the shared helper in `tests/archived_forge_contracts.py` now skips with a message naming that commit and no longer reuses a half-built mirror (that reuse caused the misleading "file not found" in the second test). With the commit supplied, both pass. |
| `test_forge_atlas_evidence_navigation.py` (1, `NameError: _incompatible`) | FIXED (test-harness defect) | The test copies selected functions out of `family_reports.py`; `_count` now calls `_incompatible`, which the copy list lacked. Added it. |
| `test_forge_test_performance.py` (1, 309 vs 267) | FIXED (stale expectation) | The display index gained PR399's four retained packets (21 + 3 + 12 + 6 = 42 attempts, pinned in `experiments/forge/pr399_scores.py`; `tests/test_pr399_scores.py` requires all 42). 267 + 42 = 309. |
| `test_pure_bcap_publication.py` (2) | FIXED (stale expectation) | The BCAP initial Adam study link moved from the main board to the BCAP family page (`family_reports.render_family` calls the pure-BCAP `display_section`; at `5737ade4` the main board appended it). The tests now check that the main board no longer carries it and that the family-page link stays bound to its receipt, including the "navigation binding" refusal. |
| `test_forge_dualnorm_smoothing_search.py` (1) | FIXED (stale expectation) | Model families such as `gan` now inherit the DualNorm BCAP trainer (`DEFAULT_RECIPE = "bcap"`, smoothing 0.001), so `get_recipe("gan", optimizer_family="formulation")` is refused. The plain formulation recipe is now `get_recipe("gan", trainer="ka2")`, the same migration other tests use (`tests/test_adversarial_losses.py`, `tests/test_bcap_default_baseline.py`). |
| `reports/forge/recipe-prior-refactor/test_workflow.py`: "Selected roster changed" (2) | FIXED (test-harness defect) | The two tests declared the campaign's controls against the live selection. The campaign's own published pins (`recipe-prior-refactor-*-v1`) and BISPF later replaced the seven leaders recorded in `baseline.json`, and `declarations_for` deliberately refuses a changed roster. The tests now use the recorded leaders and separately check the refusal; 42 declarations, 7 studies and exact aliases are unchanged. |
| `test_forge_publication_memory.py` freshness (1) | FIXED by regeneration in this change | The compiled memory pinned the old `knowledge.py` hash and listed two study readouts, `ring16-duration-20261011-readout` and `ring16-schedule-20261011-readout`, whose record and evidence files were never committed, so a clean compile cannot reproduce them (their text remains in the memory at commit `224958a3`). The memory is recompiled with `compile --summaries-only` after the view revisions. If those studies should stay in memory, their files must be recovered and committed. |
| `test_archived_forge_contracts.py` published-view replay (1) | FIXED in this change | It compared the archived view fixture with the live publication's input hashes, which differ for every revised view. It now compares the fixture with `family_progress.input_hashes` recorded at commit `5737ade4`, and skips with a reason naming that commit when it is absent. The fixture keeps the original bytes of each revised view. |
| History-only, failing on purpose without old commits: `test_bcap_integration_compatibility` (63 errors, needs `5737ade4`), `test_bcap_default_baseline` (18 errors, `git archive 5737ade4`), `test_recipe_prior_task_migration` (1, `03466efa`), `test_critic_balance_runner`, `test_generator_step_runner`, `test_routed_generator_batch_identity` (1 each, `git update-ref` to absent commits), report-local `test_publication_roster` (7, `a0493ca3`), `test_publication_migration` (9 errors, `a0493ca3`), `test_workflow::test_baseline_is_actual_committed_develop_bytes` (1, `03466efa`) | OUT_OF_SCOPE | With the old commits available, `test_bcap_default_baseline` (18), `test_publication_roster` (7), `test_baseline_is_actual_committed_develop_bytes`, 39 of the 63 `test_bcap_integration_compatibility` tests and 7 of the 9 `test_publication_migration` tests pass. Still failing then, for reasons unrelated to the view revisions: 24 `test_bcap_integration_compatibility::test_inactive_original_component_hosts_are_bitwise_develop` cases (the applied recipes now carry `prior_coulomb_*` fields that commit `5737ade4` did not have), `test_recipe_prior_task_migration` (a `grid100_14k` task-card difference from `03466efa`; no task declaration changed in this change), two `test_publication_migration` tests (next row) and the three `git update-ref` tests. Those three fail here only because this checkout is itself shallow: `git clone --shared` of a shallow repository does not share its object store, so each test's private clone lacks the old commit even when this checkout has it. They are expected to pass in the full-history CI clone. |
| Report-local `test_publication_migration`: `test_ordinary_whole_rows_preserve_original_card_and_history_without_ranking`, `test_archived_presentation_restores_guard_and_cannot_hide_new_measurement` (2 of its 9 history errors) | OUT_OF_SCOPE (found once full history was available) | With commit `a0493ca3` present they stop at the selection roster check (`trainer_families.current_family_candidates`, reached from `publication_migration.advance_selection`): the campaign's historical selection card has no pin for BISPF, a family added later. This is independent of the view revisions. Past that check, `advance_selection` also requires the live discriminator_stability view to be revision 9 ("current view changed"), so a fix should also decide whether it reads the archived revision 9, as `workflow.ordinary_contract` now does. |
| Report-local `test_publication_migration`: `test_final_registration_and_actual_initial_laws_are_bound`, `test_changed_registration_source_roster_or_initial_law_is_rejected[initial_law]` (2) | OUT_OF_SCOPE (cause differs from the listed one) | These are not history failures. `workflow.resolve_plans` rebuilds the campaign's requests from the live checkout and compares them with the frozen registration. It now refuses first with "Use the explicit revision-9 recipe-owned ordinary view", because planning reads the live discriminator_stability view, which is revision 11. Past that guard the live source digest, candidate revision, study admission and task keys have all changed since the campaign, so they fail even with full history. |
| Library behaviour outside Forge: `test_coulomb_prior` (pair count 16, expected 8; intermittent, it passes in some runs), `test_r1_rotation` (plain optimizers now require disabled guard, anchor, latent damping and direct particle gain), `test_particle_native_2d` (1 error) and `test_toy_api_diagnostics` (2) ("direction blend step requires protected-loss backward"), `test_bcap_compatibility_adapter` (`SimpleNamespace` has no `DEVELOP`) | OUT_OF_SCOPE | Recorded only; these follow recipe/library API changes outside the Forge reporting code. |
| Missing PyYAML: 33 test files fail to import, plus `test_mog_api` (2) | OUT_OF_SCOPE | The shared Python runtime has no PyYAML and must not be changed. |

The committed-page freshness guard
(`tests/test_forge_family_reports.py::test_committed_pages_and_every_drilldown_link_match_the_generator`) failed after
`579573b0` only because the committed publication pins the hash of the edited `experiments/forge/pr399_scores.py`;
the `--refresh-publication` regeneration in this change updates that pin.

### Tests updated for the view revisions in this change

These tests pinned the old views or the old leaderboard text; each now pins the decided result.

- `tests/test_forge_views.py`: discriminator_stability revision 11 (counted 6/10/1, held 0/11/0, `ring_hold` a removed
  phase; the archived revision 9 still counts 6/21/2) and clockfree_continuous revision 4 (`clockfree_audit` a
  removed duplicate; the three 14k tests and `target_shift_recovery` held; the archived revision 3 still counts
  `clockfree_audit`).
- `tests/fixtures/archived-forge-contracts.json`: the five revised views it pinned by their live bytes now have their
  original bytes stored in the fixture, with the same sha256 and Git blob. `tests/test_archived_forge_contracts.py`
  compares the fixture with the input hashes of the publication at commit `5737ade4` and skips with a named reason
  when that commit is absent.
- `reports/forge/recipe-prior-refactor/workflow.py`: the campaign's revision-9 contract reads the archived
  `configs/forge/view-history/discriminator_stability-v9.json` instead of requiring the live view to be revision 9.
  Its planning paths (`resolve_plans` and an installing `prepare`), which resolve the live view, still refuse any
  later revision with the revision-9 message, so plans can never be built from revision 11 and recorded as revision 9.
- `tests/test_forge_current_technique_inventory.py`: the check that keeps an old Atlas "7/8" progress score off the
  board now matches only a standalone "7/8", because BISPF's new total "77/81" contains those characters.
- The tier-report count dictionaries gained `"held": 0` (commit `856db97f`).
- New in `tests/test_forge_dispositions.py`: every hold and removal note in every declared view comes verbatim from
  one applied record of [dispositions.json](dispositions.json), `test-dispositions check` reports no drift, and every
  explanation link resolves to an explicit anchor in this README.
- After review: the published-view replay test runs its checks that need no history before it skips for the
  missing commit `5737ade4`; `tests/test_forge_attempt_coverage.py` expects "1 diagnostic attempt" (singular, as "1
  error" and "1 timeout" already were); `reports/forge/recipe-prior-refactor/test_workflow.py` checks that the
  campaign's planning refuses the live revision 11 with the revision-9 message.

<a name="calibration"></a>

## The vector calibration

[calibration/calibrate_vector_gates.py](calibration/calibrate_vector_gates.py) is a declared measurement calibration:
CPU-only Monte Carlo draws from each task's own true target law, scored by the unchanged scorer and graded by Forge's
own grader. No training, generator or recipe is involved. It ran against the `224958a3` sources (the scorer hashes
are recorded in the summary). [calibration-full-summary.json](calibration/calibration-full-summary.json) holds every
panel's record (its `panels` list), pass rates and metric quantiles. While it runs, the script also writes the same
panel records as a tail-able `calibration-<full|quick>-panels.jsonl` log; like other JSONL traces, that log and the
25,600 per-check rows (about 30 MB) stay outside Git.

Rerun it from the repository root, CPU only (about 5 minutes; `--quick` runs a 4-repetition smoke test):

```sh
CUDA_VISIBLE_DEVICES= python reports/forge/test-dispositions-20261015/calibration/calibrate_vector_gates.py
```

It rewrites the summary and the panels log next to the script (or in `--output-dir`). Every draw uses a named seed, so
the pass rates and quantiles reproduce exactly; only the timing fields and the recorded source hashes of edited files
change. `--observations <path>` also writes the per-check rows; keep that file outside Git.

<a name="unresolved"></a>

## Unresolved (needs the owner's scientific input or training)

1. **`target_shift_recovery`'s 400-update deadline** (`RECOVERY_DEADLINE`) is undocumented. Is it the intended
   adaptation speed? BISPF needs 620 updates. Any change would be a new task version.
2. **May a global recipe use a cell-based objective** (cell-moment aggregate or cell-quota) under the portability
   rule? These objectives build nearest-neighbour cells around real samples in data space. They are the cheapest
   re-entry probes for the vector tests and for `gaussian1d_stability`.
3. **Should BISPF be re-run** in its own ordinary cohort at the current source, for the counted suite plus ring
   endurance? Its recorded measurement is stale. This needs GPU training and was not done.
4. **No counted task represents the noisy-law 100-blob question** that the Atlas line historically passed. If it
   should be visible, it needs its own uncounted task version.
5. **Two pytest decisions:** whether hidden families (Halloween) may appear in historical listings on the main board,
   and whether the scoped "19/19" sentence stays on the board (see the [pytest ledger](#pytest)).
6. **The policy view remains a separate scope** that can never share the main board's global recipe. Is a counted
   policy score still wanted at all?

<a name="follow-ups"></a>

## Follow-ups (not delivered in this change)

- **Phase rendering** from fields that are already stored: the Gaussian phases (handoff state, 72 stationary checks,
  re-acquisition, 24 shifted-hold checks, frozen control), the four target-shift parts and the ring phases
  (acquisition window, settling failures, first hold failure and its HQ, extension checks and lowest HQ), with
  BISPF's diagnostic ring PASS next to its selected-cohort status and a `ring_extension` cell whose hold phase failed
  labelled "FAIL (hold phase)".
- **14k labels:** replace "NOT RUN / unmeasured declaration" (test-performance page) and "UNKNOWN" (family pages)
  with "BLOCKED: parent <test> FAIL (N refused starts)", linking the nonqualifying rescue diagnostic.
- **"undefined: blob missing":** change BCAP's "unavailable" shape values on `rotated100` and `staggered100` to
  "undefined: one blob received fewer than 2 points (decisive failure)". The verdict is unchanged.
- **TIMEOUT detail:** show timed-out runs with the last update reached and the metric at that point.
- **Oracle witnesses:** register the true-target references (100-blob tests) and the real-picture oracle (image
  tests), replacing "no indexed witness".
- **Per-blob vector display:** per-blob counts, share ratios, flatness, full versus core spread and spill at each
  check, labelled by which sub-gate failed; count bit-identical BCAP replays once (curve hash).
- **Image display:** per-check modes and HQ, particles per real picture, labelled 4-of-5 near misses (verdicts
  unchanged) and the `img_bars4`/`img_stripes2` anti-correlation.
- **Policy view revision:** apply the seven recorded policy decisions as `tier1_policy_coverage` revision 2, after
  updating the policy-view replay tests and the scoped-publication archive lookup. Let the policy family pages show
  the existing FAIL receipts instead of "UNKNOWN".
- **Cross-device coverage tolerance:** `rotated100_14k`'s rescue coverage audit reports INVALID instead of FAIL,
  because one point lies 0.0000078 inside the scoring circle and the GPU and CPU disagree about it. Tolerating that
  needs a new evaluator revision, with old grades kept.
- **Two-pole prior-policy proof:** regenerated two-pole policy runs lack the prior-policy proof field; add it for
  recipe-owned priors in a separate change (future runs only).
- **Registration path:** fix the render-before-write order described in the [pytest ledger](#pytest).
- **Recipe-prior-refactor publication migration:** resolve the BISPF roster check and the revision-9 binding in
  `publication_migration.advance_selection` (see the [pytest ledger](#pytest)).
- **Hold summary wording, in the next view revision:** the notes are fingerprinted view policy, so they were not
  edited here. Scope the "recorded" and "ever" claims to the current declaration (`img_bars4`, `img_blobs4`,
  `img_intensity2`, `vector_anisotropic`, `vector_overlap`, and "the best recipe ever" in the three 100-blob notes,
  which means the best 7,000-update result), and replace `gaussian1d_stability`'s "Neither selected recipe" with
  "Neither BCAP nor BISPF".
- **Per-view goal tables:** `reports/forge/leaderboards/*.md` (linked from the experiments-by-tier report and the
  compiled memory) are rewritten only by a full `python -m experiments.forge compile`, which this change did not run.
  Until then they show the previous view revisions; after it they also show each view's held count beside its
  status.
- **Two study readouts missing from the memory:** recover and commit the records and evidence of
  `ring16-duration-20261011-readout` and `ring16-schedule-20261011-readout` if they should stay in the compiled
  memory (see the [pytest ledger](#pytest)).

## Operations note for merging

The attempt-fill service (`particlegan-attempt-fill.service`) runs from a separate checkout with `--watch --refresh
--completion-hook ops/publish_attempt_coverage.py` and opens generated-report pull requests to develop. When this
change merges, pause the service or update its checkout to the merged develop first; otherwise it may republish a
board built from the old views. The attempt universe itself is unchanged (35 tasks per selected family), because held
and removed assignments are still selected for attempts.

## Committed record

- [dispositions.json](dispositions.json): one record per decision (schema `forge_test_dispositions_v1`). It is the
  input to `test-dispositions check|apply` and provenance; grading never reads it, so view fingerprints never depend
  on it. Each view entry pins its base revision's policy fingerprint (`from_fingerprint`). Records with
  `"applied": false` (the policy questions) are validated but not applied.
- Revised views: adaptation 3, clockfree_continuous 4, discriminator_stability 11, formulation_comparison 2,
  host_profile_transfer 5, quality_coverage 3. Archives of the previous revisions, byte for byte:
  `configs/forge/view-history/adaptation-v2.json`, `clockfree_continuous-v3.json`, `discriminator_stability-v9.json`,
  `formulation_comparison-v1.json`, `host_profile_transfer-v4.json`, `quality_coverage-v2.json`.
- [calibration/](calibration/): the declared vector calibration script and its summary, which includes every panel.
- Code and reports: commit `a748ae22` (held and removed-from-count roles, the HELD status and the archive-first apply
  command), commit `856db97f` (held and removed tests on the leaderboard, family pages and experiments-by-tier
  report), commit `579573b0` (pre-existing test repairs) and the review-repair commit (`fix: address review
  findings`). The regenerated pages come from this change's `--refresh-publication` run.
