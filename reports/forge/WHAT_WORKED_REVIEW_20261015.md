# What actually worked: Forge leaderboard work, 2026-10-02 to 2026-10-15

A read-only review of the work since the Forge leaderboard was centralised (#232, 2026-10-02): about 190 PRs (#219 to #423) and the local experiment directories under `/ml2/hypergan`. Five independent reviewers each covered one area: the BCAP selected recipe, BCAP research recipes, Atlas/E22, the other families plus test and harness changes, and a test-by-test pass matrix built from `attempt-coverage.json`. The key claims were then re-checked against the PRs and the code. No experiments were run for this review.

**Summary.** A handful of trainer changes produced real, matched gains. A large share of the Tier 1 movement came from redefining tests rather than from trainer changes. Most research directions after 10-09 repaired one test and broke another. Every result is a single seed (seed 0), and several margins are thin.

How to read the counts:

- "71/152" counts *view requirements*. A test required by six views counts six times. BCAP's distinct-test record is 6/6 core Tier 1 plus 9/21 Tier 2.
- "Matched" means the same tests, task-owned budget, seed and source. Many leaderboard rows compare different source digests and runtimes. `technique-inventory.json` itself says "No ranking across incompatible source or runtime cohorts."

## 1. What actually worked, ranked

### 1.1 DualNorm with constant rates (BCAP's foundation)

DualNorm is an optimizer that rescales each update to a fixed size. In the 96-configuration search (#355), DualNorm was the only optimizer family with a 6/6 Tier 1 survivor: 4 of 64 DualNorm configurations, against 0 of 20 Adam configurations (best 4/6) and at most 3/6 for every other family (`bcap-tier2-search/README.md`).

Two small additions were required:

| Addition | What it did |
| --- | --- |
| Numerical-rank truncation (#331/#332, now the default) | Ring16 passes in an ordinary uninterrupted run: acquired at update 684, then 47 terminal passes. |
| Tiny smoothing, 1e-5 to 1e-3 (#345/#350) | Restored Gaussian smoke after serial autograd (#339) broke it. The margin is thin: confirmed KS 0.0448 against a 0.05 limit. |

Settings that mattered:

- Momentum hurts. DualNorm momentum of 0.5 or 0.9 gives 2/6 (#303).
- All four survivors used the same rates: G 0.012, D 0.018, prior 0.03 (#310 made these the default; that change alone took Tier 1 from 3/6 to 4/6 via two-pole).
- Loss choice is second-order. Non-saturating against relativistic swaps word hold for mode hold.
- Constant rates are why BCAP and BISPF pass the clock-free audit. Every family with learning-rate annealing (K3P, KA2, R1/R2, GAN v3) fails it.

### 1.2 Direction blend (#361, #367, #377, #383)

This is the cleanest gain of the period. Tier 2 went from 7/21 to 9/21 and no test was lost.

| Test | Before | After |
| --- | --- | --- |
| trajectory, identity error | 0.2399 | 0.000258 |
| residual_student | 0.0610 | 0.000249 |

On tasks where direction blend never fires, the trained tensors are byte-identical to the baseline.

### 1.3 Prior moment tether with a small-prior exemption (the real source of BISPF's lead)

BISPF (#415) scores 77/152 against BCAP's 71/152. That gap is a single test, `mode_hold` (BCAP 3 of the 5 required terminal passes, BISPF 12), counted once in each of six views.

- BCAP with a moment tether at 0.01 (#390) already turned `mode_hold` FAIL into PASS (streak 3 → 7). It also broke word hold (25/25 → 23/25).
- BISPF uses the same strength (`prior_reg=0.01`, `prior_regularizer=sbnp`). Its SBNP regularizer returns zero when the prior has at most `2*k_nn + d` rows (`particlegan/bispf.py:172-176`), which exempts the 5-row word table.
- The likely cause of the gain is therefore the tether plus the exemption, not the Born-Infeld chord regularizer, the Sinkhorn chords or harmonic capture. **No ablation has been run.**
- #415's headline omits that rotated100 precision drops (0.290 → 0.266); grid100 (0.258 → 0.370) and staggered100 (0.300 → 0.342) improve.
- #393 explains the mode_hold-versus-word trade-off. The fixed tether's force relative to the adversarial force grows from 0.011 to 0.075 to 0.341 as the word adversarial signal shrinks. The proposed relative-force limit has not been run.

### 1.4 Longer horizons: many failures are early stops

- **K3P, KA2, R1/R2.** All three pass the full ring16 gate at 6,400 updates with their full schedule (`EXPERIMENT_MEMORY.md`, `ring16-schedule-20261011`). Terminal passes: K3P 66, KA2 47, R1/R2 exactly 5. Their matched 400-horizon baselines all fail. The board stops them at 1,600 updates.
- **Atlas.** Two-pole fails at 80 updates and passes at 320 (#313). Ring16 fails at 400 and passes at 1,600 (#313). The Gaussian still fails at 2,000 (#312).

Annealing plus a longer horizon conflicts with the clock-free requirement, so this is a policy decision as much as a technique question (§5).

### 1.5 Atlas local population control (#223): wins only on its own suite

| Measure (against PR155 E22) | E22 | Atlas |
| --- | --- | --- |
| Toy precision | 71.6% | 96.5% |
| Toy mass TV | 0.285 | 0.052 |
| MNIST embedding Fréchet distance | 1.88 | 0.54 |
| MNIST recall | 71.9% | 84.7% |

- Rotating 100-Gaussian task: Atlas reaches 96.1% early quality where E22 has 88.1%, but E22 wins at the endpoint.
- A faithful replay reproduces 19/19 gates (#266).
- On Forge tests, Atlas and E22 get identical verdicts. The leaderboard's 0/152 is mostly a blanket constructor refusal (`experiments/forge/api.py` `task_policy_blockers`). When the refused tasks are run anyway, they mostly FAIL.

## 2. Changes that moved the counts but are not technique gains

| Change | Effect | Note |
| --- | --- | --- |
| #316: ring16 rebuilt from BCAP's passing run (256 particles, MoG sigma 0.1, 1,600 updates, 96 checks; defaults to the exact BCAP DualNorm config) | Before (per the #289 Tier 1 run, not re-verified here): K3P and KA2 passed ring and BCAP failed. After: BCAP passes; K3P, KA2, R1/R2 and GAN v3 fail. | Swaps the family ranking. This one test now keeps K3P (11/21 Tier 2), KA2 and R1/R2 out of the counted totals. |
| #322: Gaussian split into Tier 1 smoke (any 1 of 24 states plus a confirming draw) and Tier 2 stability | Gaussian 0/9 → 19 PASS / 3 FAIL | Easier Tier 1. Stability has 0 passes in every selected family. |
| #347: word test split into acquisition and hold | Words 2/9 → all five runnable families pass | Easier Tier 1 |
| #339: serial autograd | BCAP Gaussian PASS → FAIL, causally isolated (#342) | Passes depend on floating-point accumulation order. |
| #263 / #280: ring, words and Gaussian moved into required Tier 1 | Lowered scores | Stricter |
| Per-view counting (#413/#419) | One test moves a total by up to 6 | Inflates gaps (BISPF +6 is one test) |
| 900 s word-smoke wall-clock limit on shared GPUs | Many TIMEOUT/INCOMPLETE results | Confidence LR's "+2" over its baseline is exactly these timeouts. |
| Stale-label display bug | BCAP is as stale as BISPF in `technique-inventory.json`, but the main table doesn't show it | The labeler looks up `bcap-dualnorm`; the row links to `bcap-pure`. |

Harness repairs that turned skips into measured FAILs, without adding a pass: #274, #275, #292, #341, #352, #414, #422.

## 3. Clearly did not work

| Direction | PRs / evidence | Result |
| --- | --- | --- |
| Five prior regularizers: proximal spring, Coulomb, SIGReg, moment tether, initial-distribution MMD | #387 to #392 | No net gain. Ring16 covariance is the gate most of them break. |
| Per-step speed limits: hydraulic, Armijo, role balancing, idle generator, bounded secant | #360, #370, #371, #381, #397 to #399 | Raise native precision (0.24 → 0.5-0.66) and Gaussian stationary holds (up to 71/72). Break covariance, ring16 or mode_hold. |
| Lookahead methods: network or raw-gradient optimism, extragradient, evidence tempering | #359, #362, #364, #378 | Consistently regress acquisition and mode_hold |
| Transport and density forces: local-v2, Sinkhorn, confidence transport, tail-moment, prior-only, anisotropic kNN | #358, #369, #376, #379, #380, #382 | Fix vector density tests; break two_pole, unused_token and ring16 |
| 2026-10-15 combination screens (direction + transport, ± whole-G or adversarial-channel history; 4 repair arms) | `combination-training-20261015/TRAINING-DIAGNOSIS.md` | All fail. Only the direction-only baseline passes both two_pole and ring16. |
| Confidence, signed and positive-evidence learning rates | #405 | Test-for-test identical to the matched baseline, apart from timeout artifacts |
| G-history attenuation | PR399 rows | Only family to pass `gaussian1d_stability` (72/72), but loses Tier 1 gates, net -2 |
| About 40 Atlas repairs, 10-08 to 10-11 (clean 100-Gaussian, images, ring) | `atlas-*-task*` dirs | All rejected. The clean 100-Gaussian failure is a training-dynamics problem; capacity is sufficient. |
| 100-mode grids (grid100, rotated100, staggered100) | 84 attempts | 0 passes. Best precision about 0.55 against 0.97; the true target scores 0.989. |
| 2026-10-14/15 backfill | `attempt-coverage.json` | 853 attempts, about 29.5 GPU-hours, 0 selected totals changed. All 130 new passes are on research rows. |

Recurring pattern: **endpoint metrics often pass while the sustained (terminal-suffix) gate fails.** Read partial wins from the gate, not the endpoint.

## 4. Each family owns a capability; no single recipe bridges them

| Capability | Who has it | What it costs |
| --- | --- | --- |
| ring16 acquisition, clock-free audit | DualNorm + constant rates (BCAP, BISPF) | — |
| mode_hold | Moment tether (BISPF); finite critic cap | Finite cap loses Gaussian smoke, stripes and two_broad. |
| Gaussian retention, vector_anisotropic, vector_overlap | Annealing (K3P/KA2); G-history | Clock-free audit, ring16 within 1,600 updates |
| vector_unequal_mass, img_blobs4, img_intensity2 | Kinetic transport and cell-moment recipes | ring16, two_pole, mode_hold |
| img_bars4, img_intensity2 | R1/R2, GAN v3 | img_stripes2 (0 of 21 recipes pass both image tests), ring16 |
| vector_unequal_width | Balanced transport (#375) only | Rare density, native precision |

Coverage of the 29 required tests:

| Combination | Tests passed |
| --- | --- |
| Best single recipe (BISPF) | 18/29 |
| Union of selected recipes | 23/29 |
| Union of all recipes | 25-26/29 |

The binding conflict: **no recipe that passes ring16 passes any held test that someone else passes.**

## 5. Where to dig next

1. **BISPF ablation.** Add only the tether plus the small-prior exemption to plain BCAP, on the current source. It is cheap, and it shows whether 77 needs any of BISPF's other parts.
2. **Relative-force tether.** Cap the tether's force as a fraction of the adversarial force (#393 proposal), aimed at keeping mode_hold without losing word hold.
3. **Horizon and clock policy.** Annealing plus a longer horizon passes ring16, but the 1,600-update cap and the clock-free audit rule it out. Candidates: a schedule driven by training progress rather than step count, or an explicit decision on the ring16 horizon.
4. **cell-quota-min-change.** The only recipe seen passing both `gaussian1d_stability` and `mode_hold`. It has never been run on ring16. The zero-pass audit names it as the cheapest re-entry probe.
5. **Hygiene.** Fix the stale-label bug; separate word-smoke wall-clock timeouts from recipe failures; recheck the thin Gaussian smoke margin.

## Sources

- PR bodies #219 to #423 (`gh pr view`).
- `reports/forge/`:
  - leaderboard data: `technique-inventory.{md,json}`, `attempt-coverage.json`, `test-performance.json`
  - findings: `families/*.md`, `EXPERIMENT_MEMORY.md`, `OPTIMIZER_TYPES.md`
  - BCAP studies: `bcap-tier2-search/`, `bcap-tier2-investigation/`, `bispf/README.md`
- Local studies: `/ml2/hypergan/combination-review-20261014/`, `/ml2/hypergan/combination-training-20261015/`, `/ml2/hypergan/test-dispositions-20261015/RECONCILED.md`, `/ml2/hypergan/atlas-*-task*/`.

Not verified for this review:

- No raw receipts were recomputed.
- Leaderboard-row gains and losses are protocol-matched, not causally matched.
- The PR399 history rows' Tier 1 gaps are largely constructor refusals.
- The K3P "about 81/152 if its Tier 2 counted" figure is an estimate.
