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
  - *Update 2026-10-16: the ablation has now been run and this guess did not hold up (§6.1). On seed 0, the tether alone does not reproduce BISPF's mode_hold pass.*
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

*Updated 2026-10-16 after the follow-up experiments in §6. All five original leads have been run once (seed 0). Four came back REFUTED and the hygiene pass was completed. None of the four trainer leads produced a new pass.*

What is now answered:

1. **BISPF ablation: answered, guess refuted.** The tether alone on BCAP does not pass mode_hold, and BISPF without its tether fails too (§6.1).
2. **Relative-force tether: refuted at rho=0.03.** It lost both mode_hold and ring16. The #393 word-hold question was never reached (§6.2).
3. **Progress-driven schedule for K3P: refuted.** Every K3P schedule fails ring16 inside 1,600 updates, including one that never anneals. The horizon cap is the binding constraint, not the clock rule (§6.3).
4. **cell-quota-min-change: refuted as a Tier 1 re-entry.** It fails ring16 on covariance and cannot run two_pole or five_word at all (§6.4).
5. **Hygiene: done.** Stale-label fix in PR #428. Most TIMEOUT/INCOMPLETE results are wall-clock hits, and three recipes would reach 6/6 Tier 1 if their word-smoke runs had finished. The thin Gaussian margin is general (§6.5).

Next steps, in priority order:

1. **Fix the Forge drain crash first.** Every follow-up drain crashed after grading in `experiments/forge/knowledge.py` (the `compile_memory` caching wrapper): a missing `deepcopy` import, then `_attempts() takes 1 positional argument but 2 were given`. Verdicts were read from each attempt's `graded-result.json`, so they stand, but queue status and leaderboards were never updated. PR #429 carries a fix and an opt-in `FORGE_SKIP_COMPLETION_COMPILE=1` switch, because the repaired compile took about 20 CPU-minutes after a 41 s job.
2. **Re-run the timed-out word-smoke cells under the GPU lease** (a lease gives one job a GPU to itself, so shared-GPU contention cannot slow it). Five runs had already confirmed a pass before the 900 s kill. Report wall-clock hits as NOT MEASURED rather than FAIL. This could move signed Confidence LR, Positive-evidence LR and the matched BCAP baseline to 6/6 Tier 1 without any trainer change.
3. **Add seeds to the thin results before building on them.** Single-seed margins now decide several conclusions: the BCAP control misses mode_hold by one check, the BCAP control's ring16 pass only stabilises from step 1184, and 28% of 180 Gaussian-smoke passes rest on 2 or fewer of 24 checks. Per the standing rule, add seeds only where a verdict depends on a thin margin.
4. **Find what links BISPF's tether to its critic.** The untested candidate is BISPF's harmonic thermal factor, which scales the tether down when d<=4 (mode_hold has d=4) and cannot exist under the BCAP critic. Test: BCAP plus the tether with that same scaling, or BISPF with the scaling switched off.
5. **Make an explicit horizon and audit decision for ring16.** This is a policy choice, not a technique one. The options from §6.3: keep both rules; add a declared 6,400-update ring16 variant (about 72 s per run); or extend the clock-free audit's declaration table to accept schedules driven only by saved internal state. The last option changes a gate and needs reviewer sign-off.
6. **Lower priority.** Export the relative cap's in-training scale before trying any other rho value. Run the DualNorm plus cell objective arm on mode_hold and gaussian1d_stability, to see whether the cell objective alone keeps the held passes.

## 6. Follow-up experiments (2026-10-15/16)

One workflow ran each §5 lead in its own worktree, with a decision rule written down before training. An independent verifier then rechecked each lead against the attempt receipts (`graded-result.json`, `terminal.json`, queue events). Verdicts below are **as corrected by the verifier**. The hygiene lead used only CPU and saved files and had no verifier pass.

Shared conditions: seed 0, one attempt per arm, an RTX A6000, deterministic CUDA, and one frozen source digest per lead. Every verdict comes from the sustained Forge gate. "Suffix" or "streak" is the number of consecutive passing checks at the end of the run, and most gates need at least 5. All results are single-seed: they show what happened on this seed, not how robust it is. No change alters a default, gate or budget, and nothing was merged.

Total GPU time across the four trainer leads: 1,041.6 s (about 17 minutes).

### 6.1 BISPF ablation: REFUTED (holds up)

**Question.** Does BISPF's mode_hold pass over BCAP come from the SBNP prior moment tether alone (`prior_reg=0.01`, off for small tables), rather than from its critic-side parts (Born-Infeld chord penalty, Sinkhorn pairing, harmonic capture)?

**Arms.** C = BCAP control; T = BCAP plus the tether only; B = full BISPF; B0 = BISPF with the tether off. B0 is exploratory: it was declared after the main verdict was known, and the readout says so.

| Arm | mode_hold | Checks passing | Suffix | Notes |
| --- | --- | --- | --- | --- |
| C BCAP control | FAIL | 10/24 | 3 | one dip to hq 0.869 at step 1050; the final check alone would pass |
| T BCAP + tether | FAIL | 0/24 | 0 | stuck at 7 of 8 modes from step 650 |
| B full BISPF | PASS | 13/24 | 12 | 8 modes from step 350 |
| B0 BISPF, tether off | FAIL | 0/24 | 0 | stuck at 7 of 8 modes from step 400 |

**Verdict.** REFUTED under the preregistered rule (committed in dccb6839 before the first attempt): on this seed the tether alone does not reproduce BISPF's pass. Stage 2 was correctly skipped. Verifier corrections:

- The reading that BISPF needs the tether *and* its critic-side parts together rests on one seed per arm, and partly on the post-hoc B0 arm. It is a lead, not a finding. T and B0 both stalling at 7 modes could also mean mode_hold is fragile at seed 0 under any change.
- T is a stronger tether than BISPF's own: BISPF scales the tether down with a harmonic thermal factor when d<=4, and T cannot have that factor.
- The readout overstates B0's hq at steps 400-650. It was 0.81-0.90, not at least 0.90.
- The control fails by a single check.

**GPU time:** 215.5 s. **Links:** branch `task/dig-bispf-ablation-20261015`, draft [PR #426](https://github.com/255BITS/ParticleGAN/pull/426), readout `reports/forge/dig-20261015/bispf-ablation.md` on that branch (local: `/ml2/hypergan/dig-20261015/bispf-ablation/READOUT.md`).

### 6.2 Relative-force tether: REFUTED (holds up)

**Question.** If the 0.01 moment tether's gradient on the latent table is capped at rho=0.03 times the other (mostly adversarial) gradient, does BCAP keep the fixed tether's mode_hold repair, avoid its word-hold regression and keep ring16? The new opt-in field is `prior_tether_relative_cap`. Its default of 0 keeps the old fixed tether exactly. It uses only gradient norms, so it is portable.

**Arms.** C = BCAP leader control; F = fixed 0.01 tether (reproduces #390); R = 0.01 tether capped at rho=0.03.

| Arm | mode_hold | ring16_acquisition |
| --- | --- | --- |
| C control | FAIL, suffix 3, 10/24 | PASS, suffix 26, 26/96, covariance error 0.497 |
| F fixed tether | PASS, suffix 7, 10/24 | PASS, suffix 9, 10/96, covariance error 0.726 |
| R capped tether | FAIL, suffix 0, 0/24, never reaches 8 modes; ends at 6 modes, hq 0.863 | FAIL, suffix 0, 0/96, covariance error 1.343 (limit 0.85) |

Word smoke, word hold and the rest of Tier 1 were not run: rule 1 stops the run after stage 1.

**Verdict.** REFUTED for rho=0.03 on seed 0. R is worse than both C and F, so it is not a middle ground between them. As in #390, these gates react non-monotonically to small changes in prior force. The #393 word-hold question is **still untested**. Verifier corrections:

- How often the cap actually engaged is unmeasured. Its in-training scale is recorded in memory but not exported to any receipt. R leaving F's trajectory around step 250 is unexpected if the cap rarely binds, and no check rules out an implementation artifact. The verifier's code read found no bug, and the 11 unit tests pass.
- The harness story in the readout is partly wrong. The `deepcopy` import was already in the frozen commit 1f89a8241. The logged first-drain crash was the `_attempts` TypeError, not a NameError. The arms still share source digest 120d9cb4.
- R also drops to 6 modes at steps 400 and 1000, not only at the end.

**GPU time:** 262.6 s. **Links:** branch `task/dig-relative-tether-20261015`, draft [PR #429](https://github.com/255BITS/ParticleGAN/pull/429), readout `reports/forge/dig-20261015/relative-tether.md` on that branch (local: `/ml2/hypergan/dig-20261015/relative-tether/READOUT.md`).

### 6.3 Horizon and clock: progress-driven schedule for K3P: REFUTED (holds up)

**Question.** Can a schedule that does not count steps give K3P the benefit annealing has on ring16, passing ring16 within the 1,600-update cap as well as the clock-free audit? The new opt-in `Recipe.progress_schedule` replaces step/horizon with an internal progress value p in K3P's learning-rate decay, input-noise anneal and output-noise warmup. p rises by at most 0.001 per update, and only while the critic's paired order accuracy (an exponential moving average) is below 0.75. p never decreases and is saved with the trainer.

**Arms.** A = shipped K3P step schedule; B = constant rates, no noise anneal; C = progress schedule; D = ungated internal counter (diagnostic). B, C and D also start the critic spike guard at update 0 (A keeps 200), so A against the rest is not a pure schedule comparison.

| Arm | ring16 (0/96 passing for all) | Final covariance error (min) | Clock-free audit |
| --- | --- | --- | --- |
| A step | FAIL; also fails hq (0.81) at the end | 2.950 (2.873) | FAIL: step_label and horizon change the state |
| B constant | FAIL; 39/96 checks pass every gate except covariance | 1.337 (1.285) | PASS, 4/4 comparisons agree |
| C progress | FAIL; 30/96 pass every gate except covariance | 1.787 (1.734) | FAIL by declaration only; 4/4 measured comparisons agree |
| D ungated | FAIL; 47/96 pass every gate except covariance | 1.895 (1.851) | FAIL by declaration only; 4/4 measured comparisons agree |

**Verdict.** REFUTED; Tier 2 skipped by the stop rule. The progress gate behaved as designed: it held p back while the critic was winning (p 0.098 at update 200), then advanced it. The learning rate was close to annealed by update 1,550 (multiplier 0.059 against a floor of 0.01), not fully annealed as the readout says. Annealing inside the cap did not help: the arm that never anneals came closest. **For K3P on ring16, the 1,600-update horizon is the binding constraint, not the clock rule.** C and D fail the audit only because its declaration table lists K3P's step-schedule fields. Changing that table is a gate change and was not made. The readout's 6,400-update policy figures were not verified. Verifier corrections: "covariance fails every time" does not fully describe arm A, which also fails hq. Every drain crashed afterwards with the existing `deepcopy` NameError, but the graded results were already written.

**GPU time:** 175 s. **Links:** branch `task/dig-horizon-clock-20261015`, draft [PR #425](https://github.com/255BITS/ParticleGAN/pull/425), readout `reports/forge/dig-20261015/horizon-clock.md` on that branch (local: `/ml2/hypergan/dig-20261015/horizon-clock/READOUT.md`).

### 6.4 cell-quota-min-change re-entry probe: REFUTED (holds up)

**Question.** cell-quota-minimum-change-forward-kl-g-v1 is the only recipe seen passing both gaussian1d_stability and mode_hold. Does it also pass ring16, which would make it a cheap Tier 1 re-entry? The recipe existed only in the unmerged Astra workspace. It was ported with additive, default-off changes and was confirmed faithful: gaussian1d_smoke first confirms at step 292 with 18/24, identical to the Astra record.

| Arm | ring16 | mode_hold | gaussian1d_stability | gaussian1d_smoke |
| --- | --- | --- | --- | --- |
| cell-quota candidate | FAIL, 0/96; 16 modes, hq 0.716, covariance error 3.13 (flat plateau) | PASS, streak 21 | PASS, 72/72 stationary, 24/24 shifted | PASS, 18/24 |
| BCAP DualNorm control | PASS, streak 26 from step 1184 | FAIL, streak 3 | FAIL, 2/72, 0/24 | PASS, 3/24 |
| Candidate's NSGDA base, no cell objective | FAIL, 15 modes, hq 0.458, covariance error 54.2 | — | — | — |
| DualNorm control + cell objective (Addendum A) | FAIL, 0/96; best hq 0.902, best covariance error 0.960 | not run | not run | — |

two_pole and five_word_joint_smoke are **BLOCKED**. The cell objective needs the public GANTrainer path, so planning refuses those tasks. The candidate can therefore never qualify for Tier 1.

**Verdict.** REFUTED. This pair is the review's ring16-against-held-tests conflict (§4) reproduced as a clean matched pair. The switch away from DualNorm causes most of the ring16 deficit, and the cell objective by itself also works against ring16. Verifier corrections:

- The control's ring16 pass is thin and late, on a single seed (covariance error 0.46 to 3.8 in the second half), so "decisive" oversells the separation.
- The times written inside the preregistration addenda are wrong (21:20 and 21:22, actually 21:16 and 21:18). The order was still correct.
- After step 400, mass_tv ranges 0.014-0.038.
- Stage C's clusters are not "over-wide" (min eigen ratio about 0.79).
- The blocker checks the `public_components` execution path, not sigma directly.

**GPU time:** 388.5 s. **Links:** branch `task/dig-cell-quota-probe-20261015`, draft [PR #427](https://github.com/255BITS/ParticleGAN/pull/427), readout `reports/forge/dig-20261015/cell-quota-probe.md` on that branch (local: `/ml2/hypergan/dig-20261015/cell-quota-probe/READOUT.md`).

### 6.5 Hygiene: DONE (CPU only, not independently verified)

| Item | Result |
| --- | --- |
| (a) Stale-label bug | Confirmed and fixed. The labeller checked only `family["id"]`, and its anchor could never match a row that carries a configuration link. A row is now labelled when its own id or its displayed cohort's `configuration_family` is stale. Regenerating with `--refresh-publication` changed only the BCAP label and two provenance hashes, and no score moved. Related test failures are 124 both before and after (the same set), plus 1 new passing unit test. |
| (b) TIMEOUT/INCOMPLETE | 20 of 22 are wall-clock hits, and no FAIL is. Five word-smoke runs had already confirmed a pass (step 834) before the 900 s kill. Counting them moves signed Confidence LR, Positive-evidence LR and the matched BCAP baseline to 6/6 Tier 1. Confidence LR's "+2" is exactly word smoke plus the blocked five_word_joint_hold. Of 13 Tier 3/ring16 hits, 5 are determined FAIL and 8 never had a passing check, so none flips. Timed-out runs ran at 2.8-13.9 updates/s against 24-28 normally, which points to GPU contention. |
| (c) Gaussian smoke margin | Confirmed: KS 0.044836, confirmation 0.033797, with 1/24 pairs passing. The thin margin is general: 16% of 180 Gaussian-smoke passes rest on 1 observation and 28% on 2 or fewer, including GAN v3 (1/24), BISPF (2/24) and BCAP (3/24). |

Caveats: the would-PASS calls are inferred from the earliest-confirmation rule, not graded, so a late guard failure is still possible. The timing comparison mixes recipe cost with contention. Recommended but not applied: report wall-clock hits as NOT MEASURED, use a contention-aware cap, and re-run under the GPU lease before counting.

**GPU time:** 0 s. **Links:** branch `task/dig-hygiene-20261015`, draft [PR #428](https://github.com/255BITS/ParticleGAN/pull/428) (stacked on #424), readout `reports/forge/dig-20261015/hygiene.md` on that branch (local: `/ml2/hypergan/dig-20261015/hygiene/READOUT.md`).

### 6.6 Notes common to all leads

- Every follow-up branch is based on this review's commit, so PRs #425 to #429 also show this document in their diffs.
- `tests/test_forge_dualnorm_smoothing_search.py::test_other_families_do_not_claim_an_active_smoothing_axis` failed in the relative-tether checkout. It touches none of that branch's code and was not re-run on clean develop.

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
