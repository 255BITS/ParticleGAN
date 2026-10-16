# BISPF refresh: do BISPF's lead and its other verdicts reproduce on the current source?

Status: PREREGISTERED (rule committed before any training). Results are added below the rule.

## Why

BISPF (`bispf-v1`) leads the Forge board (77/152 today, 77/81 after the zero-pass audit). Its record is
stale: it was measured on source digest `fafd6615...` with a different Python/numpy runtime, and several of
its task contracts have changed since ("Test definition changed since this run" on the board). The selected
BCAP recipe (`recipe-prior-refactor-bcap-pure-v1`, configuration family `bcap-dualnorm`, BISPF's parent) is
equally stale (source `a5eafc73...`; its stale label was fixed in #428). BISPF's only extra pass over BCAP is
`mode_hold`.

Words used here:

- **Counted test**: a test that still counts toward the leaderboard after the zero-pass audit
  (`/ml2/hypergan/test-dispositions-20261015/RECONCILED.md`). Held tests (four vector tests, three image
  tests, grid100/rotated100/staggered100, gaussian1d_stability, the Tier 3 tests) are skipped.
- **Verdict**: the Forge grader's PASS/FAIL, rebuilt from the run's curve. Sustained gates need at least 5
  passing checks in a row at the end of the run ("terminal suffix"); the word and Gaussian smoke tests use an
  earliest-confirmed gate (a passing check confirmed by an independent redraw, with all updates completed).
  Endpoint values are never used for verdicts.
- **Tier rule** (runner policy `complete_current_tier`): every runnable job of a tier finishes even if one
  fails; any non-PASS in the tier stops spending on the next tier.
- **Frozen source**: the source digest that `enqueue` freezes; both recipes run on the same one.

## Hypothesis

H: on the current source (develop + the compile_memory crash fix, #430), seed 0, CUDA RTX A6000, every
recorded verdict of BISPF and of the selected BCAP recipe on the 17 counted Tier 1/Tier 2 tests reproduces,
in particular BISPF `mode_hold` PASS vs BCAP `mode_hold` FAIL (BISPF's lead).

Recorded verdicts (from `reports/forge/technique-inventory.json`, selected cohorts):

| Tier | Test | BISPF recorded | BCAP recorded |
| --- | --- | --- | --- |
| 1 | clockfree_audit_measurement_v1 | PASS | PASS |
| 1 | two_pole | PASS | PASS |
| 1 | unused_token_hold | PASS | PASS |
| 1 | ae_gan_hold | PASS | PASS |
| 1 | gaussian1d_smoke | PASS | PASS |
| 1 | ring16_acquisition | PASS | PASS |
| 1 | five_word_joint_smoke | PASS | PASS |
| 2 | five_word_joint_hold | PASS | PASS |
| 2 | unipolar | PASS | PASS |
| 2 | trajectory | PASS | PASS |
| 2 | residual_student | PASS | PASS |
| 2 | cover_leftover | PASS | PASS |
| 2 | mid_scale_identity | PASS | PASS |
| 2 | vector_spiral | PASS | PASS |
| 2 | vector_two_broad | PASS | PASS |
| 2 | img_stripes2 | PASS | PASS |
| 2 | mode_hold | PASS | **FAIL** |

## Design

- Two candidates, unchanged: `bispf-v1` and `recipe-prior-refactor-bcap-pure-v1`. No code, default, task,
  gate or budget changes; only opt-in research-diagnostic views, studies and one campaign are added.
- Two research-diagnostic views (all tasks `diagnostic`, so these runs never fill ordinary cells and the
  leaderboard is not regenerated):
  - `dig2-bispf-refresh-tier1-v1`: the 7 counted Tier 1 tests, cheapest first, plus
    `five_word_joint_hold`. The word hold is a Tier 2 test that continues the `five_word_joint_smoke`
    checkpoint, so it has to run in the same request as the smoke test. Its verdict is admissible only if the
    recipe passes all 7 Tier 1 tests; otherwise it is reported as NOT ADMISSIBLE.
  - `dig2-bispf-refresh-tier2-v1`: the other 9 counted Tier 2 tests, cheapest first.
- Ordinary path: `forge enqueue <candidate> --study <study>`, then `forge drain --gpus $LEASED_GPU
  --workers-per-gpu 1 --request <id>` under the workflow GPU lease, one request per lease.
- Both recipes' Tier 1 requests are enqueued from the same commit, so they share one frozen source digest;
  Tier 2 requests are enqueued from the same commit as well. If the frozen digests differ between the two
  recipes for the same tier, the comparison is flagged as unmatched.

## Preregistered decision rule

1. Tier rule: a recipe's Tier 2 request is enqueued only if all 7 counted Tier 1 tests PASS on this
   source. Inside a tier, all jobs run to completion.
2. Per cell (recipe x test): REPRODUCED if the current verdict equals the recorded verdict; FLIPPED if it is
   the opposite PASS/FAIL; UNDECIDED if INCOMPLETE, TIMEOUT, INVALID, BLOCKED or NOT_RUN (reported with the
   exact error; no reruns, no other seeds).
3. BISPF's lead is REPRODUCED iff BISPF `mode_hold` = PASS and BCAP `mode_hold` = FAIL on this source. It is
   NOT REPRODUCED if either cell flips (both pass, or both fail, or reversed).
4. Overall verdict on H:
   - SUPPORTED: all 34 cells REPRODUCED.
   - REFUTED: any decided cell FLIPPED (the readout names each flipped cell and says separately whether the
     lead itself reproduced).
   - INCONCLUSIVE: no cell flipped but at least one cell is UNDECIDED.
5. Budget: at most 2 GPU-hours of attempt wall time in total. If the running total would exceed it, remaining
   requests are not launched and their cells are NOT_RUN (UNDECIDED).
6. Margins are reported for every sustained gate (passing checks, terminal suffix vs required), because a
   one-check margin (like BCAP's earlier `mode_hold` miss by one check) means a small perturbation can move
   the verdict. Margins do not change verdicts.

Prior context (motivation only, not evidence for this rule): the 2026-10-15 BISPF ablation reran `mode_hold`
for both recipes on frozen digest `5ea30969...` and got BISPF PASS (12 terminal passes) and BCAP FAIL (3
terminal passes, one check short, a single hq dip to 0.869).
