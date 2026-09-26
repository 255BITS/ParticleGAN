<!-- Superseded: preserved PR155 body before the September 26 continuous-learner audit. -->

## What we are doing

Find one GAN formulation that keeps K3P's stability (22/22 frozen GPU toys, 1200-update hold + 300 extension) while re-acquiring a shifted target fast enough to pass all 81 deadline checks — without depending on a training-horizon schedule. K3P itself is the selected base: full hold and 22/22, but only 28/81 recovery. The search resumed September 26 on the OpenCode/Muse engine after the September 25 pause; the current lead is R2 (below). Full details: [leaderboard](https://github.com/255BITS/ParticleGAN/blob/codex/k3p-continuous-search/reports/toy100/continuous-practical-leaderboard.md) · [search state](https://github.com/255BITS/ParticleGAN/blob/codex/k3p-continuous-search/reports/toy100/continuous-search-state.json) · [round-3 inventory](https://github.com/255BITS/ParticleGAN/blob/codex/k3p-continuous-search/reports/toy100/continuous-round-3/INVENTORY.md).

## Best results for the task, sorted by best

Recovery counts are passing checks out of the required 81 (all 81 required); hold is own hold+extension. Ranked with R2 first: schedule-dependent releases are disqualified (the task is horizon-independent learning), rejected/out candidates shown for the record. Among qualified contenders R2 wins outright — the only candidate pairing a schedule-free release with measured hold and recovery.

| # | Formulation | Live recovery | Hold | Schedule-free release? | Standing |
|---|---|---:|---:|---|---|
| 1 | **R2 moment-surprise — raw recovery best** | **72/81, delay 490** | **114/120** | **Yes — surprise-driven, no clock/budget reads** | **Top recovery; 6 hold + 9 deadline checks short; 4/4 toy screens, frozen twin moves** |
| 1j | **ka2 asymmetric-Kalman — best JOINT, LEAD** | **50/81, delay 1120 (back-loaded, stable 3520)** | **120/120 FULL** | **Yes — certainty-gated EMA, asymmetric time constants** | **Only full-hold + real recovery; bundle persisted; combo round live** |
| — | RP1 | 81/81 | 1200/1200 + 300/300 | Rejected before schedule review | REJECTED: image stability + native accuracy FAIL |
| DQ | PM1 / PM3 | 79/81 each | Both pass | No — scheduled noise remains | DISQUALIFIED: schedule-dependent |
| DQ | PB2 / DI2 / P3 (PB1 too) | 77/81 (PB1 79/81) | Pass (PB1 pre-hold FAIL) | No — scheduled components remain | DISQUALIFIED: schedule-dependent |
| — | TD3 | 53/81 | FAIL (stationary 0/5) | Fails on own merits | Out |
| — | B2 unguarded re-seed | 40/81 | 33/120 FAIL | n/a (hold broken) | Out: speed without stability |
| ref | **K3P — selected** | **28/81** | **1200/1200 + 300/300; 22/22 toys** | **No — still depends on schedules; must earn invariance** | **Selected base; recovery is the outstanding problem** |
| — | EP1 / PD1 | 16/81 / 0/81 | EP1 passes; PD1 does not converge | — | Out |

R2's mechanism: K3P penalty blend with `s = 0.5` fixed (no LR clock); anchor switch W driven by Adam second-moment surprise (`surprise = RMS(grad)/sqrt(v-hat)`, release at ratio > 3.0, re-anchor below 1.75); EMA updates only while W == 1 plus guarded re-seed. Final live state: all 8 modes back at HQ 0.997. Active follow-ups: R2 settling-window fix and novelty-gated G-boost (first motion with hold intact: 45/120 and 38/120 with 8-mode finals).

## Stability-since-arrival (delay-agnostic) — full table [here](https://github.com/255BITS/ParticleGAN/blob/codex/k3p-continuous-search/reports/toy100/continuous-practical-leaderboard.md)

Per user direction the comparison cares about the END, not speed: stability from each run's own arrival to end of window, plus a stable-end boolean (final 8 modes HQ >= .90 on a live passing streak). Everyone who arrives, stays:

| Candidate | Arrived | Since-arrival | Final | Stable end? |
|---|---|---|---|---|
| B3-belief | 2880 | 73/73 = 100% | 8 / 1.0 | TRUE |
| R2 | 2890 | 72/72 = 100% | 8 / 0.997 | TRUE |
| B2 | 3210 | 40/40 = 100% | 8 / 0.988 | TRUE |
| SG3 | 3480 | 13/13 = 100% | 8 / 0.919 | TRUE |
| ka2 | 3520 | 9/9 = 100% (109 pending extended run) | 8 / 0.996 | TRUE* |
| G1 | never | — | 8 / 0.891 intermittent | FALSE |

G1 is exposed: visits without staying. Compute cost is count-based (wall seconds measure contention on a shared box): eval-units = pure-A calls + 2x blended calls from mechanism receipts — ka2/R2/B3-belief/SG3/G1 all tie at 6401 per shift run, so cost binds only across architectures. Full definition on the [leaderboard](https://github.com/255BITS/ParticleGAN/blob/codex/k3p-continuous-search/reports/toy100/continuous-practical-leaderboard.md). With delay ungraded the top rows tie and the tiebreak returns to hold — ka2 is the only arrival with a full 120/120 hold.

Seed-fragility context: the 22/22 base reproduces only at its own seed (~2/8 ring/hold/stay across declared seeds; a 1e-6 init nudge gives NOT_CONVERGED 0/1200 vs repo-seed PASS 1200/1200). Failing gates are now retried at declared seeds before drop verdicts; fragile is not broken, robust wins.

## Background (unchanged)

OpenCode/NanoGPT launcher update: `--engine opencode` now selects `nano-gpt/meta/muse-spark-1.3-contributor` for individual attempts, formulation batches and pools. It uses the saved NanoGPT login (with environment-key fallback), isolated runtime config, readable stream logs, final-message capture, usage/cost reporting and dashboard support. The launcher snapshot includes setup instructions and offline integration tests.

Validation: 10 offline integration checks passed; live Muse Spark checks passed for both a text response and a shell-read/file-write workflow. Dry-runs verified all four existing/new engines and all eight K3P lanes. The existing STOP marker remains active and rejects real attempts with exit 75; this integration launched no research training.

K3P selected in #139 passes all 22 frozen GPU toys and sustained hold, but only 28/81 target-shift deadline checks. This follow-up investigated adaptation without requiring a final training horizon while preserving those gates. **No replacement meets the complete bar; K3P remains selected.**

**Search paused at the user's request on September 25; resumed September 26 on the Muse engine** (5 batches, 21 candidates, `--minutes 0` no-timeout launcher update). The eight latest tracked agents had already exited when the stop request was handled, several without final reports. Their complete measurements and partial work are archived separately.

Round 1 records 24 formulations and round 2 records 21, with no winner and explicit incomplete protocols. Round 3 has 42 formulations in reported groups, with 83/84 canonical protocols complete, plus EP2's unfinished attempt. EP2 last logged hold 650/1200 at update 2050; it has no final result and no shift run. Including EP2 gives 43 benchmark-started formulations and 83/86 complete protocols. Source-only drafts and unchanged-parent instrumentation are excluded from those counts.

The final eight-attempt archive retains original ledgers, recovered unledgered PD1 results, benchmark logs, and draft source with hashes. The evaluator repair's two-update instrumentation identity check passes, but independent equivalence to the canonical driver and full review remain unfinished: **NOT_READY**. The Codex strict canonical-control audit exited 0 without delivering an implementation or report. No 9000/30000 candidate qualification was run.

The unchanged final bar requires one same formulation to pass own hold+extension, the complete canonical shift with full-state matched frozen control, all 22 toys, substantial horizon invariance, delayed/repeated changes, and 30,000 uninterrupted updates. The native matrix is **grid100, rotated100, and staggered100 each 4/4 seeds 1234–1237**, every cell requiring full 7000-update coverage AND accuracy. Selected K3P has historical evidence for **6/12** cells: grid 4/4, rotated 1/1, staggered 1/1. Six rotated/staggered cells remain unmeasured. No partial candidate is promoted.

- [Round 1](https://github.com/255BITS/ParticleGAN/blob/codex/k3p-continuous-search/reports/toy100/continuous-round-1/README.md)
- [Round 2 and A3 verification](https://github.com/255BITS/ParticleGAN/blob/codex/k3p-continuous-search/reports/toy100/continuous-round-2/README.md)
- [Round 3 historical record](https://github.com/255BITS/ParticleGAN/blob/codex/k3p-continuous-search/reports/toy100/continuous-round-3/README.md)
- [Final attempt evidence](https://github.com/255BITS/ParticleGAN/blob/codex/k3p-continuous-search/reports/toy100/continuous-round-3/stop-inventory/manifest.json)
- [Paused search state](https://github.com/255BITS/ParticleGAN/blob/codex/k3p-continuous-search/reports/toy100/continuous-search-state.json)
- [Full leaderboard](https://github.com/255BITS/ParticleGAN/blob/codex/k3p-continuous-search/reports/toy100/continuous-practical-leaderboard.md)

Validation: launcher syntax passes and attempts are rejected while stopped; all 127 recovered-file hashes, archived ledger snapshots/receipts, inventory links, and selected K3P source hashes checked. Draft research remains separate from #139's completed selection.
