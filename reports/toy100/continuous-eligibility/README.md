# Continuous-learning eligibility: no qualified winner

**The requirement is one learner that can point at a target and keep running.**
The caller must not choose a training end, switch from acquisition to maintenance,
or restart learning when the target changes. Automatic rate reduction is allowed
if the learner can also restore useful learning and settle again on its own.

This September 26 audit supersedes PR #155's “R2 wins”, “KA2 best joint” and
“schedule-free release” eligibility claims. Those labels described selected
components or finite research scores, not the complete continuous learner.
**No current entry qualifies for this requirement.** K3P remains a comparison
baseline; it is not a qualified continuous-learning winner.

## Decisions for the entries under consideration

Disqualification applies to the exact tested configuration. A changed descendant
must earn its own results. Existing measurements remain valid for their original
protocols; an old 81-check deadline score is not the disqualification rule.

| Entries | Decision | Reason |
|---|---|---|
| **KA2, R2, B3-belief, SG3** | **DISQUALIFIED: scheduled complete learner** | Their automatic anchor/memory signals leave the inherited one-way LR and horizon-based noise schedules in place. |
| **G1** | **DISQUALIFIED: scheduled base learner** | Its reversible generator boost multiplies the scheduled generator step. Critic/prior schedules remain. |
| **B2, B3 guarded reseed** | **DISQUALIFIED: scheduled complete learner** | Releasing or reseeding memory does not replace inherited LR/noise schedules. B2 also loses pre-shift retention. |
| **K3P** | **DISQUALIFIED for continuous selection; reference only** | Uses the prescribed acquisition/decay schedule. Its released trainer also enforces a finite recipe budget. |
| **PB1, PB2, DI2** | **DISQUALIFIED: scheduled base learner** | Automatic mobility adjustments still sit above horizon-based rates/noise. |
| **PM1, PM3, P3, AP3, PX3, EP1, PD1** | **DISQUALIFIED: remaining horizon coupling** | Their rate rules can close/reopen internally, but the complete tested learner retains driver-horizon noise schedules. |
| **RP1** | **REJECTED: measured quality failure** | Its automatic rates and absolute startup noise are not source-disqualified here; a matched horizon-prefix audit passed. Image stability and native accuracy fail. |
| **TD3** | **REJECTED: measured acquisition/retention failure** | Its autonomous rates and absolute startup noise are not rejected merely for having initialization. Whole-learner horizon invariance remains unverified. |
| **A3 reversal strength** | **REJECTED: measured quality failure** | Full verification records 4/22 passes and 18 failures. |
| **EP2 and unfinished drafts** | **UNVERIFIED; ineligible for promotion** | Incomplete execution/evidence does not establish either successful continuous learning or a measured failure of the missing test. |

The remaining older leaderboard sections are archived comparisons, not additional
eligible winners. This audit covers the entries in PR #155's current comparison
tables and candidate state; it does not pretend to have newly qualified every
historical draft.

## Keep the scores as research leads

**Disqualified does not mean discarded.** Preserve the exact source, scores and
failures. Future agents should reuse useful mechanisms while correcting the
reason the complete configuration was rejected. A descendant cannot inherit
its parent's passes.

These are the original research measurements, not results from the public API.
The recovery fractions below are historical passing observations in the old
81-check window, retained as scores rather than an all-81 selection gate.
“Own hold” means the separate 1200-check hold plus 300-check extension; it must
not be conflated with 120 sampled pre-shift observations.

| Research entry | Recorded retention | Original recovery count | Useful lead / limitation |
|---|---|---:|---|
| KA2 asymmetric-Kalman | Pre-shift 120/120 | 50/81 | Adaptive critic memory; later extension is 105/109 after reported settled arrival, with dropouts. |
| R2 moment-surprise | Pre-shift 114/120 | 72/81 | Surprise-driven release; faster recovery with weaker retention. |
| B3-belief | Pre-shift 114/120 | 73/81 | Shadow belief statistics as a signal; full learner still scheduled. |
| SG3 graded memory | Pre-shift 114/120 | 43/81 | Graded memory/reseeding; retains the same hold failures. |
| G1 finalized v15 | Pre-shift 120/120 | 47/81 | Reversible generator boost; scheduled base remains. |
| B2 unguarded reseed | Pre-shift 33/120 | 40/81 | Reseeding enables motion but damages retention. |
| B3 guarded reseed | Pre-shift 120/120 | 0/81 | Guarding preserves retention but this version does not recover in the window. |
| K3P | Own hold 1200/1200 + 300/300; declared 22/22 toys | 28/81 | Strong historical stability baseline; scheduled learner. |
| PM1 / PM3 | Both own holds and extensions pass | 79/81 each | Autonomous mobility rules; inherited noise still horizon-based. |
| PB1 | Pre-shift hold fails | 79/81 | Recovery improvement does not erase loss of retention. |
| PB2 / DI2 / P3 | Own holds and extensions pass | 77/81 each | Penalty balance, data innovation and reopening mechanisms worth separating from inherited schedules. |
| AP3 | Own hold and extension pass; pre-shift 120/120 | 72/81 | Partial reopening; remaining horizon-based noise. |
| PX3 | Own hold 1200/1200 + 300/300; pre-shift 120/120 | 71/81 | Bounded reopening followed by return to the floor; useful starting point, not a qualified learner. |
| RP1 | Own hold 1200/1200 + 300/300; pre-shift 120/120 | 81/81 | Automatic reopening and horizon-prefix evidence; image/native failures remain decisive. |
| TD3 | Acquisition/retention fails | 53/81 | Discrepancy sensing alone does not solve stable updates. |
| EP1 | Own hold and extension pass; pre-shift 120/120 | 16/81 | Excursion-controlled rates; remaining horizon-based noise. |
| PD1 | Own hold not converged; pre-shift 33/120 | 0/81 | Failed projected-discrepancy configuration. |
| A3 | Pre-shift 89/120; full suite 4/22 | 71/81 | A better recovery count can conceal broad regressions. |
| EP2 | Partial hold only: last recorded 650/1200 | NOT_RUN | Unfinished evidence; no inferred outcome. |

Sources and original scope are retained in the
[scored leaderboard](../continuous-practical-leaderboard.md),
[previous PR body](pr155-body-before.md), and each entry's linked report in
[audit.json](audit.json). Old raw FAIL labels remain unchanged; current
eligibility is a separate judgment. No score or failed run was deleted by this
audit.

## Why an automatic anchor was insufficient

For R2, KA2, B3-belief, SG3, B2 and G1, the exact config and shift-driver hashes
match the committed K3P versions. The driver supplies `.01/.05` LR floors and
runs `mode='scheduled', noise_horizon=1200`. The common host applies cosine
network/prior rates, which reach their floors and do not autonomously reopen.
See [source receipts](source-receipts.json), the
[shared driver](../gap-fill-20260925/sources/k3p/shift.py),
[host control](../../../benchmarks/toy100/continuous_probe.py), and
[rate calculation](../../../benchmarks/toy100/schedule.py).

An automatic signal inside that learner does not remove its external schedule.
B3-belief specifically uses **Adam plus shadow belief statistics**, rather than
an AdaBelief optimizer. G1 has a real reversible boost, but that boost alone does
not make its remaining scheduled roles autonomous.

The newer [public KA2 experiment](https://github.com/255BITS/ParticleGAN/blob/fa511ce010120b502f494d717d01b14b8551eed8/reports/ka2-default-candidate/constant-lr-api/README.md)
also rules out simply removing LR decay from that API configuration:

| Actual public KA2 run | Pre-shift retention | First reaches changed target | Passing observations afterward |
|---|---:|---:|---:|
| Constant rates | 61/120 | 120 updates after shift | 126/209 through update 4600 |
| Decay diagnostic | 120/120 | 1690 updates after shift | 48/52 through update 4600 |

The constant run repeatedly loses the original and changed distributions.
The decay run is not an automatic reversible rate policy. Neither qualifies.
These are actual `GANTrainer` runs, separate from the smaller research host.

KA2's research extension is also complete: **105/109** passing observations
from its previously reported settled arrival at 3520 through 4600, including
failures at 4280, 4300, 4310 and 4320. The earlier assertion that every arriving
candidate stays forever is withdrawn. A final passing suffix is chosen
retrospectively and cannot prove future stability.

## What counts as evidence for a replacement

- The whole learner operates without a caller-provided end time, phase switch,
  reset, target-change notification or access to benchmark quality scores.
  A finite evaluation budget, initialization or smoothing window alone is not
  a disqualification. Automatic reversible internal states are allowed.
- Measure acquisition and stationary retention, first arrival after each target
  change, every later departure and sustained stability. Report both first
  arrival and the final passing suffix. Do not require 81/81 deadline checks.
- Demonstrate repeated and delayed changes plus a long uninterrupted stationary
  control, including useful learning after any automatic reduction in rates.
- Exercise the actual public API, checkpoint continuation, and matched K3P
  comparisons on the same model/data/runtime. Complete broader quality checks
  before recommending one library default; do not inherit an ancestor's 22/22.
- Run no seed sweeps. Preserve failures, configurations, sources, applied rates
  and training signals. Finite tests support an indefinite-use design; they
  cannot literally prove infinite-time stability.

## Authorized next work

The user authorized **three distinct approaches at a time**, replenished after
review until one meets the requirement, using the external **Codex / Astra / max**
launcher. After the requested compaction, all three external model sessions
started on **September 26 at 21:12 UTC**. Their model headers confirm Astra/max.
The [launch receipt](launch/first-launch.json) and [supervision handoff](launch/README.md)
preserve the commands, attempts and log paths. Qualification remains pending;
PRs remain unmerged.

[Machine-readable decisions](audit.json) ·
[Previous PR body, preserved as history](pr155-body-before.md) ·
[RP1 quality rejection and horizon audit](../continuous-round-3/rp1-rejection.md) ·
[PX3's reversible rates and retained scheduled noise](../continuous-round-3/completed-prox/attempts/k3p_prox_release/result.md)

Offline audit: `python reports/toy100/continuous-eligibility/verify.py`.
On the original workspace, add `--originals` to verify the six local candidate
source bundles against their receipts as well. Neither command trains a model.
