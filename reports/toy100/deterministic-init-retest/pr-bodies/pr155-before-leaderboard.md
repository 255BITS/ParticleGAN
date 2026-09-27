**Search stopped at your request. No candidate is qualified as the default.** Both PRs remain draft and unmerged; the API integration PR195 targets `develop`.

Both branches include develop’s new deterministic network and prior initialization. Sampling remains stochastic. **49 API configurations and 75 original research configurations were retested.** The remaining research cases are explicitly untested; this is a partial historical rerun. [Complete closeout and coverage](https://github.com/255BITS/ParticleGAN/blob/codex/k3p-continuous-search/reports/toy100/deterministic-init-retest/closeout.md).

The new initialization changes results:

- **RP12 improves:** the same quick coverage test goes from **0/24 to 19/24**, and its old image-intensity failure becomes a pass. But it then fails the four-bar image test **0/24**.
- **RP14 and RP15** also pass the quick screen and three image tasks, then fail the four-bar task **0/24**. **DV12** passes the quick screen but fails unequal-mass sampling **0/24**. These failures block default selection.
- **DV16 regresses:** its old quick-screen pass **11/24** becomes **0/24**.
- **Public KA2, constant-rate KA2 and K3P** each score **0/24**. Original asymmetric-Kalman research KA2 also fails this strict screen.
- **Research SN3 is a promising lead:** its own old 24-point prefix scores **5/24**, compared with **11/24** now. It uses autonomous rate/noise controls. Its longer stability follow-up was **not run before the requested stop**, so it is not a qualified release candidate.

Recovery is still **time to reach the new distribution, then stability**. DV2/DV3 reach the changed target after **380/500 updates**, then retain **183/183 and 171/171** checks. Neither reaches full quality on the original target before the change, so that later success does not establish complete qualification. There is no 81/81 deadline requirement.

[API scores](https://github.com/255BITS/ParticleGAN/blob/codex/k3p-continuous-search/reports/toy100/deterministic-init-retest/leaderboard.md) · [Research scores](https://github.com/255BITS/ParticleGAN/blob/codex/k3p-continuous-search/reports/toy100/deterministic-init-retest/research-leaderboard.md) · [Follow-up evidence](https://github.com/255BITS/ParticleGAN/blob/codex/k3p-continuous-search/reports/toy100/deterministic-init-retest/followup-results.json) · [Stop and untested-work receipts](https://github.com/255BITS/ParticleGAN/blob/codex/k3p-continuous-search/reports/toy100/deterministic-init-retest/retest-closure/).

The target remains one learner you can point at a distribution and run indefinitely, without choosing a training end or switching phases. Automatic rate reduction is acceptable if useful learning can return on its own. Short research passes and historical 22/22 scores do not demonstrate that complete behavior through the API.

**All earlier scores and sources remain preserved for future agents.** No new search round is queued.

<details>
<summary>Earlier scores and selection rationale — old initialization</summary>

**The scores below use the earlier initialization.** They remain valid historical evidence; they are not results for the new initialization. Neither PR is merged into develop, and no default winner is selected.

**No current entry qualifies.** This corrects the earlier R2/KA2 winner language. An automatic anchor-release mechanism inside a scheduled learner does not meet the whole-learner requirement.

**DV16 passed 11 broader tests, then failed long-term stability.** Its 30,000-update run found a late loss of the unchanged distribution around update **16,900**: nine failing checks, dropping to **six of eight modes**. It recovered at 17,000, but that does not erase the failure. The full declared run is complete; this tested old-initialization version cannot be selected as the default.

Its strengths remain useful: all six vector tasks, all four image tasks and small-particle coverage pass independently. It retains **693/693 checks** in the shorter 7,500-update run, and its own checkpoint/budget checks pass. In the single-change test it reaches the new target after **290 updates**, has one settling miss, then passes **190 consecutive checks**. All scores are preserved; the later failure is why the longer test matters.

C13-R1's fast recovery remains useful evidence (**130 updates, then 208/208 checks**), but it also fails small-particle coverage, ending with seven of eight modes. Its roughly 16 game evaluations per update also make this a recovery-in-updates result, not a runtime-speed claim.

**DV15 is not ready for release.** It stays stable for **699/699 checks** after learning an unchanged target, and a longer check confirms that its uneven-distribution result was slow settling. But it fails a separate image test and never learns all eight modes in the small-particle test. Its successes and failures are preserved.

**API-RP5 is a useful lead, but its current version is rejected.**

- It completed **30,000 uninterrupted updates**, recovering after three target changes in **350, 450 and 250 updates**, then staying stable through each remaining observation window. It reduces and restores learning rates itself; later changes can also recover while rates remain reduced.
- It passes **all four image tasks and all six vector tasks**. It fails a separate small-particle task: **0/24**, ending with only **5 of 8 modes**. The test implementation was independently verified. Rates remain fully open, so this is a coverage failure, not premature decay.
- On the matched recovery test, the **actual released K3P API** does not reach the new-distribution threshold within **2,200 updates after the change**. RP5 reaches it after **270 updates**, then passes **194/194 checks**. This is a measured recovery advantage, with the released K3P schedule and the same fixed starting model, data and scoring.
- Both also pass the uneven-distribution task. K3P first passes earlier and finishes more accurately; RP5 starts sustained passing earlier (**350 versus 450 updates**). The [full API comparisons and limits](https://github.com/255BITS/ParticleGAN/blob/codex/k3p-continuous-search/reports/toy100/continuous-api-search/k3p-comparison-results.json) remain available. RP5’s separate coverage failure still blocks default selection.
- **DV7 is rejected:** it completed the long recovery run and passed all four images, but failed to learn enough spread inside the rare component of an uneven distribution. DV9 fails the same task. Their strong and weak scores remain available as research leads.

The **released K3P API also fails the matched small-particle test: 0/24, ending with 6 of 8 modes**. That makes this a limitation of the released baseline too. It does not remove the coverage requirement for the new default.

The old-initialization search has been superseded by the new-initialization retest. [All measured scores, failures and exact sources](https://github.com/255BITS/ParticleGAN/blob/codex/k3p-continuous-search/reports/toy100/continuous-api-search/README.md). No broader or K3P pass is borrowed from an ancestor.

| Entries | Current decision | Why |
|---|---|---|
| **KA2, R2, B3-belief, SG3, B2, B3 guarded reseed, G1, K3P** | **DISQUALIFIED in their tested configurations** | Complete learners retain timed LR/noise schedules. G1's reversible generator boost does not remove its scheduled base. K3P is reference only. |
| **PB1, PB2, DI2** | **DISQUALIFIED** | Automatic mobility adjustments still sit above horizon-based rates/noise. |
| **PM1, PM3, P3, AP3, PX3, EP1, PD1** | **DISQUALIFIED** | Rates can close/reopen automatically, but the tested learner still has horizon-based noise. |
| **RP1, TD3, A3** | **REJECTED on measured quality** | RP1 fails image stability/native accuracy; TD3 fails acquisition/retention; A3 passes only 4/22. RP1's horizon-prefix audit passes—its rejection is not a blanket ban on automatic decay or initialization. |
| **EP2 and unfinished drafts** | **UNVERIFIED; not eligible for promotion** | Required execution/evidence is incomplete. |

These decisions apply to the exact tested configurations, not every possible descendant. The old results remain archived. Finite evaluation runs and initialization/smoothing windows alone are not disqualifying.

**Recovery means time to reach the new distribution, then stability—not 81/81 deadline checks.** Report retention, first arrival, later departures and sustained stability. A retrospective final passing suffix does not prove the learner stays indefinitely.

The actual public KA2 check reinforces the gap: constant rates retain **61/120** pre-shift observations, first reach the changed target after **120 updates**, and pass **126/209** observations afterward. The subsequent decay diagnostic retains **120/120**, arrives after **1,690 updates**, and passes **48/52** afterward. Neither demonstrates the required autonomous continuous behavior. The research extension is also complete: **105/109** after the previously reported arrival, with four failing observations; the old “everyone who arrives stays” claim is withdrawn.

**Keep these scores as research leads.** Disqualified configurations remain useful evidence; their sources, strengths and failures are preserved. The following are original research scores, separate from the public API runs above. Recovery counts are the old recorded window fractions, not a current all-81 gate.

| Research entry | Recorded retention | Original recovery count | Useful idea |
|---|---|---:|---|
| KA2 asymmetric-Kalman | Pre-shift 120/120 | 50/81 | Adaptive critic memory |
| R2 | Pre-shift 114/120 | 72/81 | Moment-surprise release |
| B3-belief | Pre-shift 114/120 | 73/81 | Shadow belief statistics |
| G1 finalized v15 | Pre-shift 120/120 | 47/81 | Reversible generator boost |
| PX3 | Own hold 1200/1200 + extension 300/300 | 71/81 | Bounded reopening and return to floor |
| RP1 | Own hold 1200/1200 + extension 300/300 | 81/81 | Automatic reopening; image/native failures still apply |
| K3P | Own hold 1200/1200 + extension 300/300; declared 22/22 toys | 28/81 | Historical retention baseline |

[All retained entry scores and limitations](https://github.com/255BITS/ParticleGAN/blob/codex/k3p-continuous-search/reports/toy100/continuous-eligibility/README.md#keep-the-scores-as-research-leads). Future agents should build on these ideas, address the precise disqualification, and earn their own results.

The original external search and its launch receipts are preserved as history. Its open-ended continuation instructions are superseded by the current user stop.

A replacement must demonstrate the complete behavior through the public API, including long stationary retention, delayed/repeated target changes, checkpoint continuation, a matched K3P comparison and broader quality verification. Scores from an ancestor or research-only training loop cannot establish those results.

[Entry-by-entry audit and sources](https://github.com/255BITS/ParticleGAN/blob/codex/k3p-continuous-search/reports/toy100/continuous-eligibility/README.md) · [Corrected leaderboard](https://github.com/255BITS/ParticleGAN/blob/codex/k3p-continuous-search/reports/toy100/continuous-practical-leaderboard.md) · [Launch/compaction handoff](https://github.com/255BITS/ParticleGAN/blob/codex/k3p-continuous-search/reports/toy100/continuous-eligibility/launch/README.md) · [Previous PR body, preserved as history](https://github.com/255BITS/ParticleGAN/blob/codex/k3p-continuous-search/reports/toy100/continuous-eligibility/pr155-body-before.md)

</details>
