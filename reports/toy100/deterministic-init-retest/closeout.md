# Retest closeout — search stopped

This is the September 26 pause record. The user later authorized the 22 stopped
screens and SN3 long hold; their [resumed results](resumed-new-init-results.md)
supersede its `NOT_RUN_USER_STOP` and SN3 follow-up status. Historical scores and
stop receipts below are retained as recorded.

The user requested wrapping up because tokens were running low. The search is
stopped; neither PR is merged and no default is promoted. PR195 still targets
`develop`. Do not restart or replenish this search without a new user instruction.

Develop's actual deterministic network and recipe-created prior initialization
is merged into both working branches. Sampling follows the existing stochastic
streams. No seed sweep or learner tuning was performed.

Completed and archived: **49 API quick-screen configurations**, plus **75
original research configurations**. Four earlier API logging-error attempts
remain recorded separately. The research screen has **4 quality passes**;
quality and continuous-use eligibility are separate decisions.

| Candidate | New evidence | Decision |
|---|---|---|
| Public KA2 / constant KA2 / K3P | Each 0/24 on the quick coverage screen; final 6 / 4 / 6 of 8 clusters | No default qualification |
| RP12 | Tiny 0/24 → 19/24; intensity 4/24 → 12/24 | Improved, then rejected by bars4 at 0/24 |
| RP14 | Intensity 0/24 → 9/24; tiny passes 12/24 | Rejected by bars4 at 0/24 |
| RP15 | Tiny 14/24; passes three image tasks | Rejected by bars4 at 0/24; no completed matching old tiny comparison |
| DV12 | Tiny 12/24 | Unequal-mass follow-up fails 0/24 |
| DV16 | Exact old tiny PASS 11/24 → new FAIL 0/24 | Regression preserved |
| DV2 / DV3 | Changed-target arrival after 380 / 500 updates, then 183/183 / 171/171 stable checks | Neither acquires original target before the change at 2400; no release qualification |
| Research SN3 (`sn3-2f595f84`) | Own old 24-point prefix 5/24, final streak 2 → new 11/24, final streak 11 | Promising; new longer stability follow-up NOT_RUN_USER_STOP |

SN3 uses autonomous rates and noise, but has no demonstrated public API or long
stability qualification with this initialization. Its old 7,500-update hold
failed after initial acquisition. The old checkpoint also omits mechanism state;
continuation must not be claimed from that checkpoint. No SN3 port or repair was
made. PNB3, JT2 and the passing ordinary-Adam research probes retain their earned
scores alongside their tested horizon-dependent configurations.

**Coverage is partial.** Remaining reviewed research cases were stopped before
execution at the user's request. Other historical definitions require additional
source binding or initialization adapters; they are explicitly NOT_RETESTED,
not quality failures. This does not claim a complete rerun of every historical
leaderboard definition. See the [coverage inventory](retest-closure/coverage-scope.json)
and [frozen queue](retest-queue.md) for exact IDs and reasons.

[API leaderboard](leaderboard.md) · [Research leaderboard](research-leaderboard.md)
· [All image follow-ups](image-runtime-review/complete-image-batch-audit.md)
· [DV2/DV3 arrival and stability](dv23-ring-runtime-review/audit.md)
· [Current configuration eligibility](research-eligibility-audits/)
· [Stop and coverage receipts](retest-closure/)
· [Archive integrity](archive-integrity.json).

The develop merge passed 60 research-branch CPU tests and 246 API-branch CPU
tests, with one CUDA test skipped in the authoritative API CPU run. Independent
source, initialization, retained-state, sampling and score audits accompany the
measurements. The early ordinary-Adam research probes do not retain final raw
checkpoints or final sampling cursors; their audit states those limits explicitly.
Old scores, failures and exact sources remain available for future work.
