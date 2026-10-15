# Signed and positive-evidence responses: Gaussian retention still fails

Two distinct Gaussian-only response variants pass the unchanged original
acquisition gate and fail the full original own-checkpoint retention gate.
Neither has a complete six-task screen at its source/revision. Neither is a
global replacement, Martyn-superior formulation or release candidate.

| Source-scoped recipe | Acquisition | Stationary hold | Reacquisition | Shifted hold | Frozen |
| --- | --- | ---: | --- | ---: | ---: |
| Preserved absolute-confidence base | PASS | 8/72 | FAIL | 0/24 | 0/48 |
| Finite critic cap | FAIL | Unstarted | Unstarted | Unstarted | Unstarted |
| Critic output orientation | PASS | 5/72 | FAIL | 1/24 | 0/48 |
| Signed confidence | PASS | 7/72 | FAIL | 0/24 | 0/48 |
| Positive excess confidence | PASS | 14/72 | FAIL | 2/24 | 0/48 |

The [signed-confidence receipts](../test-performance-evidence/pr399-signed-confidence-2.json)
replace the absolute-gap response `Phi(abs(EMA gap)/SE)` with
`Phi(EMA gap/SE)`. The inherited span-10 filter, actual G/prior learning-rate
application, gradients, moments, tasks and gates remain intact. Acquisition
has independently confirmed scheduled steps 459, 542 and 1,000. Retention
restores its own final step-1,000 state, completes 5,000 additional real
updates, then fails 7/72 stationary checks, 0/24 shifted hold, 0/48 frozen
evaluation and deadline reacquisition. Final shifted CDF KS is `.105793`.
The signed gap is a confidence proxy, not a restoring-gradient certificate.
Executed source digest is
`9ea2a30dafacbe7a8c4a962212760d57995ff8f0289047d5cfa5255c604a3aff`,
candidate revision is
`0e6430b8d559246ce38a20199242eb693e237c549f93f986c2c22530272a73ae`,
and score-ready packet SHA-256 is
`20b8e44e611507dd2d7a65c49c3bb440b4f814e67cd655ecf7f78fe05354f6e0`.

The [positive-evidence receipts](../test-performance-evidence/pr399-positive-evidence-2.json)
use `max(0,2*Phi(EMA(real-fake)/SE)-1)` with the same inherited filter. It
removes a neutral half-rate proposal without clearing gradients or optimizer
moments. At step 6,000, 654 zero-evidence proposals produced no proposal
parameter motion while real optimizer histories advanced; these are not
fabricated updates. Acquisition confirms scheduled steps 667 and 792, even
though its final step-1,000 state is outside the width bound. Retention
restores that exact final state, adds 5,000 updates and fails 14/72 stationary,
2/24 shifted, 0/48 frozen and deadline reacquisition gates. Final shifted
mean `2.947908`, width ratio `.964398` and CDF KS `.041174` meet numerical
endpoint bounds; that endpoint does not override the failed full gate.
Executed source digest is
`b4e88927c2934d079eee02eece080cfb9c221a63973117e6c6d7e9e827844831`,
candidate revision is
`65d349deee7ee709085212fdd6f7a6fd274881fc23b074e238468668cfcd815e`,
and packet SHA-256 is
`caffd13091504f5f2a5bc5a94db3a42b4c68676cfd01ebf693916c1cac3dbce7`.

Each variant keeps one global recipe, protocol seed 0, public initializer,
original tasks, target batches, prior, sampler, horizons, time limits and
evaluator. Both restore their own full state and named RNG streams, without
best-checkpoint substitution. The two executed sources differ from the
successful confidence-LR source only in the response helper and Recipe mode
validation; all other 1,212 frozen members match. They are separate source
cohorts, so the earlier 6/6 screen is not reused as current-source
qualification. The implementations remain in the coordinator archive and
are not part of this evidence-only branch.

The cumulative packet SHA-256
`db33d7b896a2f09e12abfbf12d221e366865b994fde1261f715302fee7dbbe2a`
contains **81 unique attempts: 35 PASS / 39 FAIL / 7 INCOMPLETE**. All 77
earlier rows are unchanged; these four new attempts are added once. This is a
source-scoped attempt ledger, not an addition to the 249-run non-Atlas ledger.
The old timed-out word control still prevents an exact cumulative optimizer
update count. The new signed and positive trials cost respectively
96.549931/99.147484 and 96.762846/99.287825 queue/supervisor seconds. Those
clocks are separate, with zero paid retries. Original grades, source/task
bindings, terminals and 32 retained artifact pins were checked; this display
import adds no training, sampling, gate change or default promotion.
