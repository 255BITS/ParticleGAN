# Later Gaussian corrections: acquisition and retention still fail

The successful confidence-LR six-task screen remains at its original source
and revision. Two later Gaussian-only correction hypotheses do **not** establish
retention or global superiority. Their original failures and unstarted work
remain visible, without borrowing the earlier 6/6 screen as qualification for
a different source/revision.

| Corrective recipe | Original Gaussian acquisition | Original own retention |
| --- | --- | --- |
| Confidence-LR + finite critic cap | **FAIL**: 1/24 primary observations, no independently confirmed scheduled pass | **BLOCKED_UNSTARTED**: zero attempts, updates and charge |
| Confidence-LR + critic output orientation | **PASS**: five independently confirmed scheduled pairs | **FAIL**: 5/72 stationary, 1/24 shifted, 0/48 frozen, failed reacquisition |

The [finite-cap receipt](../test-performance-evidence/pr399-gaussian-finite-cap-1.json)
is one complete 1,000-update seed-0 run under the unchanged acquisition gate.
Its final mean is `1.993633`, standard-deviation ratio `.928876` and CDF KS
`.055605` against a required maximum `.05`. The finite critic guard is active,
but the original confirmation gate fails. The dependent retention question was
never launched and has **no fabricated cell or grade**. This recipe uses the
same executed source digest as the successful confidence-LR screen,
`d48060b87f539c649d493d39c1bba957c098ae66e1e3bb84f8f162859375dbf5`,
but a distinct candidate revision
`460c3c14b2fcb65fdf338a9c978f73916c2377a2a376441f15b36ca2013c025f`.
Its packet SHA-256 is
`156e7397a1bc028cca73a575fc18912ce75bb0251f62716dda1e272b89aeb6c1`.
The earlier confidence-LR Gaussian PASS was not rerun or relabelled.

The [critic-orientation receipts](../test-performance-evidence/pr399-critic-orientation-2.json)
belong to a different source digest,
`d0f88ea803be98440c359954636eb250ad5f726a8d5ae43611821233c40fa6f5`,
and candidate revision
`aca1d7c63375b7d825c2f92e211b027c5845e9d0f128f5b2c2e47e59a8868868`.
The existing span-10 score-gap EMA chooses the critic output sign using the
current batch and exact discriminator-loss comparison, then transports the
confidence mean across a sign change. The rule is active: 94 sign changes by
step 1,000 and 766 by step 6,000. Acquisition has confirmed steps 292, 334,
500, 625 and 959, so it passes the original any-confirmed-scheduled-pass gate;
the final step-1,000 state itself is outside original mean/CDF bounds. The
own-state continuation restores that final checkpoint, completes 5,000 new
updates and fails retention. Final shifted mean `3.042173` and CDF KS `.081933`
improve on the archived confidence-LR baseline but do not satisfy stationary,
shifted or reacquisition requirements. Sign switching's causal role remains
unmeasured. The packet SHA-256 is
`66c4af74d8e27993121dd42fdbc0a11cc9d3b7d627b9c2055d6b8ac1a00571a3`.
Its other five required Tier 1 tasks are unmeasured at this source; it cannot
inherit the earlier six-task screen.

The cumulative packet SHA-256
`63ab82b74469ffb541dcbaa2761885babe22cd57028684dff2d19beb8545b661`
contains **77 distinct attempts: 33 PASS / 37 FAIL / 7 INCOMPLETE**. Its 74
earlier rows are exactly preserved, followed by the one finite-cap and two
orientation attempts. This is a source-scoped attempt ledger, not a pooled
winner or an addition to the 249-run non-Atlas execution count. The old timed
out word control still prevents an exact cumulative optimizer-update total.

The three new GPU attempts cost 26.026444/27.432728 queue/supervisor seconds
for finite cap and 99.088163/101.675703 for orientation. Those distinct clocks
are not summed into a speed comparison. All three original grades, task/source
bindings, requests, terminals and 24 retained file pins were independently
checked; this evidence-only import adds no new training, sampling, scoring or
default change. The coordinator's implementation remains outside this branch.
