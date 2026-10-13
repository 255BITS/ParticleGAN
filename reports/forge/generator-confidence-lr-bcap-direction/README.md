# Confidence-scaled generator response: complete screen, failed retention

**Result:** the corrective confidence-LR recipe passes all **6/6 original
required Tier 1** seed-0 tasks with complete actual optimizer histories. Its
separate original word hold passes **25/25**, but Gaussian retention fails
**8/72** stationary and **0/24** shifted-hold checks, with failed reacquisition.
The six-task screen is a real bounded result. It is not full Martyn superiority,
Tier 2/3 qualification, a global winner or a release/default decision.

| Original required task | Grade | Complete actual optimizer history |
| --- | --- | --- |
| `unused_token_hold` | PASS | D/G 200 |
| `ae_gan_hold` | PASS | D/E/G/prior 250 |
| `two_pole` | PASS | D/prior 80 |
| `gaussian1d_smoke` | PASS | D/G/prior 1,000 |
| `five_word_joint_smoke` | PASS | D/E/G/prior 20,001 |
| `ring16_acquisition` | PASS | D/G/prior 1,600 |

The opt-in response retains the earlier span-10 EMA but replaces hard dropped
gradients with a positive actual G/E/prior learning-rate gain
`Phi(abs(mean gap)/SE)` in `[0.5, 1]`. Gradients and optimizer moments remain
active, and scheduled learning rates are restored after every step. This is a
confidence proxy, not a calibrated probability. Frozen task architecture,
target/data law, seed, prior, sampling, horizons, original gates and time limits
remain unchanged. The only source changes relative to the prior EMA source are
the public response helper and Recipe mode validation; evaluator, host, model,
task and sampler bytes remain pinned. The executed source digest is
`d48060b87f539c649d493d39c1bba957c098ae66e1e3bb84f8f162859375dbf5`;
candidate revision is
`d3c50d1c7fce44d9d738f5de4ae127e4f380fad921da1d4cb1933d52915a5f0d`.
Its implementation is held in the local coordinator archive and is **not**
part of this evidence-only patch.

The [six-task publication](../test-performance-evidence/pr399-confidence-lr-tier1-6.json)
pins original grades, actual optimizer histories, requests, terminals and
retained artifacts. Its score-ready packet SHA-256 is
`7b33467bc3c006645afcd90a53480903f79dd9e3af3ba2240ba9f11c2d0b8c06`.
The archived PR399 control remains **5/6** because word timed out; hard EMA
remains **4/6** because Gaussian and word failed their original optimizer-history
completion requirements. Their original source identity, cost and negative
evidence are preserved. They were not rerun or recast as current-source controls.

The [own retention publication](../test-performance-evidence/pr399-confidence-lr-own-retention-2.json)
pins full state/RNG restorations of the recipe's *own* Gaussian 1,000-step and
word 834-step prefixes. Gaussian continues for 5,000 new steps and fails
retention: final CDF KS `.259749`, mean `3.491709` against target `3.0`, and
standard-deviation ratio `1.344586`. Word continues for 4,000 new steps to
4,834 and passes its 25/25 hold checks. The retention packet SHA-256 is
`6a79bb0b4cf731ad6daa10fd425df623baa16a80f754c89fc3f6a98d5c62428e`.

The cumulative packet SHA-256
`491c357aaa710442213e8c3210c43df6e1a28c339b4e8b8dc3744061f1066723`
has **74 unique attempts: 32 PASS / 35 FAIL / 7 INCOMPLETE**. All previous 66
attempt rows are unchanged; six new screen and two own-continuation attempts
are added once. This is a source-scoped research ledger, not an addition to the
249-run non-Atlas ledger or an exact optimizer-update total: the older timed-out
control word still has only a verified lower bound.

Eight actual GPU attempts cost **956.941839 queue / 965.845355 supervisor**
seconds for the screen, and **261.705458 / 264.992214** seconds for the two
continuations. These are separate clocks, not sums or controlled speed claims.
There were zero paid retries. The coordinator's original state/output audits,
all 58 retained file pins and chronological cumulative identities were checked
without new training, sampling or regrading during this display import.

The later [extra-critic-reaction analysis](../confidence-critic-reaction/README.md)
rejects a proposed correction before any new training. It adds zero attempts.
