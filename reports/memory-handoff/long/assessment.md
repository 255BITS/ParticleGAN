# Final result: pointwise handoff round and 5k checks

All 12 jobs completed successfully: ten 2k scouts, then two exact continuations
to 5k. Both queues are empty and finished. No generated training trajectories,
cold/warm losses, or trajectory discriminator were used in any new run. The only
long rollouts were end-of-run evaluations. All 48 focused tests passed before the
scouts, including an active exact B-cap ownership/equivalence check and exact resume.

| Variant | Updates | Cold256 | Cold1024 | Late stopped, cold1024 | Prefix8 / prefix32 reference-orbit passes |
|---|---:|---:|---:|---:|---:|
| Dense default | 2k | 0% | 0% | 68.8% | 0% / 0% |
| Input noise .10 | 2k | 0% | 0% | 4.7% | 0% / 0% |
| Input noise .10 | 5k | 0% | 0% | 75.0% | 0% / 0% |
| Zero-prefix fraction .25 | 2k | 0% | 0.78% | 46.9% | 0% / 0% |
| Zero-prefix fraction .25 | 5k | 0% | 0% | 30.5% | 0% / 0% |

Every other 2k scout also had zero full cold256/1024 and zero original-orbit passes.
The solitary 2k long-horizon pass did not persist at 5k. The success diagnostic is
not necessarily monotonic with horizon: it uses horizon-wide averages and a late
window, so a path can pass at 1024 while failing at 256. This is not a guarantee
that every shorter segment passes.

Training cost: 20.6–78.7 seconds per 2k scout; reducing real-prefix length reduced
cost substantially. The two follow-ups each added approximately 117 seconds for
3k updates. These measurements are for this tiny batched toy on the current GPUs,
not a scaling guarantee. The first ten jobs took 6.59 minutes of queue wall time.
Caching real context and removing generated training rollouts makes further local
formulation exploration inexpensive.

## Interpretation and recommendation

Dense supervision reduced initial handoff error (prefix32 .098 reference radii
versus .118 for the old one-point baseline), but local predictive accuracy did not
produce stable autonomous memory dynamics. The two longer checks did not repair
the failure. Noise .10's reduced stopping at 2k was not stable with further training.
These experiments do not establish that handoff-only GANs cannot work; no tested
configuration solved this task at the tested budgets.

Do not promote these runs further solely on the current evidence. Respect the
user's no-trajectory-loss constraint. A useful next local formulation scout is to
train D with temporally mismatched real points as additional conditional negatives,
testing whether stronger pressure to encode temporal relationships in M helps G.
This can stay entirely pointwise without a generated training rollout. Keep it a
separate configured loss and validate causal timing, since it changes D's negative
distribution. A second option, if wanted later, is bounded one-step generated
feedback exposure; it would be a distinct experiment, not a full path critic.
Neither proposal has been implemented or tested in this round.

[Complete leaderboard and cost table](leaderboard.md).
[Initial scout assessment](../round1/assessment.md).
