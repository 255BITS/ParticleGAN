# Raw-gradient optimistic game dynamics

One global candidate adds only `optimizer_optimism="raw_gradient"` to the exact
Phase 2 incumbent recipe. The control inherits that recipe with zero overrides.
Neither arm enables transport, protected-step projection, finite critic guards,
or CPU SVD. Protocol seed is 0. The original sixteen tasks, prior/data laws,
architecture, public initialization, observations, schedule horizons, consumed
batch streams, updates and numerical gates are retained.

For each owned network parameter application, forecast
`h_t = g_t + (g_t - g_previous)`, then apply the existing smoothed FullDualNorm
map and scheduled step: `theta <- theta - eta * N(h_t)`. The first application
uses `h=g`. Histories store raw gradients, never normalized directions or
learning-rate-scaled steps. Every critic, generator, encoder and learned-prior
parameter participates. The original word host owns encoder parameters in a
generator-labelled group; parameter history remains independent.

Sampled prior rows use their last own application, with independent Boolean
history masks. Duplicate sampled IDs are unioned by the existing public optimizer
and the aggregate autograd gradient is consumed once per row. Unsampled rows
neither move nor acquire history from dense standardization/regularizer terms.
Missing gradients leave history unchanged; an actual zero gradient is a consumed
gradient and can produce an anticipatory nonzero step. History and counters
checkpoint exactly, including Boolean masks restored after Torch's dtype cast.
The algorithm adds no draws, data requests, forwards or backwards.

[Daskalakis et al.](https://arxiv.org/abs/1711.00141) motivate anticipation through
bilinear game cycling. [Mertikopoulos et al.](https://arxiv.org/abs/1807.02629)
provide coherent-game optimism analysis using an extra-gradient construction.
Our alternating, sampled-row, normalized nonconvex update does not satisfy those
proof assumptions. DualNorm largely removes forecast magnitude, but the polar,
bias and row directions retain its changed orientation. The numerical software
control explicitly distinguishes `N(2g-g_previous)` from `N(g)`.

The archived moving-critic diagnostic at source
`da3b0470918fc0045175441ce24e69777b5d5990`,
`reports/toy100/lrfree-search/streaming-smallbatch-toy/RESULTS.md`, found
extrapolated online payoff direction error ratios 1.0016 (null) and 1.0034
(shift), rejecting its predeclared go/no-go. It trained no optimistic controller.
That evaluator-only category-payoff/mass-flow probe differs materially from this
actual all-player raw parameter-gradient update. Historical scratch Optimistic
Adam instead extrapolated preconditioned optimizer directions. Neither supplies
qualified evidence for this candidate. Round-five empirical transport negatives
also remain unchanged under source `653c38045618ad240524237a9c141c8d06b28c03`.

An additional archived raw-gradient-before-Adam package in
`reports/toy100/formulation-round-20260924/game_dynamics` also failed: unequal
mass passed only updates 550–650, then ended with minimum eigen ratio .033979
(bound .15), despite mass ratio .58594. Its declaration's mechanism SHA-256 is
`5c95a2476278c0c06a130c7cdfc9d019afd4013ab2a1bb0462086263e35a055a`.
The material difference here is the downstream operator: smoothed spectral polar
directions and sampled-row normalization have no accumulating Adam second moment.
The forecast changes orientation without changing a historical second-moment
preconditioner. This directly tests whether the earlier loss of shape depended
on that preconditioner and its scale, rather than repeating the Adam package.
The sampled-row history law and current public initialization/prior also have
distinct identities; they are limitations on cross-archive attribution.

The hypothesis is that orientational anticipation reduces rotational lag enough
to repair unequal-mass sustained density while retaining the incumbent's six
original Tier 1 passes. The registered scalar prediction is terminal full
component covariance error <=0.85; >0.85 falsifies that prediction. A covariance
endpoint alone is insufficient: the original complete sustained task gate must
PASS to count as a repair. Gaussian all-72 stationary checks, deadline
reacquisition and all-24 shifted hold checks are secondary full-gate outcomes.
The competing explanation is amplification of minibatch gradient differences,
especially after normalization, stale sampled-row history, or nonrotational
shape/critic-information failures. Failure/regression stops this exact revision.

This is one two-arm research diagnostic, not an ordinary qualification lane.
Own Gaussian and word holds require their own fully completed passing producers.
Eleven original Tier 2 questions are outside scope. Full reservations are
22,920 seconds per arm, 45,840 paired, within the 48,000 paid ceiling. Separate
software allowance is 300 seconds. No retries, seed runs, tuning or additional
scientific configurations follow automatically. Publication uses certified saved
observations and adds zero updates/draws. Raw logs/checkpoints/media stay under
`/mnt/ml7tb/ParticleGAN-forge/bcap-tier1-repair-next/moonshots/optimistic`.

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-tier1-repair-next/moonshots/optimistic/logs/driver.log /mnt/ml7tb/ParticleGAN-forge/bcap-tier1-repair-next/moonshots/optimistic/phase3-queue/queue/events.jsonl
```
