# What sustained critic refinement would require

The [critic-refinement candidate](pr84_critic_refinement.py) has a finite, strong
stationary result: its source-bound continuation passed all 1,200 dense checks
from update 1,201 through 2,400, with minimum HQ `0.939453125` and all eight
modes present. Its cold conditional trajectory also passed with identity MSE
`0.000909835`. These are observed horizons, not a guarantee at every future
update. The first cold ring attempt stopped during a nonfinite L-BFGS trial
after roughly 460 updates; that is an unclassified numerical exception, not
a measured ring-quality failure. Its exact replay and solver policy need to
be settled before drawing a ring conclusion.

## A sufficient condition that does not need an exact critic optimum

Let `z` contain **all persistent state**: G/prior and D parameters, both Adam
moment tensors and counters, EMA tensors, and the update/noise clock. Let
`T_t(z, ξ)` be one *implemented* outer update with the current minibatches and
Gaussian draws `ξ`: D Adam and its bound, bounded empirical penalized D fit
with its finite-point selection, G's frozen stencil and bound, then EMA.
The fit can remain inexact. A local, time-uniform sufficient condition is a
complete finite map on safe sets `S_t`, and a distance `d` and reference
trajectory `z*_t` such that, for every `t` in the intended hold and every
admissible draw,

```
d(T_t(z, ξ), T_t(y, ξ)) <= q d(z, y),   q < 1,  z,y in S_t;
d(T_t(z*_t, ξ), z*_(t+1)) <= b.
```

If each closed `d`-ball of radius `R >= b/(1-q)` about `z*_t` lies within
`S_t` and within a **certified** eight-mode/HQ-passing region, successive
balls are forward invariant. This is a direct triangle-inequality argument;
`b` may absorb
persistent critic fitting error and sampling disturbance. Matching clocks
and Adam counters are required between compared states; their deterministic
advance is represented by the reference trajectory. This assumes neither
an exact D best response nor convergence to a still generator. A local
input-to-state variant can split the deviation into G functional error and
critic-field error, bound their next-step magnitudes by a nonnegative 2×2
gain matrix plus disturbances, and require that matrix's spectral radius
below one. Such a bound would have to concern the *partial G field used by
the host*, not a total gradient through a hypothetical D optimum.

None of these sufficient conditions is established here. Gaussian samples
are unbounded, the nonlinear critic and best-finite-iterate L-BFGS selection
can switch branches, and a finite observed HQ margin is not a certified
state-space ball. The ring exception also means the current source has not
yet shown a finite update map on every encountered state. A weaker
common-noise mean-square contraction with RMS innovation `σ` would yield a
uniform **per-time** bound of the form
`RMS_t <= q^t RMS_0 + σ(1-q^t)/(1-q)`; it would not guarantee that no
sample path ever crosses the quality boundary. Even the simple stationary
recursion `x[t+1] = q*x[t] + Gaussian noise` has nonzero probability outside
any finite interval. It is therefore possible in principle for an inexact
empirical D fit to support stable, constant-rate behavior while finite
passing runs alone cannot prove pathwise indefinite quality.

This distinction matches the assumptions in the primary literature.
[Cothren, Bullo and Dall'Anese (2026)](https://epubs.siam.org/doi/10.1137/24M1684736)
obtain uniform disturbed tracking under contractive fast and reduced
subsystems and a bounded time-scale parameter; those properties are not
verified for this stochastic MLP/Adam/L-BFGS game.
[Kwon et al. (2025)](https://proceedings.mlr.press/v258/kwon25a.html)
prove geometric convergence to a joint stationary distribution with nonzero
constant-step bias and variance for **linear** two-timescale stochastic
approximation. That supports a distributional interpretation, not a
pathwise-quality theorem for this host.
[Zeng and Doan (2024)](https://proceedings.mlr.press/v247/zeng24a.html)
assume a strongly monotone lower-level root operator; measured finite D
loss decreases and coherent held-out G directions do not verify it.

The [hold fork's per-step dynamics receipt](../../artifacts/continuous-learning/round5/critic-refinement-hold/forks/refinement.json)
shows G-bound factor medians declining approximately
`0.157 -> 0.136 -> 0.110 -> 0.088` in windows 1,001–1,200,
1,201–1,600, 1,601–2,000 and 2,001–2,400, while
critic sharpness rises about `0.792 -> 0.899` and fitted-D displacement
falls about `0.250 -> 0.198`. The nominal learning rate remains constant.
This is a state-dependent bound outcome, not a clock-based decay. Adam's
second-moment history can also change the pre-bound proposal. Judge the
accepted **clean-output movement**, current D fitting error, and response
to perturbation, rather than treating a smaller factor or zero movement as
failure by itself.

## One cheap falsification probe for a later source-bound checkpoint

The existing hold summary records full-state hashes but does not serialize a
late candidate state. When an exact validation replay next captures one
passing late **pre-step** state, preserve parameters, both optimizer states,
EMA, counters, all RNG streams and noise scale. Verify an unperturbed
four-step continuation against its archived step receipts first. Then use
two cloned perturbations, each the size of one *recorded accepted* change:
one along the previous G/prior parameter change and one along the accepted
D-to-fitted-D parameter change. Keep their starting Adam moments identical
to the saved state. If a direction is exactly zero, report that instead of
inventing an amplitude. Use the same four future minibatches and output
noise tensors in baseline and perturbed continuations. Restore each clone
before the next branch. No target centers enter the update or the probe's
response metric.

At offsets 0, 1 and 4, record clean 12-particle support RMS separation on
fixed latent indices, centered D-logit RMS separation on one frozen
real/fake cloud, full parameter/Adam/EMA block differences, accepted
functional G-step differences, finite fit losses and closure counts, bound
factors, and selected L-BFGS branches. Predeclare a dimensionless augmented
state distance as the Euclidean sum of each persistent tensor block's
relative difference, dividing by `max(reference block L2 norm, 1)`; demand
exact counter equality. Include the functional separations separately so a
contracting parameter distance cannot hide a harmful output displacement.
Four steps for a baseline and two perturbations require 12 fitted updates,
roughly seven fit-seconds at the measured warm cost before replay overhead;
no new cold training is needed. A disjoint four-step noise continuation from
the same state can estimate innovation separately if the map response looks
promising.

If either paired separation grows over four steps, this falsifies a
four-step contraction claim **in the declared metric and tested direction**
at that checkpoint. It does not rule out another metric, another local
region, or eventual recovery after a nonnormal transient. Shrinkage is only
suggestive: one state and two directions cannot establish a uniform
Lipschitz bound, a bounded noise tail, or a quality-safe radius. Repeat at a
second late state before suggesting time uniformity because Adam bias
correction and moments change even with fixed nominal learning rates. A
sample-noise response comparable to the observed margin warrants a finite
horizon risk statement, not a forever claim. The candidate's current
nonfinite ring trial must also be replayed and classified before applying
any contraction interpretation across that update.
