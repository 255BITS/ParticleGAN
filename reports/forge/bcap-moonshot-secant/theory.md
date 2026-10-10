# Bounded secant proposal lengths: frozen hypothesis

The selected incumbent has a constant normalized step, with no active projection
or transport. Saved Gaussian diagnostics associate continued loss of fit with
constant normalized motion, while Phase 2 shows that adding transport, projection,
CPU SVD and critic guards can regress ring retention. This track changes only
`optimizer_secant_mode="bounded"`; it keeps native SVD and every incumbent field.
It estimates a local response timescale from the already consumed gradients.

[Malitsky and Mishchenko, Adaptive Gradient Descent without Descent](https://arxiv.org/abs/1910.09529)
limit gradient-descent steps by parameter displacement divided by twice the
gradient difference norm, together with a slow-growth condition. Their result
assumes a convex objective and local smoothness. The present stochastic game
and normalized directions satisfy neither that algorithm nor its guarantee.

For each parameter tensor (or each actually owned sampled prior row), let
`g` be the current raw gradient, `s=x-current_previous_x`, and
`y=g-previous_g`. The unmodified optimizer first makes its actual rounded
proposal `d`. Its equivalent normalized direction length is `||d||/eta_nominal`.
The adaptation therefore bounds the **length fraction**, rather than pretending
that a raw-gradient learning rate controls normalized motion:

```
curvature_fraction = ||s|| ||g|| / (2 ||y|| ||d||)
growth_fraction = sqrt(1 + previous_theta) previous_eta / eta_nominal
alpha = max(1/16, min(1, curvature_fraction, growth_fraction))
x_next = x + alpha d
eta = alpha eta_nominal
theta = eta / previous_eta
```

Zero `y` or zero proposal leaves curvature unrestricted. The first observation
uses alpha=1. Zero nominal rate leaves motion zero. Full-scale proposals preserve
the original rounded tensor exactly; reduced proposals retain its displacement
ray subject to parameter-dtype rounding. Existing parameter step clocks advance
once. Each sampled row checkpoints its own observation/visit clock, previous
position and gradient, rate, theta and validity; unsampled histories and positions
remain unchanged. Convolution kernels use the full actual assembled proposal,
including existing channel and offset factors. No forwards, data draws, gradient
evaluations or RNG streams are added.

The fixed 1/16 floor bounds starvation from noisy differences, and the nominal
ceiling avoids an unregistered acceleration. Both are fixed for every task. The
floor relaxes the curvature inequality. Changes in minibatches, other parameters
and the critic appear in `y`; sparse revisits especially confound the estimate.
This is a finite-step CFL-inspired heuristic, not a measured Lipschitz constant.
Its benefit would be reduced overshoot; the competing explanation is indiscriminate
damping of response, with slow adaptation after target changes.

The numerical prediction is unequal-width final component covariance error
at most .85, with its original five-check terminal suffix and all other task gates
required for a repair. A covariance error above .85 falsifies the registered scalar
prediction. Any failed original gate prevents a repair claim even if the scalar
prediction passes. Preserve all six incumbent Tier 1 gates; report Gaussian and
word own-state continuations, rare allocation, native precision and runtime
separately. Floor-hit frequency, mean applied fraction and owned history counts
will show whether the controller actually acted and whether confounded damping
is a plausible explanation.

This differs from the archived moving-critic optimism negatives: it predicts no
future direction, and never substitutes `2g-previous_g`. It also differs from
round-five balanced assignment, prior-only transport and CG4 force filtering:
there is no assignment, empirical transport loss, output-kernel solve or new
objective. Those source-bound failures remain unchanged at
`653c38045618ad240524237a9c141c8d06b28c03`. Phase 1/2 source identities, clean
served-law gates, and the explicit fixed two-pole initialization cohort remain
in the original reports and `baseline.json`.

One paired seed-0 comparison freezes all sixteen original questions, their public
initialization, data laws, priors, seen batches, budgets and scoring cadence. Each
arm reserves 22,920 seconds; 45,840 total within a 48,000-second paid ceiling.
Targeted software verification has 300 seconds. No retries, tuning, seed studies,
ordinary qualification or default adoption follow these measurements.
