# Critic tracking and a clockless responsiveness test

The current evidence favors reliable opponent tracking over another scalar
damping grid. Original PR84 can have a coherent wrong G field; finite D
refinement changes that field and passes the borrowed-state hold through
2400. Cold acquisition is a separate unresolved condition. The
[saved472 response diagnostic](pr84-profiled-field472.md) also shows why a
small own-curvature estimate cannot justify raising G's gain: the responding
critic introduces substantially more restoring curvature than frozen D.

Recent primary research gives useful distinctions, not a ready convergence
certificate for this host:

* [Kwon et al., AISTATS 2025](https://proceedings.mlr.press/v258/kwon25a.html)
  analyze linear two-timescale stochastic approximation with constant
  steps. The result concerns convergence to a joint stationary distribution,
  with nonzero step-dependent bias/variance, rather than exact resting at a
  point. A nonlinear Adam game with nonconverged MLP fits is outside that
  analysis. Persistent movement alone is therefore not a failure criterion;
  continued quality and response to meaningful error are the relevant tests.
* [Sarkar and Aggarwal, June 2026](https://arxiv.org/abs/2606.14488)
  identify bias induced by nonlinear fast-variable tracking error and a
  correction for a contractive normal form. Their theorem uses diminishing
  separated steps and additional structure. It motivates measuring response
  error, but does not authorize importing their schedules or subtracting an
  unmeasured bias in this host.
* [Yoon et al., September 22, 2026](https://arxiv.org/abs/2609.26581)
  show that a nonvanishing stochastic Polyak extragradient can fail to reach
  the mean operator's root when component operators do not share a solution.
  We have neither a verified monotone game nor common minibatch roots here.
  A state-adaptive scalar by itself supplies no responsiveness guarantee.
* [Fiez et al., ICML 2020](https://proceedings.mlr.press/v119/fiez20a.html)
  distinguish implicit-response dynamics for general-sum games. This host
  has non-saturating Rp G loss, G-only spatial smoothing and a D-only slope
  cap, so its two losses are not one zero-sum value. Neither extra D fitting
  nor the profiled partial-field secant implements a total Stackelberg
  gradient. [Metz et al., ICLR 2017](https://arxiv.org/html/1611.02163v4)
  separately compare unrolling with and without differentiation through D's
  update. That chain term is a distinct, testable next mechanism if cold
  acquisition fails; their zero-sum optimum simplification does not directly
  apply to this general-sum host.

A useful cheap responsiveness test should begin from a **passing complete
snapshot**, preferably the candidate's own cold-acquired state. Keep the
dataset, all moments, EMA, RNG states, noise horizon and nominal rates fixed.
Make one deterministic model perturbation: add `(0.35,0)` to the final clean
G output bias, i.e. five real-noise standard deviations. This is an output
translation of the model, not a target-distribution shift. Do not choose its
direction or size from the quality result.

First compare one actual update at the unperturbed and perturbed states,
using cloned identical data/noise streams. Record the raw G/prior fields,
post-Adam denominator, unbounded proposal, own-curvature factor, accepted
clean-output movement, D-fit residual and held-out field change. A factor
near zero is not by itself a rejection: the question is where an available
corrective field is lost. A nonzero raw field suppressed by the metric,
a vanishing shared-network pullback, and an excessive curvature shrink are
different mechanisms. If the first update suggests a response, a bounded
20-update continuation with every live update observed can compare the
perturbed learner against an identically perturbed frozen-G/prior control
(D may continue fitting), plus the unperturbed learner. Recovery is assessed
with the unchanged quality gate; returning individual particles to their
old identities is not required.

For explicit clock dependence, a still cheaper deterministic proposal check
can replay identical tensors, moments, cached batches and fixed late noise
at two outer-step labels. Preserve Adam's saved step/bias correction. The
late noise schedule and cap are already constant, so changing only that
label should not change a proposal. This separates an elapsed-time switch
from state-induced throttling, but cannot by itself rule out a state-induced
freeze. The model perturbation is the behavioral test. No positive-movement
floor, new seed, clock schedule, target-center controller or gain sweep is
part of this design. These responsiveness tests are proposed, not executed.

A one-step total-gradient diagnostic must also handle a concrete source
issue: `GradRegularizer._grad_norm` detaches fake coordinates. Ordinary D
gradients are correct, but naively differentiating that recorded gradient
omits how the cap field changes with generated data. A separate functional
cap must preserve the original value and D gradient while retaining this
mixed derivative, and the full virtual-step surrogate requires a directional
finite-difference audit. Nonsmooth activation or cap crossings must be
reported; they do not justify silently relabeling a partial derivative.
