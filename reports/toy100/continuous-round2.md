# Continuous learning: second research round

No replacement qualifies yet. The original scheduled production recipe stays
unchanged. This round tests mechanisms rather than another learning-rate grid.
Rest at a matched target is acceptable; cold acquisition, sustained live quality,
and response to a new learnable signal determine success.

The inherited passing-state fork is a cheap, deliberately conservative rejection
filter. Its failure does not prove a method could never reach a different stable
state from scratch. Likewise, a trajectory failure rejects a shared recipe; it
does not by itself prove failure to acquire the ring. Any exception to the
filter order needs a concrete mechanism and a declared diagnostic scope, without
claiming promotion credit or changing the production gate.

## Loss-budget experiment

The measured generator drift motivates controlling how much apparent loss
improvement a single proposal spends. For ordinary zero-momentum Adam proposal
`delta`, use `alpha = min(1, max(L_G - log(2), 0) / (-g dot delta))` and apply
`alpha * delta`. A zero descent denominator gives zero movement. D remains
ordinary Adam; all optimizer moments still advance once per host update.
Two predefined scopes distinguish the G network from G plus the learned prior.
There is no elapsed-time input, minimum step factor, host center or quality oracle.

This is inspired by [preconditioned stochastic Polyak steps](https://arxiv.org/abs/2310.02093),
but `log(2)` is only an equilibrium reference. It is not a valid sample-loss
lower bound for a fixed, possibly stale critic. We make no SPS convergence claim.
[Orabona and D'Orazio, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e45879046fd900c2536e419e361c94c0-Abstract-Conference.html)
provide both a surrogate-loss interpretation and negative convergence results
for Polyak variants. The newest related primary source,
[Yoon et al., September 22, 2026](https://arxiv.org/abs/2609.26581),
studies Polyak extragradient for monotone root finding and shows a stochastic
nonvanishing-step failure without a common component solution. Those assumptions
are not established for our nonlinear GAN, and its decreasing-step remedy does
not fulfill this task.

| Update | Passing warm checks | Minimum HQ | Mean step factor | Cold continuation |
| --- | ---: | ---: | ---: | --- |
| Scheduled identity | 200/200 | .99634 | scheduled | Control only |
| Observed constant Adam | 6/200 | .00537 | 1 | Not run |
| Network loss budget | 54/200 | .14795 | .71568 | Rejected before cold |
| Joint G/prior loss budget | 98/200 | .32275 | .64225 | Rejected before cold |

Both candidates fail at the first filter. The reference gap remains positive
at every warm update, so neither ever rests; 34/200 network and 24/200 joint
proposals take a full step. In the joint case, the first failure at update1005
has budget .07056 versus predicted improvement .02486, accepting the full step.
This criterion does not sufficiently identify damaging motion near the passing
state. No arbitrary smaller-budget sweep followed.

The scheduled identity has exact final model, moments, EMA and RNG parity with
the uninterrupted control. The observation-only controller reproduces the known
constant-rate final state hash. Actual warm rates remain G/D .00425 and prior
.0085. Three analytic tests verify the linear loss budget, untouched ordinary
moments and prior behavior, exact observation control, and rest followed by a
response to synthetic signal. That last test is an implementation check, not
the required training distribution-shift gate.

Reproduce with `python -u reports/toy100/loss_budget_probe.py --output NEW_PATH`
using the [handoff environment](continuous-learning-handoff.md).
[Results](loss-budget-results.json) and
[exact compressed evidence and source archive](continuous-evidence/round2/loss-budget-warm/)
are preserved. The driver supports cold testing only for warm survivors; neither
current candidate qualifies for that path. All evidence is scratch-only.

## Generator functional metric

The earlier scalar output-motion bounds failed cold acquisition. This experiment
changes the direction as well as the size of the G-network update, using the
clean generator Jacobian on the current training batch. For Adam's positive
diagonal metric `P`, Jacobian `J` and batch size `N`, solve
`(P^-1 + J.T J / (N eta)) delta = -gradient`. A Woodbury solve implements this
using the rounded ordinary Adam proposal as its right-hand side. The fixed
`eta=.029` uses the existing configured output-noise scale as a single declared
functional step. D and the learned prior retain full ordinary Adam updates.
This penalizes proposed output displacement, not the parameters or critic
gradient at zero. No clock or training-quality oracle enters the update.

The connection between network geometry and GAN dynamics is motivated by
[Franceschi et al., ICML 2022](https://proceedings.mlr.press/v162/franceschi22a.html).
[C-CHAIN, May 2025](https://arxiv.org/html/2506.00592v1) connects functional churn
and NTK geometry in continual RL. Our exact empirical generator damping is an
experimental adaptation, not either paper's method or a convergence guarantee.

The warm filter passes **200/200**, minimum HQ .9375, final 8 modes/HQ1.
Mean linearized proposed clean movement .06037 becomes .002127 after correction;
measured nonlinear movement is .002126. The full host costs 9.19 seconds.
The independent audit confirmed the Woodbury formula, exact duplicate-row
weighting, old-parameter Jacobian evaluation, unchanged prior/D, and RNG checks.
The correction is local: nonlinear displacement is measured, not bounded.

The next, cheaper cold trajectory gate **fails**, final identity MSE .037305
against .02, with 0/24 passing checkpoints. MSE is .05093 at100 and .03731
at200, then fluctuates around .037 through400. This is not evidence of a
candidate that merely needs a few more frozen-budget updates. The cold host
costs 15.44 seconds; no ring acquisition or longer continuation followed.
The Jacobian has 192 output rows and 6544 network parameter columns on this
host, in addition to ordinary training gradients. These extra derivatives
must not be confused with the single Adam moment update per outer step.

Three analytic tests compare against an independent primal matrix solve, exact
duplicate-row weighting, zero-field rest, and a linear generator with an
unmodified prior update. Only deterministic buffer-free MLPs are supported;
exceptions after the Adam proposal are fatal, not recoverable trial rejections.

The first cold run completed training but failed JSON serialization because a
`Recipe` object was included in context. The corrected driver serializes the
explicit applied-policy and noise receipts. The same candidate was rerun to
save its result; this is a serialization repair, not a seed experiment.
The failed attempt's exact sources and error log are retained alongside
[warm evidence](continuous-evidence/round2/functional-metric-warm/) and
[the complete cold rerun](continuous-evidence/round2/functional-metric-cold-v2/).
