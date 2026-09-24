# Continuous learning: second research round

No replacement qualifies yet. The original scheduled production recipe stays
unchanged. This round tests mechanisms rather than another learning-rate grid.
Rest at a matched target is acceptable; cold acquisition and sustained live
quality on a fixed target determine success.

**Scope clarified September 23, 2026:** adapting to a changed target distribution
is a separate problem and is no longer a promotion requirement for PR #60.
Historical shift declarations and measurements below remain intact as optional
diagnostics. Removing that requirement changes none of the eight verdicts:
each first failure is fixed-target acquisition or stationary quality. The
trajectory host uses a fixed conditional dataset; its name does not imply a
distribution shift.

The inherited passing-state fork is a cheap, deliberately conservative rejection
filter. Its failure does not prove a method could never reach a different stable
state from scratch. Likewise, a trajectory failure rejects a shared recipe; it
does not by itself prove failure to acquire the ring. Any exception to the
filter order needs a concrete mechanism and a declared diagnostic scope, without
claiming promotion credit or changing the production gate.

The work used Sol max and Astra max in separate worktrees, with local CPU tests
and no seed sweeps or GitHub CI dependency. Eight additional configurations
were tested; none qualifies:

The [round-two ledger](continuous-round2-results.json) is separate from the
first round's counts. **170 integrated local tests passed in 25.29 seconds**;
the [test log](continuous-evidence/round2/integrated-tests.log) is retained.
Production trainer/configuration files and PR #60's scheduled head are unchanged.

| Candidate | Warm checks | Cold trajectory | Cold ring | First disqualifying evidence |
| --- | ---: | --- | --- | --- |
| Network loss budget | 54/200 | Not run | Not run | Warm quality |
| Joint G/prior loss budget | 98/200 | Not run | Not run | Warm quality |
| Generator functional metric | 200/200 | MSE .037305, FAIL | Not run | Shared acquisition |
| Fresh-data D allocation | 7/200 | Not run | Unplanned diagnostic, FAIL | Warm quality |
| Marginal energy gate | 200/200 | MSE .2764, FAIL | Not run | Shared acquisition |
| Conditional energy gate | 200/200 | MSE .003682, PASS | 0/5, FAIL | Ring acquisition |
| Conditional energy backtracking | 198/200 | Not run | Not run | Warm quality |
| Projected implicit skew response, repaired | 197/200 | Not run | Not run | Warm quality |

Controls, fixed-cloud diagnostics and serialization repairs are not additional
candidates. Unplanned downstream work receives no promotion credit. A small
number of accepted proposals is an observation, not itself a failure criterion.

The [energy-signal report](energy-signal-dynamics.md) and
[D-allocation report](continuous-audit-adaptive-allocation.md) provide independent
lane details. Energy backtracking accepted a full proposal at update 1174 that
improved both measured energy halves while HQ fell from .9990 to .7998. The
[fixed-cloud counterexample](energy_objective_conflict.py) also improves energy
while HQ falls below .9, preserving all eight modes. The signal can therefore
reward a change that violates the required quality metric; this is stronger
evidence than merely observing few accepted updates.

The allocation arm used a new fresh-data e-process from
[Kim et al., August 2026](https://arxiv.org/html/2608.10096), fixing Adam rates
while allowing one to three D updates per ordinary G update. In the warm
window it always reached the three-update cap and failed, so this run became
a 3D:1G comparison. Its driver automatically ran cold ring after warm failure.
That protocol deviation is archived and excluded from promotion; the driver
now stops at the failed warm gate.

## An exactly representable equilibrium diagnostic

The original ring cannot exactly match its target law, but that fact alone
does not explain the drift. A [new matched-population diagnostic](matched-population-diagnostic.md)
freezes the twelve live generator outputs as target centers and matches real
and generated Gaussian noise at .029. It starts the critic at constant zero,
preserving Adam variance, EMA and all RNG streams identically across forks.
An analytic population MMD² is initially zero; no finite evaluation sample
noise enters this metric. This changes the target and critic for attribution,
and does not replace any production gate.

Ordinary constant Adam passes 14/20 hold checks at the predeclared MMD²≤.01
bound; functional damping passes 20/20. Neither sustains recovery after the
same target shifts by +.35. Each has its own exactly matched frozen sibling.
Functional damping finishes at MMD² .20521 versus frozen .19188, showing that
better stationary hold does not establish responsiveness in that separate
adaptation study. Shift failure is not a PR #60 veto. The original noise
horizon stays 1200, and there is no optimizer reset at the shift. This gives
future agents a small test with a representable starting law in addition to
the unchanged real-task acquisition and quality gates.

Run it with `python -u reports/toy100/matched_population_diagnostic.py --output NEW_PATH`
in the pinned [handoff environment](continuous-learning-handoff.md).

## Coherent feedback, objective cancellation and rotational dynamics

The [mechanism report](continuous-mechanism-round2.md) contains the complete
per-row force measurements, exact replays, projected-skew formulas, five
analytic/host audit tests and all runtime source archives.

The exact local replay of the earlier competitive method showed that all
twelve particles retain their nearest ring-mode assignments throughout the
warm suffix. The loss of HQ comes from moving away from centers, not changing
mode assignments. The first failing particle drifts from distance .0506 at
update 1180 to .2538 at 1194, beyond the .21 quality radius. Those failing steps
still improve both players' losses against the proposed opponent. A simple
per-player loss-improvement guard would therefore not directly catch them.

The earlier cross-only cold trajectory near miss is also more specific than
uniformly slow learning: ten identities become nearly exact while rows 2 and 3
swap targets. At update 400 their adversarial and set-cover output forces have
opposing cosines about -.995 and similar norms around .085, leaving a combined
norm near .008. The critic points toward the correct identity, while the
existing set-cover objective pulls toward the wrong matched set member. No
host objective or threshold was changed to remove this conflict.

Astra then tested a rank-two implicit correction for the antisymmetric part
of the cross-player Jacobian in the frozen Adam metric. Central same-batch
finite differences estimate two genuine cross-field directions; a 2×2 inverse
skew solve corrects their rotation while bounding metric displacement by the
ordinary joint proposal. Exact zero fields remain still, and symmetric fields
are unchanged in the measured plane. This is a new projected method inspired
by SGA, not a reproduction of
[Vater et al.'s low-rank SGA, revised July 2026](https://arxiv.org/abs/2510.25716).
It has no target oracle, elapsed-time schedule or zero-centered loss penalty.

The first version passed 199/200 warm checks, failing only the initial
transplant update 1001. It passed all five original terminal checks and every
subsequent dense check. That isolated transition justified one **explicitly
declared cold diagnostic exception** to the conservative warm filter, testing
whether its own cold dynamics could reach a good attractor. The warm FAIL was
retained; no promotion credit was granted. The cold diagnostic stopped at
update 8 with a finite-difference rounding error, before any acquisition verdict.
A shared perturbation epsilon made one player block too small to measure.

The repair uses a separate representable perturbation size and quotient for
each perturbed player, retaining the same mathematical method and the same
5% rounding guard. Independent math review found no defect. With the repaired
source, the warm arm passes 197/200, failing updates 1001,1142,1143; final HQ
.99976 and all five original terminal checks would again hide real excursions.
The single-transition exception no longer applies, so no repaired cold run was
launched. The invalid first cold attempt is not labeled acquisition failure.

These results argue against tuning only a scalar residual cutoff, movement
floor or energy threshold on these same examples. A useful next mechanism must
identify when current critic feedback supports an achievable improvement while
preserving quality near a learned fixed target. The matched-population diagnostic
separates equilibrium drift from the original ring's target mismatch; its
post-shift segment addresses the separate adaptation problem. Any future
candidate still needs the original shared acquisition, continued live quality
and production common-22 gates.

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
proposals take a full step. In the joint case, the first failure at update 1005
has budget .07056 versus predicted improvement .02486, accepting the full step.
This criterion does not sufficiently identify damaging motion near the passing
state. No arbitrary smaller-budget sweep followed.

The scheduled identity has exact final model, moments, EMA and RNG parity with
the uninterrupted control. The observation-only controller reproduces the known
constant-rate final state hash. Actual warm rates remain G/D .00425 and prior
.0085. Three analytic tests verify the linear loss budget, untouched ordinary
moments and prior behavior, exact observation control, and rest followed by a
response to synthetic signal. That last test is an implementation check, not
evidence of training distribution-shift recovery.

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
against .02, with 0/24 passing checkpoints. MSE is .05093 at 100 and .03731
at 200, then fluctuates around .037 through 400. This is not evidence of a
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
