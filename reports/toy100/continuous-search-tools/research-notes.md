# Candidate directions after the early continuous-search failures

These are untested hypotheses for K3P-derived proposals, not qualification claims.
K3P remains the shared base. A3's round-2 verification is **4/22 PASS, 18/22 FAIL**;
its successful later ring hold does not replace acquisition or transfer gates.
Read measured lane reports before declaring a proposal to avoid duplicates.

## Local-curvature step control

Malitsky and Mishchenko's adaptive gradient descent uses a growth bound together
with a local secant estimate:
`eta_next = min(sqrt(1 + eta/eta_previous) * eta,
norm(x - x_previous) / (2 * norm(g - g_previous)))`.
The iteration does not require a final training horizon. Their convergence
analysis concerns convex minimization with locally Lipschitz gradients; it does
not establish convergence for this stochastic nonconvex GAN.
[Paper, Algorithm 1](https://proceedings.mlr.press/v119/malitsky20a/malitsky20a.pdf).

Our proposed test: use a declared, reversible local-step rule rather than an
ever-growing historical gradient denominator. Sampling noise and movement of the
opponent can corrupt a secant estimate. Any smoothing, shared-sample evaluation,
preconditioning or positive lower bound is a new declared formulation, not the
paper's theorem. Log actual growth/contraction and retain K3P's particle behavior.
An estimator memory or initial calibration period is allowed; a required final
training duration or a scripted reset at the known target change is not.

## Negative momentum in the applied update

Gidel et al. study alternating game updates with negative momentum and establish
convergence for their analyzed bilinear setting. The update uses the previous
actual displacement: `x_next = x - eta * game_gradient + beta * (x - x_previous)`,
with negative beta in the stabilizing regimes considered. This is not a theorem
for K3P, its Adam preconditioner or a changing target distribution.
[Paper](https://proceedings.mlr.press/v89/gidel19a/gidel19a.pdf).

Our proposed test: preserve the alternating order and distinguish recursive
negative displacement memory from the already failed round-2 GD1 correction
`raw_delta + .5 * (raw_delta - previous_raw_delta)`. A new proposal must explain
that distinction, declare its handling of generator/prior groups and inactive
sparse rows, and keep Adam-moment changes explicit. No beta coefficient grid.

## Shared-sample extragradient

Mishchenko et al. use the same stochastic sample for the predictor and correction
evaluations to approximate an implicit step. Their analysis uses monotone
operators; their GAN experiments also account for the added iteration cost.
[Paper, Algorithm 1](https://proceedings.mlr.press/v108/mishchenko20a/mishchenko20a.pdf).

Our proposed test: a consistent game predictor/corrector, reusing the same real
batch, latent/noise draws and declared loss, with one committed update per role.
The predictor must not silently advance Adam moments, critic EMA, latent history
or optimizer counts. Snapshot and restore all temporary state; declare every extra
forward/backward evaluation. Preserve the frozen outer-step budgets. If a clean
implementation cannot be qualified within the attempt, report that limitation
instead of substituting hidden extra training.

## Evidence-driven controller direction

Round-1 RCM3 retained ring hold/extension and pre-shift hold but recovered 0/81
after reopening the early penalty at small rates. RUC3 retained gated hold but
its full-rate reopen lost recovery precision and it failed pre-shift checks.
These are diagnostic observations, not verified new bases. A possible K3P-derived
proposal separates moderate reversible mobility from restoring the early R1
mixture. It must earn acquisition and all gates itself. Do not combine scores
from these candidates or repeat their unchanged failed rules.

Every promising live result still needs hold+extension, stationary and pre-shift
checks, 81/81 deadline recovery, a matching frozen control, all 22 toy gates,
horizon-independence evidence and delayed/repeated changes. Paper references
explain hypotheses; only the executed frozen protocols establish qualification.

## Training discrepancy independent of critic optimization noise

RP1's grid center error improved to .20631 sigma by update 5500, then worsened
following a rate increase on a stationary target. Its raw-gradient signal is
therefore a questionable proxy for remaining distribution error. AC1 and AC2
instead used gradient or adversarial-score persistence; both passed the short
image screen. AC1 never converged on the ring, and AC2 lost a mode after 187 good
hold checks. Those are separate failed mechanisms,
not qualified bases.

Gretton et al. define maximum mean discrepancy for comparing two distributions
through kernel expectations and give statistical tests and computational
estimators. [Primary paper](https://www.jmlr.org/papers/v13/gretton12a.html).
This motivates an untested K3P controller: compare current real/generated
training minibatches, and compare real/real splits to estimate sampling variation.
Use a declared kernel mixture and bounded deterministic batch subsets. A mismatch
signal could retain acquisition or reopen mobility while avoiding reactions to
critic optimizer noise. It must earn every original gate.

This proposed adaptive, temporally dependent control is not the paper's test or
its guarantee. Do not treat a nominal one-shot significance threshold as a valid
sequential error guarantee. No benchmark evaluator, held-out score, true mode
center, task identity, known change time or training horizon enters the learner.
Do not replace GAN training with assignments to known targets. Preserve the
selected K3P learning mechanism and account for every extra computation.

## Correcting the game update direction

The completed confidence lane found that half-batch gradient agreement can stay
high while the ring fails, and a margin-variance controller can close before
acquisition. The joint-trust lane also failed by constraining displacement.
These results motivate testing an update-direction correction, not another
threshold on gradient size.

Mescheder et al. add a gradient of the squared joint game vector field to their
GAN updates. Their ascent notation is `v - gamma * grad(||v||^2 / 2)`; for a
descent field `F`, the corresponding step is along
`-(F + gamma * grad(||F||^2 / 2))`. The gradient includes cross-player terms.
Their analysis is local and conditional; the paper also discusses minibatch
bias. [Primary paper, Section 4 and Algorithm 2](https://www.nowozin.net/sebastian/papers/mescheder2017gannumerics.pdf).

A K3P-derived test must declare its field, sign, role ordering, preconditioning,
and treatment of critic regularization and sparse prior history. The paper does
not establish convergence for this Adam/EMA learner or changing distributions.
Computing only a critic's own gradient norm is not the full joint correction.
Use a small analytic game to catch missing cross terms before ring training.
Bound and report extra evaluations; no extra optimizer updates or altered gate
budgets. This is a hypothesis, with no transferred gate passes.
