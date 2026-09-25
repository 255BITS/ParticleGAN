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
