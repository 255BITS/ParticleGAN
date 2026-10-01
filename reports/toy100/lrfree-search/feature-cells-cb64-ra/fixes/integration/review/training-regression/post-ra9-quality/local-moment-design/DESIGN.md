# Reserved local moment birth proposal

Status: **pre-measurement design only**. Consider this single alternative only if
the complete original RA9 grid gates fail. No production implementation,
checkpoint read, forward, optimizer step, chart fit, emission or numerical test
has been performed for this design. Root selects any subsequent diagnostic.

## Source finding

`DataDriftController.observe_pair` acts only for dv8/dv9. The current dv12 policy
uses data innovation, game payoff/alignment and latent geometry; a persistent
local real/fake moment mismatch does not directly supply its mobility signal.
The feature-cell law tests cell mass, cell inside/outside mass and aggregate
inside/outside mass, rather than conditional means/covariances within a cell.
RA9's finer chart can expose some shape errors as cell-mass errors, but equal
category counts do not imply equal local means or covariance.

`latent_to_anchor` aims at one observed real representative. Its first supported,
inside, same-cell point is accepted, even before any target progress. An eligible
seed therefore exits at iteration zero. Changing the target alone cannot make
this helper perform a moment correction.

## One conservative repair

Use local real moments to improve **at most four ordinary copy births that the
unchanged count law has already authorized**. Replace an accepted planned copy
with a separately labeled new latent birth in exactly the planned destination
fine cell. Preserve ordinary copies whenever a moment proposal fails. Do not
create a new action from a moment discrepancy or an absence of count discovery.

This refinement can correct local centers and spread through an existing
stream of certified replacements. It has no authority when all relevant count
quotas are zero. That limitation is a first diagnostic rejection criterion,
not a reason to add another trigger or relax a count cutoff.

## Real fit, objective and noise

Work in the same current-D chart, effective rank r<=8. The current even training
reference rows alone fit the chart, topology, local real mean mu_R and covariance
C_R. Odd rows retain their original count/support-calibration roles; do not use
them to choose moment targets, groups, ranks or thresholds. Use fine-cell
destinations for accounting and real-only topology groups for a pooled objective.
No class labels, heldout data, oracle modes or serving-quality scores enter it.

Rank moment opportunities using a **sum** of nonnegative standardized local
mean errors and covariance errors. For example, on active real covariance axes:

    E = sum_g w_g [ ||C_R,g^(-1/2)(mu_emit,g-mu_R,g)||^2
                   + ||C_R,g^(-1/2) C_emit,g C_R,g^(-1/2)-I||_F^2 ]

Here w_g is the even-real mass of group g. This does not cancel opposing biases
in different groups. It is an empirical optimization objective, not a new
hypothesis test, global equivalence certificate or discovery of a particular
group's direction. It does not authorize per-group equality claims. Numerical
null covariance axes are omitted; rank remains at most eight.

Account for the original learned noise and bounded latent perturbation instead
of fitting clean anchors to the full real variance. In a possible future fixed
reaction diagnostic, retain the existing emitted sample's chosen row IDs and
pair its feature vectors F_j with the already measured clean q[pick_j]. For a
group whose paired draws stay in that real group, estimate

    delta_mu = mean(F_j) - mean(q[pick_j])
    W        = Cov(F_j) - Cov(q[pick_j])
    target clean mean       = mu_R - delta_mu
    target clean covariance = C_R - W

W includes current curvature/cross-covariance terms; it is not presumed to be
sigma^2 I in learned feature space. Skip a group with cross-group paired draws,
insufficient real/fake rows for its active rank, nonfinite statistics, or a
nonpositive target covariance on the active axes. Do not clip an infeasible
target covariance to manufacture a fit. This is a local plug-in noise model:
its accuracy after changing latent coordinates is an unproved assumption.
The production noise formula, floor, streams and perturbation API stay intact.

For each chosen cell, whiten/recolor a supported seed toward this target mean
and covariance in its real topology group. The unique symmetric Gaussian
covariance map can be formed by eigendecompositions of at most 8x8 matrices:

    A = C_q^(-1/2) (C_q^(1/2) C_target C_q^(1/2))^(1/2) C_q^(-1/2)
    target_feature = mu_target + A (q_seed - mu_q)

Only use active axes where both covariances are numerically resolved. A convex
combination of separate modes is not assumed to be valid support: final original
support/inside/same-fine-cell checks must reject a target in a hole or another
cell. Do not project the output to force a benchmark mode or serving choice.

## Bounded latent solve and acceptance

Use the existing differentiable scalar-head-input callback for arbitrary G/D
output shapes. At most four distinct destination cells, four truncated
least-norm Jacobian linearizations per model, and at most eight feature axes;
no all-table Jacobian. Retain the current prior-spread trust bound. A new target
solver must demand strict finite target progress plus the existing p>Q,
inside-region and planned-cell acceptance. It cannot stop only because its
starting seed is eligible. No step size, level, seed, horizon or cutoff sweep.

Current FAST and EMA need their own current-chart moment/proposal calculations;
a FAST covariance map cannot silently stand in for EMA geometry. Both endpoints
must accept in the planned fine cell. If the EMA calculation is unavailable or
the paired proposal worsens its empirical local moment objective, retain the
ordinary copy. Do not reuse a prior-reaction chart with a newly updated D.
Current EMA statistics require at most one extra chunked full-N query before
planning; keep the existing post-action serving measurement exact. Extra reads
must preserve owned/global RNG, registered buffers, modes and gradients as in
the reviewed observational helper. This cost must be measured before adoption.

## Accounting, state and replay

Start from the already planned copy child/parent pair and its exact certified
destination. Reserve its parent as a solver source seed; the parent remains
untouched and is not reported as a copied parent after conversion. Reserve both
source and child from every later phase. The new coordinates must have the same
destination cell/category as the removed planned copy; gross certificate
spending is based on the true endpoint, never the seed. If a planned copy's
category cannot be preserved, leave the copy intact.

Each conversion consumes one existing ordinary action slot, not an additional
slot. Keep total ordinary actions <=floor(Q*N), unique children/sources/parents,
the same cell/group supported ledger and rare-survivor reservations. A rejected
proposal cannot erase or reserve an extra donor, consume a quota twice, or
replenish an opposite-direction quota. Original isolation reservations and its
small-guard rule remain unchanged. Parent/death count quotas and the common
3K+2 family, including Q/(3K+2), remain exactly the existing law.

Apply each conversion through the existing new-latent birth semantics: paired
FAST/EMA coordinates, zero child row optimizer/history/evidence, invalidate child
lineage edges without linking its solver seed, and include the child in moved_rows.
The trainer's current population rebase/reset hook must revoke that child's
incarnation. Keep all G/D/prior optimization rates, average/serving/population
laws, generator weights and learned sigma update rules unchanged by this phase.

If implemented, this is a new birth policy requiring explicitly versioned
backend settings/schema and typed semantic diagnostics. Its current-D geometry
and moment caches may remain ephemeral; decisions already made, counters, row
actions and private-stream continuation must serialize. Clear derived caches on
load. No target map may survive a D epoch unless that epoch/chart is semantic
state. Define deterministic row ordering/ties and candidate budgets before tests.
Do not claim unchanged training: accepted new latents intentionally alter it.

## Finite-sample limits and rejection diagnostic

At small local sample sizes, a local mean/covariance target can follow reference
noise rather than generator bias. The even fit and real topology are estimated;
rows generated by a learned table are not independent clean samples. Current
noise residual estimates are correlated with the generator and can change after
a birth. There is no distribution-free finite covariance accuracy guarantee
without tail assumptions. A Gaussian/sub-Gaussian interpretation would be a
working assumption, not a positive certificate. Aggregate squared errors can
have more signal than one conservative test per support, but aggregation cannot
certify every support or all covariance tails.

Count power is unchanged: actual K=128 raises the current common family to 386
hypotheses and can leave fine-cell deficits undetected. This design adds no
moment tests, so it cannot remedy lack of certified replacement capacity. The
original conditional iid count assumption and its stated adaptive limitations
remain. No-discovery means insufficient evidence, not local moment equivalence.

If the final grid fails and root authorizes a diagnostic, freeze **one** fixed
current-chart mechanical reaction input and the exact source/input maps first.
Select at most four cells once by E and planned-copy capacity; no sweeps. Stop
and reject the proposal if:

1. Existing certified copy slots do not reach the groups contributing the
   moment mismatch, or do not leave unique sources/children/rare survivors.
2. Noise-aware target covariances are infeasible or unresolved, pairing crosses
   groups, or local fit samples cannot resolve the active rank.
3. Bounded paired solves cannot reduce the finite moment proxy and pass original
   support/inside/planned-cell checks in both models within four iterations.
4. Any true endpoint violates quotas, shared budget, cell/group ledger, optimizer
   reset, source reservation or replay/state invariants.

Existing raw coordinates and mode annotations may describe whether the proposed
learned-feature direction corresponds to the observed center/spread issue;
annotations cannot enter target fitting, action ranking or acceptance. This
small diagnostic is a mechanical plausibility/rejection test, not a quality
run. Historical fitted charts/emitted reaction draws were not serialized:
a fresh CPU current-chart probe is not historical CUDA action reconstruction.
No such diagnostic or new emitted count sample is authorized by this design.

## Cost and minimal code surface

Local moment reductions cost O(N*r^2); active local eigendecompositions cost
O(G*r^3), r<=8 and G<=actual K. Retained moment arrays cost O(G*r^2), plus the
existing O(N*r) chart arrays. No N-by-N storage or global output-space covariance.
The solver bound is four cells x two models x four rank<=8 linearizations.
The extra current-EMA full-N query is O(N) model/head work in existing chunks;
its wall time for arbitrary models is unknown and may dominate reductions.

Possible implementation surface, if selected: a moment-target helper/new bounded
target-progress solver in anchor_birth.py; an adapter to convert selected ordinary
copy slots to existing birth records; explicit birth/count ledger integration in
birth_phase.py/FeatureCellSnapshot.ordinary_transport; typed policy/versioned
backend diagnostics. Existing copy equations, main optimizer/controller,
serving dispatch and noise code remain the same. If no certified slots are
available, pursuing this design would require a different action-authority law;
that is outside this proposal.
