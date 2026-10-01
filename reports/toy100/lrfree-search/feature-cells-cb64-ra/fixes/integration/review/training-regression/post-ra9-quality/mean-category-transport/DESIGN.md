# Fixed mean-directed within-category copy prototype

Status: source-only preparation for one scratch CPU run. This is not a
production patch, population certificate, equality certificate, or quality
evaluation. RA9 and all earlier freezes remain read-only.

## Inputs and one fixed law

Use exactly the final RA9 grid and final RA9 toy trainer states. Decode the
saved FAST and EMA weights/tables directly; do not load a trainer that swaps
its serving view. Functional fixture adapters reproduce the saved host
architectures but are not branches in the proposed generic policy. They
produce clean G-to-D-head features; no sample, `_generate`, output noise,
quality cloud, oracle label, or second latent perturbation is used.

Fit one current-D `FeatureCellSnapshot` on the saved complete real FIFO, with
the saved requested 128 cells, rank 8, chunk 256, and the existing even-only
finite-fit cell cap. Its fitted geometry, inside boundary, and real topology
are unchanged. Clone the saved CPU RNG into one private generator, use it for
this CPU chart, then for at most one bounded batch of paired copy jitter.
This is a new fixed current-chart CPU diagnostic, not reconstruction of the
historical CUDA chart or CUDA stream. No seed is introduced or changed.

Let r be its effective rank, Q=.05, and R=sqrt(r/Q). For each fitted real
topology group g, even references define a projected-feature center c_g and
scalar scale s_g=sqrt(mean_even ||x-c_g||^2 / r). Define
`psi_g(x)=radial_clip((x-c_g)/s_g, R)`.
Every group must have at least two even rows and positive finite scale.
Invalid geometry, duplicate guard failure, nonfinite features, rank zero,
missing groups, or insufficient odd rows veto the entire witness. No group
or odd observation is removed in response to its score.

The even clipped mean a_g, current EMA clipped mean m_g and unit-or-zero
direction u_g=(a_g-m_g)/||a_g-m_g|| are fixed before reading odd witness
observations. EMA means use all clean EMA rows assigned to that real group;
missing EMA group supply vetoes. Exact zero residual uses direction zero.
All vectors psi and m have norm at most R, so the scalar observation
`X_j=u[group_j] dot (psi(odd_real_j)-m[group_j])`
lies in [-2R,2R]. All odd rows, including zero-direction rows, remain in the
sample. There are no odd-count importance weights. Its expectation follows
the true real group mixture under a declared conditional iid law. The action
objective below uses even empirical masses, which is a different weighting.

For n odd rows, unbiased sample variance v (ddof=1), alpha=Q/(3K+3), and
t=log(2/alpha), the sole trigger is strict positivity of
`LCB=mean(X)-sqrt(2*v*t/n)-(7/3)*(4R)*t/(n-1)`.
This is the scaled lower-tail form of the bounded empirical Bernstein law in
[Maurer and Pontil, Theorem 4](https://arxiv.org/pdf/0907.3740).
The current learned D, EMA, and FIFO share training history. Conditioning on
the fitted score is not an independent prospective experiment here. The
formula is conditional algebra and empirical negative evidence, with no
repeated adaptive or population-stationarity guarantee.

The null concerns clean finite-table learned features. Nonlinear D and psi,
latent jitter and output noise mean that physical output-mean equality does
not imply this feature null. Clean-objective progress does not prove emitted
feature or physical mean progress. This limitation changes neither the fixed
score nor the original live-noisy quality evaluations.

## Common family and prior actions

The original K mass, 2K inside/out, and 2 global count hypotheses plus this one
global witness have actual multiplicity 3K+3. Recompute the original raw
conditional count pvalues and every decision mask at Q/(3K+3). Keep all empty
categories. Never reuse archived discovery masks or spend another Q budget.
With no saved emitted iid feature cloud, this run's clean FAST categorical
counts are explicitly descriptive; their pvalue calculation supplies no
additional inferential authority or prior actions in the scratch prototype.

The maximum combined ordinary action count is floor(Q*N). Conservatively
subtract the saved last reaction's ordinary moves. The final grid has zero
earlier actions and therefore an empty reserved-row prefix. The toy has 51
ordinary moves in its 51-slot budget and has no residual slots. Earlier row
IDs are not serialized; a partially spent, nonzero prefix without explicit
reservation IDs would veto planning, not infer reservations. The generic
helper accepts the union of all earlier copy/isolation/newborn/source-seed
row IDs and the earlier ordinary count. It reserves them all. Isolation's
existing separately declared law is not increased or repurposed.

## Bounded proposals and objective

All original child and parent rows must have finite latent coordinates and
features and be eligible (p>Q, no support flag, and inside the fitted count
boundary) in BOTH FAST and EMA. Root selected inside-only before measurement,
tightening the broader initial same-inside/out-category draft. A pair has
identical fine cell and inside category within each view, and one common real topology group
across both views. FAST and EMA fine cell IDs may differ. Protect one eligible
survivor per occupied fine cell in each view. Children and parents are unique
and globally disjoint from each other and all earlier reservations.

Within each (FAST category, EMA category) signature, take at most 64 parents
with largest EMA unit-direction projection, and at most 64 remaining children
with smallest projection. Parent pool size is min(64, floor(bucket_rows/2));
no N by N pair matrix is built. Match descending parents to ascending children.
Keep only positive exact pre-jitter squared-mean gain, then stable sort by
gain descending, child ID, parent ID. Freeze at most the residual ordinary
budget candidate pairs before drawing. There are no retries or work sweeps.

The objective is E=sum_g w_even_g ||a_g-m_g||^2, where
w_even_g=even_group_rows/even_total. A candidate changes one EMA group mean
by `(psi(actual_new_EMA_feature)-psi(old_child_feature))/EMA_group_rows`.
Use a virtual mean ledger, fixed group counts and the exact actual clipped
features. Accept sequentially only when the actual E strictly decreases.
Unjittered directional gain is a proposal priority, never acceptance.

Draw one reaction-stream noise matrix for the frozen candidates. Both tables
use that same draw through their own existing `BoundedLatentGeometry` and
shared saved lineage graph. Prepare every displacement against the same
pre-action table epoch, then observe G/EMA_G followed by the chosen D head
directly on the exact proposed latent coordinates. Require both actual
offspring to retain their child's original cell/category/common group and
finite latent coordinates, features and eligible inside support in BOTH
views. No prior writes occur during this preview.

Feature observations preserve recursive modes, registered buffer mappings,
values and nonpersistent registration, gradient objects/values, global CPU
and owned-device RNG, and all owned trainer/reaction streams. The intended
jitter draw happens outside that guard. Arbitrary unregistered Python state
and external RNG are outside this ownership contract. Geometry sort caches
and work counters are derived; their extra bounded work is reported.

## Exact copy packet and scratch application

The ephemeral packet pins table/chart/lineage/model epoch, paired IDs, exact
FAST/EMA coordinates and parent optimizer/history rows. Validate the whole
packet before any write. Commit those bytes once to scratch priors, inherit
optimizer tensors only when their shape exactly equals the table shape, copy
latent_history rows, and register copy lineage once. Do not call `_move`
after preview because it would redraw. Scalars and model weights stay fixed.
Packets are never persisted or committed across a checkpoint boundary.

Re-evaluate accepted coordinates to prove the virtual objective equals the
final actual objective and both category/group/supported count ledgers are
unchanged. Refresh accepted FAST rows and identify the complete moved-row set.
Production, if separately selected, must include those rows in counters and
the original trainer row-evidence reset/population rebase hook, and refresh
the paired-average lease only after all phases. This scratch run does not
implement production phase orchestration, serving decisions or trainer steps.

The run reports the witness, the bounded legal pair supply, residual action
budget, actual preview retention/progress, exact packet application checks,
state/RNG neutrality and work. A missed bound or empty legal capacity is a
negative result. The clipping, range, alpha, budget, inputs and law will not
be altered after measurement.

## Execution and qualification

Freeze all helper/protocol/source/input hashes before any Torch fixture run or
PT interpretation. Math/state and lineage/ownership reviewers inspect the
same stable source first. One CPU invocation processes grid then toy; no
GPU context, optimizer step, training, emitted cloud or scorer is allowed.
Accepted packets may mutate scratch clones only. The original loaded states
and files must have exactly unchanged typed hashes afterward.

Root chooses any later production policy. This prototype alone cannot qualify
a candidate or change the original toy/grid terminal and holdout gates.
