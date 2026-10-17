# Mean copy preview and commit: source review requirements

This is a source-only integration review for the count owner's fixed scratch
prototype. It supplies no statistical certificate, numerical result, production
implementation, or quality claim. The existing RA9 package remains immutable.

## Existing copy semantics

`FeatureCellBirthDeath._move` draws one tensor from `birth_death.stream`, evaluates
FAST and EMA displacements with that same tensor and each table's own bounded
geometry, writes both children, copies optimizer tensors whose shape exactly
equals `prior.z.shape`, copies `latent_history`, and calls
`lineage.register_copies` once. Shared optimizer scalars are untouched. A copy
is not a novel birth: `apply_anchor_births` zeroes optimizer rows and creates no
parent edge, which is a different law.

The enclosing reaction refreshes every moved FAST row, invalidates affected
cell evidence, and reports all moved rows. The trainer then rebases population
participation and resets `row_evidence` for those rows. Inherited optimizer and
history bytes never imply inherited positive participation or certification.

## Minimal preparation

1. Work on the actual training FAST view. An eligible full trainer load calls
   `_serve_apply`, which substitutes EMA G/prior into the live objects between
   updates. A private scratch trainer must release that view before obtaining
   FAST features and the epoch. Production `step` already releases before the
   reaction. Do not measure two EMA views and call one of them FAST.
2. Freeze the candidate row pairs and the finite preview budget before drawing.
   Reserve all earlier ordinary and isolation children and parents, newborn
   children and source seeds. Require unique children, unique parents, and
   disjoint source and destination sets across the complete reaction. Total
   ordinary moves, including novel births and accepted mean copies, must be at
   most `floor(Q*N)`. Isolation keeps its separately declared existing law.
3. Pin the current chart and table epoch: snapshot/cache versions, table object
   identities and tensor versions, lineage tensor version, and model/controller
   inputs used for the preview. A pre-count packet cannot silently become an
   after-count packet. Any earlier writes require a new preview from the actual
   current state, or an explicit law that prepares all affected actions against
   the same prior state.
4. Draw copy noise once from the dedicated reaction stream. Compute both
   displacements before any row writes, with the selected parent IDs passed to
   the shared lineage graph. Keep detached clones of the exact proposed FAST
   and EMA coordinates, paired IDs, and parent optimizer/history rows needed by
   commit. Rejected draws may be consumed once by the declared planner; no
   implicit stream rewind or repeated draw is allowed.
5. Evaluate the two generators directly on those already perturbed coordinates
   and obtain the selected D-head features. `sample`, `_generate`, and a capture
   with `jitter=True` would perturb the coordinates again. The preview uses no
   output-noise override or fresh evaluator cloud.
6. In each view, candidate parent and overwritten child must retain that view's
   own fine cell and inside category, finite supported status, and the required
   shared real-only topology group. FAST and EMA fine cell IDs need not equal
   each other. Test the actual proposed features against those original IDs;
   parent eligibility alone does not prove offspring retention.
7. Measure sequential objective progress against a virtual EMA feature/moment
   ledger containing previous accepted proposals. It must reflect the actual
   proposed EMA features. Do not temporarily edit priors to simulate a move.

## Forward observation boundary

`birth_phase.learned_latent_features` restores hooks and module modes only.
It is not a complete observational guard. The new preview should preserve
registered buffer mappings and values (including nonpersistent registration),
all recursive G/EMA_G/D modes, parameter gradient objects and values, CPU and
owned device global RNG, the trainer's dedicated streams, and the reaction
stream around the forward observations. The intended copy draw occurs outside
that guard so that it remains a single owned draw. Existing RA9
`_record_paired_average` supplies the source pattern for this supported owned
state boundary. Arbitrary Python state and external RNG are outside that claim.

Restoring buffer bytes with `copy_` increments buffer tensor versions, even when
their values are unchanged. An epoch captured before the guarded observation
must not reject that expected restoration. Check the exact restored mappings and
values, or pin post-restoration buffer versions after proving restoration;
parameter, prior and lineage versions still must be unchanged. Pin the owned
stream state after the intended draw. A later accidental draw or forward-state
change must invalidate the packet.

Bounded geometry sort caches and work counters are derived state. Building them
does not require undoing weights, moments, history, graph, or row evidence, but
the extra preview work and cache behavior must be declared and accounted for.

## Later production commit, if selected

Validate the whole packet and its epoch before the first write. Commit the exact
coordinates and inherited row-state bytes from that packet; do not invoke the
old `_move`, which would draw and recompute. Register copy lineage once, so old
child incidents are removed and bounded symmetric parent links are created.
Mark the packet consumed. Include accepted mean rows in the enclosing reaction's
full moved-row set, refreshed FAST cache, cell invalidation, ordinary counters
and shared budget. Recompute the paired-average lease only after all actions and
cache refreshes. The existing trainer rebase and row-evidence reset must receive
the complete moved-row set, with no ancestor evidence retained.

Declare a fourth `ordinary_mean_moves` phase in any later production totals and
action ledger. Old checks that sum only mass, support and global copies, or
require a `3K+2` count family, need a separately declared audit extension before
that candidate runs. Existing frozen auditors and receipts must remain intact;
an extension must preserve their other balances, gates and source checks.

Keep packets ephemeral within one reaction; no checkpoint may occur between
preparation and commit. Introducing the `3K+3` family, the mean policy, or new
persistent diagnostics later requires declared schema/settings and atomic
checkpoint validation. Those production changes are not authorized by this
scratch prototype review.

## Review scope

The state reviewer owns the clipped score, direction selection, finite-sample
bound, and common family math. This reviewer checks preview, state ownership,
cache epochs, row reservations, and eventual commit obligations. The count
owner implements and runs the one fixed saved grid/toy scratch prototype only
after its helper/input freeze and both source reviews.
