# Conditional RA10 mean phase integration inventory

Source-only plan after the fixed prototype; no production code, test, PT
interpretation, model forward, law implementation or candidate qualification
was performed for this inventory. Root owns candidate selection and the count
owner owns implementation. Earlier review sources and seals remain unchanged.

## Reaction order

1. **Freeze the statistical witness before prior actions.** Use the even-only
   current real chart, pre-prefix clean EMA group means and fixed unit
   directions. Read the odd scalar once at the declared common `3K+3` cutoff.
   Neither odd-driven count actions nor later action progress may reorient or
   retest this witness. This is separate from the post-prefix action objective.
2. Compute original mass/local/global and novel/isolation plans with the
   declared common family. Apply their existing copies and novel births.
   Refresh every earlier moved FAST row and invalidate its affected cell
   evidence, as current `maybe_apply` does.
3. Form the full reservation union of ordinary children and parents,
   isolation children and parents, newborn children and source seed rows.
   Source seeds are protected inputs, not eligible ancestors by implication.
   Compute remaining ordinary slots as `floor(Q*N) - earlier ordinary moves`,
   including all earlier copy and novel phases. Isolation's existing separate
   law is unchanged. Partial reservation information must veto the new phase.
4. Obtain current post-prefix FAST and EMA features and group counts. Refresh
   EMA means/counts for the action objective; retain the original even geometry
   and statistical witness. Keep the original fixed witness direction for
   action ranking; evaluate actual progress against refreshed objective means
   and counts. Prepare unique, disjoint paired
   sources/destinations outside the full reservation union, in the same own
   fine cell and inside category for both views and a common real-only group.
5. Preview one fixed candidate batch against this **post-prefix epoch**. Draw
   one reaction-stream noise matrix, compute both own-prior bounded jitters
   with parent IDs, and observe exact proposed coordinates directly. Require
   actual supported/finite/inside cell/category/group retention in both views.
   Advance only a virtual nonlinear EMA objective in the declared sequential
   order. No temporary prior, optimizer, history, graph or evidence writes.
6. Validate the entire accepted packet and epoch before any writes. Commit its
   exact coordinates and existing copy optimizer/history inheritance once;
   register lineage once. Refresh new FAST rows and invalidate affected cells.
   Set the complete unique moved-row union of earlier and mean actions.
7. Produce one final paired-average lease from the current post-all-action
   EMA features and refreshed FAST cache. The existing trainer rebase and row
   evidence reset must receive the complete moved union exactly once.

Current source ordering matters: `maybe_apply` records the lease before it
returns; `GANTrainer.step` then rebases the prior tester and resets row evidence
before serving. These hooks change participation/evidence, not coordinates or
the learned chart. Minimal integration retains that single geometry measurement
and single existing hook. If a physically later lease is required, a deferred
finalizer must run once after that hook; do not duplicate rebase or measurement.

## State and ownership boundary

Use actual FAST state after the normal serving release. Additional G, EMA_G
and D observations must preserve recursive modes, registered buffer objects,
bindings, values and nonpersistent registration, parameter gradient objects
and values, CPU/device global RNG, every trainer-owned stream and the reaction
stream. The intended copy draw sits outside the observation guard; its
post-draw stream state is the packet epoch. No output-noise override, extra
sampler call, parameter proposal, optimizer step or serving swap is needed.

Pin table/lineage/chart identities and versions, model parameter identities
and versions, exact restored buffer mappings/bytes, bandwidth and post-draw
reaction stream state. Buffer restoration with `copy_` advances `_version`;
it must not trigger a false stale-packet rejection when bindings/bytes match.
Packets stay ephemeral within one reaction and cannot cross checkpoints.

Copy children inherit the old exact-shape optimizer row tensors and latent
history; shared optimizer scalars stay intact. They receive fresh row evidence
and population participation through the existing moved-row hook. Positive
ancestor participation is never copied. Novel births retain their distinct
zeroing and no-parent-link semantics.

## Bounded work and GPU synchronization inventory

Reuse the real chart and real features already fitted for this reaction. The
new statistical query needs pre-prefix EMA data; the action objective and final
lease need current post-prefix/post-mean data. Any reuse with row patches must
prove equality to the declared current query; state-neutrality alone does not
prove that cached feature outputs match a fresh query. Do not silently use a
pre-prefix witness cache as the final serving lease. Account for any additional
full-table EMA forward explicitly.

Keep 64-row source/destination pools per signature and at most the residual
ordinary budget preview rows. Use existing bounded 64-coordinate plus lineage
geometry; no all-N pair matrix, repeated chart fit, unbounded retry or per-copy
whole graph scan. Scratch whole-graph validation is review evidence, not a
production operation to repeat on each copy.

The scratch planner's per-pair CUDA `bool`, `int` and `float` reads and per-cell
`nonzero` sizing cannot be copied into production. The owner has declared
CPU-float64 sequential decision arithmetic for the prospective policy. Build
bounded device results, transfer a single decision packet containing group
means/weights, retained flags/group IDs and actual proposed feature deltas,
then execute the same sequential objective order on CPU. Transfer accepted
indices once. Pool bookkeeping should similarly use stable batched sorts and
one bulk host packet. CPU/GPU action/RNG contracts must qualify this declared
execution policy; there is no existing GPU arithmetic parity claim.

Packet coordinate/history storage is proportional to accepted rows times
latent width. Feature storage/query costs, buffer/gradient preservation and
bulk host transfers must appear in work/runtime reporting. No speed claim
follows from the scratch CPU result.

## Declared metadata and validation work

Expose a fourth `ordinary_mean_moves` phase, its child/parent IDs, action kind,
ordinary budget use and unchanged count-category/group ledgers. Preserve exact
balances:

`ordinary = mass + support + global + novel + mean`

`copy = mass + support + global + mean`

`all moved = legacy copy children + newborn children + mean children + isolation children`

All sources and children remain unique/disjoint under the complete reaction
reservation law. Include mean copies in matched/copy/ordinary/total counters
and reset coverage; do not hide them in another phase.

The count owner must update every common-family guard, cutoff, policy tag and
persisted diagnostic validation to actual `3K+3`, while preserving individual
raw count tests. A changed backend law needs schema/settings and atomic old-law
rejection. Mean witness/action diagnostics must have strict scalar types and
JSON-safe values. Derived geometry/chart/packet caches remain unsaved. Small
population reference fallback remains unchanged.

The old toy audit assumes three count phases and `3K+2`; root must freeze a
declared extension before launch. Preserve all original source/init/cursor,
noise, emission, replay and quality-gate checks and all previous ERROR/FAIL
receipts. The later candidate needs source/ownership, observer/packet atomicity,
complete moved-row reset, final lease, load/resume and action/RNG contracts.
Those tests are not performed by this source inventory.

## Source locations

- RA9 `particlegan/feature_cells.py`: `_move`, `maybe_apply`, `refresh_rows`,
  `_record_paired_average`, common count family and checkpoint guards.
- RA9 `particlegan/training.py`: serving release, moved-row rebase/reset,
  checkpoint atomic validation and final serving application.
- RA9 `particlegan/birth_phase.py`: newborn/source reservations and distinct
  novel birth reset/lineage semantics.
- Frozen scratch `mean-category-transport/{transport.py,run_prototype.py}`:
  prepared packet and declared CPU prototype, not production orchestration.

No oracle center, target gate, benchmark name, step-horizon cut, noise-disable
or forced-EMA condition belongs in the new production action policy.
