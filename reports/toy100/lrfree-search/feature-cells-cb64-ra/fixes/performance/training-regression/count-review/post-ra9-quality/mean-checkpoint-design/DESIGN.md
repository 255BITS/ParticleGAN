# Mean-transport checkpoint and auditor contract

Status: source-only contract for the separately selected RA10 integration.
This file supplies no production code, test result, new statistical law, or
candidate qualification. The unchanged toy and full Grid100 gates both remain
required.

## Small state extension

Keep GANTrainer schema 5, the population tester law, and paired-average schema
1 unchanged. Feature-cell backend schema becomes 9. Its exact settings stamp
identifies the fixed mean policy, inside-only paired eligibility, and common
`K + 2K + 2 + 1` family at `Q/(3K+3)`. Preserve the existing resolution policy.
Backend 8 and earlier checkpoints, incorrect settings, and missing or malformed
mean metadata reject before any model, optimizer, tester, buffer, RNG, stream,
or serving-view mutation. No migration or relabeling of an old state is allowed.

Use one scalar-only `last.mean_transport` dictionary. It describes the last
reaction, never a reusable authorization. Suggested exact keys are:

| Fields | Exact types / purpose |
| --- | --- |
| `schema`, `policy` | int 1; the fixed witness policy string |
| `status`, `reason` | `initial`, `invalid`, `veto`, or `firing`; source-defined str or None |
| `step`, `snapshot`, `cells`, `rank`, `observations` | nonnegative int, never bool |
| `alpha`, `radius`, `known_range` | finite float; None only in defined initial/invalid cases |
| `mean`, `variance_ddof1`, `variance_penalty`, `range_penalty`, `lower_bound` | finite float for valid witnesses, otherwise None |
| `attempts`, `moves` | nonnegative int; previewed candidate count and actual committed copy count |
| `objective_before_mean`, `objective_after_mean` | finite nonnegative float for an evaluated action baseline, otherwise both None |

Every reaction replaces the complete dictionary, including invalid and veto
cases. Return/load it without mutable aliases. Store no chart, group means,
directions, candidate packet, model closure, row-evidence copy, or extra RNG
state. Additional timings remain outside semantic state; retain the existing
`last.eval_seconds` exception. The new cumulative counter can be just
`mean_moves`; other existing copy/work counters retain their defined meanings.

## Case and count consistency

Initial means snapshot/step/chart dimensions/observations and action counts
are zero, scalar fields are None, and reason is `not_evaluated`. It cannot
fire. An invalid reaction records the actual current chart and odd row count,
has no scalar witness and no mean attempts/moves. A valid veto has n>=2,
positive rank, all finite scalar fields, and lower_bound<=0; it makes no mean
draw or copy. A firing witness has lower_bound>0. It may still apply zero
copies because of exhausted budget, missing current group supply, failed
retention, or lack of actual objective progress. Dry run never increments
actual moves.

For every noninitial reaction, mean step/snapshot/cells/rank must equal `last`
and the existing paired-average reaction stamp; observations equal the full
odd calibration count. Thus the existing trainer check of the paired step
against completed_steps also rejects a future mean stamp. Radius is
sqrt(rank/Q), known_range=4*radius, and alpha is the actual common cutoff.
Variance and both penalties are nonnegative; the lower bound follows the
fixed unbiased-variance formula. Validate arithmetic with only floating-point
roundoff allowance and derive status from the stored strict sign.

Retain exact even-fit cell-cap/rank and count-partition validation. All count
fields, decision masks, transport guards, and novel-birth metadata use actual
categories=2K, multiplicity=3K+3, cutoff=Q/(3K+3), including when the mean test
is invalid or vetoed. Never restore the old cutoff for such reactions.
`birth_phase.py` has its own hardcoded family guard and must change with the
feature-cell paths. Source inventory must find every old `3K+2` law stamp.

## Fourth copy phase and fresh incarnations

Freeze the witness's even definitions and pre-reaction clean EMA offset/unit
directions before odd-count-driven actions. Do not refit or retest it after
earlier phases. Separately obtain the actual post-prefix EMA action means and
group counts before the mean phase: mass, global, and novel moves can change
them. Never use stale witness denominators for the action ledger. The recorded
objectives refer to this pre-mean baseline and the actual final state.

Mean copies are the fourth copy phase after mass/local/global; novel births
keep their existing separate identity. Action kind 3 already denotes novel
births, so mean kind is 4. All sources/children are disjoint from each other
and every prior copy, isolation, novel child, and source-seed reservation.
Both original views and offspring satisfy inside/support/category/group
retention. Exact previewed coordinates commit once, without redraw. Mean
copies preserve both views' category/group counts and do not restore any
spent count certificate.

Per-reaction identities are:

```
ordinary_copy_moves = mass_moves + support_moves + global_moves + mean_moves
ordinary_moves = ordinary_copy_moves + ordinary_novel_birth_moves <= floor(Q*N)
moves = ordinary_moves + iso_moves
last.mean_transport.moves = ordinary_mean_moves
```

Increment actual copy, birth/death, matched, within-group, and total counters
consistently; mean moves are within-group. Include every mean child exactly
once in the complete `moved_rows` union and event total. Refresh the FAST
cache, register copy lineage once, and record paired-average eligibility only
after all phases. Existing trainer hooks then rebase the prior tester and
reset RowEvidence for the entire union.

Copy optimizer moments and latent regularizer history from the parent as the
existing copy law requires. This inheritance is not population participation
or a newborn's own gradient evidence. Rebase removes that child's completed
pairs/stationary membership; RowEvidence resets its own accumulators. The
unchanged 95% coverage/expiry rule still governs population scheduling. Mean
metadata cannot mark rows participating, close the table tester, undo its
expiry, or enable serving.

## Narrow independent qualification once source is frozen

Audit typed initial/invalid/veto/firing metadata, actual family fields, totals,
JSON serialization and atomic rejection in the complete trainer. Check exact
packet/sample/resume state and reset behavior on the selected saved fixtures;
do not infer reset correctness from optimizer inheritance. Production packet
objects and coordinate caches never survive load. Saved-checkpoint auditing
uses the original artifact/input/source checks, with only declared backend9,
policy/family, and scalar diagnostic adaptations. Evidence validity stays
separate from unchanged scorer quality. No per-group, emitted-equivalence,
stationarity, or serving-quality certificate is added.

## Source inventory

| File | Required scope |
| --- | --- |
| `particlegan/feature_cells.py` | common comparison/transport/settings stamps; fourth phase and totals; mean diagnostics; backend9 atomic checks; final cache/lease refresh |
| `particlegan/birth_phase.py` | matching common-family guard and diagnostic policy; preserve birth law |
| Fixed witness/preview helpers | same frozen score/range/bound and one-draw exact packets; generic callbacks and bounded work |
| `particlegan/training.py` | schema5, complete-union existing rebase/reset hooks, serving dispatch and checkpoint prevalidation; expect unchanged bytes |
| `particlegan/continuous.py`, `row_evidence.py` | existing participation/expiry and own-row reset laws; expect unchanged bytes |
| RA9 config, harness/scorers/fixtures | unchanged bytes and gates |
| Private API/checkpoint auditor | declared metadata adaptation only; never rewrite earlier frozen receipts |
