# Forge hypothesis-to-decision contracts

`forge new` creates an idea with schema version 2 and a draft decision contract.
Planning exposes the exact public-API bindings and proposed delta; admission
rejects the draft before writing queue state or reserving execution. This is a
small contract around existing ideas, task budgets and readouts. It launches no
training and does not adopt a screening profile.

Read [experiment memory](../reports/forge/EXPERIMENT_MEMORY.md), the relevant
source-bound failed receipts and [field ownership](forge-field-boundaries.md)
first. Published summaries can motivate a question; they cannot qualify it.

## Prepare one reviewable question

1. Create a successor with `python -m experiments.forge new --id YOUR_ID
   --parent EXISTING_ID --goal GOAL_ID`. Edit its hypothesis, actual formulation
   and `changed_factors`. Seed-only, prose-only and inactive-knob changes remain
   insufficient.
2. Run `python -m experiments.forge plan YOUR_ID --device cpu --through-tier 1`
   for a CPU question, or select its intended CUDA model. Inspect
   `decision_contract.actual_bindings` and `expected`. The task owns the prior,
   initialization, host, training allowance and public sampling law; candidate
   reference settings cannot replace them.
3. Set `prior_evidence` to repository-relative compact JSON sources, each with
   its byte SHA-256, a JSON selector and exact identifying fields. Keep the
   original record/trial ID and `use: "motivation_only"`. A selected source that
   changes or contradicts its declared identity blocks admission. No raw-log
   hydration is required.
4. Review and copy the expected control/candidate binding hashes, task map and
   complete `substantive_delta`. Freeze all authorized tasks, view, tier, source,
   protocol, backend, runtime/compute hash and complete grouped-job hash into
   `scope`. Choose finite candidate and campaign caps that cover the complete
   authorized task allowances. `max_rounds` must be 1.
5. Name a final scalar metric on an authorized task, its comparator and a
   numerical predicted threshold. Name a falsifier and a competing explanation
   that can change the interpretation. Review these choices before marking
   `status: "ready"`; retain the generated terminal rules. Planning must show
   `READY` before ordinary enqueue.

The plan prints expected hashes for inspection; copying them acknowledges a
reviewed question, not experimental success. Bindings retain complete task
execution/evaluation fingerprints and initialization receipts, including fixed
fixtures and native component policies. Descriptive initialization provenance
alone does not establish an exercised mechanism: a fixed host still consumes its
frozen source, and any causal interpretation requires reviewing saved diagnostics.
A descriptor-only initialization edit is insufficient as the substantive delta.

A signature fragment for the software fixture in
[`test_forge_decision_contracts.py`](../tests/test_forge_decision_contracts.py) is:

```json
{
  "prediction": {"task_id": "t1", "metric": "score", "op": ">=", "threshold": 0.8, "phase": "final"},
  "falsifier": {"task_id": "t1", "metric": "score", "op": "<", "threshold": 0.5, "phase": "final"},
  "competing_explanation": "A transient gain can arise from contraction rather than sustained quality."
}
```

These illustrative numbers test software branching. They propose no GAN run or
new scientific gate. The fixture demonstrates a complete ready contract using
actual plan bindings and a source-bound original failure identity.

## Bounded execution and terminal decisions

Within one queue root, the candidate round cap accumulates paid attempts and
current physical reservations across campaign IDs, compatible subscribers and
explicit infrastructure retries. Grouped jobs count once. The round identity
uses actual candidate/task binding, source, protocol, runtime/compute and backend;
legacy revision strings and job-key namespaces cannot reset it. Its budget and
job-key set become immutable on first admission. A metadata edit that changes an
old compatibility key must reuse the registered job identities or is rejected.
Transferring a running job's payer does not reserve the same physical work twice.
A retry still needs the existing explicit infrastructure authorization and its
full allowance; it cannot erase the original charge.

The campaign cap continues to apply per campaign definition. It is not an
aggregate cap across distinct candidates, controls or queue roots. A control's
binding and prior evidence identify the comparison; they do not authorize a paid
control run. Changed task/source/runtime cohorts require a newly reviewed scope.
There is no global research-spending promise across independent queue roots.

Readout evaluates frozen final metrics separately for each contract/compute
cohort and preserves the original result hashes and attempt IDs. It does not
reconstruct current defaults or regrade saved scientific evidence. Missing,
invalid, non-finite or duplicate unsuperseded task rows yield `incomplete`.
Certified infrastructure repairs may supersede the failed attempt's decision
row; its paid cost and provenance remain in the ordinary readout.

| Observation | Deterministic next action |
| --- | --- |
| Missing or ambiguous evidence | `request_missing_evidence` |
| Falsifier satisfied | `stop_revision` |
| Prediction satisfied, falsifier absent | `review_saved_diagnostics` |
| Neither signature satisfied | `stop_and_readout` |

The falsifier wins if signatures overlap. These outcomes have
`qualification_input: false`, `execution_authorized: false` and
`causal_judgment: "requires_review"`. Prediction-observed is not a qualification,
a causal result, a calibration acceptance or permission to continue training.
The usual declared scientific gates remain authoritative.

## Compatibility boundary

Saved v1 evidence is read under its original identity without migration. New
ordinary research admissions can use v1 only when the declaration exactly
matches [`legacy-ideas-v1.json`](../configs/forge/legacy-ideas-v1.json); deleting
or downgrading the version of a new independent-grading request grants nothing.
Edited legacy ideas need a v2 successor. Lightweight scheduler unit protocols
without a research idea schema remain supported.

Existing registered calibration and promotion requests retain their exact
original validator, source snapshot and bounded campaign semantics. Existing
finite configuration search remains available only after its complete grid,
resolved declaration, tasks/jobs, source/runtime/protocol and campaign are bound
in the immutable search registration; a caller-provided `configuration_id`
provides no exemption. A new search derived from a v2 base inherits its question
for review and will block if tuning changes its bindings; fill a distinct ready
contract for each derived card rather than copying stale hashes. This change does
not migrate or run any existing study.
