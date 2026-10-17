# Forge grading review — 2026-09-28

Review scope: source/evidence identity, tier reduction, activation guards, and
continuation caps. No campaign training was run for this review.

## Corrected before the pilot

- AE scoring now preserves the enclosing observation step's output-noise
  schedule. A bounded test checks initial, intermediate, and terminal sigmas.
- Behavioral hosts reject explicit resource/objective overrides they would
  ignore or replace. The shared planner can call `behavior_preflight`.
- A method invocation or `hooks_exercised: true` alone cannot certify a
  mechanism. Receipts now include actual requested/enabled/call/eligible/applied
  counters for penalty, anchor, guard, A2, and direct-particle gain. A requested
  lazy penalty with zero applications blocks qualification; disabled ablations
  make no activation claim.
- Delayed or data-dependent hooks have tiny deterministic public-component
  checks with explicit synthetic state, separate from host training evidence.
  Tests verify optimizer parity and zero global-RNG consumption. Actual host
  optimizer-update counts never include those checks.
- Malformed optimizer-count objects produce incomplete evidence rather than
  crashing the guard reducer.

Validation after the follow-up fixes below: 101 bounded view/artifact/adapter/
behavior/mechanism tests passed. Scalar adapters consume the same activation
audit.

## Follow-up corrections completed

1. **Bulk native artifacts are bound by content.** Native evidence now includes
   every input file's relative path, byte size, and SHA256 plus a manifest
   digest. The raw-envelope certificate covers that manifest. Grading verifies
   the complete tree before and after the original evaluator. Added, removed,
   or changed inputs invalidate evidence. The manifest supports relocating a
   complete tree; receipts explicitly retain local bulk-artifact requirements.

2. **Unsupported evaluator-constant edits are rejected.** Task/view validation
   and direct grading check repeated observation counts, sustained counts,
   native coverage/accuracy limits, holdout counts, ring thresholds, paired
   control windows, clock-free comparison sets, and evaluator identities.
   Altering descriptive metadata cannot silently leave an easier predicate.
   Native artifacts must also match the task's early-observation schedule.

3. **Self-reported checkpoint verification cannot enable continuation.** A
   boolean and matching state labels now remain BLOCKED, including unsupported
   restore-proof stamps. Fresh-process qualification requires future concrete
   restoration instrumentation. Uninterrupted ring hold/extension remains
   supported and bound to one run.

4. **Dependency failures propagate in topological order.** Presentation order
   remains configurable, while every dependent PASS is reduced only after its
   prerequisites. A reversed three-node chain test verifies blocked descendants
   and the required denominator.

The queue already charges grouped ring elapsed time once and validates that
uninterrupted group members stay within the same qualification tier/cap. Those
previous concerns are resolved.
