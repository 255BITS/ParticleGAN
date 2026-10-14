# Bounded extended training results

An original deadline grade remains immutable. A separately preregistered extension
can record a later result such as `FAIL (original deadline); PASS after +N training
updates (eventual_recovery_window)`. It grants no original benchmark-task pass,
qualification, promotion or change to family totals. The original source, task,
protocol, evaluator, sampling law and numerical criteria remain bound.

This capability reports **certified normalized adapter receipts** and verifies
restored/final checkpoint files. It does not enqueue training, extend ordinary
jobs, alter task declarations or install an executor for every host. A host must
restore its full state with its public loader and provide independently verified
sample/evaluator and guard receipts. Compact reporting assumes those upstream
certificates have already been validated; boolean flags alone do not establish
that a real run happened. No historical extension result is published by this PR.

## Plan before running

Use `extended_results.plan_extension(task, original, ...)` before execution. Freeze
its returned policy in the scientific admission record. Both training-update and
measured-wall-time caps must be positive and finite. The phase suffix schedule is
explicit and bounded. Native cadence remains the original interval; dense ring
checks retain every update. Original observation positions remain byte-for-byte
in the normalized receipt: increasing `execution.steps` and regenerating the old
24-point schedule is invalid. This module never modifies `execution.steps`.

The normalized original receipt retains `gate_status`, `attempt_id`,
`candidate_revision`, `source_digest`, `protocol_sha256`, `schedule_horizon`,
`executed_updates`, `observation_steps` and the complete original phase evidence.
Retain the original certified raw receipt and its hashes alongside this projection.
The resumable checkpoint supplies `schema`, `step`, `sha256`, `state_sha256` and its
artifact path/manifest provenance. Supported state schemas are `public_context_v1`
and `joint_word_v1`; a provenance-only state is not a continuation certificate.
A task without a supported certified checkpoint is `BLOCKED`. Behavioral hosts
remain `BLOCKED`: their existing provenance state is not a supported resume path.
Nontraining parity/API/source/parameter gates are `NOT_APPLICABLE` and spend no
extension work. Unknown numeric grader kinds remain `BLOCKED`.

A training host must preserve the original recipe/schedule horizon, all model and
optimizer history, named data/prior/noise/evaluation streams, and policy state.
`extended_continuation.verify_extension_prefix` compares the original certified
checkpoint with the public host's restored state and checks actual optimizer
counters. It accepts a failed parent **only in this separate capability**.
`continuation.verify_native_prefix`, ordinary checkpoint dependencies, Gaussian
and word passing-parent continuation remain unchanged. `verify_extension_final`
checks the resulting checkpoint clock and static formulation/RNG binding law.

## Keep clocks and costs distinct

The report separately exposes:

- Restored checkpoint step and absolute end step.
- First numeric hit and first full phase/window confirmation.
- Each phase's end-state result and terminal passing observation-check count.
- Accepted extension progress from the restored checkpoint.
- Every newly executed update, including interrupted work and retries.
- Total original plus newly charged training work and measured extra wall seconds.

`+N` is accepted checkpoint-relative extension progress, not endpoint minus the
nominal deadline. Word acquisition executes 20,001 updates but can export an
earlier confirmed state. Resuming that earlier state has a different absolute
clock and still charges all original work. Execution-group identities deduplicate
shared prefixes; conflicting charges are rejected. Retain the already charged
`execution_group_ids` in the original projection: those groups cannot be charged
again as new extension work. Absolute checkpoint steps and original newly
charged updates are independent clocks; imported/shared prefixes need not equal
the original task's newly charged work. Interrupted/retry work counts
against both caps, even when it contributes no accepted progress. A first hit or
passing partial window in an interrupted run remains `INCOMPLETE`.

## Preserve each grader's meaning

Gaussian/word smoke is **confirmed acquisition at one independently sampled
state**, not sustained convergence. Both primary and independent same-state
confirmation values must meet the unchanged bounds. Gaussian revision-9 stability
retains its original stationary violations and full original `FAIL`; the later
five-check reacquisition plus 24-check hold is separately named
`eventual_recovery_window`, never a rewritten stability task or new Tier-3 pass.

Native accuracy requires five terminal coverage clouds and an independent
100,000-sample fidelity holdout from the same final state, using the original
coverage thresholds and accuracy limits in their respective phases. Coverage
alone cannot qualify. Ring recovery retains its dense 200-check confirmation,
first later 1,200-update uninterrupted hold, and any required 300-update endurance
extension. A failed first later hold cannot be replaced by a selected later
suffix. Original first-hold/shift/frozen evidence remains immutable. Transfer
sustained hold keeps every predeclared observation; each phase uses the original
numeric bounds. The paired-adaptation host is currently `BLOCKED`: its coupled
active/frozen grader has no normalized numerical extension adapter. Its original
first-hold/shift/frozen evidence remains preserved; active quality alone cannot
substitute for that paired gate. No tolerance is loosened to admit a lucky endpoint.

## Report without rewriting the original

`evaluate_extension` consumes the frozen task, normalized original, policy and
continuation receipt. It verifies policy/identity/horizon bindings, prefix proof,
original positions, all exact appended observations, certified final state and
costs. Every observation contains its original evaluator hash, independent
certificate/guard status, state/sample artifact hashes and numeric values. Smoke
also supplies independent confirmation values/state; native holdout supplies the
exact sample count. Missing, duplicate, reordered, off-cadence or out-of-cap
observations are invalid evidence. A complete passing terminal window produces a
separate result wrapper with the copied original unchanged; later collapse is
`FAIL` even if earlier hit/window times are present.

The reporting-only command accepts one normalized certified adapter receipt:

```sh
python -m experiments.forge.extended_results receipt.json --output extended-result.json
```

The output must differ from the input. It contains `original_result` and
`extended_result`, including phase details and continuation provenance. Software
fixtures live only in `tests/test_forge_extended_results.py`; they are not research
measurements. There is no production `extended-result-board.json` in this change.

An optional source-bound `reports/forge/extended-result-board.json` can contain
`rows` of `{task, original, policy, continuation}` for the main report renderer.
Unsupported/nontraining rows need only `{task, original}`. The renderer revalidates
each normalized result and displays original grade, later scope, first hit/window,
end-state checks, absolute clocks, accepted/charged updates and measured cost.
It never writes or recalculates original family selections or pass counts.
