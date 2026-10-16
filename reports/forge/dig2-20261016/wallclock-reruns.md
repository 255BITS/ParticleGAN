# Dig 2: wall-clock reruns (preregistration)

Status: PREREGISTERED. Written and committed before any GPU job of this lead ran.

## Terms

- **Wall-clock limit**: a task's `resources.timeout_seconds` (900 s for `five_word_joint_smoke`,
  300 s for `ring16_acquisition`). The attempt supervisor kills a job that reaches it.
- **Wall-clock hit**: a TIMEOUT verdict, or an INCOMPLETE whose raw terminal status is `timeout`
  at or past the limit (definition from round-1 hygiene, #428).
- **Solo rerun**: the same attempt request (same frozen source snapshot, recipe revision, task card,
  timeout and request bytes, hash-checked by the worker) run by the unchanged
  `experiments/forge/attempt_worker.py supervise`, under the workflow GPU lease, which allows one job
  of this workflow per GPU. Jobs outside the workflow (attempt-fill service, llama-server, desktop)
  can still share the GPU; every rerun samples `nvidia-smi` every 15 s to record co-tenancy.
- **Per-update time**: wall seconds per training update. Reruns: from timestamped observation lines,
  (t_last_obs - t_first_obs) / (step_last - step_first), plus total seconds / completed updates.
  Earlier timed-out runs have no timestamps, so their rate is bounded: per-update time lies between
  900 / (last_logged_step + one observation interval) and 900 / last_logged_step.

## Cells (in priority order; seed 0 only, no seed sweep)

| # | Recipe | Task | Earlier verdict |
| --- | --- | --- | --- |
| 0 | bcap-default-baseline-direction-v1 (matched BCAP baseline) | five_word_joint_smoke | INCOMPLETE (wall-clock, acquired at step 834) |
| 1 | signed-confidence-lr-gaussian-v1 | five_word_joint_smoke | TIMEOUT (acquired at 834) |
| 2 | positive-evidence-lr-gaussian-v1 | five_word_joint_smoke | TIMEOUT (acquired at 834) |
| 3 | generator-confidence-lr-bcap-direction-v1 (Confidence LR; same-runtime control) | five_word_joint_smoke | PASS at 846.6 s |
| 4 | kinetic_transport_local_v2 | five_word_joint_smoke | TIMEOUT (acquired at 834) |
| 5 | kinetic-local-dualnorm-g-history-v1 | five_word_joint_smoke | TIMEOUT (acquired at 834) |
| 6 | confidence-lr-finite-cap-gaussian-v1 | five_word_joint_smoke | TIMEOUT (undetermined) |
| 7 | kinetic-local-dualnorm-g-history-reference8-u-v1 | ring16_acquisition | TIMEOUT (determined FAIL, 0/94 checks) |

Budget: the declared timeouts of all claimed cells must fit 7,200 GPU-seconds (8 cells: 6,600 s max).
The 3,600 s Tier 3 timeouts are out of budget and are not rerun.

## Decision rule

R1 (verdict). A rerun produces a **real verdict** only if the worker grades it PASS or FAIL under the
original declared limit, with the task's own gate (earliest-confirmed gate for word smoke: a confirmed
scheduled check and completion of all 20,001 updates; ring16: its sustained gate). Nothing about the
grading, the limit or the recipe is changed. A rerun that hits the limit again stays
**NOT MEASURED (wall-clock)**; it is then called **slow at the declared limit** if its solo
throughput, extrapolated, cannot complete the updates within the limit, otherwise **contended**.

R2 (contention). An earlier wall-clock hit is attributed to contention when the solo rerun's per-update
time is at most 1/1.5 of the earlier lower-bound per-update time (900 / (last_logged_step + interval)),
i.e. at least 1.5x faster solo. Otherwise it is attributed to recipe cost.

R3 (scores). A leaderboard cell changes only where R1 gives a real verdict. A Tier 1 "6/6" claim for a
recipe holds only if its rerun is PASS and its other five Tier 1 cells are already PASS. Predicted
outcomes from round 1 are not counted. `five_word_joint_hold`, which depends on the smoke checkpoint,
stays BLOCKED/unmeasured in this lead (not in budget) and is excluded from paired deltas.

R4 (paired delta). The Confidence LR minus matched-baseline delta on `five_word_joint_smoke` is restated
only from the two same-runtime reruns (cells 0 and 3). If either is NOT MEASURED, the pair is excluded.

R5 (reporting change, CPU only). Implement, opt-in and reviewable: a NOT MEASURED (wall-clock)
outcome kept out of fail totals and paired deltas, and a contention-aware limit (extend the declared
limit by the measured slowdown factor against a solo reference, capped at a hard ceiling). Success:
(a) unit tests for both pass; (b) with the feature off, regenerated reports are byte-identical
except provenance hashes; (c) no score changes except the R1 real verdicts.

## Amendment A1 (committed after cells 0 and 1 finished, before any hold run)

Disclosure: written after seeing that cells 0 (baseline) and 1 (signed Confidence LR) PASSed their reruns.
It adds measurements; it does not change R1-R5 for the cells above.

A1. If GPU budget remains after cells 0-7 (actual lease seconds used + 300 s per hold must stay within
7,200 s), run `five_word_joint_hold` (declared limit 300 s, 4,000 updates, checkpoint dependency on
`five_word_joint_smoke`) for recipes whose smoke rerun is PASS, continuing from that rerun's own saved
checkpoint and bound exactly as the attempt-coverage runner binds a new-attempt parent
(`prerequisite_bindings`). Priority: baseline, signed Confidence LR, Positive-evidence LR. R1 applies
unchanged (real verdict only under the declared limit). The earlier BLOCKED hold verdicts are replaced in
the overlay only by a real PASS/FAIL. R4 is extended: the Confidence LR minus baseline delta over
{smoke, hold} uses the rerun smokes plus Confidence LR's recorded hold and the baseline's rerun hold;
since Confidence LR's recorded hold ran on its own (earlier) smoke checkpoint, that comparison is
labelled cross-runtime.
