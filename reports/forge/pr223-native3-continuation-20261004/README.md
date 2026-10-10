# Original PR223 native3 continuation — declared, not executed

This envelope admits only three fresh native executions. It reuses the maintained
full19 helper's scientific construction, updates, observers, scorers, import
guards, lease checks and fenced child. The original study remains **16 accepted
PASS, grid100 INVALID, two NOT_RUN**. Its results are provenance and cost history,
never new passes. No model, sampler, scorer, queue or real preparation was run to
implement this envelope.

| Fresh execution ID | Original slot | Whole inclusive cap |
| --- | --- | ---: |
| pr223-native3-continuation-v1-native-grid100 | 17, grid100 | 1470 s |
| pr223-native3-continuation-v1-native-rotated100 | 18, rotated100 | 1440 s |
| pr223-native3-continuation-v1-native-staggered100 | 19, staggered100 | 1380 s |

Each case retains the complete original config
[atlas.json](../../../configs/100gaussians/atlas.json)
(`a3ee5c67ac6594014feeb1ec333131abb4b1d86832510b69923100ebd8510ad4`),
the original architecture/initialization, seed 1234, N 20,000 / z 2 / batch 2,048, all 7,000
updates, 34 existing observations of 20,000 points, the terminal five clocks
6000/6250/6500/6750/7000, and the independent 100,000-point holdout. Noisy
state-selected serving with the actual public DV12 and learned output kernel is
primary; clean and forced EMA remain diagnostics. Original coverage **and**
accuracy determine the numerical gate. Existing nine media clocks are
0/50/750/1750/2750/3750/4750/5750/7000; capture adds no scientific read or draw.
`Recipe.total_steps` remains None. The full recorded Recipe is required, rather
than a preset or three-field substitute.

## One retained budget

The committed stopped17
[results](../pr223-original-full-retest-stopped17-20261004/results.json),
[final cost](../pr223-original-full-retest-stopped17-20261004/FINAL_COST.json), and
[verification](../pr223-original-full-retest-stopped17-20261004/verification.json)
are byte pinned by `native3_contract.HISTORY_PINS`. Prior case charge
3165.841891122982 s is included once. The canonical existing ledger is
`/ml2/hypergan/.pg-pr223-full-original-retest-20261004.pr223-full19-metadata-cost.json`.
Its closed parent boundary has SHA ce949407c9a11b2d2873be0d51b2d147261f8b60cee411ea33381c8c5c6f4369,
2675 bytes, twelve complete phases and 44.37232269323431 s paid. Root supplies a
separate immutable copy of those exact bytes before entering any new phase.

New planning, freeze, copied preflight, registration, parent verification,
sealing and publication use the **same 180 s** ledger and its complete phase
prefix. Missing/foreign/reset history is refused. Old metadata is not added
again: campaign charge is prior case charge + new case paid/residual reserve +
the inclusive current ledger charge. All three new caps total 4,290 s. Worst case
is 3165.841891122982 + 4290 + 180 = 7635.841891122982 s inside the original 10,800 s
ceiling. There is no grace or retry. Complete durable terminals charge measured
time even if a numerical verdict is unavailable; interrupted/missing terminals
reserve the residual up to their full cap. Overrun is retained within measured
cost and halts another admission, rather than being charged twice. Final
accounting still records an actual overshoot. The separate named-family and
historical campaigns are not part of this cost join.

## Root-only interface

Review and commit the new source before these commands. Replace placeholders
with root-selected fresh locations and the integration interpreter. They are
instructions, not evidence of invocation. All commands below enter phases of
the same canonical ledger; no command creates another 180 s allowance.

```sh
# Exact source-only declarations, then source preparation. Root first preserves
# the SHA-bound CLOSED parent ledger copy at CLOSED_PARENT_LEDGER.
CUDA_VISIBLE_DEVICES='' python -B reports/forge/pr223-native3-continuation-20261004/run_native3.py \
  --plan --plan-file NEW_PLAN.json --closed-metadata CLOSED_PARENT_LEDGER \
  --closed-metadata-sha256 ce949407c9a11b2d2873be0d51b2d147261f8b60cee411ea33381c8c5c6f4369
CUDA_VISIBLE_DEVICES='' python -B reports/forge/pr223-native3-continuation-20261004/run_native3.py \
  --prepare-only --output NEW_OUTPUT --queue-root SHARED_FORGE_QUEUE \
  --closed-metadata CLOSED_PARENT_LEDGER \
  --closed-metadata-sha256 ce949407c9a11b2d2873be0d51b2d147261f8b60cee411ea33381c8c5c6f4369
# Run from the ACTUAL copied source. Source guards reject a foreign helper.
CUDA_VISIBLE_DEVICES='' python -B SNAPSHOT/reports/forge/pr223-native3-continuation-20261004/run_native3.py \
  --copied-preflight NEW_OUTPUT_PARENT/.NEW_OUTPUT_NAME.pr223-native3-prepared.json
```

The copied proof validates the actual relocated import owners without executing
them, compiles the exact original wrappers, and exercises the exact copied
helper/scorer prefix in private inert hosts. It stops before scoring and requires
four outcomes: repaired PASS, predecessor ambiguous namespace refusal, foreign
namespace refusal, and cached candidate refusal. The real scorer is read as
source only. Portable tests' exact 1,660-byte inert scorer fixture/README are also
explicit snapshot inputs. No Torch/model import, scientific scorer, saved array
or numerical credit is allowed in this proof.

Root sends the frozen source, exact prepared packet, persisted copied-boundary
proof and carried budget to authenticated Ember for the required fit verification
**before any paid learner**. Root claims/acknowledges that response. This is an
operational prerequisite; no invented approval flag/schema replaces it. Only
after that verification:

```sh
CUDA_VISIBLE_DEVICES=1 python -B SNAPSHOT/reports/forge/pr223-native3-continuation-20261004/run_native3.py \
  --output NEW_OUTPUT --queue-root SHARED_FORGE_QUEUE \
  --closed-metadata CLOSED_PARENT_LEDGER \
  --closed-metadata-sha256 ce949407c9a11b2d2873be0d51b2d147261f8b60cee411ea33381c8c5c6f4369
```

`--max-new-attempts` may be 1..3 for an intentional closed prefix; it never
shortens a case. GPU 1 / cuda:0, CPU 1, memory fraction .2, no autocast/TF32,
12288 MiB free and temperature ≤82°C retain the original resource conditions.
Both inherited study/attempt FDs, actual source, durable token and absolute
inclusive deadline remain mandatory. Only the already-paid launch wait is
excluded from active parent metadata time. The child uses the maintained
`run_retest.py --child REQUEST --lease-fd FD`; scientific loops are not copied
into this runner.

## Wire and passive publication boundary

`native3_contract.plan(root, helper, closed_anchor)` returns
`pg_pr223_native3_continuation_v1`, with exactly three active rows/definitions,
`required=3`, `original_required=19`, and the unchanged original19 catalog as
reference. `protocol.original19_catalog` is the complete original protocol
object, including its full config, common Recipe, runtime and all 19 rows. Active protocol ordinals 1/2/3 explicitly retain original19
ordinals 17/18/19 and parent retest IDs. `prepare(..., native3_anchor=pin)` and
`run(..., native3_anchor=pin)` are the narrowly scoped maintained-helper hooks;
their default full19 behavior is unchanged. Unknown/malformed subsets and old
grade injection are refused before model construction.

The sidecar is `.OUTPUT_NAME.pr223-native3-prepared.json`; the required copied
receipt is `.OUTPUT_NAME.pr223-native3-copied-preflight.json`, schema
`pg_pr223_native3_copied_source_preflight_v1`, bound to prepared/source/protocol/
maintained-helper bytes, count 3 / 21,000, actual eight native import owners and
`native_scorer_boundary_control`. The source manifest includes the three compact
parent files, inert portable fixtures and the immutable closed anchor at
`reports/forge/pr223-native3-continuation-20261004/inputs/closed-parent-metadata.json`.
The advancing live ledger is never a snapshot input. The byte-exact closed12
[control fixture](fixtures/README.md) is also included explicitly, so copied
portable controls need no live or archived external source access.

Requests retain `{packet,row,target,command,worker}`. New case attestation schema
is `pr223_native3_case_attestation_v1`, with the existing complete Recipe,
grade/artifact/media/source/request/token/deadline joins and explicit
`execution_scope`, `full_original19_credit=false`,
`fresh_full_original19_only=false`. Grade keys are literally
`status`, `original_gate`, `original_protocol_gate`, `reported_original_status`,
`native_gates`, `completed_steps`, `metric_observations`,
`full_protocol_complete`, `result_path/result_sha256`, `artifacts`, `final_metrics`,
`clean_diagnostic_status`, `sampling`, `qualification_input`.
`native_gates.noisy.{coverage,accuracy}` jointly controls acceptance;
`native_gates.clean` is separate. Raw accuracy PASS with noisy coverage FAIL
retains `reported_original_status=PASS` and accepted `original_gate=FAIL`; a
complete scientific failure is not erased as unavailable. Raw result/grade/GIF
are retained even when
attestation/owner/source/cap acceptance is unavailable. A case overrun gets no
accepted numerical or GIF credit. No official regrade is performed by these
metadata validators.

The new passive interface is separate from the frozen full19 prefix publisher.
Root will supply a trusted
`pg_pr223_native3_terminal_publication_card_v1` pin: canonical study directory,
fresh source origin/digest, required 3 / original_required 19, parent-reference pins,
and immutable copies of prepared/study/CLOSED ledger/copied preflight/source
manifest and optional root source-proof pins. The new consumer binds an explicit
trusted card SHA and expected source origin/digest; no new approval schema is
introduced. It must reject RUNNING rows or an open cut. Original
canonical `closed_live_inputs` paths are provenance only; an advancing LIVE
ledger is not claimed equal to the closed copy. A final post-publication cost
addendum pins that same live ledger after its phase closes. Final current case
cost plus prior case cost plus inclusive metadata is projected once. Original
16 PASS / grid INVALID / two NOT_RUN never becomes a new full19 grade.

## Structural controls

Python ≥3.10 is required by the unchanged ledger. Run only the portable model-free
suite below with CUDA hidden and one thread. These synthetic PASS/FAIL labels
test validators, not numerical convergence or capacity. Tests refuse external
archived source dependencies and all live ledgers/locks/temporary siblings, raw
studies and queues (including aliases). The synthetic phase test replaces the
ledger loader itself, so repeated module loads cannot escape the fake. The scorer fixture is inert `.py.txt` with its
original SHA and cannot be imported as a Python module.

```sh
# New native3 metadata controls and unchanged budget accounting controls.
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -B -m pytest -q \
  reports/forge/pr223-native3-continuation-20261004/test_native3.py \
  reports/forge/pr223-original-full-retest-20261004/test_budget_ledger.py
# Exact portable retained bootstrap selection; excludes external plan/parity tests.
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -B -m pytest -q \
  reports/forge/pr223-original-full-retest-20261004/test_native_scorer_imports.py \
  reports/forge/pr223-original-full-retest-20261004/test_run_retest.py \
  -k 'native or import_source or import_guard or derived_wrapper or original_scorer or copied_preflight_uses or matching_copied or canonical_normalization or fresh_pythonpath'
```

Independent review must inspect subset/source/carry accounting, strict namespace
ownership, actual generated-prefix negative controls, unchanged scientific
functions, copied proof before registration, and same-ledger final cut handling.
The real copied-source proof and Ember fit verification remain root prerequisites;
local synthetic tests do not replace either.
