# ParticleGAN Forge migration

Implementation is in progress on `codex/tiered-experiment-qualification`, based
on `develop`, in [PR #221](https://github.com/255BITS/ParticleGAN/pull/221).
The [accepted plan](../../docs/better-experiment-automation-plan-2026-09-28.md)
defines acceptance; this status file does not narrow its scope.

## Develop integration — 2026-09-29

Fetched `origin/develop` at `a8b9d3977701ca700d9918ac66d40ac814b9f9ba` and merged
it into this feature branch. Public sampling now combines clean output by default
with Forge's component enumeration, learned MoG kernel and isolated RNG streams.
Scalar, image, native and adaptation scoring explicitly omit training output
noise; native live/EMA/holdout receipts use the same law. Frozen behavioral-host
noise policies remain unchanged. Enumeration blocks must fit inside the table;
the previous wrapping description was incorrect.

The vector evaluator source changed with protocol v5. Forge retains its explicit
full-component quality thresholds under `forge-vector-full-component-v1` rather
than granting uncalibrated finite-atom shape exemptions to nonzero-width MoG
priors. Each of the six vector tasks binds the new source hash and records this
policy. A regression demonstrates that a rare-component collapse still fails
the retained Forge gate even when the upstream resolved-component gate passes.

**576 checks passed** across Forge, public training/clean sampling, recipe
defaults and the upstream vector/critic changes. The merged code is a new source
cohort. Older studies, registrations and receipts retain their original identity;
new execution requires a newly frozen profile and bounded registration.

The [merged-source quick-screen study](QUICK_SCREEN_STUDY.md) now freezes all
three candidate revisions with no old-source imports. Its three-task K3P lane
is registered with a 5,400-second ceiling. A separate
[physical pilot registration](MULTI_GPU_PILOT.md) selects two substantive vector
cells under the same source. Neither has launched; device ownership is tracked
in [the updated audit](LEGACY_CONSUMERS.md#ownership-update--2026-09-29).

The import adds 24 source files from develop's vector-protocol research as two
context cards, preserving all 174 earlier cards byte-for-byte. Seven JSONL
schemas remain explicitly unnormalized. The compiler contains 184 records and
no pending readouts or conflicts; 14 history checks passed after this addition.

Legacy grid/native-runner compatibility passed **26 tests and 13 subtests**.
The first check encountered the missing optional `matplotlib` dependency; the
successful check used a temporary validation environment with that dependency.
The frozen experiment runtime and repository source were unchanged.

## Frozen inputs and ownership

- Package/inventory baseline: `92dc0319`.
- LR-free source: `0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b` (PR #155).
- Implementation start: `2509ab53ff750860096b16d3c47458adbdf79941`.
- Later dt075 continuation / EMA / sensitivity receipts: import gaps until exact
  source artifacts are located; no substitution with similarly named packages.
- Coordinator: contracts, immutable source capture, queue, integration, Git and
  the single migration readout. Subagents: history/imports; public API/RNG;
  task declarations/views. All edits occur in this worktree with disjoint owners.
- Compute allowance for foundations: zero training. CPU parity/unit checks are
  allowed. Calibration and GPU pilot need a recorded bounded campaign.

At kickoff GPU 0 had active HyperGAN training and GPU 1 had an active viewer.
No legacy Forge/LR-free pool process was found in this machine's process list.
This is an observation, not a capacity reservation. No process was stopped and
no old queue ownership was changed. Recheck before the pilot; reserve capacity
outside active pools and retain existing experiment requests.

## Progress

| Work | State | Evidence |
| --- | --- | --- |
| Freeze source revisions / branch / PR | Complete | PR #221 open, base `develop`; pins above |
| Shared file contracts and source snapshots | Implemented, under integration review | `experiments/forge/contracts.py`, `sources.py`; mutation/reuse tests |
| History mapping and source coverage | Complete classification at checkpoint `3a3ef24a` | 7,252 scoped paths; 174 cards, 112 scientific and 62 family-context; pinned #155 included |
| MoG public API, capabilities, paired RNG | Implemented and tested | Public trainer/prior/A2, named component streams, checkpoint parity |
| Task definitions / tier views / independent graders | Implemented and tested | 29 tasks, four views; no active monotonicity gate |
| Queue, adapters, compiler, CLI and root guide | Real CPU execution and resource enforcement verified | Real process-group tests, atomic CPU/RAM reservations; final integration checks recorded below |
| Clock audit, paired adaptation and native continuation | Implemented; bounded state/protocol verification | Saved state manifests, measured optimizer counters, exact prefix/restore/RNG comparison; `CONTINUATION_REVIEW.md` |
| Administrative lifecycle and display filters | Implemented and tested | Immutable abandon/supersede receipts; explicit cancellation repair; family/provenance filters preserve full qualification |
| Automation telemetry | Implemented; historical missing measurements remain unavailable | Unique paid costs, avoided work/reuse/errors/concurrency; process RSS, CUDA allocator peaks and instrumented phase timing |
| Calibration diagnostics and robustness registration | Implemented; no production adoption claim | Registered diagnostic namespace, report-bound calibration, fixed promotion contract and forgery/retry tests |
| Phase D calibration / bounded multi-GPU pilot | Historical calibration replayed; current calibration and GPU pilot pending | Adoption remains blocked; CPU pilot below |
| Root-guide onboarding | Fresh-checkout walkthrough complete | `FRESH_CHECKOUT.md`; penalty ablation rejected after 7.371 seconds, all remaining tasks unlaunched, concluded readout and zero reservations |
| Cutover | Pending accepted calibration and reserved multi-GPU pilot | Existing launchers retained |

No candidate is promoted by implementing the engine or by passing its tests.

## First bounded CPU pilot

Campaign `implementation-pilot-cpu-v1` reserved at most 900 wall seconds for
one candidate through Tier 1. It used one CPU worker and changed no legacy queue
or GPU ownership. Request `043465ae751e26417aad2e9a`, attempt
`b766e89d1a0843ba9af94f2ce94e4273`, candidate revision
`87d849138dd0ae385260a77223e9a70a7c2a8a112733b2312f556b462cc164e9`.

| Task | Verdict | Relevant measured value | Cost |
| --- | --- | --- | --- |
| two_pole, 80 updates | FAIL | mean_abs 0.10652 < 0.30; grad_med 0.06372 <= 1.0 | 5.2552 s supervised wall time; 0.3122 s host execution |
| Remaining smoke / higher tiers | NOT_RUN | Required earlier gate failed | 0 executions |

This verifies the real request → immutable snapshot → worker → independent
grader → centralized metrics → durable receipt path and cheap rejection. It
does not establish the smoke profile's predictive value. Preserve the failed
attempt as a negative result for this exact public-component task/prior/config;
do not interpret it as a blanket rejection of historical K3P evidence.

Logs: `runs/forge/implementation-pilot/events.jsonl`. Durable evidence:
[`attempts/b766e89d1a0843ba9af94f2ce94e4273/result.json`](attempts/b766e89d1a0843ba9af94f2ce94e4273/result.json).

## Onboarding and remaining adoption work

The independent [onboarding walkthrough](ONBOARDING.md) used the documented
scaffold/plan/enqueue/drain/logs/board/readout/compile path for an anchor ablation.
It failed `two_pole` after 5.9746 CPU wall seconds, spent on exactly one attempt,
and left every later task unlaunched. Its frozen source and explicit particle-cloud
host exception remain visible; neither CPU pilot supplies learned-MoG calibration.

The [completion audit](COMPLETION_AUDIT.md) and
[continuation review](CONTINUATION_REVIEW.md) retain concrete findings and their
verification. Current calibration now ingests certified new-cohort evidence;
diagnostic lanes may collect selected independent reference cells after a screen
failure without granting qualification. No current profile has passed adoption.

- [x] Refresh source coverage and compiled memory at checkpoint `3a3ef24a`: 7,252/7,252 paths, 176 records, no conflicts or pending readouts. Repeat after subsequent source additions.
- [x] Freeze three substantive controls and a bounded first current-cohort
  diagnostic selection: all nine smoke cells measured; `CURRENT_SMOKE_READOUT.md`.
  The metadata-only noise-receipt correction has a completed v2 source cohort: all
  nine verdict/metric dictionaries match v1, at 53.211 CPU seconds. Both batches
  remain separate. The [bounded CPU reference batch](CPU_REFERENCE_READOUT.md)
  subsequently measured three K3P task passes for 20.865 seconds; full-reference
  decisions remain unknown. With every lineage failing smoke, v2 cannot satisfy
  both positive-reference and zero-false-rejection criteria. Stop filling this
  profile solely for adoption; register a separate screen study.
- [x] Declare the [separate quick-screen study](QUICK_SCREEN_STUDY.md) with
  unchanged thresholds/criteria and full reference denominator. Explicitly reuse
  three exact CPU receipts and their costs; register its bounded baseline lane
  without enqueueing. This does not constitute accepted calibration.
- [ ] Meet the unchanged acceptance criteria or revise the screen with a new
  declared profile and repeat its necessary calibration.
- [ ] Reserve non-overlapping GPU capacity or one shared resource owner, then
  record the real multi-GPU/cancellation/restart pilot.
- [x] Complete a fresh-checkout walkthrough; `FRESH_CHECKOUT.md` binds the exact
  source, commands, candidate, evidence and original readout.
- [ ] Reconcile actual legacy consumers/queues before making Forge the default
  entrypoint. The consumer inventory is complete; ownership is unchanged.

No legacy process has been stopped or adopted by Forge. No robustness seed
experiments have run; the registered promotion stage is conditional on a future
finished candidate and is not required merely to test the engine.

The [legacy-consumer inventory](LEGACY_CONSUMERS.md) maps active repository
entrypoints, CI, the external GPU owners, and rollback requirements. Both GPUs
were actively training at the latest inspection; the HyperGAN queue can launch
three pending jobs after its current child, so child completion alone is not a
capacity reservation. No ownership was transferred.

Final software freeze: **441 tests passed** (all Forge tests plus public recipe
defaults), including lifecycle, family filters and telemetry. The two warnings
are Python 3.14 fork deprecations in the concurrent reservation test. No scientific
qualification or GPU-pilot claim follows from these software checks.

Corrected-source follow-up: **90 focused tests passed**, fresh-checkout rejection
was reproduced with truthful seed metadata, and all nine v2 smoke cells have
concluded readouts. The registered two-cell GPU pilot has zero submissions and
zero launches pending the requested joint ownership window. See
[`CURRENT_SMOKE_READOUT.md`](CURRENT_SMOKE_READOUT.md).

Follow-up validation: **474 Forge and recipe tests passed** after fixing explicit
cross-profile diagnostic reuse and documentation-only registration idempotency.
The reducer preserves original receipts, complete selected outcomes and repair
costs; missing or altered imported evidence blocks adoption. Both bounded GPU
lanes have successful read-only plans against the preserved source checkout and
zero submissions. The live checkout has a newer orchestration source digest;
the older scientific receipts retain their exact cohort.
