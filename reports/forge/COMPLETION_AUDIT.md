# Forge completion audit — 2026-09-28

The accepted [implementation plan](../../docs/better-experiment-automation-plan-2026-09-28.md)
is not complete. Inventory, shared public components, conservative grading,
file-backed orchestration, and real CPU fail-fast execution are substantial
implemented foundations. Declared tasks, software tests, and imported historical
passes do not substitute for runnable scientific contracts, accepted calibration,
or the expressly required bounded multi-GPU pilot.

This audit read code, tests, declarations, saved receipts, and GPU/process status.
It launched no training, changed no queue ownership, and altered no implementation.
Other agents are actively implementing fixes; the findings below describe the
inspected state and should be closed with linked evidence, not removed by
narrowing the plan.

## Follow-up status after the software fixes

The 2026-09-29 merge of develop `a8b9d397` at `fe1dd2e1` passed 576 integration
checks; the subsequent history extension passed 14 checks. Clean public sampling
retains MoG kernel noise, and vector gates keep their explicit full-component
policy. The [new source study](QUICK_SCREEN_STUDY.md) imports no older receipts.
Its baseline screen and two-cell physical pilot were registered separately.
Calibration, the real two-GPU pilot and cutover remain outstanding.

Full CI then exposed six legacy image-host failures from the public trainer's
blanket RNG uniqueness guard. The [compatibility repair](MIGRATION.md#ci-compatibility-repair)
preserves legacy shared draws while retaining Forge isolation, with 167 affected
tests and 504 broader checks passing. Replacement develop **v2** registrations
bind that new source; the unexecuted v1 artifacts are retained. Full CI on
`3a06e09c` passed 1,582 tests and 18 subtests, with 16 explicit skips, plus all
packaging and Python-version smokes.

The [first develop GPU baseline](DEVELOP_QUICK_SCREEN_READOUT.md) then completed
three scientific FAILs in 38.915 seconds on GPU 0, with complete observations,
zero unintended RNG deviations, cross-checkout deduplication and zero remaining
reservations. All 16 independent references remain unknown. The subsequent
adapter audit found a missing real-image clamp, an obsolete sampling-law claim,
and a CUDA peak-memory initialization failure. Prospective fixes and strict
versioned sampling receipts create a new cohort; the original results and costs
remain archived. The next bounded diagnostic is corrected `img_intensity2`
alone. These one-device checks do not satisfy the physical two-GPU pilot.

The original inspection below is retained as an audit trail. These subsequent
changes close implementation gaps without claiming the still-missing scientific
or GPU-pilot outcomes:

- Runtime dispatch and conservative preflight now cover paired adaptation,
  clock probes and native continuation. `adaptation.py`, `clockfree.py` and
  `continuation.py` bind saved state/curve artifacts; short tests exercise paired
  frozen controls, a delayed-clock rejection and prefix/continuation equivalence.
  The original five missing-runtime finding is superseded. These short fixtures
  are not canonical 3,600-, 7,000- or 14,000-update qualification receipts; inspect
  the final integration tests for fresh-process coverage as well.
- `calibration.py` now ingests certified current learned-MoG attempts, independently
  grades raw evidence, preserves complete positive/negative and cost denominators,
  and accepts only matching source/task/prior/seed/runtime/compute cohorts.
  Certified repair costs remain paid; renamed smoke predicates cannot become an
  independent reference. Historical cloud replay stays separate. The original
  hardcoded-empty-current-cohort finding is closed.
- `calibration_lane.py` registers immutable, explicitly bounded diagnostic
  selections. Namespaced evidence cannot confer ordinary qualification.
  Calibration verifies registration and exact selected task scope. Diagnostic
  views always return tier 0 and ineligible, including when every raw metric passes.
- Promotion now calls `verify_calibration`, binding and replaying an accepted
  current report, its profile and criteria, cohort and exact required Tier 1
  task set. The manually editable accepted-string gap is closed.
- The combined current calibration, view, diagnostic-lane and promotion suite
  passed **160 tests**. These use saved synthetic receipts and bounded software
  fixtures; this subagent launched no experiment training. The concrete current
  study and its full independent reference denominator are documented in
  [CURRENT_CALIBRATION_PROTOCOL.md](CURRENT_CALIBRATION_PROTOCOL.md).

- Administrative lifecycle now has immutable abandon/supersede receipts, exact
  revision and successor binding, readout requirements, queue admission guards,
  and explicit cancelled-request repair. Shared evidence does not reopen a
  disposed alias. Real queue/CLI integration tests preserve verdicts and costs.
- `stats` and compiled `automation.json` now report unique paid attempts, cost to
  observed rejection/attainment, tier rejects, avoided requested work, reuse,
  execution errors and measured process overlap. Instrumented adapters report
  exclusive phase timing and scoped process/allocator peaks. Older unmeasured
  fields stay unavailable; pinned verdicts are not regraded by telemetry.
- Family and evidence-quality filters operate on whole rows and preserve their
  complete qualification denominators. All four views explicitly declare
  tier-only ordering with raw metrics; no undeclared metric aggregate is ranked.
- The [legacy-consumer inventory](LEGACY_CONSUMERS.md) identifies repository/CI
  callers and actual external GPU owners. The [physical pilot procedure](MULTI_GPU_PILOT.md)
  is prepared, with a capacity-window question pending and no GPU launch.

- Explicit cross-profile diagnostic imports now resolve the queue/reducer
  mismatch: compatible jobs can retain one paid attempt while a separately
  frozen profile authorizes its original evidence and costs. Missing, altered,
  unlisted or scientifically incompatible inputs cannot confer qualification.
  The [quick-screen study](QUICK_SCREEN_STUDY.md) demonstrates real saved-result
  reuse without new training. Documentation-only source-origin changes also
  preserve calibration/promotion registrations and their original provenance.
  **474 Forge and recipe tests passed** after these fixes.

**Still required for completion:** real compatible positive/negative current-MoG
calibration evidence meeting the unchanged criteria, the bounded physical
multi-GPU pilot, exact outstanding historical claim artifacts, the conditional adoption/cutover decision. Inventory coverage and both
fresh-checkout walkthroughs are now recorded. Neither the
software fixes nor diagnostic registration changes adoption to PASS. GPU
capacity and legacy ownership must be rechecked at execution time.

## Original blocking findings at the initial inspection

1. **Five declared tasks have no runtime adapter.** `run_task` lacks
   `clockfree_audit`, `paired_adaptation`, and `native100_continuation`; the latter
   serves all three native 14k cards. Their graders and view assignments exist,
   but these views cannot complete through Forge. The planner does not yet turn
   these unavailable adapters into explicit preflight blockers. The correct
   runtime fallback is BLOCKED, which protects claims but does not complete the
   required implementation.
2. **Phase D correctly reports adoption BLOCKED and remains incomplete.** The
   initial historical profile pairs 6/10 lineages and falsely accepts 1/3
   independently failing references; the alternate pairs 8/10 and falsely
   accepts 1/5. Those pooled numbers are descriptive, not cohort acceptance.
   No new learned-MoG calibration cohort has been measured. Several historical
   cohorts lack positives, complete timing, or compatible smoke receipts. The
   six proposed missing smoke runs alone cannot meet these criteria.
3. **The current calibration path cannot ingest the future evidence it needs.**
   `calibrate` reads `history-*.json` and constructs the new-MoG cohort from an
   empty list unconditionally. A real current-cohort receipt ingestion path is
   needed before a bounded calibration campaign can lead to accepted adoption.
4. **Promotion's adoption prerequisite is not yet evidence-bound.**
   `_finished` checks `view.calibration.status` for an accepted string but does
   not verify a passing report, its criteria/profile hashes, or the candidate's
   source/host/prior/runtime cohort. Accepted calibration must be a checked
   artifact reference, not a manually editable status.
5. **No real multi-GPU pilot evidence is saved.** The plan explicitly requires
   bounded multi-GPU draining in Phase C and a candidate/reference pilot before
   cutover. Fake GPU slots, real CPU workers, and single-task CPU failures prove
   useful separate properties; they do not satisfy this acceptance item.

## Requirement-by-requirement state

| Plan requirement | Inspected evidence | Completion judgment / next evidence |
| --- | --- | --- |
| A: classify every scoped source, including report drivers | `history.py`, catalog, source pins, coverage tests | Implemented. Current coverage audit reports 7 newly tracked Forge files missing from the catalog; refresh after final staging and require zero missing/unclassified paths. |
| A: import initial and #155 exact lineages and useful failures | 175 compiled records; historical import tests preserve cloud/fixture/package/sampling identities, raw refusals, row-EM cost, structural14k prefix evidence | Implemented for bound sources. Later dt07514k, EMA .995 D-tracking, and centre-sensitivity claims remain explicit gaps; exact receipts are still needed for their required calibration/scoring review. |
| A: deterministic memory, recall, conflicts, negative controls | `knowledge.py`, generated memory/boards, deterministic/conflict tests | Implemented; refresh derived artifacts after final source/record changes. |
| B: one shared public API and explicit extensions | `FormulationContext`, public MoG/capability support, public component bindings, extension registry | Implemented foundation. No private optimizer replacement is needed for supported hosts. A changed mechanism outside public fields still needs a reusable public API implementation and binding. |
| B: unsupported variables blocked before spending | Resource/objective checks, extension checks, capability exceptions | Partial. Missing adapter availability and clock-free-host restrictions should be surfaced by plan/enqueue before a worker is reserved. A global capability flag is not proof that a task implements it. |
| B: learned MoG default and explicit cloud exceptions | Defaults; task priors; actual public sampler/update/checkpoint tests | Implemented. Seven mathematical behavioral exceptions and four finite-centre image exceptions remain explicit. AE supplies actual MoG smoke coverage; historical cloud passes never transfer to it. |
| B: same initializer and component-scoped RNG | Named family/component/purpose streams; isolation, changed-shape, evaluation and checkpoint unit checks | Implemented for supported public paths. Real multi-GPU placement and full archived replay stream-parity evidence remain part of the outstanding pilot/calibration. |
| B: tier-only edits reuse unchanged evidence, pinned campaigns | Independent task/evaluator/policy hashes; retier and queue tests; dependency/cycle/group validation | Implemented. Uninterrupted ring group must move together; that is a necessary cap constraint rather than a copied experiment. |
| B: administrative lifecycle distinct from qualification | Queue/readout states, concluded readout binding, retained failed attempts | Implemented foundations. Superseded/abandoned workflow and successor linkage have less exercised CLI coverage than ready/running/concluded. |
| B/C: all initial 19 transfer plus three native quality hosts | Dispatch, 29 task cards, short integration tests | Runtime paths exist for the 22 quality tasks. Bounded tests validate wiring; no complete new-MoG 22-task candidate/reference evidence is yet present. |
| B/C: original observation/terminal/live predicates | Existing transfer/native/ring evaluators, immutable measurement checks, saved artifact manifests | Implemented; tampering/truncation/EMA-switch/late-failure tests are present. |
| B/C: actual intended updates and active mechanisms | Per-mechanism counters, lazy-no-op blocking, explicit ablations, tiny synthetic public-component activation checks | Implemented. Synthetic activation checks prove hook execution only; they are neither host-quality evidence nor extra seed experiments. |
| B/C: own-state ring hold plus extension, original schedule | One grouped uninterrupted run, 7,500 cap, separate hold/extension grades, short failure/prefix tests | Implemented and conservatively graded. No full-budget current-candidate endurance result is saved. |
| Clock-free: state-only eligibility and source/decision audit | Claim metadata and a receipt-checking grader exist | Missing executable audit producer, public state-only formulation contract and bound rate/decision trace checks. Scheduled K3P and constant LR must remain insufficient. |
| Clock-free: step-label/horizon/evaluation/restart invariance | Generic public checkpoint/prefix unit checks; synthetic clockfree receipt test | Missing actual audit runs against a permitted-state fixture and a deliberately hidden-clock failure. Hand-authored equal hashes are not the required production instrumentation. |
| Clock-free: own-state native7k→14k and late-release diagnostics | Three continuation declarations and prefix validator | No runtime adapter. Implement continuation with source/parent/full state/RNG/next-update proof, or one uninterrupted grouped execution; preserve the 7k prefix and preregister dense checks around releases. |
| Adaptation: preserved pre-shift state and frozen negative control | Task declaration plus strict raw-curve/control grader | No public runtime adapter. Emit matched active/frozen controls, actual update counts, pre-shift state identity, full 3,600-step diagnostic curves, and original deadline windows. |
| Live/EMA policy separation | Live gates are fixed; native EMA samples/metrics are diagnostic | No qualifying EMA view/adapter path exists. Preserve this as unresolved work; never switch a live failure to EMA. Bind the exact later EMA evidence before selecting an EMA protocol. |
| C: one file-backed queue, dedup, atomic cost/lease ownership | Queue/worker/source tests, immutable snapshot runner, shared subscribers | Implemented and tested with fake allocations/real subprocesses. Multiple real worktrees and real CUDA placement still need bounded pilot receipts. |
| C: limits, cancellation, timeout, restart, no speculative tiers | Process-group tests, fencing, tier/campaign/candidate reservations; saved CPU fail-fast attempts | Implemented software evidence. Exercise coordinator restart/cancel during the bounded GPU pilot without interrupting legacy work. |
| C: centralized tail-friendly metrics/logs and durable results | Collector, event filters, per-attempt logs, saved CPU attempts | Implemented CPU evidence. Simultaneous real GPU child streams have not been demonstrated. |
| C: resource-aware scheduling and legacy ownership | Free-memory checks, one drain owner and lease files per queue | Queue-local control exists; there is no cross-legacy-scheduler GPU owner. Pilot must explicitly reserve non-overlapping capacity or use a common resource owner first. |
| D: independent smoke/reference calibration with cost/error criteria | Initial/alternate matrices, frozen criteria, blocked denominators | Matrix is present and honestly blocked; actual accepted calibration is outstanding. Missing evidence cannot count as zero cost or scientific failure. |
| D: controlled deeper diagnosis of rejected candidates | Plan describes a separately bounded calibration lane | No executable current calibration-lane contract was located. It must permit selected diagnostic work without conferring qualification or turning each rejection into a full sweep. |
| E: guide, discovery links, small declaration, readout | `EXPERIMENTATION.md`, root AGENTS/README links, real CPU attempt/readout path | Mostly implemented. The onboarding document initially recorded an existing-worktree preparation, not a fresh-checkout completion. Finish and bind the documented walkthrough, including its final readout and compilation. |
| E: fixed finished-candidate robustness stage | Registration/planning/enqueue/reduction code and synthetic contract tests | Implemented contract foundation, correctly no real promotion claim. Bind accepted calibration first; an actual robustness stage is required only when a finished public-default candidate is promoted. No speculative seed runs are needed for engine completion. |
| E: default adoption, legacy queue cutover and rollback | Migration document explicitly says pending; old launchers retained | Outstanding after Phase D/pilot. Inventory live consumers, stop new old-queue claims only within authorized ownership, reconcile pending work and retain rollback receipts. |
| Metrics for the automation itself | Actual wall time, optimizer counts, device cohort, FLOPs explicitly unavailable | Partial. Publish cost-to-reject/qualify, tasks avoided, reuse and incomplete/error rates, concurrency, and measurable peak memory. Separate training/evaluation/sampling cost where available; do not call aggregate adapter time pure training time. |
| Goal-specific comparisons and filtering | Four views, compatibility cohorts, tiers/raw metrics/cost, partial/pinned rows | Implemented basic views. Family/evidence-quality filtering and domain-balanced ranking need explicit acceptance coverage if advertised. Retain separate cost and quality, not a universal aggregate. |
| Future monotonicity and broader production/domain tracks | Explicitly marked future in plan | Correctly inactive; no monotonic GAN-loss predicate should be added to close this audit. Preserve future scope without pretending current views establish production readiness. |

## Current scientific and compute evidence

The saved implementation pilot `b766e89d1a0843ba9af94f2ce94e4273` ran
`two_pole` through its full 80-update host budget and failed: mean_abs 0.10652
against >=0.30, with 5.2552 seconds supervised wall time. It launched no remaining
smoke or higher-tier tasks. A second CPU receipt,
`c0178716dd3c4d5695b391919bf9ab81`, likewise records a completed `two_pole` FAIL
and about 5.9746 seconds. These are exact revision-specific negative results and
useful workflow evidence, not proof that historical K3P fails or that the initial
screen is well calibrated. The known useful historical reference failing a new
public-component smoke path needs diagnosis of intended mechanism/prior/host
differences before adopting the screen.

Read-only capacity snapshot during this audit found two NVIDIA RTX A6000 devices
(49,140 MiB each). GPU 0 reported 3,126 MiB used and 83% utilization with active
HyperGAN training; GPU 1 reported 1,152 MiB used and 16% utilization with desktop/
viewer processes. Another legacy `run_grid` process was also present in the
process list. Free memory is not a GPU ownership reservation. No process was
stopped and no device was claimed. Recheck capacity immediately before a pilot,
establish non-overlapping ownership, and wait if the required devices are not
available.

## Smallest faithful next work

1. **Close software blockers before expensive runs.** Add a shared adapter
   availability/preflight registry; implement public paired adaptation and the
   executable clock/state audit contract; implement verified native continuation
   rather than accepting self-reported restore booleans. Add bounded tests using
   a known state-only fixture, a hidden-release counterexample, a matched frozen
   negative control, and a real fresh-process restore comparison. Keep unchanged
   original metric predicates and explicit unsupported cases.
2. **Make accepted calibration attainable and evidence-bound.** Ingest certified
   current task receipts into named prior/source/runtime cohorts, with an
   independent reference excluding the smoke predicates. Bind profile, criteria,
   task/evaluator, source and receipt hashes in the report and accepted-view
   reference. Promotion must verify that binding. Do not mark an empty MoG cohort
   accepted or manually relabel the provisional view.
3. **Predeclare the smallest informative calibration matrix.** First recover
   existing exact fixture/package/cost artifacts. For each cohort targeted for
   adoption, choose at least the frozen required positive and negative references
   and complete missing smoke/reference cells within explicit task/candidate/
   campaign limits. New-MoG comparisons are a separate cohort, with the single
   screening seed. The proposed six historical negative-lineage smoke jobs are
   useful gap filling but not a sufficient adoption campaign. If known positives
   are rejected, diagnose bias or revise the profile under a new revision; do not
   weaken the active predicate after observing failure.
4. **Run the required real multi-GPU pilot when owned capacity exists.** Use two
   substantive candidate/reference workloads and compatible scientific tasks,
   not seed variants. Preserve normal prerequisite gates; any deliberately
   deeper diagnostic work must use the explicit nonqualifying calibration lane.
   Record simultaneous physical device assignments, budgets/reservations, logs,
   source snapshots, restart/cancel behavior, dedup across submitters, central
   event integrity, and final readouts. Retiering/reuse checks launch zero extra
   training. CPU fallback cannot be labelled the multi-GPU pilot.
5. **Finish evidence and adoption rather than changing the milestone.** Refresh
   inventory and compiled records, complete the fresh-checkout walkthrough,
   publish measured automation costs and unresolved scientific outcomes, then
   meet Phase D acceptance or explicitly remain adoption-blocked. Only after
   that should Forge become the default entrypoint and replace covered legacy
   launchers. A useful unfinished implementation is still unfinished.

No successful default GAN, EMA policy decision, or multi-seed robustness claim is
required merely to prove the engine works. Conversely, implementing the engine
does not waive its accepted scientific API, calibration, GPU-pilot, or migration
acceptance criteria.

## Latest measured checkpoint

The corrected v2 cohort has all nine smoke cells measured (five PASS, four FAIL),
53.211 seconds total CPU wall time, identical saved metrics to v1, concluded
readouts and zero reservations. Its 48 independent reference cells remain unknown;
false accept/reject rates cannot yet be measured. The two-cell physical GPU pilot
is registered but neither submitted nor launched. Software evidence is 441 checks
at the broad freeze plus 90 focused checks after the receipt-only correction.
