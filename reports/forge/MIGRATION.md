# ParticleGAN Forge migration

Implementation is in progress on `codex/tiered-experiment-qualification`, based
on `develop`, in [PR #221](https://github.com/255BITS/ParticleGAN/pull/221).
The [accepted plan](../../docs/better-experiment-automation-plan-2026-09-28.md)
defines acceptance; this status file does not narrow its scope.

## Current checkpoint

Develop `a8b9d397` is merged; [CI at `1b2d8915`](https://github.com/255BITS/ParticleGAN/actions/runs/36672467580)
passes 1,807 tests and 18 subtests, including the concluded diagnostic receipts.
The engine, root guide and physical two-GPU pilot are implemented and verified.
Current scientific calibration has one correctly rejected negative, no complete
positive reference, and 50/57 unknown cells. The
[two-control screen](CONTROL_MODE_READOUT.md) failed both cells for 39.242 seconds;
all three declared lineages now fail smoke. This exact profile cannot meet its
positive-reference and zero-false-rejection requirements together. Stop filling
it solely for adoption; register a new justified screen/profile before further
calibration. The controls' independent reference outcomes remain unknown.
**Default adoption remains blocked.**
The [baseline host readout](HOST_PROFILE_TRANSFER_READOUT.md) and control readout
contain measured metrics and costs; [consumer ownership](LEGACY_CONSUMERS.md) records which legacy
entrypoints and external jobs remain outside Forge. Earlier checkpoints below
retain the evidence available at their dates.
The separately registered [A2-off probe](A2_OFF_NATIVE_READOUT.md) also failed
for 85.368 seconds, worsening coverage and mass/shape accuracy; its control was
reused. The engine recorded a useful negative without a control rerun or a sweep.
The [current independent audit](CURRENT_IMPLEMENTATION_AUDIT.md) closes the
original software findings with code, test and artifact evidence. It retains
scientific calibration and conditional adoption as the remaining requirements.
The [wider reference-source audit](POSITIVE_REFERENCE_SOURCE_AUDIT.md) found no
compatible full-reference positive. Its discovered legacy MoG envelope study is
now preserved in [supplemental context](supplemental/local-mog-envelope-v1/README.md)
and compiled memory, including all 26 rows and failed configurations. It retains
its historical EMA criterion and supplies no current qualification.
The latest compiler includes 193 records, with no conflicts or pending readouts;
both exact control revisions are concluded. The two-control independent audit
verified frozen grades, named initialization/RNG declarations and paid costs,
with its missing-tensor/final-state verification limit stated explicitly.

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
cells under the same source. At that checkpoint the v2 baseline had completed
and the physical pilot was unlaunched. Its later completed proof is recorded
below. Device ownership is tracked
in [the updated audit](LEGACY_CONSUMERS.md#ownership-update--2026-09-29).

The import adds 24 source files from develop's vector-protocol research as two
context cards, preserving all 174 earlier cards byte-for-byte. Seven JSONL
schemas remain explicitly unnormalized. The compiler contains 184 records and
no pending readouts or conflicts; 14 history checks passed after this addition.

Legacy grid/native-runner compatibility passed **26 tests and 13 subtests**.
The first check encountered the missing optional `matplotlib` dependency; the
successful check used a temporary validation environment with that dependency.
The frozen experiment runtime and repository source were unchanged.

## CI compatibility repair

The first merged full CI run recorded 1,566 passes and six failures: legacy
image harnesses intentionally share a training generator, which Forge's new
blanket public-trainer guard rejected. Commit `b4fae98a` preserves that legacy
draw sequence and enforces distinct named streams at Forge's boundary instead.
Evaluation stays isolated; model RNG isolation rejects direct/global aliases
that would lose draws. Checkpoint restoration rejects conflicting shared/global
states before mutation, preserving the existing schema and same-binding rule.

The affected suite passed **167 tests**, with one unavailable-CUDA check skipped;
the broader Forge/recipe/K3P suite passed **504 tests**. The final normalized-RNG
restore check passed 61 tests with one CUDA skip. Full CI on `3a06e09c` passed
1,582 tests and 18 subtests, with 16 explicit skips, plus packaging and version smokes.
The unexecuted develop v1 registrations remain frozen. Replacement **v2** baseline
and physical-pilot lanes bind the repaired source, preserve all budgets/criteria,
and import no old measurements. Both later executions are recorded below;
the pilot was still unlaunched at this earlier checkpoint.

## GPU-discovered adapter and receipt repairs

The first screen exposed three prospective fixes: reference image batches now
clamp noisy templates to `[0,1]`; CUDA peak-memory instrumentation initializes
the allocator before resetting counters; and all 29 tasks declare a versioned
sampling policy matched against actual adapter receipts. Queue submission and
runtime recheck candidate and grouped-task policies against frozen source.
Archived unversioned requests retain their original semantics and costs.

**525 Forge tests passed**, including reference-batch/RNG parity, sampler
attestation, grouped-task boundary validation and historical compatibility.
Full CI on `cd5308ec` also passed **1,643 tests and 18 subtests**, with 16 explicit
skips, Python 3.10/3.11 wheel smokes and release packaging:
[run 36663613925](https://github.com/255BITS/ParticleGAN/actions/runs/36663613925).
A real GPU 0 allocation-only probe measured 4,096 allocated / 2,097,152 reserved
bytes without training. Old missing memory measurements remain unavailable.
The repairs require a new cohort and registrations before any further training.

## Explicit host-profile transfer

The image adapter now uses one validated profile resolver and model factory.
The opt-in `img_intensity2_residual16` card binds the published architecture source
by hash; it keeps the current public K3P recipe, named initialization/RNG, clean
enumeration and original 600-update gates. The separate `host_profile_transfer`
view exposes it as a diagnostic. Existing tasks and default views are retained.
Constructor tests verify actual shapes before any optimizer update. Receipts bind
the resolved host card, model classes, parameter counts/shapes and initial-state
hashes. Unsupported declarations or a changed profile source block preflight.

This enables an explicit architecture transfer measurement; it does not make
the historical positive a current result. Vector critic and native architecture/
initialization differences are documented in [the audit](HOST_PROVENANCE_AUDIT.md).

The [registered residual16 intensity probe](IMAGE_PROFILE_TRANSFER_READOUT.md)
then passed its final eight checks for 11.198816130 seconds. It measured both
modes, HQ 0.90625 and mass TV 0 with zero unintended RNG deviations. Its readout
is concluded and reservations are zero. Only one cell in that new 57-cell
matrix is measured; this does not establish full-reference positivity or adoption.

CI on `68754bac` passed **1,668 tests and 18 subtests**, with 16 explicit skips,
Python 3.10/3.11 wheel smokes and release packaging:
[run 36665456793](https://github.com/255BITS/ParticleGAN/actions/runs/36665456793).

Six explicit published vector variants now use the shared discriminator
factories, with current MoG initialization and full-component gates retained.
Their two source declarations are included in snapshots independently of view
selection. Raw tasks and original views retain their previous architectures;
the variants are diagnostics in `host_profile_transfer`. Construction and
integration checks cover source tampering, snapshot portability, unchanged raw
RNG draws and dispatch into the public batch-distance discriminator before any
optimizer update. No training result is implied by these checks.

The published image resolver also materializes stripes, bars and blobs variants
through configuration alone. Three native parent profiles and their matching
continuation cards explicitly select identity affine G, Fourier-3 D, named
Xavier weights and uniform-square initial MoG locations. The additive public
initializer stages and validates writes; existing deterministic initialization
retains its behavior. Native checkpoint metadata binds the component policies
and measured initial tensors/streams. These are declared transfers, without
historical fixture parity. Full 7k/14k scientific gates remain unchanged; the
14k cards retain their clock-free prerequisite and are not included in the
scheduled transfer view.

Frozen-source queue and runtime checks now recompute host semantics, actual
recipe resolution, candidate revision, grouped job identity, resource locks
and continuation parent compatibility before dispatch. Existing frozen requests
retain their original contract. Further science must use the joined source;
software checks alone do not approve calibration.

The joined Forge and public initializer suite passed **703 tests**. A
[GPU 0 construction check](native-profile-cuda-init-check.json) found exact
CPU/CUDA equality for G, D, prior locations and initialization receipts,
unchanged training streams and sigma, and zero optimizer updates. Full CI for
the preceding vector-profile commit `a0f7499b` passed **1,709 tests and 18 subtests**,
with 16 explicit skips. The joined native source `93804cbc` also passed full CI:
**1,807 tests and 18 subtests**, 16 explicit skips, Python 3.10/3.11 wheel smokes
and release packaging ([run 36668104913](https://github.com/255BITS/ParticleGAN/actions/runs/36668104913)).

The [joined-source diagnostic readout](HOST_PROFILE_TRANSFER_READOUT.md) records
five measured cells for 156.284 paid seconds: bars4, intensity and unequal mass
pass; mode-hold and native grid fail. The native model acquired accuracy and then
lost shape quality. Mode-hold makes this one correctly rejected reference-negative
lineage, with 52/57 cells still unknown. Missing reference costs stay unavailable.
The [independent audit](HOST_PROFILE_INDEPENDENT_AUDIT.md) reproduced the first
three verdicts and verified the native artifacts/checkpoint. No positive reference
or calibration acceptance follows from this diagnostic pairing.

## Physical pilot completed

The [v3 pilot](PHYSICAL_GPU_PILOT_READOUT.md) passed physical two-GPU overlap,
fresh-clone deduplication, coordinator recovery with unchanged live workers,
cancellation and one repair. Both vector cells passed. All three attempts cost
45.000919218 seconds total, with zero final reservations and desktop processes
preserved. The corrected intensity diagnostic separately failed for 16.837 seconds.
Both exact-revision readouts are concluded; no scientific result was retried.

Calibration remains blocked. The historical positive intensity fixture uses a
different architecture, so host provenance needs review before expanding the
current matrix. Full reference quality and legacy cutover remain outstanding.

## First merged-source GPU screen

After [full CI passed](https://github.com/255BITS/ParticleGAN/actions/runs/36660755141)
on `3a06e09c` (1,582 tests and 18 subtests; 16 explicit skips), the unrelated
NPC process exited and GPU 0 became idle. The user-authorized GPU 0 screen then
completed **three scientific FAILs for 38.915 seconds**, with no execution errors,
zero reservations and a concluded readout. Duplicate submission from a fresh
checkout reused one request and all three results without additional launches.
See [the complete readout](DEVELOP_QUICK_SCREEN_READOUT.md).

The run exposed unavailable CUDA peak telemetry; the allocator probe ran before
CUDA initialization. The repair passed 30 focused tests and an actual GPU 0
allocation check (4,096 allocated / 2,097,152 reserved bytes; zero training
updates). The receipts remain unchanged. The adapter audit also found missing
clamping of noisy image training data and stale candidate sampling declarations.
Prospective fixes need their own source/task identities before further execution.
At that point the physical two-GPU pilot was unlaunched and all 16 independent
baseline references were unknown. Its later v3 execution is recorded above.

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
| Task definitions / tier views / independent graders | Implemented and tested | 45 tasks, five views including opt-in host-profile diagnostics; no active monotonicity gate |
| Queue, adapters, compiler, CLI and root guide | Real CPU execution and resource enforcement verified | Real process-group tests, atomic CPU/RAM reservations; final integration checks recorded below |
| Clock audit, paired adaptation and native continuation | Implemented; bounded state/protocol verification | Saved state manifests, measured optimizer counters, exact prefix/restore/RNG comparison; `CONTINUATION_REVIEW.md` |
| Administrative lifecycle and display filters | Implemented and tested | Immutable abandon/supersede receipts; explicit cancellation repair; family/provenance filters preserve full qualification |
| Automation telemetry | Implemented; historical missing measurements remain unavailable | Unique paid costs, avoided work/reuse/errors/concurrency; process RSS, CUDA allocator peaks and instrumented phase timing |
| Calibration diagnostics and robustness registration | Implemented; no production adoption claim | Registered diagnostic namespace, report-bound calibration, fixed promotion contract and forgery/retry tests |
| Phase D calibration / bounded multi-GPU pilot | Historical calibration replayed; physical v3 pilot passed | Current calibration remains blocked; full reference evidence still required |
| Root-guide onboarding | Fresh-checkout walkthrough complete | `FRESH_CHECKOUT.md`; penalty ablation rejected after 7.371 seconds, all remaining tasks unlaunched, concluded readout and zero reservations |
| Cutover | Pending accepted calibration and legacy reconciliation | Existing launchers retained; physical pilot passed |

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
- [x] Reserve available GPU capacity and record the real multi-GPU/cancellation/
  restart pilot: [v3 proof and readout](PHYSICAL_GPU_PILOT_READOUT.md).
- [x] Complete a fresh-checkout walkthrough; `FRESH_CHECKOUT.md` binds the exact
  source, commands, candidate, evidence and original readout.
- [x] Record all 12 legacy consumer dispositions and 22 trainer entrypoints in
  [the versioned index](legacy-consumers.json); add scoped guide pointers. The
  inspected local HyperGAN queue is finished. Uncovered and historical commands
  remain available; remote queue ownership stays unverified and unclaimed.
- [ ] Complete default adoption after accepted calibration. Any actual supported
  pending-request transfer still requires an owner handoff and reconciliation;
  no such local transfer is currently identified or enacted.

No legacy process has been stopped or adopted by Forge. No robustness seed
experiments have run; the registered promotion stage is conditional on a future
finished candidate and is not required merely to test the engine.

The [legacy-consumer inventory](LEGACY_CONSUMERS.md) maps repository entrypoints,
CI, external GPU owners and rollback requirements. Its dated original snapshot
had both GPUs occupied and three HyperGAN jobs pending. The newer observation
records that queue empty, GPU 0 available for bounded Forge work, and unrelated
training on GPU 1. No application ownership was transferred.

## Archived pre-develop validation checkpoints

These observations predate the develop integration and completed physical pilot.
The current evidence and remaining acceptance are stated at the top of this file.

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
