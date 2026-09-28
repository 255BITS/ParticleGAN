# ParticleGAN Forge: the experiment engine

**ParticleGAN Forge** turns new ideas into comparable evidence through cheap gates,
shared leaderboards, and a memory of what the project has already tried.

Date: 2026-09-28. Status: implementation plan. Inventory baseline: `92dc0319`.

## Goal and decisions

Make it easy for an agent to propose an idea, implement its changed mechanism,
run the cheapest useful tests, and compare its evidence with everything already
tried. Bad candidates should stop before consuming expensive training budgets.
Candidates that clear the gates should advance automatically within the declared
campaign budget. Every outcome should teach the next agent something.

The first deliverable in implementation is a classification of the existing
experiments and their evidence. Build the automation around that inventory,
reusing the current evaluators and their thresholds.

Decisions from this discussion:

- Three qualification tiers: **smoke, quality, endurance**. Higher tiers require
  lower-tier passes for the same candidate and goal.
- Multiple leaderboard views over shared evidence. Start with **discriminator
  stability**; monotonically decreasing metrics are a **future** view, with no
  monotonicity gate in the first release.
- A modeled experiment lifecycle, including useful negative results and
  incomplete work, rather than a collection of launch scripts.
- A durable queue that agents can submit to and workers can drain across multiple
  GPUs, with automatic result recording and centralized, tail-friendly logs.
- Repo files are the source of truth. No database, hosted service, or private
  agent memory is required to propose, execute, compare, or learn.
- One compiled, searchable experiment-memory file explains what worked, what
  failed, what remains unknown, and what to try next.
- No seed-only experiments. Preserve existing seed evidence without launching
  new seed sweeps. Prefer metrics and leaderboards over inspecting images.
- Refactoring and sweeping changes are allowed. Preserve existing evidence and
  reproducible protocols while moving new research onto the common interface.

## 1. What already exists

At this revision there are 181 tracked paths under `benchmarks/`, 100 under
`experiments/`, another 467 Python/shell files under `reports/`, 544 report
Markdown files, and 673 config files. These are **file counts, not independent
experiment counts**: many are shared helpers, frozen source copies, reports,
configs, or repeated descriptions of the same run.

Important foundations:

| Existing component | Reuse in the new system |
| --- | --- |
| [Grid runner](../experiments/run_grid.py), [runner contract](experiment-runner.md) | Config validation, immutable resolved requests, output locks, source/config fingerprints, completion manifests, preserved failed attempts |
| [Transfer protocol](../benchmarks/transfer_suite/protocol.py) | Independently reconstructed live-weight verdicts, complete observation schedules, sustained success, explicit missing evidence |
| [Transfer suite guide](../benchmarks/transfer_suite/README.md) | Required/ranking/diagnostic importance, solvability controls, development/reserved families, domain-balanced comparisons |
| [Toy100 coverage gate](../benchmarks/toy100/gate.py), [accuracy gate](../benchmarks/toy100/accuracy_gate.py) | Full-budget coverage and fidelity qualification, terminal sustained checks, saved-sample audits |
| [Continuous probe](../benchmarks/toy100/continuous_probe.py) | Uninterrupted hold, extended training, target-shift recovery, matched frozen controls |
| [Selected K3P declaration](../reports/toy100/current-research-base.json) | Exact formulation, complete mechanism/response/latent drivers, known passes and remaining failures |
| [Gap-fill evidence](../reports/toy100/gap-fill-20260925/README.md) | Candidate lineage, task-level evidence, source hashes, historical cross-candidate comparison |

The current transfer protocol already calls importance levels `tier`. The new
cost/progression field must be named `qualification_tier`; retain importance as
a separate dimension. An expensive diagnostic does not become a requirement
merely because it is assigned to Tier 3.

K3P and `b_cap` have substantial tracked evidence. This inventory did not find a
tracked R3P declaration by that name. Record it as unresolved until its exact
implementation/configuration is located; do not infer that another name is R3P.

### Initial classification

Classify **techniques**, **experiment declarations**, **benchmark tasks**,
**execution attempts**, **evaluators**, and **reports** separately. K3P is a
candidate formulation; `b_cap` is a technique/configuration axis; `two_pole` is a
task. None of these is interchangeable with an individual execution attempt.

| Family | Proposed role and tier suitability |
| --- | --- |
| Mechanism/configuration checks and deterministic initialization | Preflight and cheap Tier 1 checks; shared helpers are support code, not scored experiments |
| Short behavioral hosts: `two_pole`, `unused_token_hold`, `ae_gan_hold` | Initial Tier 1 screen using their existing complete short protocols |
| Remaining behavioral, vector, and procedural-image hosts | Tier 2 quality and stability coverage; intentionally weak or ambiguous diagnostic cases remain diagnostic |
| Native `grid100`, `rotated100`, `staggered100` | Tier 2 full coverage **and** accuracy at the established 7,000-update budget |
| Own-state hold, ring extension, longer training | Tier 3 endurance; detect late collapse after ordinary quality gates pass |
| Target-shift recovery and reserved transfer families | Tier 3 adaptation/transfer tasks; attach to views that require those capabilities |
| Learned LR, smart descent, architecture/regularizer/noise searches | Candidate-generation families; qualify their outputs through shared tasks instead of treating a whole sweep as one result |
| Paired transport, MoG, sparse/denoising studies | Additional objective families with their own quality metrics and calibrated tier profiles |
| CIFAR/image training, trajectory/transition, Gym/control studies | Domain-specific qualification tracks; production evidence where relevant, not universal blockers for every GAN idea |
| Plotting, rendering, analysis, launchers, archived drivers | Support/evidence roles; retain lineage and dependencies, do not count them as separate scientific trials |

Inventory all tracked experiment/config/report families, including executable
drivers under `reports/`. Record untracked/local result roots as discovery gaps,
without silently importing them or treating them as reproducible repo evidence.
Every scoped file must map to a family/role or an explicit `unclassified` entry.
The coverage audit must fail on newly introduced, unclassified scoped files.

## 2. The agent-facing workflow

Keep the authoring burden small. An agent supplies a hypothesis, parent/reference,
changed factor(s), implementation/config reference, and desired goal. The system
fills in identifiers, protocol defaults, source hashes, tasks, budgets, logs,
receipts, and leaderboard wiring. Do not require agents to copy a launch matrix
or implement a new evaluator for each idea.

Proposed commands below describe the intended interface; they do not exist yet:

```sh
# Read relevant prior successes, failures, and unresolved evidence first.
python -m experiments.forge recall --goal discriminator_stability --query "critic anchoring"

# Scaffold a declaration and standard candidate adapter from a known parent.
python -m experiments.forge new --id critic-anchor-v2 --parent k3p \
  --goal discriminator_stability

# Implement the idea, then show exact gates, reuse, estimated cost, and blockers.
python -m experiments.forge plan critic-anchor-v2

# Stop at the first blocking failure. Default to Tier 1 unless a campaign says otherwise.
python -m experiments.forge run critic-anchor-v2 --through-tier 1

# Submit without waiting; GPU workers execute only currently eligible work.
python -m experiments.forge enqueue critic-anchor-v2 --through-tier 3 --campaign campaign.json
python -m experiments.forge drain --gpus 0,1 --workers-per-gpu 1
python -m experiments.forge queue
python -m experiments.forge logs --follow

# Compare successes, failures, and partial candidates; rebuild shared knowledge.
python -m experiments.forge board --goal discriminator_stability
python -m experiments.forge compile
```

Before launch, show exact duplicates and closely related prior attempts. Reuse
compatible completed evidence automatically. A changed hypothesis string alone
must not cause identical training to run again. A substantive code/config change
creates a new candidate revision with explicit lineage. A controlled rerun for a
broken execution remains an attempt of that revision, with its reason recorded.

Agents may propose new views or tasks, but cannot weaken an active protocol to
turn their own failure into a pass. Such changes create a new protocol version
with a rationale and calibration evidence.

Use one candidate adapter contract for recipe/architecture changes and optional
optimizer, regularizer, latent, response, and checkpoint hooks. Task adapters own
the frozen host and evaluator. Scaffolding should inherit a complete parent
adapter, so an agent implements its changed mechanism once and participates in
all compatible tasks. Incompatible tasks receive an explicit applicability
decision; they are not silently skipped to improve a score. Bind each candidate
to an immutable source snapshot when enqueued, including any uncommitted changes
from its idea worktree, so later edits cannot change queued or running experiments.
Worktree isolation alone does not freeze executable source.

## 3. Repository records and lifecycle

Proposed layout, using the repo's existing code/config/report conventions:

```text
experiments/forge/                    # CLI, scheduler, adapters, map/reduce
configs/forge/
  catalog.json                            # task/family/source inventory
  protocols/<protocol-id>.json             # immutable versioned gates and budgets
  views/<view-id>.json                     # goal, task selection, ranking, eligibility
  campaigns/<campaign-id>.json             # resource limits and candidate selection
  ideas/<idea-id>.json                     # concise agent-authored declaration
reports/forge/
  records/<record-id>.json                 # normalized historical or new experiment card
  attempts/<attempt-id>/
    request.json                          # fully resolved immutable request
    events.jsonl                          # lifecycle and execution events
    result.json                           # terminal result, cost, evidence references
    evidence.json                         # hashes, runtime, evaluator receipt
  leaderboards/<view-id>.json              # generated data
  leaderboards/<view-id>.md                # generated reviewable view
  EXPERIMENT_MEMORY.md                     # single compiled agent reading file
  compilation.json                        # input hashes, coverage, conflicts, reducer version
runs/forge/                               # durable local operational state; ignored by Git
  queue/                                  # pending requests, claims, terminal receipts
  workers/                                # worker identity, GPU allocation, heartbeat
  claims/<compatibility-key>/              # prevent duplicate work across campaigns
  events.jsonl                            # centralized structured log
  status.json                             # current queue, workers, resource use
  <campaign-id>/
    progress.jsonl                        # campaign-filtered event stream
    <attempt-id>/run.log                   # complete child stdout/stderr
```

Declarations, normalized records, compact evidence, and generated readouts live
in Git. Bulk checkpoints, verbose logs, and large sample arrays stay in repo-local
ignored run directories by default; keep their paths and hashes in receipts.
Copy/adopt evidence needed for portable regrading into the durable report, or
explicitly mark the record as dependent on unavailable local artifacts. Do not
claim that a Git checkout can replay evidence it does not contain. No required
state should live in `/tmp`, an external agent session, or an undocumented path.

An experiment card records:

- Stable idea ID, exact candidate revision, parent, technique tags, hypothesis,
  changed factors, and links to related/duplicate ideas.
- View/protocol versions, task identity, fixed seed/fixture policy, full effective
  config, implementation/harness hashes, and training/evaluation budgets.
- Raw metrics and their meanings/directions, live versus EMA, complete versus
  partial windows, controls, artifact references, and provenance quality.
- Outcome per task and view; failure reason/first failing step where available;
  measured compute, elapsed time, and resources.
- Conclusion, limitations, and next action: advance, investigate, revise,
  deprioritize, or preserve as a useful negative result.

Separate administrative lifecycle from scientific qualification:

| Lifecycle state | Meaning |
| --- | --- |
| `proposed` | Idea exists; hypothesis and prior-art references can still change |
| `ready` | Candidate revision and protocol are resolved, validated, and frozen |
| `running` | One or more eligible tasks have active attempts |
| `awaiting_readout` | Execution stopped or exhausted its permitted work; conclusions need recording |
| `concluded` | Metrics, comparison, explanation, and recommendation are published; a failed idea can be fully concluded |
| `superseded` / `abandoned` | Administrative disposition with a reason and successor link where applicable; evidence remains visible |

An execution attempt separately reports `queued`, `running`, `completed`,
`error`, `timeout`, or `cancelled`. A gate reports `PASS`, `FAIL`, `INCOMPLETE`,
`INVALID`, `NOT_RUN`, or `BLOCKED`. Infrastructure failure is not scientific
evidence against the idea. Missing evidence never earns a pass.

Qualification is a derived field **per candidate revision, protocol, and view**:
highest fully passed tier, next eligible task, blocker, and evidence completeness.
A Tier 3 failure leaves its Tier 2 result visible; it does not rewrite history.
Passing a stability view need not pass an adaptation or future monotonicity view.

Append attempt events and write final artifacts atomically. Concurrent agents own
different idea/attempt files. Reserve work by candidate revision, task, and evidence
compatibility key, not just attempt ID: two requests for the same work should
attach to one execution. Use per-attempt locks and a single reducer for shared
outputs. Recovery after a crash inspects the saved request, process/lock state,
and completion receipt before resuming or creating a new attempt. Never silently
overwrite failed evidence or count two attempts as two independent ideas.

## 4. Tier gates: cheap rejection, then stronger evidence

Preflight is a zero-training validation step, not a fourth qualification tier.
It checks adapter availability, resolved config, resource limits, provenance,
duplicate work, and applicability to the chosen view.

| Tier | Question | Initial discriminator-stability profile | Progression |
| --- | --- | --- | --- |
| **1 — smoke** | Does the mechanism execute correctly and preserve basic bounded adversarial learning? | Existing `two_pole` (80 updates), `unused_token_hold` (200), `ae_gan_hold` (250), plus finite-state and intended-update checks | All required checks pass before any Tier 2 launch |
| **2 — quality** | Does useful live output quality survive across the declared problem families? | Complete the 19 transfer hosts and three 7,000-update native coverage/accuracy gates, reusing the three identical Tier 1 host results | All required checks pass before Tier 3 |
| **3 — endurance** | Does a qualified solution remain good after acquisition and extended training? | First-convergence ring hold for 1,200 updates and the following 300-update extension; add other long-budget hosts as their protocols are calibrated | Qualifies the measured endurance scope; production promotion is a separate decision |

This is an initial profile to calibrate, not a declaration that these smoke tests
already predict production success. Tier 1 contains only **530 host updates**,
using already short complete tests rather than shortening a full native run.
No wall-time or FLOP target is claimed until measured on the reference runtime.
If any proposed smoke criterion rejects an established useful reference, examine
whether it is detecting a relevant failure or an accidental implementation bias
before freezing the protocol.

The ring endurance task has a **maximum total budget of 7,500 updates**: 1,200
initial updates, up to 4,800 settling updates seeking the first 200 consecutive
qualifying live checks, then 1,200 hold updates and 300 extension updates. The
confirmation is inside the settling allowance. Reserve the full possible cost,
not just the 1,500 updates after convergence. Preserve the existing
[convergence declaration](../reports/toy100/gap-fill-20260925/sources/k3p/convergence_gate.py).

Keep target-shift recovery as a separately visible Tier 3 capability. An
adaptation view can require it, including the matched frozen negative control.
The first stability profile should not silently equate recovery after a changed
target with remaining stable on a stationary target.

Protocol rules:

- Freeze required tasks, thresholds, observation cadence, evaluation sample
  counts, budgets, schedules, and permitted early stops before a campaign.
- Recompute verdicts from complete recorded evidence using existing evaluators.
  A process exiting successfully or a trainer writing `PASS` is insufficient.
- Score live weights for live qualification. Preserve EMA and best-checkpoint
  metrics as separate diagnostics. A good earlier checkpoint cannot hide late
  collapse, missing modes, or poor final distribution fidelity.
- Preserve complete observation/sustained-pass requirements. For transfer hosts,
  the existing gate uses 24 observations and a terminal passing suffix; for native
  tasks, retain both coverage and accuracy evidence.
- Endurance must continue the candidate's own trained state, optimizer moments,
  random streams, and original schedule horizon. Do not restart or stretch
  annealing simply because the requested training duration increased. The current
  [`--steps` override changes schedules](toy100.md#run-every-problem-or-inspect-one),
  so it cannot serve as an exact-prefix screen.
- A frozen or nonlearning system must not qualify merely because its numbers are
  steady. Check actual intended optimizer updates and nondegenerate output quality.
- Verify candidate-specific hooks are installed and exercised by suitable smoke
  checks. A delayed guard or anchoring phase that never activates provides no
  evidence about that mechanism; declare expected inactivity on individual hosts
  and exercise it in a targeted cheap mechanism check when needed.
- Archived K3P drivers have module-global mechanism/EMA/LR/latent/response state
  beyond ordinary model and optimizer checkpoints. Run endurance uninterrupted
  until complete fresh-process state restoration is independently verified.
- Timeout, invalid evidence, or a failed **required** task blocks downstream
  qualification. Diagnostic errors/failures remain visible without vetoing required
  passes. Any process that exceeds its execution budget must still be stopped.
- Changing code, config, view requirements, or protocol starts a new identity;
  previously passing tiers carry over only when evidence compatibility is proven.

Tests should reject as early as the protocol permits. Within a tier, order tasks
by observed rejection value per measured cost, while preserving a frozen order
for each campaign. Do not early-stop on an unvalidated noisy signal. Cap tasks,
candidates, and campaigns independently so a faulty mechanism cannot spend an
unbounded budget before producing its first useful result.

## 5. Queue, GPU workers, and centralized logs

Submission and execution are separate. Agents should enqueue a resolved candidate
once and return to other work. A repo-local queue coordinator decides which tasks
are eligible and dispatches them to GPU workers. `run` is a convenient submit-and-
wait wrapper around the same path, not a second execution implementation.

The first implementation supports one machine with several GPUs and multiple
submitting agents/worktrees. Keep coordinator state in one configured repository
root; worktrees submit there and retain their own immutable source references.
Default to one worker per GPU. Permit explicit sharing only when memory/resource
estimates and campaign policy allow it. Multi-machine execution can later use the
same request/result contract; do not require a distributed service initially.
Enforce one active coordinator per queue; additional drain invocations attach to
it or fail clearly. Reserve GPU capacity and campaign budget atomically across
workers, counting outstanding reservations as well as completed spending.

Queue behavior:

1. **Enqueue:** validate the idea, freeze the request, assign an idempotency key,
   and persist it atomically. Repeated submissions return the existing request or
   reusable result. Record requested maximum tier, GPU/memory requirements,
   campaign, priority, and total resource allowance.
2. **Claim:** atomically reserve an eligible task and its candidate/task
   compatibility key across campaigns. Allocate GPU(s), memory headroom, CPU
   threads, and a worst-case task budget before starting a child. Pass device
   assignments explicitly, including `CUDA_VISIBLE_DEVICES` where appropriate.
3. **Execute:** spawn in the declared source environment, record PID/process
   group and worker heartbeat, stream logs, and persist incremental progress.
   A child receives only its own output directory and bounded execution allowance.
4. **Record:** atomically save raw result/evidence references, exit status, elapsed
   resource use, and independently graded verdict. Every exit path emits a terminal
   receipt, including errors, timeouts, cancellations, and unavailable evidence.
5. **Advance or stop:** update the reducer; enqueue the next eligible required
   task only if all prerequisites pass, the requested tier cap permits it, and
   the remaining campaign budget can cover it. A Tier 1 failure never reserves
   downstream Tier 2/3 GPU jobs.
6. **Drain:** exit when submitted work and its permitted automatic promotions are
   finished or explicitly blocked. An optional `--watch` mode waits for new ideas.
   Surface unschedulable GPU/memory requests as blockers instead of hanging silently.

Prioritize cheap screens while aging queued quality/endurance work so survivors
are not starved. Allow filtering/draining by campaign, goal, or priority. Expose
queue inspection, cancellation, bounded retry of execution failures, and campaign
pause/resume. Retry neither a scientific failure nor a seed-only variant as an
automatic recovery strategy.

When several requests share one execution, record its subscribers and cost owner.
Charge actual compute once under a declared accounting policy. Cancelling one
subscriber detaches its request; it must not kill work another subscriber still
needs. If the paying campaign withdraws, transfer the remaining reservation only
after another campaign accepts it, or stop the work. Shared jobs must not evade
resource limits through double counting or uncharged ownership changes.

Budget checks apply before **every** task, including after successful promotion.
`--through-tier 1` must stop after Tier 1 even when it passes. Timeouts and forced
cancellation terminate the entire child process group. A graceful drain shutdown
stops claiming new jobs and records the disposition of active work.

Heartbeats alone do not prove a training process died. Before reclaiming expired
work, check the process/lock state and terminate or fence the old worker; stale
workers cannot publish a second accepted completion. A restarted coordinator
reconstructs queue state from durable requests and receipts, reconciling active
workers before relaunch. Resume from a checkpoint only when the task adapter can
restore all required state; otherwise preserve the attempt and restart explicitly.

All workers emit structured records with timestamp, campaign, idea/revision,
attempt, worker, GPU, tier, task, event type, latest metrics, and cost. Workers own
separate streams; one collector writes `runs/forge/events.jsonl` and materializes
`status.json`, preventing interleaved JSON writes. Keep full child stdout/stderr
in stable per-attempt logs and make the central stream link to them. Provide
`logs --follow` filters for candidate, task, worker, and GPU, with a bounded replay
of recent events so a new agent can quickly see what is happening.

On each completion, update durable records and materialized leaderboard views.
Refresh the compiled memory after the experiment readout, or include an explicit
pending-readout entry in the interim. A user should be able to tail one file while
several agents feed several GPUs without assembling logs from each worktree.

## 6. Goal-specific leaderboards and useful compute

Store one candidate-by-task evidence matrix. Views select requirements and
ranking rules from it; creating another view must not trigger another complete
training sweep. Shared evidence is reusable only when task, candidate revision,
measurement protocol, budget, and runtime compatibility match.

Every board should show candidate/parent, exact revision, highest passed tier,
per-tier status and pass denominator, missing/blocked tasks, principal raw
metrics, first blocker, cost, and next action. Show failed and partial candidates
by default, with filters for tier, goal, family, outcome, and evidence quality.
Keep unmeasured candidates visible as unranked rows.

Compare only compatible cohorts. Full-quality candidates should not lose to a
short screen because the latter omitted difficult tests. Filter by qualified
tier first, then apply predeclared goal-specific metrics within the cohort.
Expose quality and cost separately; preserve existing domain-balanced ranking
where appropriate. Avoid a universal average that lets easy tasks cancel a
required failure or lets a large family outvote the others.

Initial views:

| View | State and interpretation |
| --- | --- |
| Discriminator stability | First implemented qualification profile: bounded learning, useful quality, and stationary endurance |
| Quality / coverage | Alternate readout of the same evidence; separates ordinary quality from endurance failures |
| Adaptation | Later profile requiring target-shift recovery and its controls |
| Monotonic progress | Future design only; later choose appropriate error/quality signals, tolerances, windows, and improvement requirements. Do not assume GAN losses should decrease monotonically |
| Production/domain readiness | Later application-specific views combining relevant toy evidence with actual target-workload tests |

Measure the automation itself: cost to reject, cost to qualify, rejection rate
by tier, tasks avoided after rejection, evidence reuse rate, and incomplete/error
rate. Measure actual updates, G/D forward/backward work where instrumented, peak
memory, device/runtime, elapsed time, and concurrency. Label FLOPs as measured,
estimated, or unavailable; **equal steps are not equal compute**, particularly for
extra-gradient, penalty, or EMA-critic work. Do not rank speed from runs with
incompatible hardware or contention.

## 7. Compiled experiment memory: map, then reduce

The main agent reading artifact is
`reports/forge/EXPERIMENT_MEMORY.md`. It should answer: what has been
tried, in what exact form, against which goals, what happened, and what a new idea
needs to change to be informative. Include successes, failures, mixed results,
and unfinished questions. A blacklist of technique names would discard too much.

### Map: normalize independently

Partition the historical inventory by report/experiment family. Subagents or
importers produce one normalized card per exact experiment revision with explicit
links to source evidence. They must not edit the shared compiled file directly.
Map workers can run in parallel because each owns different records.

Prefer structured configs, result snapshots, event streams, and evaluator outputs.
Use Markdown to recover hypotheses, explanations, and recommendations, citing the
specific source section. Mark narrative-only claims as such. Do not infer a pass
from positive prose, a failure from a missing file, or a scientific outcome from
the number of scripts in a directory.

Record source hashes and mapper/schema versions so unchanged sources need not be
mapped again. Preserve distinctions between negative controls and candidate
failures, and between a failed configuration and its broader technique family.
Expose conflicts, ambiguous identities, absent raw evidence, and unmapped sources
in the coverage report; do not resolve contradictions by majority vote.

### Reduce: compile and compare deterministically

Validate cards, resolve exact duplicates, retain attempt/lineage relationships,
and materialize per-view qualification and comparison tables. Evidence matching
uses hashes and configuration identity, not a short label such as `K3P`.
Conflicting evidence stays visible and blocks unsupported promotion.

Generate the single Markdown memory file with:

1. A compact index and goal/family summaries for quick agent orientation.
2. One concise entry per experiment revision: hypothesis/change, outcome and
   scope, strongest metrics, failure mode, limitations, next action, evidence links.
3. A section of unresolved ideas, contradictory reports, and missing evidence.
4. Compilation provenance and inventory coverage, so omissions are detectable.

Keep entries concise and link raw detail rather than copying whole reports.
The `recall` command selects relevant entries from the same records for a small
agent context; the complete single-file history remains available. Compilation
must be deterministic, incremental, and independent of training or image access.
Exclude generated outputs from mapper inputs to prevent recursive self-ingestion.

After each attempted idea, including early failures, the lifecycle requires a
short readout and recompilation. Agents should find closely related prior work
before execution and cite it in their declaration. A retry of a failed idea is
valid when it names a substantive changed factor and the failure it addresses.

## 8. Historical calibration examples

The existing evidence already demonstrates why separate tiers and views matter.
These are **historical results**, not newly earned passes under the proposed
protocol. Import them with their original scope before considering reuse.

| Exact historical formulation | Declared quality suite | Standard ring hold | Following extension | Target-shift recovery |
| --- | --- | --- | --- | --- |
| K3P, selected base | 22/22 | 1,200/1,200 pass | 300/300 pass | Fail: 28/81 deadline checks |
| K3G | 22/22 | 1,200/1,200 pass | Fail: 63/300 passing | Fail: 22/81 |
| P1, step-keyed K3 formulation | 22/22 | 1,200/1,200 pass | Fail: 63/300 passing | Fail: 22/81 |
| RG5 + `b_cap`, no A2, base floors | 18/22 | Not converged | Not reached | Fail: 0/81 |
| RG5 + A2 + `a_r1r2`, .1/.1 floors | Not qualified as a full 22-task configuration | Different shift-protocol hold | Not run | Paired pass: 81/81 live versus 0/81 frozen |

Sources: [gap-fill comparison](../reports/toy100/gap-fill-20260925/README.md),
[gate-by-gate evidence](../reports/toy100/gap-fill-20260925/qualification-summary.json),
[paired recovery audit](../reports/toy100/gap-fill-20260925/rg5-recovery-pair.json).
The quality total uses the declared runs; historical additional seed results stay
separate. No new seed experiment is needed to build this history.

The memory should explain these outcomes: K3G and P1 are examples of ordinary
quality passing while endurance fails; K3P separates stationary stability from
adaptation; RG5's recovery improvement belongs to its particular configuration.
Neither `b_cap` nor critic anchoring should receive a universal works/doesn't-work
label from these rows. The frozen control's failure is expected supporting
evidence for recovery, not another failed technique.

Use these and other archived positive/negative references to calibrate gate
selection. Estimate how often an early rejection would hide a later useful
candidate. Where historical evidence is insufficient, run a small, budgeted
calibration set with fixed declared fixtures, changing tasks or mechanisms rather
than seeds. A separate, explicitly budgeted **calibration lane** may run deeper
diagnostics on selected rejected candidates to detect an overaggressive screen.
This is the sole declared exception to normal prerequisite scheduling: those
results cannot bypass failed gates or retroactively qualify that candidate under
the same protocol. They can motivate a future protocol revision. Do not silently
turn every rejection into a full-suite run. A gate is a compute-saving heuristic,
not proof of production readiness or a reason to stop improving the test suite.

## 9. Implementation sequence and acceptance criteria

| Phase | Deliverable | Acceptance criteria |
| --- | --- | --- |
| **A. Inventory and history** | Machine-readable catalog, mapped experiment cards, initial compiled memory | Every scoped tracked source is classified or explicitly unresolved; known positive/negative examples retain exact identities and evidence; compilation needs no GPU |
| **B. Lifecycle and views** | Schemas, declaration scaffolding, reducer, read-only plan/board/recall | Separate attempt state from scientific verdict; show all candidates, blockers and next steps; future monotonicity remains inactive |
| **C. Queue and gated execution** | Adapters around existing runners, fixed smoke profile, durable queue, multi-GPU drain, centralized logs | Agents can enqueue concurrently; required Tier 1/2 failures block later tiers; duplicate submissions reserve one execution; results are recorded on every exit; crashes/source edits cannot certify stale results |
| **D. Calibrate and pilot** | Historical replay plus a small declared reference/challenger campaign | Known quality-versus-endurance failures remain visible; report rejection cost, false-rejection findings and reuse; no seed sweeps or image-based ranking |
| **E. Adopt and expand** | Agent instructions, migrations, additional domain views | One small idea declaration and implementation are enough to join comparisons; completed/failed attempts update memory; new tasks require protocol calibration |

Implementation should include meaningful tests for:

- Lower-tier failures, missing evidence, and infrastructure errors blocking higher
  tiers; diagnostic failures not doing so; partially qualified rows staying visible.
- Changed source/config/protocol invalidating reuse, exact duplicate detection,
  attempt locking, atomic writes, interrupted-run recovery, and budget enforcement.
- Concurrent submissions from separate worktrees sharing one queue, correct GPU
  assignments, memory-aware dispatch, lease recovery without duplicate execution,
  and complete centralized logs without interleaved/corrupted event records.
- Requested tier caps and campaign-budget exhaustion preventing new launches;
  timeout/cancel killing all child processes; seed-only proposals rejected before
  training; expected calibration exceptions remaining outside qualification.
- Regrading failures hidden by a best checkpoint, EMA, missing terminal checks,
  or a passing standard hold followed by a failing extension.
- Deterministic map/reduce, complete inventory coverage, conflict reporting,
  negative-control handling, and no accidental merging across configurations.
- Adding a new view over existing evidence without rerunning compatible tasks;
  a view with unknown requirements cannot invent passes or rank missing metrics.

Expose one stable campaign log and one per-attempt log, all unbuffered, with
timestamps, candidate, tier, task, metric/verdict, elapsed cost, and next action:

```sh
tail -F runs/forge/events.jsonl
tail -F runs/forge/<campaign-id>/progress.jsonl
tail -F runs/forge/<campaign-id>/<attempt-id>/run.log
```

Verify scheduling first with serial fake workers, then require bounded multi-GPU
draining in Phase C. Never dispatch a candidate's expensive downstream tiers
speculatively. Parallelize historical mapping and candidate implementation with
disjoint files, then reduce centrally.

The first usable milestone is **inventory + memory + read-only goal views**.
The next is **one command that tries an idea through cheap gates and publishes a
useful readout even when it fails**. Broader production tracks can then use the
same lifecycle and evidence without forcing every new idea through every study
the repository has ever accumulated.
