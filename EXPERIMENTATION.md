# Experimentation with ParticleGAN Forge — read before running ideas

Read this before proposing or running an experiment. Forge records ideas,
checks cheap prerequisites before spending on larger tests, and compares exact
candidate revisions through goal-specific leaderboards.

Start with the [compiled experiment memory](reports/forge/EXPERIMENT_MEMORY.md)
and the [current technique leaderboard](reports/forge/technique-inventory.md).
The [implementation plan](docs/better-experiment-automation-plan-2026-09-28.md)
defines the migration and adoption criteria.
For a new research question or host, follow the
[experiment creation guide](docs/forge-new-experiment.md), with worked ring and
joint BiGAN examples, scorer controls, registration and artifact publication.

For the combined E22/Atlas/Forge API, read
[the develop integration notes](reports/forge/DEVELOP_INTEGRATION.md). Presets,
external budgets and policy serving remain explicit; current Forge task
definitions block E22/Atlas policy cohorts before reservation until separate
policy-aware tasks are frozen. Keep bulk execution logs outside Git as required
by [`AGENTS.md`](AGENTS.md).

## Readiness and scope

**The framework is ready to use.** The
[final readiness audit](reports/forge/FRAMEWORK_READINESS_AUDIT.md) verifies the
requested engineering scope. [Full CI](https://github.com/255BITS/ParticleGAN/actions/runs/36677816874)
passed 1,807 tests and 18 subtests; the [physical two-GPU pilot](reports/forge/PHYSICAL_GPU_PILOT_READOUT.md)
and [fresh-checkout walkthrough](reports/forge/FRESH_CHECKOUT.md) verified the
queue, recovery, cancellation, reuse, logs and readout workflow. Use the public
API and learned MoG defaults; particle-cloud exceptions must be explicit in the
task. Do not run seed-only experiments.

**Screening profiles remain provisional; scientific default adoption requires
calibration.** Framework delivery does not require finding a winning formulation.
The [current control readout](reports/forge/CONTROL_MODE_READOUT.md) records 7/57
measured cells, 50 unknown, one true rejection and no complete positive reference.
All three declared lineages fail smoke, so this exact profile cannot meet both
the frozen positive-reference minimum and zero false rejections. Stop filling it
solely for adoption. A new justified screen/profile and its bounded calibration
are required; the controls' independent references and false-reject rate remain
unknown. [Migration status](reports/forge/MIGRATION.md) retains the earlier
studies, metrics, costs, source cohorts and future research/adoption work.

The [2026-09-30 bounded reference study](reports/forge/SCIENTIFIC_CALIBRATION_20260930_READOUT.md)
adds one substantive training-output-noise-removal lineage while preserving all
three host-profile smoke gates, 16 independent references and frozen criteria.
Its sole full 7k native diagnostic failed for 93.747 new paid seconds. Shape
improved but terminal quality and sustained coverage failed. The new profile has
8/76 measured cells and is also infeasible: every lineage is smoke-negative or
reference-negative. Stop filling both failed profiles; no robustness or promotion
run is eligible. Original evidence and the concluded failed revision remain in
compiled memory.

Task declarations describe intended coverage. Preflight and execution check the
public API, formulation, host and frozen source; unsupported capabilities remain
`BLOCKED` in the required denominator. Ordinary prerequisite failures stop later
tiers. Selected deeper comparisons require the separately registered calibration
lane described below and confer no ordinary qualification.

Historical cards preserve successes, failures, raw errors, negative controls and
missing evidence under their original protocols. [Initial](reports/forge/calibration/initial.md)
and [alternate](reports/forge/calibration/quick-discriminator.md) historical
screens also remain blocked under the [frozen criteria](configs/forge/calibration/criteria-v1.json).
Historical cloud, noisy served-model or EMA passes cannot fill a current MoG/clean-live
cell. The [newer #155 archive](reports/forge/supplemental/pr155-current-archive-v1/README.md)
adds 31 native arms and six context families, preserving noisy/clean scoring,
failed ideas, host refusals and OOM attempts in the compiled memory. The
[newer #155 audit](reports/forge/calibration/pr155-current-followup-source-audit.md)
still leaves the exact dt075 14k, EMA .995 D-tracking and centre-sensitivity
experimental receipts [unbound](reports/forge/import-gaps.json); a later written
sensitivity assertion is separately preserved. That research is still developing;
import its receipts when available using the existing framework.

Incoming E22/KA2 changes the public API and model of record. Its inspected
positives use particle clouds and noisy state-selected serving. Follow the
[pinned compatibility checklist](reports/forge/UPSTREAM_E22_COMPATIBILITY.md)
when that API lands in develop, preserving MoG support, named streams, external
budgets and complete policy checkpoints. Current receipts retain their frozen
source and formulation identity.

Run commands from the repository root in the project's Python environment.
The [registered formulation comparison](reports/forge/studies/FORMULATION_COMPARISON_V1.md)
compares fixed R1/R2, BCap and full released v0.7 GAN v3 with K3P on declared
native hosts. `Recipe.reg_arm` exposes the two fixed L2/autograd penalties through
the shared public trainer. Paired noisy artifacts retain the required clean/live
grade and grant no extra reference credit. Optional calibration `diagnostic_tasks`
bind separate host identities and report their costs without entering smoke or
reference decisions. Existing profiles without this field retain their semantics.

The [completed formulation comparison](reports/forge/FORMULATION_COMPARISON_READOUT.md)
records all five full 7k native runs as FAIL for 408.765 new paid seconds.
Independent audits pass, and the K3P control exactly reproduces the previous
clean-MoG trajectory and named training streams. BCap improves centering but
contracts further; R1/R2 improves shape but loses centering/coverage. Released
GAN v3's matched MoG adaptation improves shape but still fails the full gates;
its separate named-cloud host also fails and supplies no MoG reference credit.
The profile has 4/95 ordinary calibration cells measured (91 unknown), plus
1/5 separate diagnostic cells (4 unknown); acceptance remains blocked.
Stop these exact experimental revisions. Inspect saved critic gradients and
per-mode moments before another supported, bounded training hypothesis. No
automatic matrix filling, tuning, seed study, continuation or promotion follows.

The history, recall, compile, validate, and board commands launch no training.
Recall includes source-bound concluded configuration and policy publications as
non-qualifying summaries. [Research memory freshness](docs/forge-research-memory.md)
is visible in plans; `forge compile --check` checks it without artifact hydration.

```sh
python -m experiments.forge validate
python -m experiments.forge recall --goal discriminator_stability --query "critic anchor"
python -m experiments.forge board --goal discriminator_stability
```

## Add one idea

Use an existing declaration as a parent. The scaffold inherits its formulation
settings and supplies the standard fields; edit the hypothesis, changed factors,
and actual mechanism/configuration before submission.

```sh
python -m experiments.forge new --id critic-anchor-v2 --parent k3p \
  --goal discriminator_stability \
  --hypothesis "A stronger critic anchor preserves terminal quality during hold"
```

New scaffolds use idea schema v2 and a draft [hypothesis-to-decision contract](docs/forge-decision-contract.md).
Bind original evidence identities, review the actual task-owned delta, freeze one
bounded round and numerical prediction/falsifier, then mark the contract ready.
Planning shows the bindings; unfinished drafts block admission before spend.
Saved v1 evidence retains its original identity.

Edit `configs/forge/ideas/critic-anchor-v2.json`. Describe the substantive
change in `changed_factors`, cite relevant prior work, and choose
`mechanism_class`: `structural`, `floor_constant`, or `sampling_only_patch`.
Keep floor/constant tuning visible separately from structural changes. A
sampling-only improvement must describe its training and public sampling laws.

Use `recipe_overrides` for existing package settings. Register new reusable
capabilities through the [shared formulation API](experiments/forge/api.py);
declare its extension and required capabilities on the idea. Avoid copying
another candidate's launcher or training loop. Missing host support must produce
an explicit blocker rather than silently dropping the mechanism.

New ordinary tasks use learned MoG priors: locations learn, while the declared
width and uniform mixture weights remain fixed by default. Particle-cloud hosts
must explicitly declare `kind: "particle_cloud"`, `sigma: 0`, and an exception
reason. Historical sigma-zero receipts retain their original identity.

Every task must define `execution.prior` with `kind`, `sigma`, `standardize`,
and `learnable`; it cannot inherit a prior from the candidate, preset or API.
`kind: "mog"` selects the public `MoGParticlePrior` code path, while
`kind: "particle_cloud"` selects `ParticlePrior`. A zero-width MoG and a particle
cloud can describe the same distribution but retain separate implementation
identities. Ordinary MoG tasks still require positive sigma. Particle clouds also
require `standardize: false` and their explicit exception reason. Parameter-only
controls declare `prior_applicability: "not_sampled"` in `execution`.

The [field ownership contract](docs/forge-field-boundaries.md) separates task
conditions, technique mechanisms, hyperparameters and the comparison protocol.
Each task also declares `execution.initializer`; fixed host fixtures and native
component policies remain its more specific initialization rules. An explicit
candidate initializer is a compatibility requirement and cannot replace the
task's policy. Native tasks declare their model and training resources, while
profiled and behavioral hosts retain their frozen host definitions and sources.

Inspect effective values and their owners before submission:

```sh
python -m experiments.forge plan k3p --device cpu --show-boundaries
```

Planning and adapters use the same task binding. Execution receipts record each
field's owner and source; inactive historical host optimizer settings remain
labelled provenance. Configuration searches preserve the base technique's
mechanisms and reject axes that are inactive or delegated on every tuning task.
Changing a mechanism requires a structural idea. Family labels describe lineage;
task priors and sampling laws remain distinct scientific comparison cohorts.
These declaration changes create new execution identities. Saved receipts and
qualification outcomes retain their original bindings and are not regraded.

Do not create seed-only ideas. Screening uses one fixed protocol seed with named,
isolated streams. Changing an initializer, sampling law, or stream binding changes
the evidence identity. A future public-default promotion requires the plan's
separately registered robustness stage after the candidate is frozen; a passing
screen is not a public-default promotion.

## Inspect cost, then submit

```sh
python -m experiments.forge plan critic-anchor-v2 --through-tier 1
python -m experiments.forge enqueue critic-anchor-v2 --through-tier 1 \
  --campaign configs/forge/campaigns/smoke.json
python -m experiments.forge queue
```

`plan` shows resolved tasks, task groups, reuse, blockers, and reserved worst-case
wall time. `enqueue` freezes the source request and returns without starting
training. An identical scientific request attaches to existing work or compatible
evidence. Changing only prose does not justify rerunning the same experiment.

The default smoke campaign caps campaign and candidate reservations at 900
seconds. Task timeouts live in each task's `resources.timeout_seconds`.
Larger campaigns need their own JSON definition and explicit budgets. The worker
reserves a complete task allowance before starting it; a remaining budget too
small for that reservation blocks another launch.

```sh
# One bounded worker per GPU; execution is a separate, explicit action.
python -m experiments.forge drain --gpus 0,1 --workers-per-gpu 1

# Or submit and wait for this campaign on CPU for an applicable smoke task.
python -m experiments.forge run critic-anchor-v2 --through-tier 1 --gpus cpu
```

Use `--device cpu` or `--device cuda` when planning/enqueuing; `run` infers the
backend from `--gpus`. CUDA model, CPU model, and backend participate in evidence
identity. Boards show distinct runtime cohorts, never a pooled CPU/CUDA pass.
The compact board's Compute column names the requested cohort. CPU-only behavioral
tasks still run on CPU in a CUDA-targeted request; use each attempt's device and
cost receipt when comparing execution time.

Use `--allow-sharing` only with an intentional resource policy when increasing
workers per GPU. Training time under different hardware or contention is not a
speed leaderboard. FLOPs remain labelled unavailable until measured or estimated
with a declared method.

Inspect measured automation costs with `python -m experiments.forge stats`.
Compilation also writes `reports/forge/automation.json`: paid attempts, rejection
and qualification costs, avoided requested work, reuse, execution errors and
observed concurrency. Missing timing and memory measurements stay unavailable.
Instrumented adapters separate optimizer updates, sampling and evaluation; process
RSS and PyTorch allocator peaks are scoped measurements, not whole-device usage.

Worktrees share the main Git repository's `runs/forge` queue by default. For an
explicit coordinator location, put global options **before** the subcommand:

```sh
python -m experiments.forge --root /path/to/worktree \
  --queue-root /path/to/repository/runs/forge enqueue critic-anchor-v2
```

Use the same queue root for submitters, workers, and log viewers.
`PARTICLEGAN_FORGE_QUEUE` also sets the shared queue location. Declarations and
durable report receipts belong to the checkout selected with `--root`.

After the develop integration, scalar, image, native and adaptation scoring uses
the public prior distribution without training output noise. Learned MoG kernel
noise remains present. Behavioral hosts keep their explicitly frozen laws.
Vector tasks record their retained full-component gates; upstream finite-atom
shape exemptions have not been calibrated for Forge's MoG prior. These changes
produce new evidence identities and do not relabel previous receipts.

New idea cards set `claim_contract.sampling_law` to `task_declared`. Each task's
`evaluation` declares `sampling_contract_version: 1`, `sampling_law`, and
`eval_output_noise`; the adapter records the policy it actually executes.
Planning rejects unsupported host/policy combinations. Missing observed policy
is `INCOMPLETE`; a contradiction is `INVALID`, before scientific metrics are
graded. Behavioral scheduled-noise and parameter-only measurements retain their
explicit exceptions. Archived unversioned receipts keep their frozen semantics.

Host architecture is part of a task's evidence identity. For example,
`img_intensity2` retains its original transpose12 declaration;
`img_intensity2_residual16` explicitly binds the published residual16 profile and
its source hash. Both use the same image factory, public trainer and gates.
The opt-in `host_profile_transfer` view exposes the variant as a diagnostic;
the default views do not automatically add it. `imageprofiles.task_from_profile`
materializes the full task card from the shared resolver without runtime
inheritance. Preflight rejects unsupported profiles or mismatched model/data
declarations. Receipts record the resolved card, actual parameter shapes/counts
and initial model-state hashes. Architecture transfer does not import an archived
positive or relax the fixed named-RNG comparison policy.
The same resolver materializes the three other `img_*_residual16` diagnostics.

The same diagnostic view includes six `vector_*_published` tasks. These bind
the published critic cards through shared factories, including the public
`BatchDistanceDiscriminator`. `vectorprofiles.task_from_profile` materializes
architecture changes while preserving the original task's data, resources,
learned MoG, named initialization and full-component gates. Both source
declarations travel with frozen jobs, including the report containing the
critic cards. The profile explicitly records that historical prior scale and
draw order are not restored. Raw tasks keep their existing architectures.

Native `*_affine_square_named_v1` cards select the identity affine generator,
Fourier-3 critic and explicit uniform-square initial locations through
[`particlegan.init.initialize_`](docs/api.md#initialize_module--method-parameter_generatorsnone-distributionsnone-gain10-stricttrue).
They retain learned MoG width/masses, current recipes, named streams and the full
7k coverage/accuracy gates. Task-owned component policies, actual tensor hashes
and per-parameter stream states enter receipts and checkpoint compatibility.
An incompatible prior, supplied fixture or conflicting initialization claim is
`BLOCKED`. The catalog's matching 14k variants retain the clock-free audit and
own-state checkpoint prerequisites; select them only in a view containing those
requirements. They are not added to the scheduled transfer view automatically.

Queue submission and runtime revalidate profiles against frozen source, including
actual recipe resolution, candidate identity, grouped tasks, budgets and matching
continuation parents. Cached preflight text cannot authorize a changed host.
Older snapshots retain their recorded execution contract.

## Tiers and views

| Qualification tier | Initial stability profile | Purpose |
| --- | --- | --- |
| 1: smoke | `two_pole`, `unused_token_hold`, `ae_gan_hold`; 530 total host updates | Cheap behavior, finite-state, intended-update and mechanism checks |
| 2: quality | Complete 19 transfer hosts plus three 7,000-update native coverage/accuracy gates | Require useful sustained live quality across families |
| 3: endurance | Own-state ring hold and extension; reserve up to 7,500 total updates | Detect late failure after acquisition |

All required lower-tier tasks must pass before higher-tier work is eligible.
Failed, missing, invalid, and blocked evidence stop downstream spending.
Diagnostics remain visible without vetoing required passes. The hold and extension
share one uninterrupted execution; the extension cannot borrow another
candidate's trained state or hide behind a passing ordinary hold.

The [experiments by tier report](reports/forge/EXPERIMENTS_BY_TIER.md) lists the
current task assignments for every view, including required, ranking and
diagnostic roles, declared budgets, dependencies and tasks unassigned to any
view. Regenerate it after adding tasks or changing tier placement:

```sh
python -m experiments.forge experiments-by-tier \
  --output reports/forge/EXPERIMENTS_BY_TIER.md
# Inspect one view or print a machine-readable inventory without writing.
python -m experiments.forge experiments-by-tier --view discriminator_stability
python -m experiments.forge experiments-by-tier --json
```

The report reads validated task and view declarations plus compact published
research indexes without accessing the queue, launching training or regrading
saved evidence. Its prior column names each task's code path and absolute sigma;
recorded outcomes and related API demos show their own saved priors, with missing
bindings labelled unrecorded. Each task links to its question, declared numerical gates,
recorded Forge configuration outcomes and related public-API training GIFs.
Related demos keep their own recipe, prior, initialization, budget, sampling
and runtime; they confer no qualification on a different Forge task.

Regenerate this same report after changing tasks/views or publishing results
and media indexes. New declarations and explicit retained-question mappings
are discovered automatically, and the report freshness test catches stale
content. A task's optional `description` supplies its current question; otherwise
the guide uses the retained question audit or the declared gate. Task variants
name their host/problem explicitly, and public-API variants retain `legacy_ids`
for the related-question join. The [current solution leaderboard](reports/forge/technique-inventory.md)
remains the single ranking for its goal; use its source-bound evidence and
the compiled memory to explain a release choice after full qualification.

Boards order current rows by attained tier, then name/cohort. Raw metrics and
cost remain separate; no aggregate metric ranking is declared. Filter whole rows
without shrinking their qualification denominator:

```sh
python -m experiments.forge board --family image --evidence-quality imported_recorded
python -m experiments.forge board --scope pinned --evidence-quality certified_pinned
```

The [explicit task-recipe adaptation guide](docs/forge-task-recipe-adaptation.md)
describes the released v0.7 successor's opt-in resource/objective binding. Its
original native-host card remains frozen; adaptation never silently discards
training mechanisms or imports qualification.

For a technique inventory with one passes/total column per tier, use the
[current technique leaderboard](reports/forge/technique-inventory.md).
This is the single published technique inventory, updated in place. It includes
the [configurable Modern GAN training baseline](reports/forge/R3GAN_BASELINE_READOUT.md)
and selects one complete configuration for each declared trainer family and
comparable runtime cohort. Alternative configurations retain their recorded
results and source identities in the companion JSON and evidence snapshots.
The [released v0.7 task-adaptation readout](reports/forge/RELEASE07_TASK_ADAPTATION_READOUT.md)
adds one measured successor with all 24 integration preflights ready. Its
required smoke failure stops the remaining ordinary tasks; the original native
recipe cards and their blockers remain frozen.
Numerical source snapshots and receipt proofs remain in
`reports/forge/technique-evidence/` as provenance, without separate leaderboard
tables. Regeneration launches no training and needs no raw-log hydration;
independent regrading of a source needs its byte-exact original receipts.
The roster is
discovered from `configs/forge/ideas/*.json` and
`configs/forge/configurations/*.json`. The explicit family registry keeps
hyperparameter trials within an existing technique row. R1/R2, BCap, K3P and the other public formulations retain
their actual resolved recipes and separate source/runtime/sampling cohorts.
The ordinary `inventory` execution command still submits declared ideas;
configuration grids run only through an explicitly requested `search` study.

Use the [configuration-search workflow](docs/forge-configuration-search.md) to
declare a bounded recipe grid without copying a trainer. A study freezes its
tuning tasks, objective, deterministic tie-break and campaign budgets before
execution. The leaderboard uses all tier cells from one selected configuration;
it never combines each task's best result from different settings. A search with
no complete smoke pass has no qualified winner. Later tiers remain untouched
during selection, and the provisional screen still requires calibration before
scientific ranking or default adoption.

```sh
# Read-only cost/coverage plan, then explicit gated execution.
python -m experiments.forge inventory plan --through-tier 3
python -m experiments.forge inventory run --through-tier 3 --gpus 0,1
# Regenerate the one current leaderboard from committed evidence.
python reports/forge/regenerate_technique_inventory.py
# After a new experiment, regrade/register its source and update the same file.
python reports/forge/regenerate_technique_inventory.py --source-commit <executed-commit>
python -m experiments.forge logs --follow --campaign technique-inventory-v1
```

The original default inventory campaign has explicit reservation ceilings for
its 12-technique roster. New techniques require checking the expanded plan and
a new immutable campaign ID with adequate budgets. The Modern GAN recipe uses
its own one-candidate campaign; it does not rerun unchanged techniques. Ordinary
failures stop later tasks, including remaining tasks in that tier; unsupported
techniques reserve no training resources. Required denominators remain 3/19/2
for the current `discriminator_stability` view, including unknown and blocked
cells. A zero passes/total cell alone does not establish a scientific failure.
Archived and calibration results appear separately and cannot fill current
qualification cells. Provisional screening still confers no default adoption.
`techniques` also renders a read-only local board or JSON. The publication wrapper
exports final metrics and receipt hashes without copying per-update diagnostics
into Git. Full new execution envelopes remain ignored and archived unchanged;
summary projections are never qualification inputs. A checkout can regenerate
the same current leaderboard from committed evidence without raw-log hydration.
For an independent regrade, hydrate exact originals using the
[archive manifest and restoration instructions](reports/forge/TECHNIQUE_INVENTORY_READOUT.md).
`--source-commit <executed-commit>` reconstructs and verifies the recorded
implementation, registers its measured evidence and updates the same leaderboard;
its results do not qualify a newer checkout. The raw `techniques --output` export
cannot overwrite this registered publication. Memory compilation links this goal
to the current inventory instead of generating a second table. Do not rerun unchanged science solely for a
merge or a reporting change. Inventory runs compile boards once after draining;
attempt receipts and tail-able events remain available throughout execution.

Receipt certification describes evidence identity, not a scientific pass.
Use `board --json` to inspect family/provenance options and unknowns.

`discriminator_stability`, `quality_coverage`, `adaptation`, and
`clockfree_continuous` select different requirements from shared task evidence.
Clock-free eligibility requires its explicit clock/state audits and continuation;
a constant LR or scheduled hold does not establish the claim. Forge measures
step-label, horizon, evaluation-cadence and restart perturbations on saved public
trainer states. Its source audit also blocks known delayed releases that a short
probe can miss. Native 14k tasks restore the candidate's certified 7k checkpoint,
keeping the original schedule and named RNG streams. Paired adaptation compares
an active continuation with a frozen copy of the same pre-shift state.
These adapters have bounded software verification; no full-budget current
clock-free or adaptation qualification has been recorded. Monotonicity is a
future view and has no active gate.

Change tier placement through view policy, preserving task/evaluator definitions:

```sh
python -m experiments.forge retier discriminator_stability \
  --task img_bars4 --tier 1 --reason "Recorded calibration supports earlier rejection"
```

This creates a view revision and recomputes eligibility without launching jobs.
Existing campaigns retain their pinned view. Changing the task's steps, prior,
sampling, or thresholds instead changes scientific compatibility. Do not shorten
`benchmarks.toy100 --steps` to claim a preserved prefix: that option also changes
annealing schedules.

## Watch and recover work

```sh
python -m experiments.forge logs --follow --candidate critic-anchor-v2
python -m experiments.forge logs --follow --task two_pole
python -m experiments.forge queue
```

`queue` reports the resolved queue location. Its `events.jsonl` is the central
timestamped stream; each campaign also has `progress.jsonl`, and each attempt has
`run.log`. For the default queue in the primary checkout:

```sh
tail -F runs/forge/events.jsonl
tail -F runs/forge/<campaign-id>/progress.jsonl
tail -F runs/forge/<campaign-id>/<attempt-id>/run.log
```

```sh
python -m experiments.forge pause <campaign-id>
python -m experiments.forge resume <campaign-id>
python -m experiments.forge cancel <request-id>
python -m experiments.forge retry <compatibility-key> --reason "Repaired the worker environment"
```

After cancellation, explicitly enqueue the same frozen request/campaign again
to reattach its subscription, then retry its cancelled compatibility key if needed.
The cancelled receipt and cost remain visible.

Retry is for a repaired execution problem; it does not erase a scientific failure.
The retry receipt binds its predecessor and repair reason. A certified repaired
attempt replaces the infrastructure verdict for qualification while both attempts
remain in the history, readout and cost. Unlinked conflicting results stay invalid.
Preserve original errors and known parity refusals. A candidate with an unsupported
capability needs adapter/parity work before another attempt. Process leases and
completion receipts protect against duplicate work during coordinator recovery.

## Finish with a comparison and recommendation

Every attempted idea needs a readout, including cheap failures:

```sh
python -m experiments.forge board --goal discriminator_stability
python -m experiments.forge readout critic-anchor-v2 \
  --conclusion "State the measured result and its limits" \
  --comparison "Compare the same task/protocol and scoring weights with the parent" \
  --next-action "Advance, revise a specific mechanism, investigate a blocker, or stop"
python -m experiments.forge compile
```

To stop pursuing a finished idea, preserve its readout and record the reason:

```sh
python -m experiments.forge abandon critic-anchor-v2@REVISION --reason "Recorded failure; no further work planned"
# Or link a genuinely revised idea:
python -m experiments.forge supersede critic-anchor-v2@REVISION \
  --successor critic-anchor-v3 --reason "The successor changes the failed mechanism"
```

Stop active work and publish its readout before either command. These immutable
administrative records preserve verdicts and costs; the disposed revision cannot
be submitted or retried again. Use an exact revision prefix when the name is
ambiguous. A successor link does not transfer qualification.

If several revisions of the name have run, use `candidate-id@revision` to bind the
readout. Finish or cancel active queued/running/paused work first. A readout covers
its exact durable attempt set; a later retry requires an updated readout. A
concluded failed idea remains useful history. Live, EMA, clean/noisy,
and calibrated-sampling outcomes remain separate; an earlier good checkpoint
cannot replace a failed required terminal window.

Compilation writes one searchable [experiment memory](reports/forge/EXPERIMENT_MEMORY.md),
per-view Markdown/JSON boards, and [input hashes and coverage](reports/forge/compilation.json).
Rows show exact identities, status/denominator, cost, blockers, and next actions.
Historical summaries can overlap individual attempt evidence; never add their
counts or costs as though they were independent experiments.
Pinned older revisions retain recorded verdicts and costs in the board. A changed
live evaluator does not regrade or promote them.

## Maintain history and coverage

The committed memory is readable immediately. Rebuilding historical imports also
needs their pinned Git objects. If the importer reports a missing revision in a
new clone, fetch the exact #155 source before rebuilding:

```sh
git fetch origin 0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b
```

Keep import gaps visible if a frozen source cannot be fetched; never replace it
with the current branch by name.

```sh
python -m experiments.forge history
python -m experiments.forge history --check
python -m experiments.forge compile
```

History import reads the pinned initial repository and #155 trees without
executing them. It classifies every scoped tracked file, including drivers under
`reports/`, and preserves package/fixture/configuration provenance on normalized
cards. New tracked sources must appear in the catalog; unknown families require
an explicit classification. Generated Forge records and boards never become
historical mapper inputs. Keep bulk checkpoints/samples in their declared durable
location; unavailable local artifacts remain evidence gaps.

## Calibrate a screen before adopting it

`calibrate` replays saved evidence without training. Follow the
[current-cohort protocol](reports/forge/CURRENT_CALIBRATION_PROTOCOL.md) to freeze
the controls, independent reference, task identities, cost criteria and prior.
Check the [read-only feasibility preflight](docs/forge-calibration-preflight.md)
before selecting paid diagnostics. New calibration-lane registrations reject
mathematically infeasible profiles and unresolved original receipt issues;
logical feasibility alone supplies no adoption or qualification credit.

```sh
python -m experiments.forge calibration-preflight --profile <current-profile> --require-feasible
python -m experiments.forge calibrate --profile initial
python -m experiments.forge calibration-lane register --contract path/to/diagnostic-contract.json
python -m experiments.forge calibration-lane plan <registration-id>
python -m experiments.forge calibration-lane enqueue <registration-id>
python -m experiments.forge drain --gpus 0,1 --campaign calibration-<registration-id>
python -m experiments.forge board --scope calibration_diagnostic
```

Registration selects explicit lineages and tasks with a reason and separate task,
candidate and campaign budgets. Only those registered diagnostics can proceed
after a failed screen. Checkpoint dependencies still require a valid passing
producer. Diagnostic jobs use a separate queue namespace: their outcomes cannot
qualify ordinary candidates. Record a readout after this work as usual.
Start from the [diagnostic contract template](configs/forge/calibration-lane-template.json)
and replace its placeholders with the frozen profile and selected lineage.

An accepted calibration is a verified report with frozen criteria and compatible
source/task/prior/RNG/runtime evidence. Editing a status field cannot approve it.
Changing the required smoke set needs a corresponding calibration before adoption.

For a separately declared screen, explicitly import already-measured compatible
diagnostics before freezing its profile:

```sh
python -m experiments.forge calibration-lane imports <original-registration-id> \
  --lineage <original-lineage-id> --tasks <task-a> <task-b>
```

This prints bindings for the new profile's `diagnostic_imports` field. It writes
nothing and launches nothing. The bindings preserve original receipt hashes,
candidate revision, task keys, costs and retry history. Calibration verifies the
original registration and unchanged scientific cohort and criteria. Unlisted
foreign diagnostics supply no credit; imported diagnostics never qualify an
ordinary candidate. A documentation-only commit preserves a registration when
its scientific bytes and execution policy are unchanged.

## Register a finished candidate's robustness stage

After all required gates, accepted calibration and a concluded readout, use the
[promotion contract](configs/forge/promotion-template.json) to preregister the
candidate, reference controls, seed set, scoring and complete budget.

```sh
python -m experiments.forge promotion register <candidate> --contract path/to/promotion.json
python -m experiments.forge promotion plan <registration-id>
python -m experiments.forge promotion enqueue <registration-id>
python -m experiments.forge promotion report <registration-id>
```

This is the sole fixed-seed screening exception. Every registered outcome counts;
there is no in-stage tuning, seed selection or live-to-EMA scoring switch.
Registration and submission revalidate the finished candidate and calibration.
