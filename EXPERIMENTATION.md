# Experimentation with ParticleGAN Forge — read before running ideas

Read this before proposing or running an experiment. Forge records ideas,
checks cheap prerequisites before spending on larger tests, and compares exact
candidate revisions through goal-specific leaderboards.

Start with the [compiled experiment memory](reports/forge/EXPERIMENT_MEMORY.md)
and the [discriminator stability board](reports/forge/leaderboards/discriminator_stability.md).
The [implementation plan](docs/better-experiment-automation-plan-2026-09-28.md)
defines the migration and adoption criteria.

## Readiness and scope

The engine is implemented. [Full CI](https://github.com/255BITS/ParticleGAN/actions/runs/36674097253)
passed 1,807 tests and 18 subtests; the [physical two-GPU pilot](reports/forge/PHYSICAL_GPU_PILOT_READOUT.md)
and [fresh-checkout walkthrough](reports/forge/FRESH_CHECKOUT.md) verified the
queue, recovery, cancellation, reuse, logs and readout workflow. Use the public
API and learned MoG defaults; particle-cloud exceptions must be explicit in the
task. Do not run seed-only experiments.

**Default adoption remains blocked by scientific calibration.** The
[current control readout](reports/forge/CONTROL_MODE_READOUT.md) records 7/57
measured cells, 50 unknown, one true rejection and no complete positive reference.
All three declared lineages fail smoke, so this exact profile cannot meet both
the frozen positive-reference minimum and zero false rejections. Stop filling it
solely for adoption. A new justified screen/profile and its bounded calibration
are required; the controls' independent references and false-reject rate remain
unknown. [Migration status](reports/forge/MIGRATION.md) retains the earlier
studies, metrics, costs, source cohorts and outstanding acceptance work.

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
sensitivity assertion is separately preserved.

Incoming E22/KA2 changes the public API and model of record. Its inspected
positives use particle clouds and noisy state-selected serving. Follow the
[pinned compatibility checklist](reports/forge/UPSTREAM_E22_COMPATIBILITY.md)
when that API lands in develop, preserving MoG support, named streams, external
budgets and complete policy checkpoints. Current receipts retain their frozen
source and formulation identity.

Run commands from the repository root in the project's Python environment.
The history, recall, compile, validate, and board commands launch no training.

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

Boards order current rows by attained tier, then name/cohort. Raw metrics and
cost remain separate; no aggregate metric ranking is declared. Filter whole rows
without shrinking their qualification denominator:

```sh
python -m experiments.forge board --family image --evidence-quality imported_recorded
python -m experiments.forge board --scope pinned --evidence-quality certified_pinned
```

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

```sh
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
