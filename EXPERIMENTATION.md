# Experimentation with ParticleGAN Forge — read before running ideas

Read this before proposing or running an experiment. Forge records ideas,
checks cheap prerequisites before spending on larger tests, and compares exact
candidate revisions through goal-specific leaderboards.

Start with the [compiled experiment memory](reports/forge/EXPERIMENT_MEMORY.md)
and the [discriminator stability board](reports/forge/leaderboards/discriminator_stability.md).
The [implementation plan](docs/better-experiment-automation-plan-2026-09-28.md)
defines the migration and adoption criteria.

## Readiness and scope

The initial smoke profile is **provisional**. The
[initial historical replay](reports/forge/calibration/initial.md) pairs 6 of 10
lineages and falsely accepts 1 of 3 independently failing lineages. The
[alternate screen](reports/forge/calibration/quick-discriminator.md) pairs 8 of 10
and falsely accepts 1 of 5. Both remain **adoption BLOCKED** under the
[frozen criteria](configs/forge/calibration/criteria-v1.json); these retrospective
fractions describe the saved subset, not population error rates. The [first current-cohort smoke batch](reports/forge/CURRENT_SMOKE_READOUT.md)
measured all nine cheap cells, including three learned-MoG AE passes. Independent
quality references are still missing, so current calibration remains blocked. Task declarations describe the
intended coverage; actual execution also requires a compatible public-API
adapter. Unsupported capabilities remain `BLOCKED` in the required denominator.
A file existing in the catalog does not mean its adapter has been qualified.

Historical cards preserve successes, failures, raw errors, negative controls,
and missing evidence. They do not automatically qualify a new Forge candidate.
See [import gaps](reports/forge/import-gaps.json), especially the later dt075
14k, EMA .995, and centre-sensitivity receipts not located in the pinned #155
snapshot. Narrative summaries are explicitly distinct from normalized results.
The [later source audit](reports/forge/calibration/followup-source-audit.json)
preserves the corrected prior-EMA-relaxation result separately: three diagnostic
host passes do not replace the canonical staggered100 failure or bind those
missing claims.

The bounded CPU pilot exercised the real queue, worker, grader, durable receipts,
and readout: `two_pole` failed after about 5.26 seconds, so the remaining smoke
tasks did not launch. Its exact frozen revision remains visible as a pinned row.
That demonstrates fail-fast execution; it does not validate the gate profile.

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
