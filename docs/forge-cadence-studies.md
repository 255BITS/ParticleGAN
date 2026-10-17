# Fixed-strength critic-penalty cadence studies

`forge cadence` compares one reusable recipe at several critic-penalty intervals
through existing Forge tasks. It uses the configuration-search machinery and
public trainers; it does not create a separate training loop. A generated report
compares whole recipes, numerical task results, training cost and observed
penalty activity. The report supplies no public-default promotion.

Read [EXPERIMENTATION.md](../EXPERIMENTATION.md), the
[compiled memory](../reports/forge/EXPERIMENT_MEMORY.md), and the
[ownership glossary](../RESEARCH_GLOSSARY.md) before declaring a study.

## Declare and inspect

Put a schema-version-1 manifest in `configs/forge/cadence-studies/ID.json`.
It names `base_candidate`, `trainer_family`, `view`, `through_tier`,
`execution_backend`, `protocol`, `intervals`, `control_interval`, a finished
`hypothesis`, and a finite `campaign`. CUDA comparisons also name `cuda_model`.
For example, the BISPF comparison uses intervals `[1, 4, 8, 16]` and control `1`.
The optional `lanes` list selects `ordinary`, `coverage`, or both; it defaults to
`["ordinary"]`. Coverage requires an explicit `coverage_view`.

The wrapper generates schema-version-2 searches with only
`grid.reg_every = intervals`. The base penalty coefficient and any coefficient
schedule endpoints remain fixed. Public penalty kernels already multiply an
applied penalty by its interval, so multiplying `reg_coeff` by the interval as
well would double-compensate. Extra recipe axes are rejected. Another technique
can use this workflow when its existing public penalty supports `reg_every`;
choose that technique's explicit base candidate and family.

Every comparison uses protocol seed 0 and the public deterministic initializer.
Task architecture, target/data law, seen batches, initial prior distribution and
capacity, sampling, update budget and scoring cadence remain fixed. The recipe
retains its prior update policy and prior regularizer. Constructor, data,
training-noise and evaluation streams stay isolated and checkpointed. Fixed
identity/zero fixtures keep their explicitly declared initialization cohorts.

```sh
python -m experiments.forge cadence plan bispf-lazy-cadence-v1
python -m experiments.forge cadence enqueue bispf-lazy-cadence-v1
python -m experiments.forge cadence run bispf-lazy-cadence-v1 --gpus 0,1
python -m experiments.forge cadence report bispf-lazy-cadence-v1 \
  --output reports/forge/cadence-studies/bispf-lazy-cadence-v1.md
```

`plan` is read-only. `enqueue` freezes declarations and source but starts no
training. `run` drains the campaign; it does not launch other experiments.
Use the study ID or its manifest path. A new source, task binding or interval
roster requires a new study identity after registration. An archived result from
a different source remains historical evidence, including the earlier BISPF
[throughput readout](../reports/forge/bispf-throughput/README.md).

## Ordinary gates and deeper diagnostic coverage

Ordinary requests complete every independent runnable job in the current tier,
then apply required failures to higher tiers. Held tasks remain measured and
visible; they do not block qualification. An unattempted task is unknown.
Checkpoint and explicit gate dependencies retain their original requirements.

An optional `coverage_view` names a separately authored `research_diagnostic`
view. It must mirror the source view's selected task identities, with every
assignment explicitly diagnostic at execution tier 1. The report retains the
original source tiers and required/held roles as display metadata, separately
from diagnostic execution and qualification. This scope allows independent
deeper measurements when ordinary tier gates would stop progress. It cannot turn
a failed parent into passing
checkpoint evidence: a dependent continuation stays blocked without its required
own parent state and gate. Diagnostic results never fill ordinary qualification.

For the first BISPF accuracy study, use `lanes: ["coverage"]` to measure
independent tasks across the original tiers under one diagnostic scope. This
avoids rerunning the same task set in two evidence namespaces. All results remain
diagnostic, even when their numerical gates pass. A later ordinary comparison of
a frozen selected whole recipe is separate authorized work with its own budget.

Coverage and ordinary jobs have separate evidence namespaces. The campaign must
cover their additive full reservations, including diagnostic parent executions.
Do not assume cross-scope reuse or combine same-named task receipts silently.
Compatible reuse within a scope remains governed by Forge's exact scientific
bindings. An infrastructure retry retains the earlier receipt and cost; a
numerical failure does not authorize a retry.

For discriminator-stability revision 12, each arm reserves 2,520 seconds in
Tier 1 (six required tests plus the clock audit diagnostic), 40,500 in Tier 2
(14 required and seven held), and 3,600 for the single Tier 3 ring execution
group. `ring_hold` and `ring_extension` share that group and are charged once.
The ordinary ceiling is **46,620 seconds per arm**. A full mirrored coverage
scope independently costs **46,620 seconds per arm**. The recommended four-arm
coverage-only study therefore needs **46,620 seconds per candidate and 186,480
seconds total (51.8 hours)**. Explicitly requesting both lanes needs **93,240
seconds per candidate and 372,960 seconds total (103.6 hours)**. These are
conservative independent reservation ceilings, not expected runtime.
The plan derives costs from declarations; a view change requires recalculation.

Revision 12 counts **6/14/1** required tasks across its tiers. Seven held Tier 2
tasks, the removed ring-hold phase, and the clock diagnostic remain visible.
Creating a cadence study does not change those assignments or denominators.

## Compare and conclude

The preregistered question is whether interval 4, 8 or 16 preserves the control's
numerical accuracy while reducing penalty work. A control PASS becoming a FAIL
is a regression. Record the failing bounds and sustained observations; do not
replace them with a selected earlier checkpoint. A FAIL becoming a PASS is a
repair under this fixed comparison. Unknown, blocked, invalid and incomplete
cells remain separate from measured failures.

Use task-specific metrics: ring/image quality and covered modes, mixture mass
error and covariance, Gaussian location/width/CDF error, word quality and inverse
accuracy, and behavioral correctness. Compare their raw matched deltas alongside
required pass counts and held results. There is no shared accuracy percentage
across these different questions. Report one complete interval recipe across
the suite rather than selecting each task's best interval.

Any separately declared speed measurement must time complete lazy cycles; for
these intervals use, for example, 32 warmup and 256 measured updates. A ten-update
sample can miss every interval-16 penalty. Compare serial runs on the same
physical device and label unavailable contention telemetry. This workflow does
not infer speed from worker cost or create a timing benchmark automatically.
Measure synchronized public training-update time separately from evaluation,
media and startup. Pair latency with penalty application counts and measured
activity; a skipped penalty evaluation need not skip other BISPF work. In BISPF,
penalty-call counters also govern internal state transitions, so a larger interval
changes those transitions' timing. Preserve that declared behavior in the
interpretation; average coefficient compensation alone does not make the trained
trajectories equivalent.

`run` and `report` write compact Markdown and its JSON companion to
`reports/forge/cadence-studies/ID.md` by default. `--output` selects an export
location; existing files must be a matching generated report pair. `plan` writes
only when `--output` is explicitly supplied. `--json` includes full per-task
details on stdout.
Retain one current generated comparison publication for the goal and link its
detail/provenance from the existing technique inventory rather than creating a
second competing leaderboard. Include a conclusion, matched control comparison,
limitations and a recommendation to retain, investigate, or stop each interval.
Passing provisional gates is separate from calibration and default adoption.

## Logs and artifacts

```sh
python -m experiments.forge logs --follow --campaign bispf-lazy-cadence-v1
tail -F runs/forge/events.jsonl
```

Use the queue path printed by the plan when it differs from the default.
Campaign `progress.jsonl` and each attempt's `run.log` provide narrower tails.
Keep stdout, per-update JSONL, checkpoints, tensor dumps and test logs outside
Git. Publish compact final metrics, provenance receipts and reproduction sources.
Actual-training GIFs must show target and observed behavior at fixed axes and
queries, with update numbers and numerical gate status. Render them from saved
scheduled observations without adding training or changing scoring cadence, in
accordance with the [public-API toy contract](../reports/toy_audit/api_contract/README.md).

The initial manifest and software checks launch no paid study runs. The generated
report must keep unmeasured results unknown until actual compatible receipts
exist.
