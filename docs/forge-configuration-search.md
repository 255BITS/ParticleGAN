# Trainer families and bounded configuration search

The [HyperGAN/Hyperchamber search audit](../reports/forge/hypergan-search-audit/README.md)
compares the historical dictionary selector with this machinery and checks
the optimizer/loss representation gaps in a preserved Halloween config.
Its recommendations are implemented by the
[finite search compiler and public role/legacy settings](forge-search-spaces.md).
The audit's original evidence remains pinned to its inspected base.

The [current technique leaderboard](../reports/forge/technique-inventory.md)
publishes one complete selected configuration per formulation family. The
generic [current family selection](../configs/forge/selections/family-current-v1.json)
pins one exact ordinary evidence row; other source and runtime cohorts remain
unranked alternatives. A configuration changes settings on the shared public trainer; it does
not introduce another training loop or a new technique row. The companion JSON
retains alternative configurations and the exact evidence behind their results.

The proposed [convergence-selection plan](forge-convergence-selection-plan.md)
asks whether each solution family can represent the required toys, then illustrates
bounded hyperparameter search for one shared defaults config, gate progression,
and selection of the quickest stable solution from comparable family finalists.
Its convergence timing and speed objective are future additions; the implemented
search behavior is described below.

The [BCAP search options report](../reports/forge/bcap-search-options/README.md)
flattens every public Recipe field and inventories optimizer/loss categories,
conditional numerical settings, current evidence and read-only compiler checks.

[`trainer-families.json`](../configs/forge/trainer-families.json) declares family
membership and a canonical fallback. Families identify formulations;
each new search also records its stricter public technique signature. Historical
ablation families retain their identities. A structural control change within a
formulation can be an ordinary idea in that family; a changed optimizer/loss
formulation requires a new family. The older matched R1/R2 penalty swap and the
Modern GAN recipe belong to the same R1/R2 family, but their historical source
cohorts remain separate evidence. The canonical fallback is an explicit choice,
not a claim that incomparable historical configurations have been ranked.
The current whole-row pin takes precedence over `active_search_by_backend`.
That setting still selects registered studies when no current pin applies and
when reconstructing archived policies. Completing a later study preserves earlier
trials; changing a selection is a reviewable declaration change.

An ordinary idea is a reusable global recipe, rather than a solution owned by
one task. Evaluate it through the unchanged full view and prerequisite gates.
It can become the configured family standard only after the complete required
Tier 1 denominator passes. The pin validates every scientific row field,
including unmeasured cells; it cannot combine outcomes across candidates or
import direct-task diagnostics. A failed replacement remains an unranked family
alternative while the recorded incumbent stays selected. Neither that retention
nor a new standard ranks incompatible sources or changes public defaults.
Calibration and independent confirmation still govern default adoption.

A `current_measurement` pin can identify an explicitly requested experimental
starting recipe after all required tasks in its declared measurement views have
PASS/FAIL outcomes. It retains the complete row, failed gates and previous
selection; it supplies no configured-standard or default-qualification claim.
The [BCAP optimizer readout](../reports/forge/dualnorm-tier1/README.md) records
such a tied starting choice separately from its original search tie-breaks.

For a new family with only unexecuted declarations, the registry may specify
`unmeasured_display_backend` and `unmeasured_display_reason`. The publisher
shows its sole canonical row on that backend and keeps every alternative.
This option rejects measured outcomes, costs, qualification and active searches;
measured families use ordinary whole-row selection.

## Declare a search

New search declarations use `schema_version: 2` and supply a finished study
`hypothesis`. They select the `protocol` by name; `protocol_hash` is generated
in the plan/report and must not be authored. The finite grid, tuning scope,
selection objective and campaign budgets belong to the search study. Derived
schema-v3 configuration cards contain only reusable training definitions and
capability/claim requirements: they inherit no embedded decision contract,
study hypothesis, budgets or first-study/report binding. Each frozen request
records its `search_plan` selection. The same content-addressed configuration
can participate in multiple registered searches. Bare configuration admission
still requires the exact finite search registration. Original schema-v1 specs,
cards, saved reports and hashes retain their compatibility path unchanged.

For example, use the existing fields shown below with `schema_version: 2`,
add `hypothesis`, and omit `protocol_hash`. All existing whitelist, mechanism,
active-axis, full-reservation, gate and deterministic whole-configuration
selection checks apply. This declaration change does not tune or launch a model.

A search declares a base recipe, a finite grid of public `Recipe` settings, a
fixed protocol/view/runtime, a tuning tier cap and immutable campaign budgets.
Coupled settings such as two endpoints of a penalty schedule belong in one
dimension. A list of grid objects declares their finite union instead of a
Cartesian product. This can refresh existing configurations whose original
override sets differ without adding redundant overrides or changing their
content identities. Duplicate choices and more than 256 configurations are
rejected. Trials inherit the same architecture, prior, initialization, named
random streams, task horizons and sampling law. Seeds and task definitions are
not search parameters.

The [field registry](../experiments/forge/boundaries.py) declares ownership and
the conservative finite search whitelist. Search resolves the complete base
and trial recipes and rejects a change that enables or disables a mechanism:
zero penalty coefficients at either schedule endpoint, optional cosine
schedules, optimizer moment activation, AMSGrad, prior regularization, and zero
terminal learning rates are structural boundaries. Positive strengths and timing
can vary within the same signature. A structural ablation needs its own idea;
a family label cannot turn it into a hyperparameter trial.
Positive LR floors, including a constant floor of one, remain settings within
the same always-declared schedule path.
Ordinary planning of a configuration card applies the same parent technique,
fixed-law and whitelist checks, even when its content hash is internally valid.

Search also rejects an axis that is inactive or task-owned on every declared
tuning task. For example, a coefficient annealing fraction has no effect when
its schedule is absent, and fixed R1/R2 does not consume `reg_kappa`. Behavioral
hosts own their auxiliary `prior_reg` objective; scalar public trainers retain
the candidate's value. A mixed study can tune that scalar value only when one
of its tuning tasks actually consumes it. Task bindings use the same pure
resolver as execution, without constructing or training models.

The existing `direct_particle_betas` pair can be searched when at least one
tuning task actually constructs a formulation optimizer for direct generated
coordinates, such as `two_pole`. It does not control sampled latent locations,
word networks or plain Adam. Positive moment strengths can vary; enabling or
removing either moment requires a structural idea. The ordinary pair validation,
fixed task laws and whole-configuration selection rules still apply. This exposes
an existing public optimizer setting without changing its update rule.

Technique signatures are additional provenance in newly planned studies.
Existing configuration hashes, saved studies, qualification results and
historical family/cohort distinctions retain their original identity.

`optimizer_smoothing` exposes fixed-scale DualNorm smoothing with a zero public
default. Enabling or disabling it crosses a structural technique boundary;
positive scales can vary within the explicitly enabled base. The
[smoothing guide](dualnorm-smoothing.md) supplies a reusable structural card,
an unexecuted CUDA Tier 1 search example, and the separate categorical comparison
path for unsmoothed versus smoothed controls. Default recipes, current family
selections and archived qualification remain unchanged.

The initial [R1/R2 search](../configs/forge/searches/r1r2-modern-toy-v1.json)
tests four substantive toy-host adaptations of the existing Modern GAN recipe:
learning rates 0.00425 and 0.0085, paired with penalty schedules 1 to 0.1 and
0.1 to 0.01. All retain plain Adam, beta1 zero, beta2 0.9 to 0.99, and the
unchanged cosine burn-in fraction of 0.2. The original paper-derived learning
rate 0.0002 has already failed the 80-update movement screen; its receipt is
preserved without an unchanged repeat.

These configurations test whether transport within the frozen short toy horizon
improves at a host-scale learning rate or weaker penalty. They do not reproduce
the paper's image architecture, Gaussian latent prior, image budget or EMA
evaluation. The [previous readout](../reports/forge/R3GAN_BASELINE_READOUT.md)
explains the original failure and scope.

## Plan, execute and inspect

Run from the repository root in the project environment:

```sh
python -m experiments.forge search plan r1r2-modern-toy-v1
python -m experiments.forge search enqueue r1r2-modern-toy-v1
python -m experiments.forge search run r1r2-modern-toy-v1
python -m experiments.forge search report r1r2-modern-toy-v1
python -m experiments.forge logs --follow --campaign r1r2-modern-toy-search-v1
```

Planning launches no training. Enqueue freezes the configuration declarations
and source before submission; run drains the existing Forge queue for the
bounded campaign. Identical scientific requests reuse compatible evidence.
Changing a declaration under an existing study or configuration identity is
rejected. All configurations in a comparison must bind the same frozen source,
tasks, protocol, runtime and serving law. Original receipts remain qualification
inputs; compact search reports and leaderboard publications are display records.

The four-trial campaign reserves at most 3,600 seconds, with 900 seconds per
configuration. Its three existing smoke tasks each have a 300-second allowance.
The original frozen requests stop a configuration after its first failed task.
New ordinary search requests freeze `complete_current_tier`: all independent
runnable jobs in that tier finish before a required non-PASS blocks higher tiers.
Per-task dependencies, complete reservations and the study's frozen budgets
still apply; changing scheduling policy does not combine configuration evidence.
Actual paid time, failed attempts, remaining unknown tasks and reuse are recorded.
Raw logs, checkpoints and per-update traces remain under ignored `runs/` or in
an artifact archive; the compact report and provenance receipts are committed.

The [K3P five-task search](../reports/forge/k3p-global-tier1-v2/README.md)
is a worked example starting from the global clean/full word-study successor.
Its [declaration](../configs/forge/searches/k3p-global-tier1-v2.json) tests twelve
complete recipes by varying positive critic coefficients and G/D rates. It
holds the nominal latent-prior rate fixed and retains all five current Tier 1
requirements. A historical word pass motivates the base; it cannot fill a new
trial's word cell. Its separately bounded follow-up restores an existing input
noise control as an explicit structural base, then searches eight numerical
configurations within that base's signature. Both bases belong to K3P because
their optimizer and loss formulation stay the same. The first required failure
stops that trial, leaving later cells UNKNOWN. A 5/5 result is needed before
replacing the current family pin.

The [continuation readout](../reports/forge/k3p-global-tier1-v3/README.md)
adds a bounded eight-recipe rate search and four-recipe direct-moment search.
Saved outputs distinguish fitting ring cores from fitting the full served tails.
An analytic direct-Adam displacement bound motivates tuning its existing beta2,
without promising actual movement. The continuation retains the same five-task
ladder, complete global recipes and incumbent selection when no recipe qualifies.

For a new v2 search, review and bind each derived card's decision contract to
its actual tuning scope before enqueueing. An inherited full-view contract
must be rebound when the study only authorizes Tier 1. The example's
[preparation script](../reports/forge/k3p-global-tier1-v2/prepare.py) materializes
cards, checks exact bindings and budgets, and records READY plans without
training. Preparation refuses to change a registered or admitted study.
Use the ordinary read-only search plan/report commands after registration.
The [execution wrapper](../reports/forge/k3p-global-tier1-v2/run.py) freezes the
reviewed commit, uses standard queue admission and grading, and defers global
report generation until publication. The coordinator allows one GPU0 worker
and one automatic CPU worker; declare that two-worker maximum explicitly.

## Select one configuration

The declared objective ranks complete required tuning-task PASS counts, with
a deterministic configuration-content identity as the tie-breaker. All tier
cells in a family row come from that same configuration. There is no per-task
or per-tier mixing of configurations, and no wall-time ranking.

A configuration is screen qualified only when every required tuning task
passes. When all trials fail, the selected display incumbent is labelled
best observed; there is no qualified winner. A tie does not establish that one
configuration is scientifically better.

This initial search uses the three Tier 1 tasks only. The 19 quality tasks and
two endurance tasks remain untouched during selection. A screen-qualified
configuration can subsequently be frozen and evaluated through the ordinary
later-tier prerequisites, reusing its compatible smoke evidence. Changing the
selected recipe after inspecting those tasks would require a new selection
protocol; those observations would no longer be independent confirmation.

The initial study also preregisters a conditional
[confirmation campaign](../configs/forge/campaigns/r1r2-modern-toy-confirmation-v1.json).
Only its one deterministic 3/3 smoke winner can enter; a study without a qualified
winner stops. The campaign retains the existing full-task 44,100-second
reservation ceiling, reuses smoke evidence and stops at the first ordinary
prerequisite failure. Tuning and confirmation costs are reported separately.

The completed [R1/R2 readout](../reports/forge/R1R2_CONFIGURATION_SEARCH_READOUT.md)
records a 3/3 smoke winner at LR 0.0085 and gamma 1 to 0.1. Its ordinary
trajectory quality check failed, so it remains Tier 1 and later requirements
remain unmeasured. The readout links the compact confirmation cost receipt and
the current leaderboard.

### Advance every smoke survivor in a new full-view study

A fresh study with `tuning_through_tier: 3` submits every grid configuration
through the existing ordinary gate ladder. Each configuration advances
independently: all of its smoke requirements must pass before quality, and all
quality requirements must pass before endurance. A failed configuration stops;
the others keep advancing. This requires no one-smoke-winner confirmation step.
Freeze the complete per-configuration and campaign ceilings before enqueueing;
retain every required task, including unknown and blocked later requirements.
Tasks used to select the defaults are tuning evidence, not independent confirmation.

The additive `progression` report lists all smoke survivors and all full-view
qualified configurations. Its outcome distinguishes `full_winner`,
`tuning_only_winner`, `best_observed` and `pending`. The existing `selection`
dictionary retains its archived PASS-count/hash objective unchanged. A full
winner is therefore a provisional whole-config gate winner, rather than a
measured fastest config or permission to change defaults. Incomplete comparisons
cannot name a final full-view winner; alternatives and full denominators remain.

`speed_selection.status` is currently `UNAVAILABLE`. Some frozen evaluators
record terminal-suffix times, while others lack per-observation seconds or bind
coverage separately from joint accuracy. Reports preserve available original
evaluator fields as `evaluator_timing`, without converting them to first
acquisition speed. Total paid wall time remains cost evidence. Hardware
contention, clock scope and a complete comparable timing contract must be
resolved before any fastest-convergence claim.

Use the same explicit coordinator queue for all families; enqueue launches no
worker. The coordinator owns GPU admission and the existing memory/resource
limits, with one worker per GPU unless an intentional sharing policy is declared:

```sh
python -m experiments.forge --queue-root /path/to/round/queue search plan NEW_STUDY_ID
python -m experiments.forge --queue-root /path/to/round/queue search enqueue NEW_STUDY_ID
python -m experiments.forge --queue-root /path/to/round/queue drain --gpus 0,1 --workers-per-gpu 1
python -m experiments.forge --queue-root /path/to/round/queue search report NEW_STUDY_ID
```

The current screening profile remains provisional. Neither this search nor a
passing smoke screen establishes calibrated ranking or public-default adoption.
Calibration and the separately registered robustness stage retain their existing
requirements. Random search and successive halving are future extensions; the
first implementation uses a finite grid with fully declared resource ceilings.

## Publish the current leaderboard

After archiving the original execution receipts and committing the frozen
reproduction declarations, independently regrade the executed source:

```sh
python reports/forge/regenerate_technique_inventory.py --source-commit EXECUTED_COMMIT --device cpu
python reports/forge/regenerate_technique_inventory.py --device all
```

Ordinary regeneration reads committed evidence and changes the same current
report. Frozen numerical snapshots preserve history without generating more
leaderboard tables. New trainer families and configuration studies can use the
same declarations, queue, selection reducer and publication command.

When the required view policy advances, register the new measured source with
`--advance-policy --source-commit EXECUTED_COMMIT`. This explicitly archives
the previous policy and its numerical cohorts, preserving their original tier
denominators. Only evidence for the current policy enters the selected rows.
Subsequent regeneration uses the normal command without `--advance-policy`.
The [all-existing-config Tier 1 refresh](../reports/forge/tier1-refresh/README.md)
uses five new study registrations for the same 32 immutable configurations and
15 existing ideas under revision 3. It refreshes selection projections without
changing the scientific configuration cards.
