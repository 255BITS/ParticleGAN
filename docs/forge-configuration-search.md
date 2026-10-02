# Trainer families and bounded configuration search

The [current technique leaderboard](../reports/forge/technique-inventory.md)
publishes one complete selected configuration per trainer family and runtime
cohort. A configuration changes settings on the shared public trainer; it does
not introduce another training loop or a new technique row. The companion JSON
retains alternative configurations and the exact evidence behind their results.

The proposed [convergence-selection plan](forge-convergence-selection-plan.md)
illustrates advancing candidates through the gates, retaining failures, and
choosing the quickest stable converging config from a comparable survivor set.
Its convergence timing and speed objective are future additions; the implemented
search behavior is described below.

[`trainer-families.json`](../configs/forge/trainer-families.json) declares family
membership and a canonical fallback. Distinct mechanisms, including structural
ablations, have explicit families. The older matched R1/R2 penalty swap and the
Modern GAN recipe belong to the same R1/R2 family, but their historical source
cohorts remain separate evidence. The canonical fallback is an explicit choice,
not a claim that incomparable historical configurations have been ranked.
An explicit `active_search_by_backend` setting selects the study used by a
family. Completing a later study preserves earlier trials; switching the active
study is a reviewable registry change.

## Declare a search

A search declares a base recipe, a finite grid of public `Recipe` settings, a
fixed protocol/view/runtime, a tuning tier cap and immutable campaign budgets.
Coupled settings such as two endpoints of a penalty schedule belong in one
dimension. Trials inherit the same architecture, prior, initialization, named
random streams, task horizons and sampling law. Seeds and task definitions are
not search parameters.

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
Ordinary prerequisites stop a configuration after its first failed task.
Actual paid time, failed attempts, remaining unknown tasks and reuse are recorded.
Raw logs, checkpoints and per-update traces remain under ignored `runs/` or in
an artifact archive; the compact report and provenance receipts are committed.

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
