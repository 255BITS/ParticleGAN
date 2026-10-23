# Research glossary

| Term | Meaning |
|---|---|
| **Technique / formulation** | A training method, such as BCAP (`particlegan.Recipe` mechanism class / family). |
| **Recipe** | A concrete configuration of a technique: losses, weights, optimizer, prior learning or freezing policy, and prior regularizer (`configs/forge/configurations/<id>.json`). |
| **Candidate / Idea** | A versioned declaration (`configs/forge/configurations/<id>.json` or legacy `configs/forge/ideas/<id>.json`) binding a `Recipe` preset/overrides, prior policy, hypothesis, and lineage. |
| **Task** | The problem to solve (`configs/forge/tasks/<id>.json`), including architecture, target data, initial prior distribution and capacity, sampling, and evaluation criteria. |
| **View / Goal** | A versioned evaluation policy (`configs/forge/views/<id>.json`) that assigns tasks to qualification tiers with roles (`required`, `held`, `diagnostic`). CLI commands accept `--view` and `--goal` interchangeably. |
| **Tier** | A progressive qualification stage (`qualification_tier >= 1`) within a view. Tier 1 screens fast smoke/stability gates; higher tiers run fuller accuracy, transfer, or continuation suites. |
| **Protocol** | The fixed comparison rules (`configs/forge/protocols/screening.json`): initialization, data sequence, update budget, evaluation cadence, and scoring (fixed seed `0` for screening). |
| **Study** | A bounded scientific comparison (`configs/forge/studies/<id>.json`) pairing a candidate against an explicit control on a declared view/tier scope with an immutable budget and stop rule. |
| **Search** | A finite, deterministic hyperparameter grid (`configs/forge/searches/<id>.json`) over `TUNABLE_FIELDS` that preserves the base candidate's technique signature. |
| **Campaign** | A resource budget envelope (`budget_seconds`, `candidate_budget_seconds`) governing queue reservations. Studies and searches embed their own campaign definitions. |
| **Attempt / Receipt** | One worker execution of a resolved job (`runs/forge/<campaign>/<attempt>/` and `reports/forge/attempts/<attempt>/`), bound by `result.json` and `evidence.json` SHA-256 certificates. |
| **Cohort** | A comparability partition defined by `(candidate_revision, source_digest, runtime_cohort)` (CPU vs CUDA and GPU model). Results from different cohorts are never silently merged. |
| **Evidence scope** | Whether an observation is `current` (regradable qualification input), `pinned` (archived source/task/runtime), `historical` (imported summary), `research_diagnostic` (bounded study/search), or `calibration_diagnostic`. |
| **Readout** | A concluded scientific record (`reports/forge/records/readout-<hash>.json`) binding a candidate's finished attempts to a `conclusion`, `comparison`, and `next_action`. Required before reusing or modifying a stopped candidate. |
| **Promotion** | A preregistered multi-seed robustness stage (`reports/forge/promotions/<id>/registration.json`) for a candidate that has already qualified and concluded its screening readout. |
| **Experiment** | A comparison of declared recipe changes under a fixed protocol. |
| **Run** | One execution of a recipe on a task. |
| **Metric** | A numerical measurement of performance. |
| **Gate** | A declared pass/fail requirement. |
| **Baseline** | The reference recipe used for comparison. |
| **Representative recipe** | A technique’s best observed complete recipe under the declared tasks, gates, and budgets. |
| **Counted requirement** | A test with the role `required` in a view. It counts toward the leaderboard total and toward qualification; a test in several views is one requirement per view. |
| **Held test** | A test with the role `held`: still declared, run and shown with its history. It is excluded from qualification and never blocks progress, but remains in the full-suite performance comparison, including its failures. Its view stores the reason, a summary, an explanation link and the rule for restoring it to qualification. |
| **Removed from the count** | A test that stays assigned as a diagnostic with a note naming why it is not counted: a phase or duplicate of a counted test, a superseded definition, or a test that cannot run as declared. |
| **HELD status** | Reported for a tier with only held tests. A view reports it when every counted tier passes but its highest tier has only held tests, or when it counts nothing; a counted failure still shows as a failure. By default it is never PASS and never eligible (`counted_eligible` records whether all counted tiers passed). |
| **Archived view revision** | The exact bytes of an earlier view revision, copied to `configs/forge/view-history/<view>-v<N>.json` before the view changes. Recorded results keep pointing to it. |

**Ownership:** The recipe owns prior updates and regularization (`experiments/forge/boundaries.py`). The task defines the initial prior distribution and capacity, and the sampling contract.

**Selection:** Compare one global recipe across the task suite. Keep failures visible; avoid selecting a different recipe for each task.

**Holding:** A test is held when no one global recipe passes it together with every counted test, when no recipe has passed its current declaration, or when it continues a held test. Holding changes qualification requirements, not any recorded result or the full-suite performance comparison. See the [2026-10-15 test dispositions](reports/forge/test-dispositions-20261015/README.md) and the [problem list](reports/forge/problems.md) for each question's intent, failures and known passing configurations.

## Repository layout for Forge

- `configs/forge/` — **Immutable & versioned declarations**: `configurations/` (v3 recipe cards), `ideas/` (legacy v1/v2 cards), `studies/` (bounded questions), `searches/` (finite hyperparameter grids), `tasks/` (problem contracts), `views/` (tier/role policies), `protocols/` (screening rules), `calibration/` (gate calibration profiles).
- `reports/forge/` — **Committed durable evidence**: `technique-inventory.md` (canonical family leaderboard), `problems.md` (per-task failure diagnoses), `records/` (concluded readouts and dispositions), `configuration-search/` & `cadence-studies/` (published study reports), `EXPERIMENT_MEMORY.md` & `compilation.json` (deterministic recall index).
- `runs/forge/` — **Repo-local operational queue** (untracked): `queue/state.json`, `events.jsonl`, source snapshots, and per-attempt `run.log` / `terminal.json` files.

## 5-minute agent quickstart

```bash
# 1. Orient: search prior art & in-flight work without loading bulk memory
python -m experiments.forge recall --query "k3p" --view discriminator_stability
python -m experiments.forge board --view discriminator_stability

# 2. Propose or probe: fast preflight probe, or scaffold a configuration + study
python -m experiments.forge probe k3p --task gaussian_1d_smoke --view discriminator_stability
python -m experiments.forge new --id my-idea --parent k3p --view discriminator_stability --hypothesis "..."

# 3. Plan & execute: verify boundaries/reuse, then enqueue or run
python -m experiments.forge plan my-idea --study my-idea-study --show-boundaries
python -m experiments.forge run my-idea --study my-idea-study --gpus 0

# 4. Conclude: publish readout to unlock the candidate and update recall memory
python -m experiments.forge readout my-idea --study my-idea-study \
  --conclusion "..." --comparison "..." --next-action "..."
```
