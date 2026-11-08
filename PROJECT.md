# Project: Formulaic Tier Evaluation Baseline, Tier 1 Resolution, and Family Progress

## Architecture
- **Benchmarking & Evaluation Framework**: ParticleGAN Forge (`experiments/forge/`, `configs/forge/views/`, `configs/forge/tasks/`)
- **Reporting & Documentation**: Markdown reports in `reports/` and `reports/forge/`, JSON coverage data in `reports/forge/attempt-coverage.json` and `technique-inventory.json`
- **Family Abstractions**:
  - BISPF: Born-Infeld Sinkhorn Particle Flow (`particlegan/bispf.py`, `particlegan/_bispf/`, `particlegan/recipes.py`)
  - Atlas: Particle-Normal Prior, DV12 stationarity policy, birth/death feature cell guard, KA2 de-clocked anchor (`particlegan/policy.py`, `particlegan/birth_death.py`, `particlegan/feature_cells.py`)
  - BCAP: recipes built on the fixed one-sided L2 critic input-gradient cap `b_cap`; no recorded acronym expansion. See the [research glossary](RESEARCH_GLOSSARY.md#acronyms-and-short-names) (`particlegan/grad_regularizers.py`, `particlegan/optim/dualnorm.py`).
- **Version Control & Integration**: Git PR workflow into `develop` using `gh pr create` and clean incremental merges.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Baseline Evaluation Report | Comprehensive baseline formulaic tier evaluation report under `reports/` documenting all 400 candidate x task results, pass/fail status across tiers and recipes | M1 | ORIGINAL_REQUEST §R1 |
| 2 | Baseline Report PR & Merge | Branch `docs/baseline-formulaic-tier-evaluation-report`, PR #500 & repair PR #501 to `develop`, passing tests and clean merge | M1 | ORIGINAL_REQUEST §R1, §R4 |
| 3 | Tier 1 Calibration & Retiering | Document empirical justification and reclassify universal 4-step diffusion failures (`model_ddgan_gaussian_chain`, `model_ae_ddgan_gaussian_chain`) to Tier 2 in `configs/forge/views/discriminator_stability.json` | M2 | ORIGINAL_REQUEST §R2, §R3 |
| 4 | Tier 1 Calibration PR & Merge | Branch `fix/reclassify-tier1-ddgan-chains-to-tier2`, PR to `develop`, passing tests and clean merge | M2 | ORIGINAL_REQUEST §R3, §R4 |
| 5 | Family Deep-Dive & Evaluation Progress | Documented evaluation results and formulation/recipe progress across BISPF, Atlas, and BCAP for failing Tier 1 tasks | M3 | ORIGINAL_REQUEST §R2 |
| 6 | Tier 1 Universal Coverage Verification | Ensure every Tier 1 task has at least one candidate recipe passing, verified by `python -m experiments.forge recall` | M3 | ORIGINAL_REQUEST §R2 |
| 7 | Family Progress PR & Merge | Branch `feat/family-tier1-progress-evaluations`, PR to `develop`, passing tests and clean merge | M3 | ORIGINAL_REQUEST §R2, §R4 |
| 8 | Integration & Repository Hygiene | Full repository test suites (`pytest`), Forge readout currency check, clean develop branch with zero untracked bulk logs | M4 | ORIGINAL_REQUEST §Verification |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Baseline Evaluation Report PR | Create `reports/baseline_formulaic_tier_evaluation_report.md`, catalog registration, verify and merge into `develop` (PR #500, #501) | none | DONE |
| 2 | Tier 1 Calibration & Retiering | Reclassify universal DDGAN tasks to Tier 2 with empirical justification, open PR, verify and merge into `develop` | M1 | DONE |
| 3 | Family Progress & Tier 1 Resolution | Documented family evaluations & recipe progress across BISPF, Atlas, BCAP ensuring >=1 passing candidate per Tier 1 task, PR and merge into `develop` | M2 | DONE |
| 4 | Final Integration & Qualification | Complete verification with `pytest`, `python -m experiments.forge recall`, and final compliance confirmation | M3 | PLANNED |

## Interface Contracts
### Evaluation Views ↔ Tasks
- `configs/forge/views/*.json` define `assignments` containing `task`, `qualification_tier`, `importance` (`required`, `held`, or `diagnostic`), and `order`. Held tests retain their results in full-suite comparisons while remaining outside qualification, as defined in the [research glossary](RESEARCH_GLOSSARY.md).
- Tasks own the problem, initial prior distribution and capacity, architecture, sampling, budget and gates; recipes own prior updates and regularization. Use the [field ownership contract](docs/forge-field-boundaries.md) and [task/study guide](docs/forge-studies.md).
- Toy tests follow the [public ParticleGAN API contract](reports/toy_audit/api_contract/README.md), including a numerical pass/fail metric and an actual-training GIF. Concluded scientific records live in `reports/forge/records/`; execution receipts bind original request/result/evidence envelopes. Retained receipt locations and [artifact restoration](docs/forge-artifact-resolver.md) preserve those identities separately from compact records.

### Reporting ↔ Leaderboard Index
- Standalone reports under `reports/` must not conflict with `reports/leaderboard-index.json`. Avoid `# ... leaderboard` title headers on non-indexed reports.
- `python ops/refresh_forge_readouts.py --check --reports-only` verifies retained publication integrity. The strict `--check` separately checks evidence against the current source; legitimate software changes can leave that scope `NEEDS_REFRESH`. Preserve the original source pins and results, report the gap, and avoid rerunning unchanged studies solely to merge a cleanup.
- All tracked files in `SOURCE_ROOTS` must be declared in `configs/forge/catalog.json`.

## Code Layout
- `reports/baseline_formulaic_tier_evaluation_report.md`: Standalone baseline report
- `configs/forge/catalog.json`: File catalog and inventory manifest
- `configs/forge/views/discriminator_stability.json`: Qualification tier assignments
- `reports/forge/families/`: Family documentation and evaluation breakdowns
- `reports/forge/technique-inventory.md`: Technique inventory and coverage summary
- `reports/forge/attempt-coverage.json`: Machine-readable evaluation matrix
