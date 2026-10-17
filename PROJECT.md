# Project: ParticleGAN N-Tier & Target Experiments Suite

## Architecture
ParticleGAN benchmark and public API architecture expansion:
- **Core Forge Infrastructure (`experiments/forge/`)**: Extensible dynamic tier framework supporting arbitrary positive integer `qualification_tier >= 1`, continuous from Tier 1. Dynamic naming (`Tier 4: stress_and_formulations`, `tier-N` fallback for $N \ge 5$), dynamic reporting tables, and strict view importance (`required`, `ranking`, `held`, `diagnostic`).
- **Public API Benchmark Suite (`benchmarks/toy_audit/`)**: Self-contained runnable target fixtures adhering to `reports/toy_audit/api_contract/README.md`.
  - `api_tanh.py` (or integrated in `api_vectors.py`/`api_diagnostics.py`): Dirac point-mass contraction (`zero_tanh_smoke`) and rail escape (`zero_tanh_rail_escape`).
  - `api_deep100g.py` (or integrated in `api_vectors.py`): 16-hidden-layer MLP (`grid100_mlp16`) and 16-layer residual MLP + scaled bounded tanh (`grid100_res16_tanh`).
  - `api_misgan.py`: 2-pair detached mask MisGAN (`misgan_block_recovery`) and 3-pair neural imputer (`misgan_block_neural_imputer`).
  - `api_bigan.py`: Continuous 16-mode ring joint BiGAN (`ring16_bigan_continuous`).
- **3-Level Solvability Audits (`tests/`)**: Automated `pytest` suites enforcing Level 1 (Oracle 200/200 pass with >=2x noise headroom), Level 2 (constructive capacity witness achieves PASS), Level 3 (destructive negative controls deterministically FAIL).
- **View Registration & Publications (`configs/forge/views/`, `reports/forge/`)**: Revision 13 views, historical view archives in `configs/forge/view-history/`, updated `EXPERIMENTS_BY_TIER.md`, `problems.md`, `technique-inventory.md`, and compiled summaries.
- **E2E Testing Track**: Independent, requirement-driven, opaque-box test suite validating all tiers and features end-to-end.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | F1: N-Tier Core Infrastructure | Generalize `experiments/forge/` (`views`, `planning`, `queue`, `studies`, `decisions`, `dispositions`, `configuration_search`, `telemetry`, `__main__`) for dynamic $N \ge 1$ | M1 | ORIGINAL_REQUEST §R1 |
| 2 | F2: Dynamic Tier Naming & Reports | Tier 4 as `stress_and_formulations`, `tier-N` fallback, dynamic table headers in `tier_report.py`, `family_reports.py`, `technique_inventory.py` | M1 | ORIGINAL_REQUEST §R1 |
| 3 | F3: View Importance & Tier Contiguity | Enforce contiguity from Tier 1; enforce view importance (`required`, `ranking`, `held`, `diagnostic`) without optional tiers | M1 | ORIGINAL_REQUEST §R1 |
| 4 | F4: Task R2 `zero_tanh_smoke` | Tier 1, Dirac point mass at `(0, 0)`, `MoGParticlePrior(sigma=0.1)`, 2-layer MLP + `nn.Tanh()`, 1000 steps, 4 error gates | M2 | ORIGINAL_REQUEST §R2 |
| 5 | F5: Task R2 `zero_tanh_rail_escape` | Tier 3, gated on `zero_tanh_smoke`, 2000 steps, target shift `(+0.9, -0.9) -> (0, 0)`, 400-step settling | M2 | ORIGINAL_REQUEST §R2 |
| 6 | F6: Task R3 `grid100_mlp16` | Tier 4, 16-hidden-layer 128-wide LeakyReLU MLP, 7000 steps, `MoGParticlePrior(sigma=0.025)`, clean live sampling, 100-mode coverage + accuracy | M2 | ORIGINAL_REQUEST §R3 |
| 7 | F7: Task R3 `grid100_res16_tanh` | Tier 4, 16-layer residual MLP + domain-scaled bounded `tanh` ($S=5.25$), 7000 steps, coverage + accuracy + non-saturation (`saturated_fraction <= 0.001`) | M2 | ORIGINAL_REQUEST §R3 |
| 8 | F8: Task R4 `misgan_block_recovery` | Tier 4, 2-pair MisGAN with detached mask gradient in $L_x$ on 8D `block` problem, 7000 steps, 2D manifold recovery, 8D subspace, mask fidelity, `ppost` Bayesian imputation | M3 | ORIGINAL_REQUEST §R4 |
| 9 | F9: Task R4 `misgan_block_neural_imputer` | Tier 4, 3-pair MisGAN with neural imputer $G_i([x*m, m, z_i])$, 7000 steps, evaluated directly on held-out ambiguous single-sensor queries | M3 | ORIGINAL_REQUEST §R4 |
| 10 | F10: Task R5 `ring16_bigan_continuous` | Tier 4, continuous 2D multimodal joint BiGAN `(G(z), E(x), D(x, z))` on 16-Gaussian ring with `MoGParticlePrior(sigma=0.1)`, 4000 steps, generation + reconstruction | M3 | ORIGINAL_REQUEST §R5 |
| 11 | F11: 3-Level Solvability Audits | Automated `pytest` suites covering Level 1 (Oracle 200/200 pass), Level 2 (constructive capacity witness), Level 3 (destructive negative controls) for all 7 tasks | M2, M3 | ORIGINAL_REQUEST §R6 |
| 12 | F12: Full-Budget API Execution & GIFs | Execute all 6 cases (7 tasks) at declared full budgets via `benchmarks.toy_audit.api_run` (protocol seed 0), producing receipts, readouts, and actual-training GIFs | M4 | ORIGINAL_REQUEST §R6 |
| 13 | F13: View Registration & Publication Refresh | Register in `discriminator_stability.json` (rev 13) and `formulation_comparison.json`, archive in `view-history/`, refresh `EXPERIMENTS_BY_TIER.md`, `problems.md`, `technique-inventory.md`, `compile --summaries-only` | M4 | ORIGINAL_REQUEST §R6 |
| 14 | F14: Git, PR, and Storage Discipline | Merge into `develop`, push to `origin/develop`, zero bulk logs in git, all validators and test suites pass | M5 | ORIGINAL_REQUEST §R7 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | N-Tier Core Infrastructure & Reporting | Features F1, F2, F3: Generalize `experiments/forge/` for dynamic $N \ge 1$, Tier 4 = `stress_and_formulations`, dynamic Tier N fallback, update reports and unit tests in `tests/test_forge_*.py` | none | **DONE** (560 unit/adversarial tests pass, 162/162 E2E pass, validate/compile clean) |
| M2 | Tanh & Deep100g Suite Implementation & Solvability | Features F4, F5, F6, F7, and F11 (tanh & deep100g): Public API fixtures, task JSONs, and 3-level solvability unit tests | M1 | **DONE** (Gate Passed: 19 solvability tests pass, all reviewers/challengers/auditor approve) |
| M3 | MisGAN & BiGAN Suite Implementation & Solvability | Features F8, F9, F10, and F11 (misgan & bigan): Public API fixtures, task JSONs, and 3-level solvability unit tests | M1 | **DONE** (Gate Passed: 9 solvability tests pass, ppost 2.88x headroom, genuine witnesses, load_state_dict, all reviewers/challengers/auditor approve) |
| M4 | Full-Budget API Execution, GIFs, View Registration & Publication Refresh | Features F12, F13: Run `api_run` for all new cases at protocol seed 0, produce GIFs/receipts, register views (rev 13), refresh publications | M2, M3 | **DONE** (Gate Passed: all 7 runs verified, goal GIFs & receipts <100KB, rev 13 view registered, rev 12 archived, publications CURRENT, validate exits 0, all reviewers/challengers/auditor approve) |
| M5 | Final Acceptance, E2E Verification & Git Publication | Feature F14: Pass 100% E2E test suite (Tiers 1-4), execute Tier 5 adversarial hardening, verify clean git tree, merge into `develop`, push to `origin/develop` | M4, E2E-Track | IN_PROGRESS |

## Interface Contracts

### M1 (Forge N-Tier Engine) ↔ Downstream Tasks (M2, M3, M4)
- Task declarations in `configs/forge/tasks/<task_id>.json` can specify any integer `"qualification_tier": N` (with $N \ge 1$).
- Tier 4 tasks declare `"qualification_tier": 4`.
- `tier_name(4)` returns `"stress_and_formulations"`.
- `tier_name(N)` for $N \ge 5$ returns `"tier-{N}"`.
- Contiguity check validates that if the highest required tier in a view is $H$, required tiers must equal `set(range(1, H + 1))`.

### M2 / M3 (Task Providers) ↔ M4 (Runner & View Registration)
- Every task has:
  1. A valid JSON task definition in `configs/forge/tasks/<task_id>.json` conforming to Forge task schema.
  2. A registered CLI case in `benchmarks.toy_audit.api_contract.PROVIDERS` invocable via `python -m benchmarks.toy_audit.api_run --case <case_id> --output <dir> --seed 0`.
  3. Output files produced in `<dir>`: `receipt.json`, `readout.md`, `goal.gif`.
- Solvability audits executable via `pytest tests/test_solvability_<suite>.py`.

### E2E Testing Track ↔ Implementation Track
- Opaque-box E2E test runner invoked via command specified in `TEST_READY.md`.
- Exit code 0 indicates 100% pass across all test tiers.

## Code Layout
- `experiments/forge/`: Forge orchestration, tier definitions, view validations, planning, queueing, reports.
  - `views.py`, `planning.py`, `queue.py`, `studies.py`, `decision_contracts.py`, `dispositions.py`, `configuration_search.py`, `telemetry.py`, `__main__.py`
  - `tier_report.py`, `family_reports.py`, `technique_inventory.py`, `research_tiers.py`
- `configs/forge/tasks/`: JSON declarations for all tasks (`zero_tanh_smoke.json`, `zero_tanh_rail_escape.json`, `grid100_mlp16.json`, `grid100_res16_tanh.json`, `misgan_block_recovery.json`, `misgan_block_neural_imputer.json`, `ring16_bigan_continuous.json`).
- `configs/forge/views/`: View definitions (`discriminator_stability.json`, `formulation_comparison.json`).
- `configs/forge/view-history/`: Historical immutable view archives.
- `benchmarks/toy_audit/`: Public-API cases, providers, and runners.
- `reports/toy_audit/api_contract/`: Committed compact receipts, readouts, and final goal GIFs.
- `reports/forge/`: Generated markdown reports (`EXPERIMENTS_BY_TIER.md`, `problems.md`, `technique-inventory.md`).
- `tests/`: Automated unit tests, solvability audits, and tier framework tests.
