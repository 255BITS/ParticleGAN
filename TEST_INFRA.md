# TEST_INFRA — ParticleGAN E2E Testing Infrastructure

## 1. Test Philosophy
The End-to-End (E2E) testing track provides independent, requirement-driven, opaque-box verification for the ParticleGAN Forge benchmark and public-API expansion.

### Core Principles
- **Dual Track Independence**: The E2E test track is designed strictly against requirements (`ORIGINAL_REQUEST.md` and `PROJECT.md`), isolated from internal implementation assumptions.
- **Opaque-Box Verification**: The test suite exercises features exclusively through public interfaces:
  - Command-line interfaces: `python -m experiments.forge ...`, `python -m benchmarks.toy_audit.api_run ...`
  - Public Python APIs: `particlegan.Recipe`, `particlegan.GANTrainer`, `particlegan.MoGParticlePrior`, `benchmarks.toy_audit.api_contract`
  - Declarative schemas: task declarations (`configs/forge/tasks/`), view contracts (`configs/forge/views/`), report markdown structures
- **Progressive Testability & Readiness Demarcation**: Tests validate contracts dynamically. Where features or artifacts are pending upstream milestones (e.g. M1 N-tier generalization, M2/M3 target tasks, M4 full-budget receipts), tests evaluate specification contract adherence, task schema compliance, and interface readiness without fragile coupling.
- **Fail-Closed Gate Integrity**: Numerical bounds, finite-value checks, and non-saturation rails reject invalid states immediately. Facade tests that pass vacuously are forbidden.

---

## 2. Feature Inventory Mapping
All 14 features from `PROJECT.md § Feature Inventory` are systematically mapped across four testing tiers:

| # | Feature ID | Name | Milestone | Tier 1 (Coverage) | Tier 2 (Boundaries) | Tier 3 (Cross-Feature) | Tier 4 (Workflows) |
|---|------------|------|-----------|-------------------|---------------------|------------------------|--------------------|
| 1 | F1 | N-Tier Core Infrastructure | M1 | `test_f1_tier1_*` (5 cases) | `test_f1_tier2_*` (5 cases) | Pairwise w/ F2, F3, F4 | Scenario 1 |
| 2 | F2 | Dynamic Tier Naming & Reports | M1 | `test_f2_tier1_*` (5 cases) | `test_f2_tier2_*` (5 cases) | Pairwise w/ F1, F6, F13 | Scenario 1, 5 |
| 3 | F3 | View Importance & Tier Contiguity | M1 | `test_f3_tier1_*` (5 cases) | `test_f3_tier2_*` (5 cases) | Pairwise w/ F1, F7, F8 | Scenario 1, 4 |
| 4 | F4 | Task R2 `zero_tanh_smoke` | M2 | `test_f4_tier1_*` (5 cases) | `test_f4_tier2_*` (5 cases) | Pairwise w/ F1, F5, F11 | Scenario 4 |
| 5 | F5 | Task R2 `zero_tanh_rail_escape` | M2 | `test_f5_tier1_*` (5 cases) | `test_f5_tier2_*` (5 cases) | Pairwise w/ F4, F11 | Scenario 4 |
| 6 | F6 | Task R3 `grid100_mlp16` | M2 | `test_f6_tier1_*` (5 cases) | `test_f6_tier2_*` (5 cases) | Pairwise w/ F2, F7 | Scenario 4 |
| 7 | F7 | Task R3 `grid100_res16_tanh` | M2 | `test_f7_tier1_*` (5 cases) | `test_f7_tier2_*` (5 cases) | Pairwise w/ F3, F6 | Scenario 4 |
| 8 | F8 | Task R4 `misgan_block_recovery` | M3 | `test_f8_tier1_*` (5 cases) | `test_f8_tier2_*` (5 cases) | Pairwise w/ F3, F9 | Scenario 3 |
| 9 | F9 | Task R4 `misgan_block_neural_imputer` | M3 | `test_f9_tier1_*` (5 cases) | `test_f9_tier2_*` (5 cases) | Pairwise w/ F8, F11 | Scenario 3 |
| 10 | F10 | Task R5 `ring16_bigan_continuous` | M3 | `test_f10_tier1_*` (5 cases) | `test_f10_tier2_*` (5 cases) | Pairwise w/ F11, F12 | Scenario 3 |
| 11 | F11 | 3-Level Solvability Audits | M2, M3 | `test_f11_tier1_*` (5 cases) | `test_f11_tier2_*` (5 cases) | Pairwise w/ F4, F5, F10 | Scenario 3 |
| 12 | F12 | Full-Budget API Execution & GIFs | M4 | `test_f12_tier1_*` (5 cases) | `test_f12_tier2_*` (5 cases) | Pairwise w/ F10, F13, F14 | Scenario 2 |
| 13 | F13 | View Registration & Publication Refresh | M4 | `test_f13_tier1_*` (5 cases) | `test_f13_tier2_*` (5 cases) | Pairwise w/ F2, F12 | Scenario 1, 5 |
| 14 | F14 | Git, PR, and Storage Discipline | M5 | `test_f14_tier1_*` (5 cases) | `test_f14_tier2_*` (5 cases) | Pairwise w/ F12, F13 | Scenario 5 |

---

## 3. Test Architecture
The test suite is structured under `tests/e2e/`:

```
tests/e2e/
├── __init__.py                  # Package marker
├── conftest.py                  # Shared fixtures, contract specifications, schemas, CLI runner helpers
├── test_tier1_features.py       # Tier 1: Feature coverage (>=5 test cases per feature for all 14 features)
├── test_tier2_boundaries.py     # Tier 2: Boundary and corner cases (>=5 test cases per feature for all 14 features)
├── test_tier3_combinations.py   # Tier 3: Cross-feature pairwise interactions
└── test_tier4_scenarios.py      # Tier 4: Real-world multi-stage workflow scenarios
```

### Shared Fixture Infrastructure (`conftest.py`)
- `repo_root`: Absolute path to ParticleGAN repository root.
- `run_cli`: Safe CLI runner invoking Python modules in a subprocess with timeouts, returning exit codes, stdout, and stderr.
- `contract_specs`: Formal dictionary containing the exact expected schemas, parameters, gates, and metadata for all 7 target tasks and N-tier requirements.
- `schema_validators`: Pure Python validators verifying task JSONs, view JSONs, and hold metadata without importing internal modules.
- `synthetic_forge_env`: Isolated temporary environment for testing dynamic view configurations, tier reports, and contiguity checks.

---

## 4. Runner Invocation
The E2E test suite is run with standard `pytest`:

```bash
# Run complete E2E test suite
pytest tests/e2e -v

# Run by tier
pytest tests/e2e/test_tier1_features.py -v
pytest tests/e2e/test_tier2_boundaries.py -v
pytest tests/e2e/test_tier3_combinations.py -v
pytest tests/e2e/test_tier4_scenarios.py -v

# Run specific feature tests across Tier 1 and Tier 2
pytest tests/e2e -k "f1_" -v
pytest tests/e2e -k "f4_" -v

# Run with short traceback on failure
pytest tests/e2e --tb=short
```

---

## 5. Scenario List

### Tier 1: Feature Coverage (70+ tests)
- **F1 (N-Tier Core)**: Acceptance of qualification tiers 1, 2, 3, 4, 5+ in view definitions and assignments.
- **F2 (Dynamic Naming)**: Correct naming of tiers ("smoke", "quality", "endurance", "stress_and_formulations", "tier-5"), dynamic table headers.
- **F3 (View Importance & Contiguity)**: Contiguity enforcement from Tier 1, valid importance roles (`required`, `ranking`, `held`, `diagnostic`), rejection of optional tiers.
- **F4 (zero_tanh_smoke)**: Dirac point mass at (0, 0), MoGParticlePrior(sigma=0.1), 2-layer MLP + nn.Tanh(), 1000 steps, 120s timeout, 4 error gates.
- **F5 (zero_tanh_rail_escape)**: Tier 3, gated on zero_tanh_smoke, 2000 steps, 300s timeout, target shift (+0.9, -0.9) -> (0, 0), 400-step settling.
- **F6 (grid100_mlp16)**: Tier 4, 16 hidden layers, width 128, LeakyReLU(0.2), 7000 steps, MoGParticlePrior(sigma=0.025), clean live sampling, 100-mode coverage + accuracy gates.
- **F7 (grid100_res16_tanh)**: Tier 4, 16-layer residual MLP, domain-scaled tanh (S=5.25), 7000 steps, coverage + accuracy + non-saturation gate (<= 0.001).
- **F8 (misgan_block_recovery)**: Tier 4, 2-pair MisGAN with detached mask gradient in L_x on 8D block problem, 7000 steps, manifold recovery + subspace + mask fidelity + ppost accuracy.
- **F9 (misgan_block_neural_imputer)**: Tier 4, 3-pair MisGAN with neural imputer G_i([x*m, m, z_i]), 7000 steps, evaluation on held-out ambiguous single-sensor queries without ppost.
- **F10 (ring16_bigan_continuous)**: Tier 4, continuous 2D multimodal joint BiGAN (G, E, D) on 16-Gaussian ring, MoGParticlePrior(sigma=0.1), 4000 steps, generation + reconstruction gates.
- **F11 (Solvability Audits)**: Level 1 (200/200 oracle pass with >=2x noise headroom), Level 2 (constructive capacity witness achieves PASS), Level 3 (destructive negative controls FAIL).
- **F12 (Full-Budget API Execution)**: Public API provider registration, CLI invocation via `api_run`, seed 0 isolation, compact receipt, markdown readout, goal GIF.
- **F13 (View Registration & Publications)**: Revision 13 views, historical view archives in `configs/forge/view-history/`, hold metadata schemas, report refresh.
- **F14 (Git & Storage Discipline)**: Zero untracked bulk logs, .gitignore verification, compact receipt size limits, clean git status.

### Tier 2: Boundary & Corner Cases (70+ tests)
- **F1 (N-Tier Core)**: qualification_tier=0, negative tiers, non-integer types, extreme large tier (N=100), qualification_tier at task root rejected.
- **F2 (Dynamic Naming)**: Tier 999 naming, empty intermediate tier column handling, single-tier report, out-of-order tier definitions.
- **F3 (View Importance & Contiguity)**: Discontiguous required tiers ({1, 3} missing 2; {1, 2, 4} missing 3), missing Tier 1, invalid importance ("optional"), held-only top tier.
- **F4 (zero_tanh_smoke)**: Saturation tolerance boundary (>0.0 fails), error thresholds (mean_abs_error=0.005001 fails), prior dispersion boundary, nonfinite output fails closed.
- **F5 (zero_tanh_rail_escape)**: Tanh saturation gradient boundary (tanh' ≈ 0.19), 400-step settling limit boundary, target shift step boundary (1000->1001), stuck-at-rail control failure.
- **F6 (grid100_mlp16)**: 16-layer gradient stability, mode count boundary (99 modes fails), precision boundary (0.969 fails), latent sigma boundary (0.025), 100k sample accuracy boundary.
- **F7 (grid100_res16_tanh)**: Saturation fraction boundary (0.001001 fails), scale factor domain boundary [-5.25, 5.25], residual gradient flow, extreme rail outputs.
- **F8 (misgan_block_recovery)**: Subspace leak boundary (off_plane=0.0401 fails), ambiguous acc boundary (acc_lo=0.699 fails), mask_tv boundary (0.0501 fails), all-zero/all-one masks.
- **F9 (misgan_block_neural_imputer)**: Max error boundary (1.01e-6 fails), std ratio bounds (0.699 fails, 1.301 fails), latent noise ignore failure, acc_lo boundary.
- **F10 (ring16_bigan_continuous)**: Reconstruction RMSE boundary (0.1201 fails), mode consistency boundary (0.979 fails), scrambled pair rejection, mode coverage boundary (<16 modes).
- **F11 (Solvability Audits)**: Oracle pass boundary (199/200 fails), headroom boundary (1.99x fails), nonfinite outputs fail, off-plane leaks fail, scrambled BiGAN pairs fail.
- **F12 (Full-Budget API Execution)**: Short run (--steps 16) rejected as full quality pass, seed divergence detection, output collision handling, nonfinite metrics fail closed.
- **F13 (View Registration & Publications)**: Duplicate task in view rejected, revision gap detected, malformed hold metadata rejected, cyclic task dependencies rejected.
- **F14 (Git & Storage Discipline)**: Forbidden untracked files (.pt, .pth, .npz, .jsonl), oversized files (>10MB), forbidden metadata filenames (AGENTS.md, GEMINI.md in teamwork).

### Tier 3: Cross-Feature Combinations (>=15 tests)
- Pairwise interactions between Core N-tier, Reports, View Importance, Tasks, Solvability, API Execution, View Registration, and Git Discipline.

### Tier 4: Real-World Scenarios (>=5 tests)
- Scenario 1: Full N-Tier Pipeline Lifecycle (declaration -> view assignment -> contiguity -> report export).
- Scenario 2: Public API Execution & Verification Pipeline (case discovery -> seed isolation -> metric observation -> receipt generation).
- Scenario 3: Solvability Audit Gauntlet (Oracle noise headroom -> constructive capacity witness -> destructive negative controls).
- Scenario 4: Multi-Tier Task Progression & Gating Workflow (Tier 1 smoke -> Tier 3 rail escape -> Tier 4 16-layer stress & MisGAN).
- Scenario 5: Repository Integrity & Publication Freshness Audit (Forge validation, publication refresh checks, git hygiene).

---

## 6. Coverage Thresholds
- **Tier 1 (Feature Coverage)**: >= 5 distinct test cases per feature (Total >= 70 tests).
- **Tier 2 (Boundaries & Corners)**: >= 5 distinct boundary test cases per feature (Total >= 70 tests).
- **Tier 3 (Cross-Feature Combinations)**: >= 15 pairwise interaction test cases.
- **Tier 4 (Real-World Application Scenarios)**: >= 5 full multi-stage end-to-end workflow scenarios.
- **Pass Threshold**: 100% pass on all implemented features and verified contracts.
