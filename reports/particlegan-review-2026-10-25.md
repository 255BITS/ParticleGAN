# ParticleGAN project review — 2026-10-25

Reviewed revision: `baa65a306` on `feat/swappable-prior-manipulation`.
Method: six parallel, read-only reviewers, one per area:

1. Core trainer, recipes and priors
2. Particle/transport algorithms
3. Feature cells and optimizers
4. Forge
5. Tests and benchmarks
6. Docs, scientific claims and repo hygiene

Findings marked **confirmed** were reproduced with short CPU scripts or a sampled pytest run. **Read** means the finding was verified by reading the code only. **Plausible** means the finding is not yet reproduced. No files other than this report were changed.

## Executive summary

- **Engineering verdict.** The numerical core is in good shape. The optimizer math checks out against SVD: polar/Newton–Schulz, the quintic, the Gram fallback, and bf16. R1/R2 penalties and checkpoint round-trips are correct.
  - The `experiments/forge` view grader is carefully conservative.
  - Recent additions are weaker: swappable priors, `particlegan_forge`, BISPF heuristics and the e2e tiers. They introduce provenance breaks that contradict the repository's own AGENTS.md contract.
- **Scientific verdict.** The central claim has no matched control in the current results: a learnable particle prior beats a fixed Gaussian. In addition:
  - The qualification denominator moves as tests are "held".
  - Baselines get far less search budget than house recipes.
  - The default recipe (BISPF) switches behaviour on task shape (dimension, batch size, step count), so its "one global configuration" is several hand-tuned ones.
  - Headline numbers have no uncertainty.
- **Maintainability verdict.**
  - **Repo size:** 1.8 GB tracked (22.6k files, 19.4k under `reports/`), plus 526k lines of vendored Python snapshots in `reports/toy100`.
  - **Leaderboards:** about 58 "current leaderboard" pages, and several of them disagree.
  - **Large modules:** god-modules of 1.4k–1.9k lines.
  - **Tests:** about 90 e2e tests cannot fail.

## Severity-ranked defects

| # | Sev | Location | Defect | Status |
|---|---|---|---|---|
| 1 | High | `particlegan/training.py:246` | `type(self.prior) is MoGParticlePrior` is the one prior check not unwrapped.<br>A wrapped MoG prior gets no `prior_noise_generator`, the caller's stream is ignored, it isn't checkpointed, and MoG noise is drawn from `latent_generator`.<br>An identity manipulator therefore changes the trajectory. | confirmed |
| 2 | High | `particlegan/recipes.py:make_prior`, `particlegan/priors.py` | Manipulator hyperparameters (temperature, jitter, k_nn, routing_mode…) are not in `state_dict` or the saved recipe.<br>A checkpoint loads silently into a trainer with different manipulator settings. | confirmed |
| 3 | High | `particlegan/priors.py:483-502` | Constructing `SimplexRoutingManipulator` calls `orthogonal_` on the global RNG, which breaks constructor-RNG isolation.<br>`learnable_projection=False` with `z_dim=None` crashes (`KeyError`, attribute already exists). | confirmed |
| 4 | High | `particlegan/bispf.py:243-272` | `SBNPPriorRegularizer` overrides the declared weight.<br>Without references, any `0<w<0.1` becomes 1.0. When `n<=64, d>=8`, the weight is ignored and repulsion is multiplied by 25.<br>Weights 0.05 and 1.0 give the same loss. | confirmed |
| 5 | High | `particlegan/bispf.py:1077, 1207, 1241` + SBNP branches | Behaviour is gated on task shape and step: `dim>=170` → "simplex"; `call_idx>=1650 && dim==2 && 64<=b_r<=512` → thermal 1e-5; `b_f>=1024` rescale.<br>The "global recipe" is task-tuned and undeclared. | read |
| 6 | High | `particlegan_forge/tiers.py:745-747` | `qualify()` checks the task fingerprint but not the candidate fingerprint.<br>Changing `recipe_overrides` keeps the stale PASS. If the adapter then raises, no new receipt overwrites it. | confirmed |
| 7 | High | `particlegan_forge/tiers.py:937-962` | `compile_reports` writes a hard-coded narrative ("fast-weight-bcap-ns 4/4 PASS", 480 steps, batch_size=1) into `EXPERIMENT_MEMORY.md` regardless of the receipts.<br>Introduced by HEAD. | confirmed |
| 8 | High | `particlegan_forge/tiers.py:393-403` | A task with no thresholds automatically PASSes, which violates the "declare a numerical pass/fail metric" contract. | confirmed |
| 9 | Med/High | `particlegan_forge` vs `experiments/forge` | Both packages use `configs/forge` and `reports/forge` with different schemas.<br>`run`/`board`/`compile` crash on this repo (`KeyError 'resources'`), and that crash is the only thing preventing #7 from overwriting the live inventory. | confirmed |
| 10 | Med | `particlegan/optim/constraint_geometry.py:104`, `strict_progress.py:97`, `direction_blend.py:31`, `critic_cap.py:71` | If the base `step()` raises, `_protected` and `_cap_panels` are never cleared.<br>Every later step fails with "losses already pending", and the critic-cap optimizer can no longer be checkpointed. | confirmed |
| 11 | Med | `particlegan/row_evidence.py:79-86` | The test labelled "Hotelling T²" is actually a sphericity statistic.<br>Under anisotropic null noise at d=8, P(p≤.05)=0.17 and BH flags 10% of rows, all false.<br>Also, `z_dim>=34` with window 50 never tests, and the gate is a silent no-op. | confirmed |
| 12 | Med | `particlegan/bispf.py:955` | The module-global `_LAST_THERMAL_SCALE` (keyed by device) couples trainers and EMA evaluators in one process. It contradicts the K3P "no module-level state" contract that BISPF inherits. | plausible |
| 13 | Med | `particlegan/bispf.py:671` | The OT barycentric target is not normalized by column mass, which is 0.94–1.08 after 5 Sinkhorn iterations. The pull therefore scales toward the origin by up to about 8%. | confirmed |
| 14 | Med | `particlegan/bispf.py:699` | `BISPFLossAdapter.d_loss` calls `d_step` without a `generator`, so instance noise and multinomial draws use the global RNG. | read |
| 15 | Med | `particlegan/training.py:419/438/560`, `policy.ServedModel.sample` | `postprocess_generated` is applied only to G-step fakes, not to D-step fakes, `sample()` or served output. The critic trains on a different distribution from the one G is scored on. | read |
| 16 | Med | `particlegan/training.py:483-491` | Strict-progress replay assumes the transform is additive (`latent + (z - loc)`). A Simplex manipulator replays wrongly without error. | read |
| 17 | Med | `particlegan/priors.py:413` | `StratifiedSamplingManipulator.sample_indices` returns a view of `_perm`. The next wrap rewrites indices already handed out (`[0,1,2]` became `[4,0,7]`). `initial_cursor` and `num_particles` are not validated against the base prior. | confirmed |
| 18 | Med | `particlegan_forge/tiers.py:770-780` | A tier with only diagnostic tasks is marked FAIL, which blocks higher tiers. `experiments/forge/views.py` marks the same tier HELD. | confirmed |
| 19 | Med | `particlegan_forge/tiers.py:672-674`, `rng.py:145-175` | Missing adapter fields default to "finite, 0 RNG deviations, all steps done". `load_state_dict` skips the seed, version and runtime checks that `experiments/forge/rng.py` has. `deterministic_scope` uses `warn_only=True`. | confirmed / plausible |
| 20 | Low | `particlegan/ka2.py:274` | `record.alpha *= game_trust` compounds across calls. Trust 0.9 cuts alpha by about 87% in steady state, not 10%. | plausible |
| 21 | Low | `particlegan/feature_cells.py:1565-1568` | `maybe_apply` zeros the evidence and advances counters before its `try`, so a failure loses the evidence. | read |
| 22 | Low | `particlegan/optim/constraint_geometry.py:78` | Frozen params in a group crash `autograd.grad`. | confirmed |
| 23 | Low | `experiments/forge/planning.py:24-46` | The binding cache key records which overrides are present but not their values. | plausible |
| 24 | Low | various | `assert` used for invariants (`output_moments`, `mean_transport`). Stale docstrings (`birth_death.py:15` clone behaviour, `bispf.py:868` cooler formula). `RoutedEvidence.flagged_rows_sum` is double-counted. | read |

## Area reviews

### 1. Core API: `training.py`, `recipes.py`, `priors.py`, `policy.py`

- **Unwrapping is duck-typed.** `getattr(prior, "base", prior)` appears in about 8 places, and #1 is the call site that was missed.
  - `ManipulatedPrior.__getattr__` forwards to the base prior. As a result, `hasattr(prior, "regularize" | "set_sigma" | ...)` is always true and hides capability gaps.
  - Use one `unwrap_prior()` helper and an explicit capability protocol.
- **Recipe ownership is bypassed.** Manipulator penalties are added to `loss_g` with no Recipe weight. Router parameters keep training under `prior_update="frozen"`. The router's optimizer group takes `lr`/`betas` with no Recipe field.
  - All three are undeclared trainer deltas under AGENTS.md.
- **`GANTrainer` vs `UpdatePolicy`.** `GANTrainer` is mostly forwarding to `UpdatePolicy`, with about a dozen `_serve_*` passthroughs and `setattr`-injected properties (`training.py:817-829`). It is unclear which object owns state.
- **The `_step` G-phase is overloaded.** It mixes sampling, postprocess, three regularizers and replay. Extract `_generator_objective()`.
- **Per-step overhead.** `regularize()` runs even when every manipulator returns zero. It does a `torch.unique` when N>1024 and a `.item()` host sync every step.
- **Smaller issues.**
  - The `inspect.signature` call on every forward is slow and hard to follow.
  - `overrides.pop("manipulators", ...)` is dead code.
  - `out.weights = w` sits inside `try/except: pass`.
  - `projection.data = ...` mutates a parameter behind the optimizer.

### 2. Particle, transport and population algorithms

- **Sound:** birth–death Beta(k,k) normal scores, meander p-values and BH; the empirical-Bernstein bound; EMA moment recursions; DDGAN posterior coefficients; RLOO+KL in the autoencoder.
- **The "anytime-valid" mixture-t e-process is only nominal.** The inputs are bounded cosines, `rebase`/`exclude` rewrite the past, and two scales are tested at once. Treat it as a heuristic gate, not as error-rate control.
- **Duplication.** There are at least six pairwise-distance/kNN implementations (three inside `bispf.py` alone) and three BH implementations.
  - `group_means` is defined twice.
  - Pooled reference moments are computed three times.
  - The Adam-surprise ratio is computed twice; the KA2 copy syncs the host once per tensor.
  - `PopulationSequentialSettleTest` copies about 5 methods verbatim from `SequentialSettleTest`.
- **Stringly variants.** Tuples of variant names (`"dv4"…"dv12"`) are repeated about 20× across modules. Derive feature flags once.
- **`bispf.py` is overloaded.** At 1.4k lines it mixes an engine, a prior, a regularizer, a loss and a penalty, with magic constants (0.065/0.050, 25×, 1650) and alias arguments (`knn_k`/`k_nn`).

### 3. Feature cells and custom optimizers

- **Math verified on CPU.**
  - Newton–Schulz polar matches SVD to about 1e-6 (tall/wide/batched/bf16).
  - The quintic interval [0, 1.2023686] is invariant.
  - The Gram/FP64 fallback matches the SVD law.
  - Conv permutation round-trips (including groups and transposed convs) are correct.
  - Optimizer state round-trips are bit-exact.
  - R1/R2 uses `create_graph` correctly on detached inputs.
- **`feature_cells.py` (1,918 lines) holds six concerns:** feasibility, lineage, DV12 geometry, hypergeometric tests, a ~950-line snapshot and a ~500-line controller.
  - The three transport planners are 63–81% identical.
  - One line is a single 1,802-character `dict(...)`.
  - Formatting styles are mixed.
- **`feature_reference.py` is a 69%-similar fork of `birth_death.py`,** and both define `ParticleBirthDeath`.
- **`NormalizedOptimizer.step` duplicates its branches.** Its batched-conv branches repeat the same plumbing three times. The wrapper chain uses `NormalizedOptimizer.step(self)` to skip parents; a single "propose → adjust → commit with try/finally" helper would fix #10 everywhere.

### 4. Forge (`experiments/forge`, `particlegan_forge`, `EXPERIMENTATION.md`)

- **Good:** `experiments/forge/views.py` independently re-grades raw evidence.
  - It refuses status-only stamps.
  - It marks conflicting compatible attempts INVALID.
  - It checks the full observation schedule.
  - HELD never counts as PASS.
- **`particlegan_forge` is an untested partial fork.** It duplicates `rng`, `contracts` and the tier logic with weaker semantics. It ships in the wheel and is not mentioned in EXPERIMENTATION.md. Either make it a thin facade over `experiments/forge` or delete it.
- **Forge reaches into trainer internals.**
  - `canonical_image_adapter.py:474-476` `exec`s rewritten trainer source to swap `GANTrainer._step`.
  - `planning.resolution_cache` monkey-patches other modules' globals.
  - `views.py:593-600` loads a grader from `reports/toy100/...`, so scoring code lives in the reports tree.
- **Sprawl.** 116 modules, several over 1k lines, plus many one-off publication modules (`pr399_scores`, `native3_*`, `common26_comparison`) still on the import path.
- **EXPERIMENTATION.md** opens with about 50 lines of changelog links before any instructions. Add a one-page quickstart (recall → new → plan → run → board → readout) and move history to an index.

### 5. Tests and benchmarks

**What the reviewer ran:** two CPU batches, about 4.5 minutes in total.
- Batch 1: 21 passed, then 1 failed (`ot`/POT not installed locally; CI installs it).
- Batch 2: 196 passed in 46 s.

**Findings:**
- **About 86 e2e tests in tiers 2–3 test a test-local mirror** (`tests/e2e/conftest.py:210-271`, `CONTRACT_SPECS`), not production code.
  - Tier-4 scenarios are tautologies. One example is the "capacity witness": a zeroed output layer under L1 loss has zero gradient at exactly 0, so the test can't fail.
  - `test_tier4_scenarios.py:255-286` fails on any checkout with a stray local `.pt`.
- **35 CUDA-only tests use `skipif(not cuda)`** instead of the declared `cuda` marker. They never run in CI, and on workstations they run silently and compete with queued experiments.
- **The 3.11 CI job doesn't run pytest.**
- **`test_det_init_registry` is slow and only checks determinism.** It takes 179 s with 350 subprocesses, and any constant output passes.
- **Brittle tests.** About 125 tests read `reports/` and 58 assert on Markdown text; these are change detectors, not defect detectors.
- **Coverage gaps:**
  - `feature_cells` (0/10 public symbols)
  - `anchor_birth` (0/2)
  - `execution` (0/2)
  - `birth_phase` (1/9)
  - `routing`: `RoutedExecution`, `RoutedEvidence` and `RoutedRowControl` are untested
  - `recipe_schedules` (2/6)
  - the `PriorProximalOptimizer` class
  - `_qr.r2_points`: no property test, although it underlies the public deterministic initializer
- **Benchmarks.**
  - The toy100 metrics are well powered (n=20k, about 200 per mode) and toy_audit has real oracle and destructive controls.
  - Gate metrics carry no CI or bootstrap.
  - `vector_tasks.score_samples` reuses the same reference draw (seed 991) and eval latents (seed 990) at every observation, so the "5 terminal observations" are highly correlated.
  - Some cases score on 128–1,024 samples.

### 6. Scientific quality, docs and repo hygiene

**Claims and evidence:**
- **No prior ablation.** README:26-29 claims a learnable prior recovers modes that a fixed Gaussian loses. No current result compares the default recipe with (a) a frozen table and (b) N(0, I), with everything else matched.
  - `examples/100gaussians_no_particle_prior.py` exists but is uncited.
  - On CIFAR, the "Direct GAN" (FID 19.5) beats "Particle AE-GAN" (20.1), with one seed and no error bar.
- **Moving denominator.** Tests the leader fails get "held", and `transfer_suite` went from "7/16" to "7/10" through a scope revision. BISPF stands at 18 PASS / 11 FAIL / 20 unmeasured out of 49 and is still promoted to default.
- **Unequal tuning budget.**
  - R3GAN: 80 updates (5.3 s), stopped at its first failure with 23 UNKNOWN.
  - BCAP/BISPF: 41- and 25-configuration searches.
  - E22 shows 0/115 compatible, yet it is featured at 98% in the README.
- **Selection and qualification share evidence.** Seed-0 tuning-tier PASSes count toward qualification.
  - The calibration criteria adopt a gate from 3 lineages (1 positive, 2 negatives). The 95% upper bound on false accepts at 0/2 is about 0.84.
  - Ties are broken by configuration hash.
- **Missing uncertainty.** Apart from the older RpGAN study (5 seeds, bootstrap CIs, multiplicity caveat; the best-quality section in FINDINGS.md), headline numbers have no uncertainty.
- **FID setup.** CIFAR FID is computed on train statistics with an ImageNet-feature critic, which is a known source of Inception-FID bias. There is no FD-DINOv2 or test-split cross-check.

**Clarity:**
- **README order.** Six paragraphs of Forge, release and throughput narrative come before the minimal example.
- **The 100-Gaussian GIF shows the default failing** (42.1% within 3σ) right before E22's 98%, so a newcomer can't tell which to use.
- **Acronyms are never expanded:** BISPF, SBNP, BCAP, K3P, KA2, E22, UCD, DV12 and "hq" are missing from RESEARCH_GLOSSARY.md.
- **FINDINGS.md is mislabeled.** It is titled after RpGAN gradient-penalty centering, mentions a "world-model repo", stops at 2026-10-06, and cites four missing reproduction files.

**Contradictory "current" pages:**
- `reports/toy100/README.md` says K3P is 22/22; the inventory says K3P is stale at 20/29.
- `transfer_suite` names b_cap3 as the baseline.
- `cifar-ddgan` says "both GPUs free".
- There are about 711 tracked leaderboard files, against the "one current leaderboard per goal" rule.

**Hygiene:**
- **Size:** 1.80 GB tracked, 2.0 GB `.git`.
- **Largest directories:** `reports/forge` (758 MB) and `reports/toy100` (533 MB).
- **Python under `reports/toy100`:** 1,979 `.py` files, with 963 byte-identical copies under `reports/` (e.g. `training.py` ×45).
- **Binaries:** GIF 638 MB, NPZ 277 MB, JSON 490 MB (including a 24 MB `technique-inventory.json` and 74 MB of `technique-evidence/*.json`), 105 `.pt` files, no LFS.
- **Ignored but tracked:** 147 files are tracked despite `.gitignore`, contrary to "never force-add ignored bulk logs".
- **Branches:** 827 total, 469 unmerged into develop.
- **Unreferenced files:** a 19.6 MB root GIF (`100gaussians_no_particles.gif`) and `five_modes.gif` are unreferenced.

## Recommendations (prioritized)

1. **Fix the provenance breaks before running anything else.** These are defects #1–#4, #6–#8 and #14. Each one means a receipt or checkpoint does not describe what actually ran.
   - Add an `unwrap_prior()` helper.
   - Store manipulator configuration in checkpoints and receipts.
   - Make Simplex initialization deterministic.
   - Make SBNP honour its declared weight.
   - Pass `generator` to BISPF `d_step`.
   - Bind candidate fingerprints in `qualify`.
   - Delete the hard-coded narrative in `compile_reports`.
   - Reject threshold-less tasks.
   - Add a regression test for each.
2. **Turn BISPF's shape and step gates into declared Recipe fields** (#5, #12). Then re-qualify BISPF under its real recipe, because existing evidence is bound to undeclared behaviour.
3. **Run the missing ablation.** Compare the default recipe against a frozen particle table and against N(0, I), matched on architecture, data, budget, seed 0 and the deterministic initializer, across the full 49-test suite and the CIFAR U-Net.
   - Publish it as one leaderboard, with a fixed, dated test list and the PASS/FAIL/unmeasured counts shown.
   - Label external baselines "untuned" or give them equal search budgets.
   - Add bootstrap or oracle noise floors to gate metrics, and vary eval latents across terminal observations. Neither is a seed experiment.
4. **Settle the Forge boundary.** Make `particlegan_forge` a facade over `experiments/forge` or remove it. Stop `exec`-patching `GANTrainer._step`, and move graders out of `reports/`.
5. **Make the test suite catch defects.**
   - Rewrite e2e tiers 2–4 against real validators.
   - Mark GPU tests with `@pytest.mark.cuda` and deselect them by default.
   - Run pytest on 3.11.
   - Add direct tests for `feature_cells`, `routing` execution/evidence, `recipe_schedules`, `PriorProximalOptimizer` and `r2_points`.
   - Fix #10 with try/finally plus an `abort_pending()`.
6. **Shrink the repo and its sources of truth.**
   - Archive `technique-evidence`, NPZ/GIF/HTML/`.pt` bulk and toy100 source snapshots using the existing `log-archive.json` pattern, keeping blob IDs.
   - Untrack the 147 ignored files.
   - Banner every superseded README to point to the single current inventory.
   - Prune merged branches.
7. **Refactor for readability, after #1–#5.**
   - Split `feature_cells.py` and `bispf.py`.
   - Write a shared kNN/BH/moments utility.
   - Merge the settle-test and birth–death forks.
   - Derive variant flags once.
   - Reorder the README (what it is → install → minimal example → one controlled results table).
   - Add an acronym table to the glossary.
   - Give EXPERIMENTATION.md a one-page quickstart.

## Remediation status (2026-10-26)

The review was worked in 14 pull requests against `develop`, all merged:
#480 (`particlegan_forge`), #481 (core prior provenance), #482 (metric uncertainty), #483 (optimizers, row evidence, KA2, feature cells), #484 (docs), #485 (BISPF declared behaviour), #486 (test suite), #487 (optimizer solver refactor), #488 (`feature_cells` split), #489 (Forge boundary), #490 (`bispf` split and shared utilities), #491 (repo hygiene), #492 (prior ablation and BISPF re-qualification), #493 (one leaderboard per goal), plus the integration PR that adds this section.

### Findings

| # | Status | PR | Resolution |
|---|---|---|---|
| 1 | fixed | #481 | `unwrap_prior()` everywhere; the trainer classifies the storage prior, so a wrapped MoG prior gets, consumes and checkpoints `prior_noise_generator`. |
| 2 | fixed | #481 | Manipulator `config()`/`manipulator_spec()` saved as module extra state and checked on load (mismatch raises; legacy states without it still load). Not added as Recipe fields. |
| 3 | fixed | #481 | Simplex projection from the deterministic `_qr.semi_orthogonal` source (`init_seed`) or an explicit `generator`; no global-RNG draw; the `learnable_projection=False, z_dim=None` crash is fixed. |
| 4 | fixed (declared) | #485 | Every SBNP weight rule is a declared `BISPFGates` field. The defaults keep the legacy law; `sbnp_small_latent_weight=None` and `sbnp_unanchored_weight_enabled=False` honour the declared weight. |
| 5 | fixed (declared) | #485, #492 | Shape/step gates are `BISPFGates` fields under `Recipe.bispf_shape_gates` / `bispf_gate_overrides`. #492 re-qualified the default (all 29 recorded verdicts reproduced) and measured gates-off (A4). |
| 6 | fixed | #480 | `qualify()` binds task and candidate fingerprints, RNG version, engine/adapter source hashes, device and seed; adapter exceptions become INVALID receipts. |
| 7 | fixed | #480 | `compile_reports` is receipt-derived and writes only under `reports/particlegan_forge/`. |
| 8 | fixed | #480 | `TaskSpec` rejects missing or non-numerical thresholds. |
| 9 | fixed | #480 | `particlegan_forge` uses `configs/`, `reports/` and `runs/particlegan_forge`; declarations carry a namespace marker. |
| 10 | fixed | #483, #487 | Shared propose/adjust/commit with `try/finally` and a public `abort_pending()`; #487 made it the single wrapper template. |
| 11 | fixed (declared) | #483 | Relabelled as a sphericity statistic (still the default; e22 is qualified on it). `row_evidence_statistic='hotelling'` is a declared option; untestable configurations warn and set `.untestable`. |
| 12 | partial | #485 | Trainer readers use a trainer-owned thermal source. Caller-owned loops in `experiments/train_*.py` and `toy_particle_native_2d` still read the legacy module-global channel (kept bit-identical). |
| 13 | partial | #485, #492 | `BISPFEngine(ot_target_column_normalize=True)` exists, but no recipe builds `BISPFEngine`, so #492 could not measure it (arm A5). |
| 14 | fixed | #485 | `BISPFLossAdapter.d_loss` draws from a declared, checkpointed stream. #492 found no effect on Forge runs. |
| 15 | fixed | #481 | `postprocess_generated` applies to critic fakes, G fakes, replay, `sample()` (live and EMA) and served output. |
| 16 | fixed | #481 | Non-additive manipulators are rejected with `strict_progress` at construction. |
| 17 | fixed | #481 | `sample_indices` returns clones; `initial_cursor` and `num_particles` are validated. |
| 18 | fixed | #480 | A tier with only diagnostic/held tasks is HELD and does not block higher tiers. |
| 19 | fixed | #480 | Missing adapter evidence grades INVALID; `NamedStreams.load_state_dict` checks seed, version and runtime; `deterministic_scope` uses `warn_only=False`. |
| 20 | fixed (declared) | #483 | Confirmed (steady-state alpha 0.098 at trust 0.9). `ka2_trust_mode='scale'` is a declared option; the default stays `compound` because e22 is qualified on it. |
| 21 | fixed (residual documented) | #483 | `maybe_apply` restores counters and evidence on failure. Stream draws and rows already moved are not rolled back; the docstring says so. |
| 22 | fixed | #483 | Frozen parameters are excluded from `autograd.grad` and get zero normals. |
| 23 | fixed | #489 | The binding cache key includes override values. |
| 24 | fixed | #483, #485, #490 | `flagged_rows_sum` counted once per update (#483); cooler docstring (#485); `assert` invariants raise `RuntimeError` and the `birth_death` docstring is corrected (#490). |

### Recommendations

| Rec. | Status | PR | Notes |
|---|---|---|---|
| 1. Provenance breaks | fixed | #480, #481, #485 | Each defect has a regression test that fails on the pre-fix code. |
| 2. Declare BISPF gates, re-qualify | fixed | #485, #492 | Under the ordinary revision-14 study BISPF stops at Tier 1, so it keeps its revision-9 qualification. |
| 3. Matched prior ablation, uncertainty | partial | #492, #482, #489 | Ablation done (toy suite and CIFAR, seed 0). Oracle noise floors and a versioned eval-draw contract (#482); fixed dated 49-test denominator, Clopper–Pearson calibration bounds and an "untuned" label for external baselines (#489). Equal search budgets for baselines are not done: no per-baseline budget metadata exists. |
| 4. Forge boundary | partial | #480, #489 | No more `exec` patching (`GANTrainer(serve_after_step=...)`), the grader moved to `benchmarks/`, the resolution cache is an explicit scope. `particlegan_forge` is hardened and namespaced but is still a separate package rather than a facade over `experiments/forge`; one-off publication modules were not moved (each is still a CLI, worker or archived-source reference). |
| 5. Test suite catches defects | fixed | #486, #483, this PR | e2e tiers use real validators; CUDA tests are opt-in (`-m cuda`); 3.11 runs pytest; new unit coverage; faster `det_init`. This PR repairs the `tests/conftest.py` module-name checks broken when `tests/` became a package. |
| 6. Shrink the repo | partial | #491, #484, #493 | 147 ignored files untracked, 33 unreferenced files (59 MB) archived, 160 merged branches pruned (1.80 → 1.74 GB). One current leaderboard per goal with an index (`reports/leaderboard-index.json`). Bulk archiving of `technique-evidence`, NPZ/GIF/`.pt` and the toy100 source snapshots is not done. |
| 7. Refactor for readability | partial | #488, #490, #487, #484 | `feature_cells.py` and `bispf.py` split; shared kNN/BH/moment helpers in `particlegan/_stats.py`; settle-test fork merged; variant flags derived once; README, glossary acronyms and Forge quickstart done. The birth–death fork stays (its constructor, checks and checkpoint config differ). |

### Default-behaviour changes

Qualified recipes keep bit-identical default trajectories. #490 checked 18 trajectory hashes, #488 the feature-cell harness and #487 277 optimizer trajectories. #492 reproduced every recorded BISPF verdict. The changes below are RNG-isolation, checkpoint or crash fixes, or apply only to configurations no qualified recipe uses:

- **Manipulated priors (#481).** A wrapped MoG prior now uses and checkpoints `prior_noise_generator`. Simplex initialisation no longer touches the global RNG. A frozen prior also freezes router parameters. Postprocessing reaches every critic and serving path. Manipulator specs live in the checkpoint extra state. No qualified recipe uses manipulators.
- **BISPF (#485).** `d_loss` uses a declared, checkpointed stream, and each trainer reads its own thermal record instead of a process-wide cache. Interleaved BISPF trainers in one process now evolve independently; a single trainer is bit-identical.
- **`particlegan_forge` (#480).** New paths, schema-2 receipts, INVALID for missing evidence, strict deterministic scope. Older receipts without a binding read as NOT_RUN.
- **Failure paths (#483).** Optimizer wrappers, `ConstraintGeometryOptimizer` and `maybe_apply` recover from a failed step. The `RoutedEvidence.flagged_rows_sum` diagnostic changes for routed runs.
- **Receipts (#482, #489).** New reporting-only fields: noise floor, `evaluation_contract`, calibration CIs and holdout claims. Gate values and verdicts are unchanged.
- **Tests (#486).** Default pytest deselects CUDA tests, and the thread budget follows `OMP_NUM_THREADS`/`MKL_NUM_THREADS`.

### Declared candidate options (default = legacy law)

- `Recipe.bispf_shape_gates=False`; `Recipe.bispf_gate_overrides` for `sbnp_small_latent_weight=None`, `sbnp_unanchored_weight_enabled=False`, `late_thermal_enabled=False`, `dense_g_rescale_enabled=False` and the cooler default query (#485).
- `BISPFEngine(ot_target_column_normalize=True)`. Engine-only, so no recipe can measure it yet (#485).
- `Recipe.prior_code_law="fresh_gaussian"`, the N(0, I) latent control (#492).
- `row_evidence_statistic="hotelling"` and `optimizer_options={"ka2_trust_mode": "scale"}` (#483).
- `evaluation.eval_draw_contract="per_observation_v2"`, which starts a new evaluation cohort (#482).
- `GANTrainer(serve_after_step=False)` (#489).
- `PriorManipulator(penalty_weight=...)` and `SimplexRoutingManipulator(init_seed=..., generator=...)` (#481).

### Science outcome

Full report: [prior-ablation-2026-10-25](prior-ablation-2026-10-25/README.md) (#492). All results are one-seed diagnostics (protocol seed 0) and grant no qualification credit.

- **Toy suite.** On the 15 tests every prior arm graded, the scores are 7 PASS for the learned table, 5 for the frozen table and 5 for fresh N(0, I). The difference is `ring16_acquisition` and `mode_hold`, both at a +0.00 margin, so it is not established. The learned table gives more precise samples: on ring16, 0.95 of samples are near a mode, against 0.61 frozen and 0.53 N(0, I). All arms fail the three 100-mode grids.
- **Re-qualification.** The default reproduces all 29 recorded verdicts. Turning the shape/step gates off loses 4 graded passes (6 counting an INCOMPLETE and its dependent).
- **CIFAR-10.** The particle DDGAN host ran 10k updates, scored by FID50k of the final EMA weights:

  | Arm | FID50k |
  | --- | ---: |
  | Learned table | 79.7 |
  | Frozen table | 108.5 |
  | Fresh N(0, I) | 82.7 |

  Evaluation noise is at most 0.21. Training noise is far larger: FID5k swings by up to 36 within a run. The ranking is not resolved. Every arm is far from the archived 49.5, which is a separate cohort (different seed and source).

### Integration fixes (this PR)

- `tests/conftest.py`: the archived BCAP comparator adapter and the renderer clock isolation matched `request.module.__name__` against bare file names. Since `tests/__init__.py` was added (f10674eb6, before this review), pytest imports `tests.<name>`. The adapter was silently skipped, which caused all 33 `test_bcap_integration_compatibility` failures. The fix matches on the file name. The adapter also gains the three new inactive Recipe defaults (`bispf_shape_gates`, `bispf_gate_overrides`, `prior_code_law`). The archived comparator bytes and all its assertions are unchanged.
- `particlegan/kinetic_transport.py`: #490 added a two-line comment. That changed the sha256 pinned by 44 Forge task and variant cards: the word and behaviour hosts and `policy_cloud_coverage_v1`. The comment now lives in `particlegan/_stats.py`, and the file is byte-identical to its pinned source again. No task card or qualification record was rewritten. This fixed the failures in `test_forge_word_metadata_migration`, `test_forge_views` (frozen thresholds), `test_sigreg` (source refresh), `test_forge_profile_dispatch` and `test_forge_taskrecipes`.

### Remaining follow-ups

- Connect `ot_target_column_normalize` to a recipe field so it can be measured, or drop it (#13).
- Bind the thermal source explicitly in the caller-owned loops (`experiments/train_*.py`, `toy_particle_native_2d`) (#12).
- Make `particlegan_forge` a facade over `experiments/forge`, and retire the one-off publication modules (rec. 4).
- Register per-baseline search budgets so external baselines can be tuned equally rather than labelled "untuned" (rec. 3).
- Archive the remaining bulk: `technique-evidence`, NPZ/GIF/`.pt` and the toy100 source snapshots (rec. 6).
- **CIFAR prior question.** Use a longer horizon or an FID averaged over the last checkpoints, not seed repeats. Check why the seed-0 DDGAN host is at about 80 FID against the archived 49.5. Add an exact resume path to the CIFAR host.
- **Forge execution defects listed in #492:**
  - AE model-toy snapshot reads
  - `zero_tanh` routing
  - MisGAN/BiGAN vector target declarations
  - native profiles for `grid100_*`
  - five-word wall-clock headroom
- `configs/forge/catalog.json` drifts on every merge that changes tracked files. Rerun `python -m experiments.forge history` with the changes staged, then `forge compile --summaries-only`.
