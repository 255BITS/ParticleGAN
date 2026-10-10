# CB64-RA canonical CUDA retest

## Purpose and frozen candidate

The user restored GPU access after disabling the Codex filesystem sandbox. At 2026-09-29 22:36:28 UTC, `nvidia-smi` sees two NVIDIA RTX A6000 GPUs, NVIDIA device nodes are present, and PyTorch 2.13.0+cu126 successfully computes on physical GPU 0. The previous CPU validation is device-specific diagnostic evidence. Its failures do not determine the canonical CUDA verdict.

Test the unchanged [CB64-RA config](../feature-cells-config-20260929/configs/overrides-CB64-RA.json) and [matching package](../feature-cells-config-20260929/pkg-CB64-RA/particlegan/feature_cells.py). Reference source and all completed CPU artifacts remain read-only. The tested READY receipt and source freeze are in the previous study. No controller, recipe, threshold, seed, fixture or quality-gate changes are authorized for this retest.

Candidate config SHA256: `d2b1018854671ded2ffe92b2f30c97a3871cf15911c6c241ceaadd281d94e7e7`.

Candidate package SHA256: `13f5bbe4de824e6899cb28ee4dff8f35d74bbaa6243bfe9c18b8df44d173a1ce`, over sorted Python paths relative to `particlegan`, NUL, file bytes, NUL. Per-file maps and READY identities take precedence over aggregate hash conventions.

## Prescribed runs

1. Four learned-model runs: nonlinear 25-mode toy and MNIST, E22 and CB64-RA, each 2,000 updates at seed 314159, N1024/z128/batch128. Reuse the original architectures, saved real streams, MNIST data, evaluator, and initial CPU-constructed parameter/prior hashes. Execute training and serving on CUDA. Save the original checkpoint schedule and evaluate the original toy and raw/active image metrics. Compare equal updates and the latest checkpoint within the common training-time budget. MNIST has no invented numerical pass gate.
2. Four saved-state replay checks: two ten-update continuations from each problem/variant's CUDA checkpoint 1000. Require every semantic state section and loss, including CPU/CUDA RNG, to match exactly; exclude only the observational birth-death evaluation duration.
3. Thirteen original CUDA portability tasks, with original budgets, streams, fixtures and scorer gates: mode_hold; img_intensity2, img_blobs4, img_stripes2, img_bars4; vector_two_broad, vector_unequal_mass, vector_unequal_width, vector_anisotropic, vector_overlap, vector_spiral; ring_shift; stationary.
4. Three native CUDA tasks: grid100, rotated100 and staggered100, seed1234/N20000/z2/batch2048, all 7,000 updates, 34 observations, five terminal 20,000-sample checks and an independent 100,000-sample holdout. Use original noisy/live coverage and accuracy scorers. Clean/EMA results are diagnostics. Preserve canonical initial tensor and stream checks; report mismatch as invalid evidence or an execution error.

Use `/ml2/hypergan/lrfree-20260926/harness/screen.py` directly with its frozen hosts/specs/scorers. Do not use the CPU adapter or `harness-absence`. Candidate options are `eval_output_noise=true`, `save_final_state=true`, `strict_streams=true`, `diagnostics=true`; preserve original behavior defaults. Unset all absence and native reduced-budget environment variables. Do not submit jobs to the global pool. Archived GPU E22 native confirmations remain noncontemporary references; no new reference-native sweep is required.

## Resources and execution

The root coordinator serializes numerical GPU jobs across both lanes. Physical GPU 0 only (`CUDA_VISIBLE_DEVICES=0`); leave GPU 1 and all other processes untouched. GPU memory fraction .2, two CPU numerical threads for learned runs/replays and one for each original screen. Use the declared cuBLAS workspace configuration and learned-model deterministic/TF32 settings. Capture CUDA-synchronized training duration, whole-run duration, allocated/reserved GPU peaks and CPU RSS separately.

GPU 0 already hosts other work. Record hardware/resource observations; do not claim dedicated hardware or equivalence to archived timings. Four paired learned runs share the current scheduling policy. No repeated seeds, restarts to improve an outcome or source tuning after measurement. Runtime errors are recorded as errors; continue other prescribed jobs without changing the candidate or gates.

## Roles and receipts

The user requested GPT6.1 max subagents. `cuda_training` owns `learned/`; `cuda_screens` owns `screens/`; `cuda_audit` owns `audit/`. They prepare runners and declare READY before quality runs. Root owns the launcher, common source freeze, environment receipts and combined reports. Save source/config/host/scorer/data hashes before measurements and check them at completion.

Final report separates canonical CUDA acceptance, learned-model quality, correctness and earlier CPU diagnostics. A GPU benchmark PASS requires all prescribed frozen verdicts to pass with valid evidence. CPU results stay archived with their original device scope. The original CPU cost/geometry experiments remain controlled synthetic diagnostics; this phase does not silently convert their timings into CUDA scaling claims.
