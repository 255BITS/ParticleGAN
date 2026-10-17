# CB64-RA original CUDA screen retest, 2026-09-29

This protocol is frozen before the new CUDA outcomes. The coordinator owns GPU scheduling and runs each task once, serially with the four paired learned-model jobs and their saved-state replay checks. This lane prepares and collects the original frozen screen jobs. Preparation executes no quality or training workload. No external pool submission, existing-job changes, GPU1 use, seed sweep, source repair, threshold edit, fixture relaxation or reduced-budget substitute is authorized.

Candidate: `/ml2/hypergan/gan-attempts/feature-cells-config-20260929/pkg-CB64-RA`, with `configs/overrides-CB64-RA.json` in that study. The source digest is `13f5bbe4de824e6899cb28ee4dff8f35d74bbaa6243bfe9c18b8df44d173a1ce` using sorted relative Python paths, NUL, bytes, NUL. Config SHA256 is `d2b1018854671ded2ffe92b2f30c97a3871cf15911c6c241ceaadd281d94e7e7`. The per-file review freeze and tested implementation READY receipt are checked and retained by hash. The implementation's historical aggregate digest uses a different scheme; per-file identity is the comparison.

The wrapper invokes `/ml2/hypergan/lrfree-20260926/harness/screen.py` through `runpy.run_path(..., run_name='__main__')`, after placing that original harness on the import path. Its SHA256 is `ee8193adbdf09e93511befae7b6491143c26de88612eddf065cbb92eb2153c3c`. Original hosts, specs, native fixture, 1200-row mode-hold gzip batch receipt, scoring code and frozen native repository source map are hashed. There is no adapter. The original CUDA G/prior tensor and prior-range checks, construction RNG checks, per-update stream checks and scorer source checks remain active. The historical native step0 snapshot hash is a diagnostic and adds no criterion.

Options are exactly `eval_output_noise=true`, `save_final_state=true`, `strict_streams=true`, `diagnostics=true`. The other original defaults resolve to plain generation, the public GANTrainer with serial backward enabled, batch-feature-zero initialization, no image prior perturbation, and no frozen ring control. Image/native budget extension options are unset. `ABSENT`, `ABSENT_START`, `ABSENT_END` and `LRFREE_NATIVE_TEST_STEPS` are removed from the process environment.

The wrapper fixes `CUDA_VISIBLE_DEVICES=0`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, device `cuda:0`, CUDA memory fraction `.2` before entering the original screen, and one numerical CPU thread. It confirms and records physical GPU0 UUID `GPU-72c1b506-891d-b8bc-b353-e020585e1c47` through CUDA device properties; a different physical device causes a resource ERROR. It fixes `CUBLAS_WORKSPACE_CONFIG=:4096:8` and the original numerical thread environment. The original deterministic algorithm, cuDNN, TF32, matmul precision and seed settings execute in the unmodified harness. Peak allocated and reserved GPU memory and the allocator limit are recorded. At most one screen can hold the lane lock; the coordinator serializes this lane with learned-model work. The wrapper rejects repeat runs in a task output directory.

## Exact original budgets

| Task | Updates | N | z | Batch | Observations |
|---|---:|---:|---:|---:|---:|
| mode_hold | 1200 | 12 | 4 | 128 | 24, every 50 |
| img_intensity2, img_blobs4, img_stripes2, img_bars4 | 600 each | 32 | 8 | 32 | 24, every 25 |
| vector_two_broad, vector_unequal_mass, vector_unequal_width, vector_anisotropic, vector_overlap | 1200 each | 256 | 4 | 128 | 24 |
| vector_spiral | 1600 | 256 | 4 | 128 | 24, original ceil schedule |
| ring_shift | 4600 | 20000 | 2 | 2048 | 460, every 10; shift after 2400 |
| stationary | 7500 | 20000 | 2 | 2048 | 750, every 10 |
| grid100, rotated100, staggered100 | 7000 each | 20000 | 2 | 2048 | 34 |

The original portability seeds and stream construction remain fixed. Native seed is 1234, with observations at 0,1,10,25,50,100 and every 250 through 7000. All three native tasks require five 20000-sample terminal quality clouds at 6000,6250,6500,6750,7000 plus the independent 100000-sample holdout. The original holdout target/noise/latent seeds are 2835/2836/2837. The source-checked original `native100_score.py` launches the official coverage and accuracy gates in an independent process. Live noisy scoring is primary; clean and EMA results remain diagnostics.

## Verdicts and collection

Original `result.json` PASS/FAIL/ERROR is preserved exactly as `primary_status`. Collector receipts independently check source identity, original options, physical GPU0, completed budget, observation schedule, strict stream deviations, and requested final state. Mode-hold or ring historical construction RNG/reference comparisons preserve their original warning-only semantics and are explicitly reported as MATCH or MISMATCH/WARNING separately. Native receipts additionally verify the mandatory canonical G/prior hashes and range, the official source map and verdicts, event schedule and complete terminal/holdout cloud shapes. Errors before the original harness writes a result produce a marked wrapper ERROR with its traceback. Runtime, mandatory fixture and evidence errors are reported honestly.

Acceptance is the original PASS or FAIL only when the CUDA fixture/evidence is VALID. Otherwise the acceptance status is ERROR, alongside the exact original primary verdict. Portability passes only with 13 accepted PASS results. Native passes only with three accepted official noisy live PASS results including coverage and accuracy. No historical result is inherited by CB64-RA.

`run_screen.py --task TASK` is the coordinator API. Optional original API arguments have locked defaults and cannot select another package, config, device, output or option set. `--check-only` verifies static receipts and command definitions without importing torch or running training. `collect.py --task TASK` collects a finished attempt; `collect.py --all --report --manifest` writes the numerical leaderboard, report and artifact hashes. Per-task raw metrics, rates, final state, native clouds and original verdicts stay in `runs/TASK/`; coordinator logs can be tailed from the root `logs/screen-TASK.log`.

The report includes live/EMA/clean metrics, native terminal booleans, coverage and accuracy status, holdout precision and mass/center/trace/KS/eigenvalue metrics, branch counters, ordinary reaction and isolation activity, row-evidence effective sample size, and GPU memory. Screen/context elapsed times include evaluation and diagnostics and are labeled accordingly. They are not paired learned-run training-only throughput. Old CPU validation remains read only and does not decide the new CUDA verdict. Archived E22 GPU confirmations are noncontemporary comparators; new E22 native jobs are unnecessary. This lane does not claim the complete S1–S6 suite or full admissibility ceremony.
