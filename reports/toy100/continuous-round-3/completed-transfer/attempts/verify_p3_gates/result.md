# Verification report

Neither candidate is promoted. K3P stays the selected base.

RP1's transfer screen stops on a real quality failure: `img_intensity2` finishes at 2 modes and HQ 1.0, but only 3 of 24 checks pass and the passing suffix is 2. The gate requires 5 consecutive stable checks. No native, stress, or extra-seed work was started.

P3's four sensitive gates passed. Its preserved shift remains FAIL 77/81. The other 18 P3 toys stay NOT_RUN.

## RP1 transfer screen

Unchanged `rp1_signal_close`, copied from the published pin and matched to both the pin and the originating candidate. `mechanism.py` SHA256 `ed49869c6e06e1ac638cf04dba36b123aa7cb3b883650d7e16aa8843aaa22d14`. Hold, shift, and natives were not rerun. The supervisor's prior claim (hold 1200+300, live shift 81/81) is not a new measurement here.

| Gate | Status | Seconds | What the run showed |
|---|---|---:|---|
| mode_hold | PASS | 54.356 | 8 modes, HQ 0.999268; 15/24, confirmed at 700. Final rate gain 0.00934, mixing s 0, applied critic LR 8.22e-5 |
| vector_unequal_mass | ERROR | 31.574 | Unchanged `policy_rate_action` raised `lr_g diverged` at critic step 449. Gain had just become 0.99 while the applied LR was still 0.00425. No quality verdict |
| vector_unequal_width | ERROR | 56.458 | Same rate-check abort at critic step 1097, gain 0.99 versus applied LR 0.00425. No quality verdict |
| img_stripes2 | PASS | 28.968 | 2 modes, HQ 0.96875, TV 0; 23/24, confirmed at 150. Rate stayed 0.00425, mixing s 1 |
| ae_gan_hold | PASS | 13.889 | recon MSE 0.008985; 22/24, confirmed at 73. Rate stayed 0.00425 |
| cover_leftover | PASS | 41.440 | content kept 0.999024, pole errors about 0.011; 14/24, confirmed at 500. Final rate gain 0.077, critic LR 3.70e-4 |
| img_bars4 | PASS | 26.860 | 4 modes, HQ 1.0; 9/24, confirmed at 500. Rate stayed 0.00425 |
| img_blobs4 | PASS | 28.865 | 4 modes, HQ 0.9375; 14/24, confirmed at 550. Rate stayed 0.00425 |
| img_intensity2 | FAIL | 27.963 | Final cells pass (2 modes, HQ 1.0, TV 0) but confirmation fails: 3/24, suffix 2, first pass at 500, no confirmed step |
| mid_scale_identity through vector_two_broad (10 gates) | NOT_RUN | 0 | Stopped after the intensity quality failure |
| grid100, rotated100, staggered100 | NOT_RUN | 0 | Native owner lane. Not started after the quality failure |

The two vector ERROR rows are the known post-step rate check reading the controller after it had already moved. They are not quality failures. Fresh adapter reruns, after those errors were saved, passed: unequal mass HQ 0.987305, 19/24, confirmed at 500; unequal width HQ 0.982178, 21/24, confirmed at 400. Adapter SHA256 `3b60a1d9e5ab679376ccfd9efe6559c80e5eef23e4813f776349c709577cdf81`. Those reruns do not clear the intensity failure, and the original ERROR rows are still in `tests.jsonl`.

### img_intensity2 trajectory

The controller never left acquisition. Across all 600 updates, generator/critic LR stayed 0.00425, prior LR stayed 0.0085, multiplier stayed 1, and mixing weight stayed 1. Warmup was on by critic step 249. The quiet counter stayed 0, so the close never armed. Input noise went from 0.5 to 0 by the absolute step-120 rule, and output noise reached 0.029 by step 240. That schedule does not depend on the task horizon; the host still passes `network_lr_horizon_cap` 1600 into the unused multiplier arguments.

Quality oscillated for the whole budget. Modes were 0 or 1 for most of the first 475 updates. The only passing checks are updates 500 (HQ 0.906), 575 (HQ 0.969), and 600 (HQ 1.0). Update 525 dropped to 1 mode and HQ 0.531, and update 550 was HQ 0.875, so the run never held five passing checks.

Observation table: `reports/toy100/verify-rp1-attempt/traces/img_intensity2-observations.csv`. Controller samples: `reports/toy100/verify-rp1-attempt/runs/img_intensity2/mechanism-receipt.json`. Full result: `reports/toy100/verify-rp1-attempt/runs/img_intensity2/result.json`.

## P3 sensitive screen

Unchanged `p3_floor_reopen` matched `reports/toy100/continuous-round-2/sources/source-hashes.json` (13 files) and the 19 gap-fill fixtures. `precision_policy.py` SHA256 `57fca1ac270e3372a647918863d8d3fb3ac30475363d265d34849f71f26b0132`. It was the installed schedule policy on every measured gate. Training was CUDA FP32, deterministic, TF32 off, one thread.

| Gate | Status | Seconds | Live result | Confirmation |
|---|---|---:|---|---|
| mode_hold | PASS | 50.154 | 8 modes, HQ 1.0 | 15/24, confirmed at 700 |
| vector_unequal_mass | PASS | 67.833 | HQ 0.988281, covariance error 0.254390 | 19/24, confirmed at 500 |
| vector_unequal_width | PASS | 52.607 | HQ 0.980957, covariance error 0.435071 | 21/24, confirmed at 400 |
| img_stripes2 | PASS | 26.761 | 2 modes, HQ 1.0 | 23/24, confirmed at 150 |

All four ended precision-locked in dwell: mixing weight 0, generator/critic LR 0.000425, prior LR 0.000850, down from 0.00425 and 0.0085. Noise is still the K3P horizon schedule: input 0.5 to 0 and output 0 to 0.029 over the task `total_steps` (1200, or 600 on stripes). Preserved and not rerun: hold 1200/1200, extension 300/300, stationary 5/5, pre-hold 120/120, shift FAIL 77/81. The other 18 toys are NOT_RUN.

An earlier `mode_hold` row of ERROR at 2.418 seconds is an observer bug from wrapping `benchmarks.toy100.models` while it was still importing. No training step ran. That row is kept. The PASS above is the rerun.

## Replay

Python `/tmp/pr38-default-env/bin/python`. Environment: `CUDA_VISIBLE_DEVICES=GPU-72c1b506-891d-b8bc-b353-e020585e1c47`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, one CPU thread, AVX2, `PYTHONHASHSEED=0`. Frozen runtime: `/ml2/hypergan/gan-attempts/claude-pool-20260925T041936Z/qualify_a2_bounded_damp/20260925T041936Z-2509885/repo/reports/toy100/qualify-a2-attempt/prepared/repos/cuda`.

Exact argv is in `reports/toy100/verify-p3-attempt/logs/GATE.command.json`, `reports/toy100/verify-rp1-attempt/logs/GATE.command.json`, and `reports/toy100/verify-rp1-attempt/logs/GATE-adapter.command.json`. Ledger: `tests.jsonl` next to this file.

```sh
tail -f reports/toy100/verify-rp1-attempt/progress.jsonl
```
