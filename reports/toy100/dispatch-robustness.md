# Toy100 dispatch robustness (#107, #140, #143)

Robustness audit only. No new mechanism and no retune. The Intel-path stay ordering below is one seed on one CPU vendor branch. It is not a 22/22 result and it is not a production configuration. `LD_PRELOAD` of `mkl_serv_intel_cpu_true` is a diagnostic, the same one Codex used, and is not a deployment fix.

## Host and the two paths

Every receipt is `reports/toy100/continuous-evidence/dispatch-robustness/**/dispatch.json` (and `*.dispatch.json`).

| field | value |
| --- | --- |
| vendor | GenuineIntel |
| model | Intel(R) Xeon(R) Processor, family 6, model 207, stepping 2 |
| torch | 2.14.0+cpu, ATen capability AVX2 |
| threads | intra-op 1, interop 4, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` |
| caps | `ATEN_CPU_CAPABILITY=avx2`, `MKL_ENABLE_INSTRUCTIONS=AVX2`, oneDNN ISA vars unset |

`mkl_serv_intel_cpu_true()` is **1** on the `intel` preload and **0** on the `amd` preload. Those integers are on every receipt.

This machine's ordinary MKL dispatch (no `LD_PRELOAD`) is the Intel path. A 3-update prefix of #140 with no preload matches the return-1 shim record for record: update-1 critic sharpness `0.38521575927734375`, G curvature factor `0.10629096826749508`. The column labeled `amd` below is therefore the forced non-Intel vendor branch (`reports/toy100/mkl_nonintel_dispatch_probe.c` returns 0), run on this Xeon because omitting the preload cannot leave the Intel path here. It is not a Ryzen measurement, and it does not reproduce Codex's ordinary-AMD trace (that trace had sharpness `0.3852115273475647` and G factor `0.10636833119977061`).

Commands are the Codex audit environment (unset `PYTHONPATH`, `ONEDNN_MAX_CPU_ISA`, `DNNL_MAX_CPU_ISA`; one thread; AVX2 pins; empty `CUDA_VISIBLE_DEVICES`) via `reports/toy100/run_dispatch_robustness.sh`. Pins: #107 `reachstall` and #140 `delayg05` at `ef4084a4`; #143 `holdw15` at `3384976c`. Seeds 1 and 2 override mode-hold, trajectory `PROTOCOL`, and `NoisePolicy` only. Seed 0 through that hook matches the published trajectory `identity_mse` `0.0009426682372577488`.

The published cold command stops the ring when trajectory fails. Seeds 1 and 2 fail trajectory on both paths, so their rings were a second run of the same probe with `--tasks mode_hold`. Stay 1210–2400 ran only where warm identity, warm method, cold trajectory, and cold ring all passed: Intel path, seed 0, all three candidates.

## Acquire table

Through step 1200 the three candidates match on every path × seed (same warm counts, same trajectory MSE, same ring curve). The dispatch split is in the shared training path, before any of these arms.

Warm counts are steps 1001–1200, pass = 8 modes and HQ ≥ 0.9. Ring "ckpt" is the step-1200 observation. "Raw" is the terminal draw stored on the mode-hold receipt.

| path | seed | warm identity | warm method | trajectory | ring ckpt 1200 | raw terminal |
| --- | --- | --- | --- | --- | --- | --- |
| intel | 0 | 200/200, min 8, HQ .990 | 200/200, min 8, HQ .921 | PASS, mse 0.000942668 | 8 / .9988, missing [] | 8 / .9172, missing [] |
| intel | 1 | 0/200, min 7, HQ .988 | 0/200, min 7, HQ .726 | FAIL, mse 0.257848 | 8 / .9983, missing [] | 8 / 1.0, missing [] |
| intel | 2 | 0/200, min 6, HQ .927 | 0/200, min 4, HQ .333 | FAIL, mse 0.306294 | 6 / .620, missing [0, 1] | 6 / .596, missing [1, 2] |
| amd (forced 0) | 0 | 0/200, min 3, HQ .671 | 0/200, min 3, HQ .370 | PASS, mse 0.000941636 | 7 / .977, missing [1] | 6 / .758, missing [0, 1] |
| amd (forced 0) | 1 | 200/200, min 8, HQ .990 | 194/200, min 8, HQ .867, fails 1155–1158 and 1160–1161 | FAIL, mse 0.273074 | 8 / .999, missing [] | 6 / .668, missing [4, 5] |
| amd (forced 0) | 2 | 0/200, min 6, HQ .979 | 0/200, min 6, HQ .919 | FAIL, mse 0.313491 | 8 / .999, missing [] | 8 / 1.0, missing [] |

#107 does fail acquisition on the forced non-Intel path at seed 0: warm identity is 0/200 (min 3 modes) and the cold ring does not finish at 8 modes. The failure is not one 7-mode endpoint repeated on every seed.

- Seed 0 ends at 7 modes on the step-1200 checkpoint (missing mode 1, HQ 0.977) and at 6 modes on the terminal draw (missing 0 and 1, HQ 0.758).
- Seed 1's step-1200 checkpoint is 8 modes at HQ 0.999, while the terminal draw is 6 modes missing 4 and 5.
- Seed 2's ring passes at 8 modes, HQ 1.0, on the forced non-Intel path. The Intel path at seed 2 is the one that ends at 6 modes.

Codex's Ryzen ordinary-AMD receipt was a stable 7-mode ring. This forced-0 run on GenuineIntel does not reproduce that record.

## Stay, only where acquire passed

Intel path, seed 0. Terminal 1000–1200 is `(1000, 8, .999), (1050, 8, 1.0), (1100, 8, .9993), (1150, 8, .9995), (1200, 8, .9988)` for all three, matching the published Intel-path receipt.

| candidate | stay 1210–2400 | min modes | min HQ | final | failing steps |
| --- | --- | --- | --- | --- | --- |
| #143 holdw15 | 115/120 | 7 | 0.622314 | 8 / 1.0 | 1560, 1810, 2110, 2210, 2380 |
| #140 delayg05 | 114/120 | 5 | 0.3625 | 8 / 1.0 | 1820, 1850, 2180, 2190, 2200, 2260 |
| #107 reachstall | 97/120 | 0 | 0.0 | 8 / 0.9714 | 23 steps, including 1770 and 1820 at 0 modes; last fail 2150 |

Forced non-Intel cells did not pass acquire, so stay was not run there. The 115 > 114 > 97 order is the Intel-path seed-0 order. It was not remeasured on the other branch.

## Where the paths diverge

Same #140 factory, seed 0, one mode-hold update, both preloads. 5704 ATen ops.

The first update that differs past 1e-6 is update 1. Inside it:

1. First kernel that is not bit-identical: `aten::addmm` at op 47, max abs `1.19e-7` (3669 of 12288 elements differ, none by more than 1e-6). The stack is `F.linear` in the generator MLP (`benchmarks/locked_shared/mlp.py`, first `Linear`). No nested `aten::mm` sits under that call, so the split is inside the addmm GEMM, which is the kernel that consults `mkl_serv_intel_cpu_true`.
2. First output whose max abs exceeds 1e-6: `aten::reciprocal` at op 717, max abs `13.47`. The three ops before it are `sqrt`, `add`, `_to_copy`, and those differ by `2.8e-9`. That is a reciprocal of a near-zero `sqrt(v) + eps` denominator (Adam's second-moment scale). Later in the same update a reciprocal reaches max abs `1.70e5`. 372 floating outputs in this update exceed 1e-6.
3. Particle `z` at the end of update 1 differs by max abs `1.19e-7`. The candidate's own sharpness scalar crosses 1e-6 at phase 2 of update 1: `0.38521575927734375` vs `0.3852182924747467` (delta `2.53e-6`). Phase-0 sharpness in that update differs by `4.47e-7`.

Mode count, on the cold ring's 50-step grid (`reachstall`; the other two candidates match):

| seed | first checkpoint where mode count or missing-mode set differs | checkpoints that differ |
| --- | --- | --- |
| 0 | step 50: Intel 0 modes, forced-0 has 1 mode | 23/24 |
| 1 | step 50: Intel 1 mode, forced-0 has 0 | 20/24 |
| 2 | step 150: Intel 1 mode, forced-0 has 0 | 22/24 |

The ring loss is a systematic kernel bias that the update amplifies, and the mode outcome depends on the seed. The GEMM error is about 1e-7 and always in the same direction at that first `addmm`. The reciprocal then turns a 1e-9 denominator error into an O(10) change before update 1 finishes. The mode curves separate at the first or third checkpoint and stay apart for the rest of the ring. That is not a single late coin-flip at the 7-vs-8 boundary (seed 0's Intel ring is already at 8 by step 850; the forced-0 curve has been on a different mode count since step 50). It is also not one fixed 7-mode attractor: the missing modes are `[1]` at the seed-0 checkpoint, `[4, 5]` on the seed-1 terminal draw, and none on seed 2.

## Leaderboard and recommendation

Intel path, seed 0, stay only (the only cells that passed acquire):

1. #143 width hold 115/120, min modes 7
2. #140 delayed G Adam lr ×0.5, 114/120, min modes 5
3. #107 reach stall 97/120, min modes 0

Recommendation: leave the three candidates as they are. The Intel-path ordering is real on this host and matches the published receipts, and it does not survive the non-Intel MKL branch. Do not retune the ×0.5 learning rate or the width hold from this table. Do not ship the preload. A Ryzen run of the same commands, without a shim, is what would show whether Codex's stable 7-mode endpoint is the AMD branch itself or something this return-0 shim on GenuineIntel does not reproduce.
