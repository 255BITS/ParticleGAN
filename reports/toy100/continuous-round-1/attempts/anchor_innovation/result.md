# Anchor innovation (attempt 3603386)

Lane: critic-memory innovation on the pinned K3P parent. The parent files under `reports/toy100/gap-fill-20260925/sources/k3p` were copied and not edited. Parent scores were reused and not rerun: ring hold 1200/1200 (min HQ 0.90723), extension 300/300 (min HQ 0.98779), target-shift recovery 28/81 with delay 1130.

Best candidate this round is **innov3**. It keeps the hold and the 300-update extension, with a higher minimum HQ than the parent on this one ring run, and it fails recovery more completely than K3P (0/81, never stable). It does not outrank K3P. No candidate advanced to the 22 toy gates.

## Leaderboard

| Rank | Candidate | Hold | Extension | Timely recovery | Toys | Notes |
|---|---|---:|---:|---|---|---|
| — | K3P parent (not rerun) | 1200/1200, min HQ 0.90723 | 300/300, min HQ 0.98779 | FAIL 28/81, delay 1130 | 22/22 prior evidence | Schedule-gated anchor |
| 1 | innov3 | **PASS** 1200/1200, min HQ 0.99316 | **PASS** 300/300, min HQ 0.99121 | **FAIL 0/81**, delay none | NOT_RUN | Reference speeds up when the anchor residual is large |
| 2 | innov1 | **FAIL** NOT_CONVERGED, max 6 modes | NOT_RUN | NOT_RUN | NOT_RUN | Full anchor from call 2; grad/Adam follow stayed near 0 |
| 3 | innov2 | **FAIL** stopped at step 3650 on a 5-mode plateau, HQ 0.997, streak 0 | NOT_RUN | NOT_RUN | NOT_RUN | Strength stayed high (mean 0.67). Driver terminal write not reached |

Qualification gates are the ring hold and the target-shift driver. innov2 was stopped during settling after modes had been 5 and HQ about 0.99 from step 2500 through 3650, so an 8-mode confirmation was not in progress. That is a real failure, not a finished `NOT_CONVERGED` receipt.

## What was tested

All three proposals keep K3P's optimizer floors (network 0.01, prior 0.05), particle response, sparse-latent rule, critic guard, architecture, seed 1234, and the `a_r1r2` / `b_cap` mix `s(r)`. The anchor penalty is applied on every critic penalty call, including while `s == 1`, so it is not waiting on the learning-rate clock. Inherited noise is still the K3P horizon schedule. These are intermediate ablations, not horizon-free formulations.

**innov1 — reference speed from Adam innovation.** `follow = clamp((grad_rms/adam_rms - 1) / 4, 0, 1)` and `alpha = 0.001 ** (1 - follow)`. Prox weight is always full. Measured follow mean 0.020 (max 0.57) and final alpha 0.001, so the reference never left K3P's slow EMA. The new piece that actually ran was a full anchor from penalty call 2. The ring reached at most 6 modes (parent has 8 by step 600) and the 4800-check settling window failed every check. Extra critic forwards: 6298.

**innov2 — penalty strength, fixed decay 0.999.** `w = R1 / (R1 + P)`, detached, times the anchor residual. `w` mean 0.67 because `P` and `R1` were the same order during acquisition, so most of the anchor stayed on. Same acquisition failure: 4 modes at step 600, then a stable 5-mode state. Stopped at step 3650 to spend the last proposal on a rule whose signal had actually moved.

**innov3 — full anchor, reference speed from the residual.** `follow = P / (P + R1)`, `alpha = 0.001 ** (1 - follow)`. This is the signal innov1 missed. Early `P` was large versus `R1`, so the reference caught up and `P` fell to about 0.001 by step 600. The ring then matched the parent's acquisition: 8 modes and HQ 1.0 at step 600, convergence at step 1400, hold 1200/1200 with min HQ 0.99316, extension 300/300 with min HQ 0.99121. Seconds 134. Extra critic forwards: 2898. Final mixing weight `s` was 0. Mean follow 0.227, mean alpha 0.0086 (K3P's alpha is 0.001).

## Target shift (innov3 only)

Driver: existing `shift.py`, shift at update 2400, deadline step 2800, 81 checks required. Seconds 139.

| Check | Result |
|---|---|
| Continued hold, updates 1200–2400 | 120/120, min HQ 0.95435 |
| Stationary window inside this run (steps 1000, 1050, 1100, 1150, 1200) | FAIL. Steps 1050/1100/1150 are 5 modes HQ 0.652, 7 modes HQ 0.814, 7 modes HQ 0.781. The separate hold driver, which scores the dense window from step 1201, still passed |
| Deadline window, steps ≥ 2800 | **0/81**, `deadline_pass` false |
| Passing checks after the shift | **0/120** |
| Recovery delay | none (`stable_from_step` null) |
| End of run, step 3600 | live 4 modes, HQ 0.488; EMA 5 modes, HQ 0.557 |
| Best post-shift point | 6 modes, HQ 0.749 at step 3400, then a decline |

After the shift, modes went 8 → 0 immediately, then 1, 3, 5, 6, and back to 4. The output was not a frozen pre-shift sample.

Optimizer audit: both optimizers had 2400 Adam updates at the shift and 3600 at the end. No counter reset. G and D learning rates spanned 4.25e-5 to 0.00425 (the 1% floor to the initial rate). Prior spanned 4.25e-4 to 0.0085. At the shift, rates were already on the floor. `s` was 0. Follow rose from 0.127 at step 2400 to 0.498 at step 2500 (alpha 0.031), then eased to 0.136 by step 3600. The reference did speed up when the residual grew, and the critic and generator kept stepping. That motion never reacquired 8 modes. K3P's slow anchor, on the parent run, did reacquire late (28/81, stable at update 3530). Speeding the reference removed that late recovery and did not create a timely one.

A matched frozen control was not run. The live recovery did not pass, so there was no pass to confirm. Sensitive screens (mode_hold, unequal mass, unequal width, stripes), the 22-toy suite, and the delayed/repeated-change stress check were not run.

## Budget dependencies that remain

The anchor rule does not read the training horizon, step index, scores, targets, or shift time. The inherited schedule still does, and on this ring driver the binding horizon is the hardcoded `noise_horizon=1200` in `hold.py` and `shift.py`, not config `total_steps` 7000.

`FixedControl` is constructed with that 1200-step horizon. `policy_multipliers` then uses `min(control_total, network_lr_horizon_cap)`. With the cap at 1600, the min is 1200, which matches the measured G/D floor at update 1200. Raising config `total_steps` or the cap does not change those rates. Lowering the cap below 1200, or changing the driver's 1200, does. Evaluated from the schedule formula, not from a second training: at step 500 the network multiplier is 1.0 for caps 1600 and 3200; at step 1200 it is 0.694 if the binding horizon is 1600 and 1.0 if it is 3200. A prefix before the anneal start (step 720 when the binding horizon is 1200) is rate-identical across larger horizons. From step 720 on, it is not.

Two further schedule pieces stay horizon-dependent and were not removed:

- Input and output noise still anneal on the driver's 1200-step horizon.
- The `a_r1r2` / `b_cap` mix `s` is still `last critic LR / max critic LR`. innov3 stops gating the anchor residual by `s`, and it does not stop gating A versus B by that clock.

No two-horizon training replay was run. The candidate is not a horizon-independent result.

## Replay

Environment on every benchmark process: `CUDA_VISIBLE_DEVICES=GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`. Python: `/tmp/pr38-default-env/bin/python`. Frozen repo: `/ml2/hypergan/gan-attempts/claude-pool-20260925T041936Z/qualify_a2_bounded_damp/20260925T041936Z-2509885/repo/reports/toy100/qualify-a2-attempt/prepared/repos/cuda`. Fixture: `/ml2/hypergan/ParticleGAN-epsilon-gan-followup/reports/toy100/direct-particle-base/initialization-fixtures/mode_hold/initial-values.pt`. Floors: `--network-floor 0.01 --prior-floor 0.05 --backend cuda --task mode_hold`.

```sh
# innov3 hold + 300-update extension (PASS)
/tmp/pr38-default-env/bin/python -u \
  reports/toy100/anchor-innovation-3603386/innov3/hold.py \
  --repo <frozen repo> --config reports/toy100/anchor-innovation-3603386/innov3/config.json \
  --task mode_hold --backend cuda --initial-state <mode_hold fixture> \
  --output <fresh dir> --network-floor 0.01 --prior-floor 0.05

# innov3 target shift (FAIL 0/81)
/tmp/pr38-default-env/bin/python -u \
  reports/toy100/anchor-innovation-3603386/innov3/shift.py \
  --repo <frozen repo> --config reports/toy100/anchor-innovation-3603386/innov3/config.json \
  --task mode_hold --backend cuda --initial-state <mode_hold fixture> \
  --output <fresh dir> --network-floor 0.01 --prior-floor 0.05
```

Paths above are relative to this attempt's repo. innov1 and innov2 use the same drivers with their own directories.

## Artifacts

- Sources: `repo/reports/toy100/anchor-innovation-3603386/innov{1,2,3}/` (`mechanism.py` is the only modified file in each; siblings match the pinned K3P hashes).
- innov1 hold: `repo/reports/toy100/anchor-innovation-3603386/runs/innov1-hold/result.json` (275.74 s).
- innov2 hold log (no `result.json`; process stopped): `repo/reports/toy100/anchor-innovation-3603386/runs/innov2-hold.log`.
- innov3 hold: `repo/reports/toy100/anchor-innovation-3603386/runs/innov3-hold/result.json` (133.96 s).
- innov3 shift: `repo/reports/toy100/anchor-innovation-3603386/runs/innov3-shift/result.json` (138.74 s).
- Gate log: `tests.jsonl` beside this file.

## Next mechanism

Do not add a full anchor during the first several hundred updates unless the reference actually tracks. Grad-versus-Adam does not do that here (follow stayed near 0 while modes were missed). Scaling the penalty by `R1/(R1+P)` also left the anchor mostly on and missed modes.

innov3 shows a slowly adapting reference can stay precise on a fixed ring: hold and extension both passed, with minimum HQ above the parent's single-run figures. The same residual-triggered speedup fired after the target change (follow about 0.50 one hundred updates later) and the run still had 0 passing recovery checks, ending at 4 modes. Chasing the reference when the residual is large gave up K3P's late partial recovery without meeting the 400-update deadline. The floor rate (1% of the initial G/D rate) was unchanged and is still a live hypothesis for why post-shift motion stayed short of 8 modes. A later proposal should keep acquisition free of a stale anchor and should not treat a large residual as a reason to glue the reference to the critic during a shift.
