# relative_update_control — ruc1, ruc2, ruc3

Pinned parent is K3P (not rerun): 22/22 toys, ring hold 1200/1200 (minimum HQ 0.90723), extension 300/300 (minimum HQ 0.98779), target-shift recovery 28/81 with delay 1130. Sources stayed hashed copies. Only `mechanism.py` changed. `latent.py`, `response.py`, and `config.json` match the pinned parent.

Host network and prior multipliers are the identity (floors 1 and 1). `policy_multipliers` returns 1 for horizons 1600, 4800, and 7000 and totals 3600 and 7500, so the cosine schedule does not move rates. Applied rate is `base_lr * response_gain * rho`, with `rho_min` 0.01 on G/D and 0.05 on the prior. Critic mixing uses the K3P map on a training signal, not last/max learning rate. The EMA anchor is that map's `(1-s)` term. Noise still uses the drivers' `noise_horizon=1200` (labeled intermediate ablation).

No toy screen, frozen control, or repeated-shift stress. Live recovery never passed, so a frozen control would not qualify adaptation. Three proposals, one GPU worker.

## Leaderboard

| Candidate | Standard hold | 300-update extension | Shift stationary | Shift window 1200–2400 | Deadline recovery | Worst live HQ in that run |
|---|---|---|---|---|---|---|
| K3P parent (published) | 1200/1200 | 300/300 | included in its shift hold | 120/120, min HQ 0.90869 | **28/81 FAIL**, delay 1130 | extension min 0.98779 |
| ruc1_energy_moment | NOT_RUN | NOT_RUN | never 8 modes | 0/120 | **0/81 FAIL** | 0 |
| ruc2_moment_leak | NOT_RUN | NOT_RUN | stuck at 6 modes, HQ 0.99 | 0/120 | **0/81 FAIL** | 0 |
| ruc3_quiet_then_leak | **1200/1200 PASS**, min HQ 0.98853, converged step 1603 | **300/300 PASS**, min HQ 0.99585 | 5/5 PASS, min HQ 0.95850 | **109/120 FAIL**, steps 1300–1400, min HQ 0.69922 | **0/81 FAIL**, no passing post-shift check | shift final 0.74951 at 6 modes |

Ranked by hold + extension + timely recovery, K3P remains ahead. ruc3 matches the parent's hold and extension on the convergence-gated driver and is behind on recovery. The 11 shift-window failures sit in the leak transition, before the convergence gate's 200-step confirmation, so the standard hold never scores them.

## What the three mechanisms did

**ruc1** set `rho` from gradient RMS over a decaying peak. The peak was the initial gradient, so by step 100 the critic rate was already 0.000577 (14% of 0.00425). Maximum modes observed: 4. Continued hold 0/120, recovery 0/120. 163 seconds.

**ruc2** leaked openness toward 0 from step 0 at rate `1-0.999`. By step 300 it had 6 modes at HQ 0.63 and then sat on that subset (step 600: 6 modes, HQ 0.9995). The parent, still at full rate, has 2 modes at step 300 and 8 modes at step 600. The shift only reached moment ratio 1.41, so `rho` barely reopened (0.30 at the logged shift sample). Maximum modes 7, unsustained. 172 seconds.

**ruc3** keeps `rho` exactly 1 for 800 quiet steps, then multiplies openness by 0.995. A moment ratio above 1.25 sets a separate boost to 1 that decays at the same 0.995 and raises `rho` without raising the openness the critic mix reads. Acquisition matched the full-rate regime: 8 modes by step 500, HQ 1.0 at the step-1200 stationary window. Anchor evaluations start at penalty call 940 (`pure_a` 939, then blend, then pure anchor from call 1720). On the shift run that is 780 blend calls plus 1881 pure-anchor calls, each with one extra real-input gradient of the EMA critic. `extra_critic_forwards` in the receipt stays 0, same as the parent implementation.

Critic applied rate and relative displacement `rms(Δθ)/rms(θ)` on the shift run:

| Update | Critic `rho` | Applied critic LR | Relative displacement | Openness (`s` input) |
|---:|---:|---:|---:|---:|
| 100 | 1 | 0.00425 | 0.08507 | 1 |
| 800 | 1 | 0.00425 | 0.01404 | 1 |
| 1200 | 0.143 | 0.000609 | 0.000821 | 0.135 |
| 2400 | 0.0103 | 4.39e-5 | 0.000214 | 0.00033 |
| 2500 | 0.995 | 0.00423 | 0.02931 | 0.00020 |
| 3600 | 0.938 | 0.00398 | 0.01621 | ~0 |

The floor step is about 400 times smaller than the step-100 update. The post-shift boost put the step back near the full rate and kept it there through step 3600, because later moment-ratio crossings reset the boost. With the anchor held on (`final_s` 0) and the step reopened, recovery passed 0 of 120 checks and ended at 6 modes, HQ 0.750. The parent's 1% floor eventually reacquired at update 3530. A sticky full-rate reopen did not.

Generator and prior rates on the hold run ended at the floors (last `rho` 0.0100 and 0.0500). The hold window minimum HQ is 0.98853 across all 1200 checks; the extension minimum is 0.99585 across all 300.

## Remaining budget dependencies

- Driver `noise_horizon=1200` still anneals input noise and warms output noise. It is not part of the update rule.
- ruc3's 800 quiet steps and 0.995 leak are fixed update-count constants. They do not read `total_steps` or `network_lr_horizon_cap`. Two 40-update prefixes with horizon caps 1600 and 4800, floors 1 and 1, matched on all 80 mobility rows and on the RNG digest (`runs/ruc3-prefix-h1600`, `runs/ruc3-prefix-h4800`). Both driver statuses stay `SETTLING` because 40 updates never reach the convergence gate.
- Inherited critic guard still waits 200 Adam steps. Anchor EMA decay remains 0.999 per critic step.

## Replay

Environment for every command: `CUDA_VISIBLE_DEVICES=GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`. Python: `/tmp/pr38-default-env/bin/python`. Frozen repo: `/ml2/hypergan/gan-attempts/claude-pool-20260925T041936Z/qualify_a2_bounded_damp/20260925T041936Z-2509885/repo/reports/toy100/qualify-a2-attempt/prepared/repos/cuda`. Fixture: `/ml2/hypergan/ParticleGAN-epsilon-gan-followup/reports/toy100/direct-particle-base/initialization-fixtures/mode_hold/initial-values.pt`.

```sh
/tmp/pr38-default-env/bin/python -u reports/toy100/relative-update-control/ruc3/shift.py \
  --repo <frozen-cuda-repo> --task mode_hold --backend cuda \
  --config reports/toy100/relative-update-control/ruc3/config.json \
  --initial-state <mode_hold-fixture> --network-floor 1 --prior-floor 1 \
  --output <fresh-dir>
```

The hold command is the same with `hold.py`. ruc1 and ruc2 use their own directories and the same floors.

## Artifacts

- Code: `repo/reports/toy100/relative-update-control/ruc1|ruc2|ruc3/` (`parent-mechanism.py` is the untouched K3P mechanism).
- ruc1 shift FAIL: `runs/ruc1-shift/result.json` (163.42 s).
- ruc2 shift FAIL: `runs/ruc2-shift/result.json` (172.29 s).
- ruc3 shift FAIL: `runs/ruc3-shift/result.json` (144.08 s). Receipt: `runs/ruc3-shift/mechanism-receipt.json`.
- ruc3 hold PASS: `runs/ruc3-hold/result.json` (130.39 s).
- Gate log: `tests.jsonl` (5 executed gates: hold PASS, horizon-prefix PASS, three shift FAIL).

## Next mechanism, not run

The acquisition failure was an early collapse of `rho`. The recovery failure on the run that did acquire was the opposite: once `r_moment` crossed 1.25, the boost reset to 1 often enough that the step never returned to the floor. A useful next rule would reopen by a bounded factor over the floor and decay that boost without resetting it to 1 on every mild crossing, while leaving the 800-step full-rate prefix and the anchor leak in place. That is a fourth proposal and was not trained.
