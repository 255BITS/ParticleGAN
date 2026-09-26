# k3p_particle_mobility

K3P stays the selected base. No candidate passed hold, the 300-update extension, and timely recovery together. Nothing was promoted. `current-research-base.json` was not edited. The pinned parent mechanism `d2eb08ee932b288cbba25cd1e7be3a9572b129bd1baf0b79718be1eb37ba9391` was copied, not modified.

Parent, not rerun: 22/22 toys, ring hold 1200/1200 (min HQ 0.90723), extension 300/300 (min HQ 0.98779), target-shift deadline 28/81, delay 1130.

Three proposals. One GPU worker. Ledger: `tests.jsonl` beside this file. Sources: `repo/reports/toy100/k3p-particle-mobility/`.

## Leaderboard

Ranked by hold, extension, then timely recovery. Toy passes are zero for every modified candidate.

| Candidate | Hold | Extension | Stationary | Continued | Deadline | Delay | Worst deadline HQ |
|---|---|---|---|---|---|---|---|
| K3P parent (published) | 1200/1200 | 300/300 | published | published | 28/81 FAIL | 1130 | published with that failure |
| **pm1_relative_mobility** | **1200/1200, min HQ 0.93823** | **300/300, min HQ 0.98999** | **5/5** | **120/120** | **79/81 FAIL** | **610** | **0.88037 at step 2990 (7 modes)** |
| pm3_prior_match | 1200/1200, min HQ 0.92505 | 300/300, min HQ 0.94653 | 5/5 | 120/120 | 79/81 FAIL | 810 | 0.89990 at step 3190 |
| pm2_dwell_anchor | 1200/1200, min HQ 0.93604 | 300/300, min HQ 0.97437 | 5/5 | 120/120 | 78/81 FAIL | 1040 | 0.88062 at step 3420 |

pm1 is the best partial lead in this lane. It is not a verified base. Sensitive gates, the 22-toy matrix, frozen recovery control, horizon-prefix equality, delayed and repeated shifts, the 12 native seed runs, and the 30000-update continuation are **NOT_RUN**.

## What the rates did

All three keep K3P's penalty body, guard (5× Adam RMS after 200 critic steps), anchor decay 0.999, latent rule, and direct response. `latent.py` and `response.py` match the parent hashes `197df635…` and `7e71d60a…`. Host floors passed to the drivers are 1 and 1. The applied multipliers replace the cosine. Critic mixing is the K3P map of the **network multiplier**, not last/max learning rate.

Shared acquire latch: both multipliers stay at 1 until the 0.98 EMA of the concatenated generator+prior gradient RMS is below 0.05 of its own peak, with a short confirmation. On the ring that latch falls between steps 500 and 550, after 8 modes already exist at full rate (step 500, HQ 1). Damping does not start during acquisition.

The ring prior group is 48 parameters and the generator group is 19298. This is the mode-hold particle table, not the 20000-particle native prior. Native behavior of these controllers is NOT_RUN.

**pm1** (`mechanism.py` `bfaace9b8f2a3cc62116060129f66036d1c92335d28a80b721d97d0b66cc9292`, `mobility_policy.py` `1b70b894637a289c8fede63096f635edf97a28a683d5b28cf62a862b54f4c8fb`). After the latch each role dwells at 0.10 of its base and reopens to 0.25 when its own raw RMS exceeds five times its own quiet floor. The floor is frozen during the reopen. Return to dwell compares RMS with that floor, not with the lifetime peak. On the shift both roles reopened together: at step 2450 network multiplier 0.247 and prior 0.246, mixing s 0.483, network RMS about 22× its floor. By step 2500 both were back in dwell because the gradient had cooled. Settled rates are G/D 0.000425 and prior 0.00085. s at dwell is 0.184. Deadline misses are only steps 2990 (7 modes, HQ 0.88037, one mode count 0) and 3000 (8 modes, HQ 0.89673). From 3010 through 3600 the checks pass. Both optimizers took 3600/3600 updates. The live ring went from 8 modes to 0 at the shift and later returned, so a frozen generator is not impersonating adaptation. Added critic forwards: shift 3074, hold 2374. Hold converged at step 1400 with 0 settling failures. Extension is steps 2601–2900.

**pm2** (`mechanism.py` `edf2663e04712f52f5c7ad53cbc960bbe050c91a13d0b8598d2749858be23e32`, `mobility_policy.py` `52ff47493b4ae55816fcbb76f13b4d080cb60e4b87ec5070fb6934b9ef6f8d8e`). Same rates as pm1. Settled dwell forces s=0 (full cap plus EMA anchor). Acquire and the brief adapt reopen still use the K3P map. This was aimed at the quiet 2990/3000 dip, and it does not raise the dwell rate (a stationary reopen to ~0.21 has already been seen to worsen native centers). Hold and extension still pass, with a lower extension minimum. Recovery gets worse: 28 failing checks, delay 1040, deadline 78/81 (misses 2810, 3420, 3430). Full anchor during dwell slowed the move. Reject.

**pm3** (`mechanism.py` `d103ff725a206dcafb482ac422f460a875d1de70aec9f73ff31728aaf10d175b`, `mobility_policy.py` `ae7e56198900bf62e22b3f0b57c0862c7f82851543e7b468a969f64609996c98`). pm1 mixing. Prior dwell fraction 0.05 so its absolute LR matches the network dwell (measured prior min 0.000425). Prior adapt stays 0.25, so a hot prior still uses the recipe 2× absolute rate. The roles do diverge: step 600 is network 0.10 and prior 0.05; step 2450 is network 0.244 and prior 0.245. Hold minimum falls to 0.925 and extension minimum to 0.947. Deadline is again 79/81, now at 3190 (HQ 0.89990) and 3200 (7 modes), delay 810. Early misses run through step 2550. Quieter settled particles slowed reacquisition and did not remove the two-check miss. Reject.

## Budget dependence

The multiplier function records `total_steps` and the horizon cap and does not branch on them. Receipts show ignored total 1200 and ignored cap 1600. A paired prefix audit of model, optimizer, controller, EMA, RNG, rates, and noise under two declared horizons was **NOT_RUN**.

Remaining scheduled component: K3P input noise 0.5 with anneal end 0.1, and output noise 0.029 with warmup 0.2, on the driver's frozen noise horizon of 1200. That noise is a labeled intermediate ablation, not a horizon-independent final rule. The 200-step guard warmup and the EMA decays are estimator memory. Search time is not an input to the rate or the mixing weight.

## Not run

Matched frozen control (no shift returned UNCONFIRMED), sensitive four, the other transfer toys, native 7000-update coverage and accuracy, the 9 extra native seeds, delayed and repeated shifts, the 30000-update continuation, and the two-horizon prefix. No score is inherited from K3P, A3, or AP3.

## Replay

```sh
export CUDA_VISIBLE_DEVICES=GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/tmp/pr38-default-env/bin/python
REPO=/ml2/hypergan/gan-attempts/claude-pool-20260925T041936Z/qualify_a2_bounded_damp/20260925T041936Z-2509885/repo/reports/toy100/qualify-a2-attempt/prepared/repos/cuda
FIX=/ml2/hypergan/ParticleGAN-epsilon-gan-followup/reports/toy100/direct-particle-base/initialization-fixtures/mode_hold/initial-values.pt
CAND=/ml2/hypergan/gan-attempts/formulations-20260925T181045Z/k3p_particle_mobility/20260925T181045Z-3769156/repo/reports/toy100/k3p-particle-mobility/pm1
$PY -u $CAND/shift.py --repo $REPO --config $CAND/config.json --task mode_hold --backend cuda \
  --initial-state $FIX --output /tmp/pm1-shift --network-floor 1 --prior-floor 1 --anneal-start 0.6
$PY -u $CAND/hold.py --repo $REPO --config $CAND/config.json --task mode_hold --backend cuda \
  --initial-state $FIX --output /tmp/pm1-hold --network-floor 1 --prior-floor 1 --anneal-start 0.6 --steps 7500 --post-window 300
```

Logs: `repo/reports/toy100/k3p-particle-mobility/logs/`. Raw results: `runs/pm1-shift`, `runs/pm1-hold`, and the same layout for pm2 and pm3.

## Next mechanism, not trained

Do not promote pm1. The shift spike is detected (about 22× the quiet floor) and the 0.25 reopen closes because the gradient cools within about 50 updates. The two deadline misses happen later, at the dwell rate, when gradient RMS is already back under the floor, so another gradient-ratio reopen cannot see them. Forcing the anchor fully on during that dwell (pm2) and halving the settled prior step (pm3) both slowed recovery and left a two-check miss. A longer or larger reopen is a poor next step: a stationary reopen to multiplier ~0.21 has already worsened native center error, and these ring misses are not inside the reopen. The next controller has to mark that quiet one-mode dropout with a training signal other than gradient RMS, without reading modes, scores, or the shift time. Replacing the horizon-1200 noise is a separate formulation and would need its own hold and shift.
