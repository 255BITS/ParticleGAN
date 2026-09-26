# k3p_balanced_updates — coupled G/D/prior mobility

K3P stays the selected base. No candidate passed hold, the 300-update extension, and the raw shift verdict together. Nothing was promoted. `current-research-base.json` was not edited. Toy gates, the matched frozen control, the horizon-prefix audit, and the delayed/repeated-change stress check are NOT_RUN.

Parent (not rerun): 22/22 toys, ring hold 1200/1200, extension 300/300, target-shift deadline 28/81. Pinned mechanism `d2eb08ee932b288cbba25cd1e7be3a9572b129bd1baf0b79718be1eb37ba9391`.

Lane: one shared multiplier on generator, critic, and prior. No per-role controllers and no sticky reset to full rate. Critic mixing reads that multiplier, not last/max learning rate. Particle latent damping and direct response are byte-identical to K3P. Host floors passed to the drivers are 1 and 1, so the cosine learning-rate schedule is the identity at every horizon. The driver's `noise_horizon=1200` is still in force. That noise schedule is a labeled intermediate ablation; the multiplier and the mixing weight do not read it.

## Leaderboard

Ranked by hold + extension + timely recovery. Parent scores are published measurements, not this round.

| Candidate | Stationary 1000–1200 | Continued hold to shift | Deadline recovery | Own hold + extension |
|---|---|---|---|---|
| K3P parent | 120/120 in its published shift window | included | FAIL 28/81, delay 1130 | PASS 1200/1200 and 300/300 |
| **kbu3_late_quiet_band** | FAIL 0/5, min 6 modes, HQ 0.698 | FAIL 101/120, stable from step 1510, 8 modes at the shift | FAIL 0/81, ended 6 modes HQ 0.990 | FAIL `POST_CONVERGENCE_FAIL` at step 2590 after 884 good hold updates. Extension NOT_RUN |
| kbu1_joint_band | PASS 5/5, min HQ 0.966 | FAIL 33/120, collapse at step 1600 | FAIL 0/81, ended 7 modes | FAIL `NOT_CONVERGED`, longest streak 120. Extension NOT_RUN |
| kbu2_quiet_leak_band | FAIL 0/5, max 7 modes, min HQ 0.920 | FAIL 0/120, min 7 modes | FAIL 0/81, ended 7 modes HQ 0.973 | FAIL `NOT_CONVERGED`, streak 0, 4800/4800 settling failures. Extension NOT_RUN. Ended step 6300 at 7 modes, HQ 0.9995 |

No toy passes. Added critic evaluations are the EMA-anchor forwards: kbu1 0 (anchor never started), kbu2 2638, kbu3 shift 2162.

## What the three runs did

All three copy hashed K3P `config.json`, `latent.py`, and `response.py`. Only `mechanism.py` changes. One multiplier `m` scales the host rates of G, D, and the prior, so the recipe prior/network ratio stays 2. Mixing is the K3P map of `m`: `s = 1` while `m >= 0.5`, and `s ≈ 0.184` at `m = 0.10`, which turns the 0.999 EMA anchor on. The guard stays at 200 Adam steps and ratio 5. Optimizer updates continued through every shift (3600/3600 on both optimizers). Post-shift outputs moved, so a frozen generator is not impersonating adaptation.

**kbu1** (`2311362b5e42f040ff2e3ff901f60a9f4c1059c0b1005f70002d83d8e78ceef0`). `m` stays 1 until critic relative displacement `||Δθ||/||θ||` is below 0.01 for 100 consecutive steps, then tracks a band `[0.10, 0.35]` from the critic Adam moment ratio. The streak never formed. Printed quiet counts on 100-step samples stayed in the single digits while full-rate displacement jittered through about 0.005–0.015. `m` was 1.0 for all 3600 critic steps, `s` stayed 1, and the anchor never started (`pure_a` 3600). Eight modes from step 500, stationary window passed, then 0 modes at step 1600. Same pattern as the round-1 full-rate loss, now measured with the joint multiplier stuck at acquisition rate. Hold ran the 4800-check settling budget and stopped at step 6300 with 6 modes, HQ 0.959. Seconds: shift 150.56, hold 313.83.

**kbu2** (`ecef4039736ac3e5f08a3f63a4145a606106cd3efb95a3be95f71681169a1531`). Addresses that miss. Above 0.35, `m` decreases only in proportion to how far displacement sits below 0.02, and it cannot climb back to 1. Inside the band it tracks the moment-ratio target at step 0.02, capped at 0.35. On the full-rate trace, displacement is already 0.017 at step 400 (4 modes), so this gate opened during acquisition. `m` was 0.95 by step 500 and the ring locked at 7 modes for the whole 3600 updates (max modes 7). G, D, and prior recorded the same `m` (min 0.100, max 1, last 0.142). Scheduled host LR stayed 0.00425 / 0.0085 while applied critic LR followed `m`. `lr_overwrite_after_scale` is 0. Anchor forwards 2638. The hold never saw 8 modes: longest streak 0, 4800 settling failures, stopped at step 6300 with 7 modes and HQ 0.9995 (EMA 7 modes, HQ 1.0). Extension NOT_RUN. At that step the mobility row's group learning rates were 4.93e-4 and 9.87e-4, still in the 2× prior/generator ratio, about 0.12× the recipe base. Seconds: shift 151.15, hold 247.38.

**kbu3** (`3a26e8c7e338507897af66f2f227dcb84f120137c3c846a4fd269c567930180d`). Tightens the leak to displacement below 0.012, which on the kbu1 trace is the post-8-mode range, with leak step 0.004. Calm motion inside the band uses that same 0.004 step; only a moment ratio above 1 uses the 0.02 reopen, still capped at 0.35. A few pre-500 steps still dipped under 0.012, so `m` was 0.988 at step 500 and the 8th mode was late: first 8-mode checkpoint at step 1340, not 500. Stationary 1000–1200 failed (6–7 modes). From step 1510 through the shift at 2400 the live ring held 8 modes (continued hold 101/120; pre-shift HQ 0.988). The full-rate collapse at step 1600 did not repeat. At step 2400 the logged moment ratio was 0.53, under 1, and the 100-step samples after the shift stayed under 1, so the reopen did not leave the floor (`m` 0.113 at the shift, 0.138 at step 3600; applied critic LR about 4.8e-4 against scheduled 4.25e-3). Recovery deadline 0/81, final 6 modes at HQ 0.990. The own hold confirmed at step 1705, then failed the hold at step 2590 on HQ 0.8909 with 8 modes still present (884 good hold updates, min modes 8). Extension NOT_RUN. Seconds: shift 150.85, hold 116.79.

## Rate and mixing audit

- Host scheduled rates on these runs stayed at the recipe bases (critic/generator 0.00425, prior 0.0085) for all 3600 steps. That is the floor-1 identity schedule. It does not depend on `network_lr_horizon_cap`.
- Applied rate is that host rate times the shared `m`. kbu2 and kbu3 logs show `scheduled_lr` fixed and `applied_lr` tracking `m`. `lr_overwrite_after_scale` is 0 on all three.
- Mixing `s` fell below 1 only when `m` fell below 0.5, while the host learning rate was still at its maximum. The LR clock is not the mixing input.
- kbu1 never left `m = 1`, so its applied rate equals the host rate and does not by itself prove a non-unit scale. kbu2 and kbu3 do.
- Remaining budget dependence: input noise reaches 0 by 0.1 × 1200 updates and output noise reaches 0.029 by 0.2 × 1200, both from the driver horizon. A two-horizon prefix comparison was NOT_RUN. No candidate reads `total_steps`, the horizon cap, scores, mode counts, target centers, or the shift step.

## Replay

Environment on every command: `CUDA_VISIBLE_DEVICES=GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`. Python: `/tmp/pr38-default-env/bin/python`. Frozen repo: `/ml2/hypergan/gan-attempts/claude-pool-20260925T041936Z/qualify_a2_bounded_damp/20260925T041936Z-2509885/repo/reports/toy100/qualify-a2-attempt/prepared/repos/cuda`. Fixture: `/ml2/hypergan/ParticleGAN-epsilon-gan-followup/reports/toy100/direct-particle-base/initialization-fixtures/mode_hold/initial-values.pt`.

```bash
CAND=reports/toy100/k3p-balanced-updates/kbu3   # or kbu1 / kbu2
/tmp/pr38-default-env/bin/python -u "$CAND/shift.py" \
  --repo "$FROZEN" --config "$CAND/config.json" --task mode_hold --backend cuda \
  --initial-state "$FIXTURE" --output runs/kbu3-shift \
  --network-floor 1 --prior-floor 1 --anneal-start 0.6
/tmp/pr38-default-env/bin/python -u "$CAND/hold.py" \
  --repo "$FROZEN" --config "$CAND/config.json" --task mode_hold --backend cuda \
  --initial-state "$FIXTURE" --output runs/kbu3-hold \
  --network-floor 1 --prior-floor 1 --anneal-start 0.6 --post-window 300
```

Paths are under `/ml2/hypergan/gan-attempts/formulations-20260925T165310Z/k3p_balanced_updates/20260925T165310Z-3675726/repo/reports/toy100/k3p-balanced-updates/`. Logs are in `logs/`. Gate ledger: `.../20260925T165310Z-3675726/tests.jsonl`.

## Not run

- Matched frozen recovery control, sensitive four, and all 22 toys. No shift reached `UNCONFIRMED` (stationary and continued hold and deadline together).
- Two-horizon prefix of model, optimizer, controller, EMA, RNG, rates, and noise.
- Delayed change and a second change on uninterrupted state.

## Next mechanism, not trained

The shift did not move the critic moment ratio through 1 (0.53 at update 2400 on kbu3), so a band keyed on `(r - 1)` never reopened and the ring settled at 6 modes. A later controller can keep the shared multiplier and the cap, and drive the reopen from a signal that actually jumps when the data move, such as gradient RMS over a slow EMA of gradient RMS, without resetting `m` to 1. The leak still has to stay at `m = 1` through the step-500 acquisition: kbu1 got 8 modes there and then collapsed at step 1600, while any earlier leak (kbu2, and a small early dip on kbu3) delayed or deleted the 8th mode. Three proposals were used.
