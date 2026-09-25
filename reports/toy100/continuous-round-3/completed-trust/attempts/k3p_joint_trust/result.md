# k3p_joint_trust — coupled displacement bound

K3P stays the selected base. No candidate passed hold, the 300-update extension, and recovery together. Nothing was promoted. The pinned parent mechanism is unchanged (`d2eb08ee932b288cbba25cd1e7be3a9572b129bd1baf0b79718be1eb37ba9391`).

Parent, existing evidence only: 22/22 toys, ring hold 1200/1200, extension 300/300, target-shift deadline 28/81, delay 1130. Not rerun.

Lane: one shared bound on realized Adam displacement of the critic, generator, and prior. Host floors passed to the drivers are 1 and 1, so the cosine learning-rate schedule is the identity at every horizon. That is not a silent constant-rate handover. jt1 reads the mixing weight from the radius. jt2 and jt3 set the mixing weight to 1 and use the displacement cap as the critic constraint, so the EMA anchor stays off. Particle latent damping and direct response are the pinned files. Driver `noise_horizon=1200` remains a labeled intermediate ablation. The radius and the mixing weight do not read it.

## Leaderboard

Ranked by hold + extension + timely recovery, then toy passes. All three candidates are unqualified. Partial confirmation is listed so the failure is specific.

| Candidate | Stationary 1000–1200 | Continued hold to shift | Deadline recovery | Own hold + extension | Toys |
|---|---|---|---|---|---|
| K3P parent | 120/120 in its published shift window | included | FAIL 28/81, delay 1130 | PASS 1200/1200 and 300/300 | 22/22 |
| jt2_growth_trust | FAIL 0/5, min 0 modes | FAIL 0/120 | FAIL 0/81 | FAIL `POST_CONVERGENCE_FAIL` at step 3436. Confirmed at 3351, 84 good hold updates, then HQ 0.871 with 8 modes. Extension NOT_RUN | NOT_RUN |
| jt3_late_growth | FAIL 0/5, min 4 modes, HQ 0.353 | FAIL 0/120, min 2 modes | FAIL 0/81, ended 7 modes HQ 0.969 | FAIL `NOT_CONVERGED`, streak 0, 4800/4800 settling failures. Extension NOT_RUN. Ended step 6300 at 6 modes, HQ 0.831 | NOT_RUN |
| jt1_shared_radius | FAIL 0/5, min 5 modes, HQ 0.4719 | FAIL 0/120, min 0 modes | FAIL 0/81, ended 5 modes HQ 0.732 | FAIL `NOT_CONVERGED`, streak 0, 4800/4800. Extension NOT_RUN. Ended step 6300 at 6 modes, HQ 1.0 | NOT_RUN |

Shift worst quality: jt1 deadline window min HQ 0 at 0 modes; jt2 stationary window hit 0 modes; jt3 stationary min HQ 0.353. No candidate reached 8 modes on the shift run (jt1 and jt3 peaked at 7, jt2 at 5). jt2's 8-mode state appeared only on the uninterrupted hold, at step 3351, after the shift's change time.

Added anchor forwards: jt1 shift 1721 and hold 3146. jt2 and jt3 added 0. Both optimizers advanced on every shift (3600/3600 critic, generator, and prior group steps). Modes changed after update 2400 on every shift, so a frozen generator is not impersonating adaptation. Host critic LR stayed 0.00425 (`mechanism-receipt.json` for jt1; live receipt for jt2 and jt3).

## What the three runs did

All three copy hashed K3P `config.json`, `latent.py`, and `response.py`. Only `mechanism.py` changes. Projection is after Adam and after the unchanged latent and direct-response hooks: `theta = theta_old + min(1, cap/rel) * (theta - theta_old)`, one scale per parameter group. Adam moments are left as Adam wrote them. A zero displacement stays zero. Guard stays at 200 steps and ratio 5.

**jt1_shared_radius** (`c76598fcb7c0de583f310c337fdbd5882c38120ae168ce8b46a79e5d6b819497`). Fixed ball. `rho` starts at 0.05, the kbu1 critic relative displacement at update 100. It shrinks by 0.98 when a smoothed ask falls below 0.5 and grows by 1.02 when ask rises above 1.5, inside `[0.004, 0.05]`. Mixing uses the K3P map of `rho/0.05`, not of learning rate. Early generator steps were clipped (step 1 critic relative displacement 0.061 projected to 0.05). The ring reached 7 modes at update 500 (HQ 0.907) and never 8. Between updates 1200 and 1400 the ball sprang back to 0.05 while modes fell from 7 to 3; those samples were inside the ball (attempted relative displacement about 0.04, scale 1). After the shift, modes went 7 → 2 → 7 by update 2800 and ended at 5. Final radius 0.0212, final mixing weight 0.8455. Seconds: shift 150, hold 277.

**jt2_growth_trust** (`b7ab6ce268577baca0ed00bab8ce0a944e310716828f97e60dce2e6f2836b03b`). Replaces the spring-back ball with `cap = min(0.10, 2 * ema)` and `ema = 0.99*ema + 0.01*max realized relative displacement` across critic, generator, and prior. Mixing weight fixed at 1. The 0.10 ceiling is under kbu1's generator acquisition peak of 0.168 at update 160, and the path left that unclipped run before update 20. Shift maximum was 5 modes. The hold did reach a 200-check confirmation at step 3351 (8 modes, HQ 0.994) and then failed the disjoint hold at step 3436 on HQ 0.871 with 8 modes still present. The following 300 diagnostic updates passed 291/301 and ended at 8 modes, HQ 0.987. That diagnostic is not a restarted hold and not the 300-update extension. Cap at the failure was about 0.015. Seconds: shift 146, hold 162.

**jt3_late_growth** (`4acc710d0bfa2211ec22c851d97b9a12bbdfdc4aa94a73228bf6a7caf7a3643e`). Same growth rule with the ceiling raised: `cap = min(0.50, 2 * ema)`, ema decay 0.995, ema starts at 0.25 so the first cap is 0.50. Mobility matched kbu1 exactly through update 100, and the update-160 generator sample matched kbu1's 0.1677 with scale 1 under a cap of 0.301. Critic mobility had diverged by update 200. An unsampled step between 161 and 199 exceeded the cap; the every-20 trace does not contain it. The first sampled clip is update 1740 (scale 0.647, cap 0.019). Shift peak was 7 modes, with a drop from 6 at update 1600 to 3 at 1900, then 5 at the shift and 2 at update 2500, ending at 7 modes HQ 0.969. Hold never recorded an 8-mode qualifying check (streak 0, 4800 failures), ending at 6 modes HQ 0.831. 589 shift groups and 1445 hold groups were clipped. Seconds: shift 139, hold 285.

## Rate and mixing audit

- Scheduled host rates stayed at the recipe bases (critic and generator 0.00425, prior 0.0085). With floors 1, `learning_rate_scale` is identically 1 for any `total_steps` and any `network_lr_horizon_cap`.
- jt1 applied the radius after the step and set mixing from `rho/0.05`. jt2 and jt3 left mixing at 1, so the anchor never started.
- Prior groups still receive the direct-response gain during the Adam step. The projection, when it binds, scales that realized step. When scale is 1 the gain is unchanged.
- Remaining budget dependence: input noise reaches 0 by 0.1 × 1200 updates and output noise reaches 0.029 by 0.2 × 1200, both from the driver horizon. A two-horizon prefix comparison was NOT_RUN. No candidate reads `total_steps`, the horizon cap, scores, mode counts, target centers, or the shift step.

## Replay

Environment: `CUDA_VISIBLE_DEVICES=GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`. Python: `/tmp/pr38-default-env/bin/python`. Frozen repo: `/ml2/hypergan/gan-attempts/claude-pool-20260925T041936Z/qualify_a2_bounded_damp/20260925T041936Z-2509885/repo/reports/toy100/qualify-a2-attempt/prepared/repos/cuda`. Fixture: `/ml2/hypergan/ParticleGAN-epsilon-gan-followup/reports/toy100/direct-particle-base/initialization-fixtures/mode_hold/initial-values.pt`.

```bash
CAND=reports/toy100/k3p-joint-trust/jt2   # or jt1 / jt3
/tmp/pr38-default-env/bin/python -u "$CAND/shift.py" \
  --repo "$FROZEN" --config "$CAND/config.json" --task mode_hold --backend cuda \
  --initial-state "$FIXTURE" --output runs/jt2-shift \
  --network-floor 1 --prior-floor 1 --anneal-start 0.6
/tmp/pr38-default-env/bin/python -u "$CAND/hold.py" \
  --repo "$FROZEN" --config "$CAND/config.json" --task mode_hold --backend cuda \
  --initial-state "$FIXTURE" --output runs/jt2-hold \
  --network-floor 1 --prior-floor 1 --anneal-start 0.6 --post-window 300
```

Paths are under `/ml2/hypergan/gan-attempts/formulations-20260925T173623Z/k3p_joint_trust/20260925T173623Z-3729093/repo/reports/toy100/k3p-joint-trust/`. Logs are in `logs/`. Gate ledger: `/ml2/hypergan/gan-attempts/formulations-20260925T173623Z/k3p_joint_trust/20260925T173623Z-3729093/tests.jsonl` (6 FAIL rows).

Declarations: `jt1/declaration.json`, `jt2/declaration.json`, `jt3/declaration.json`.

## Not run

Matched frozen recovery control, four sensitive gates, all 22 toys, native 7000-update coverage and accuracy, the two-horizon prefix, and delayed or repeated changes. No shift reached `UNCONFIRMED`. The supervisor rule for this round says not to open stress checks or extra grid seeds on a failed candidate.

## Next mechanism, not trained

A fixed or slowly tightened cap on joint relative displacement either clips acquisition or lets the update-1600 critic jump (0.147 on kbu1) through. jt3 shows the every-20 acquisition trace is not a safe envelope: the run matched kbu1 through update 100 and matched the update-160 generator sample, then diverged before update 200. jt2 shows a later 8-mode state can appear under a tight cap and still fail the hold on a single HQ sample of 0.871. A later controller that still wants a displacement bound has to leave every acquisition step, including the unsampled ones, unprojected, and arm only after realized motion has already been inside the hold band kbu1 measured from updates 500–1500 (critic relative displacement at most 0.017). Three proposals were used.
