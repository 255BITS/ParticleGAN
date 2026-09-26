# k3p_local_curvature

K3P stays the selected base. Nothing here is promoted. `current-research-base.json` was not edited. The pinned parent mechanism is still `d2eb08ee932b288cbba25cd1e7be3a9572b129bd1baf0b79718be1eb37ba9391`.

Parent scores were not rerun: 22/22 GPU toys, ring hold 1200/1200 (min HQ 0.90723), extension 300/300 (min HQ 0.98779), target-shift deadline **FAIL 28/81**, delay 1130.

Three proposals, one GPU worker. Each ran its own canonical hold (with the 300-update post window) and its own 3600-update target shift. No candidate passed hold, so extension, frozen control, sensitive gates, the 22 toys, horizon-prefix equality, and delayed/repeated shifts are **NOT_RUN**.

## Leaderboard

Ranked by own hold, extension, and timely recovery. Partial scores are not passes.

| Candidate | Own hold | Extension | Shift stationary / continued | Deadline recovery | Final live | Toys |
|---|---|---|---|---|---|---|
| K3P parent (not rerun) | 1200/1200 | 300/300 | parent evidence | FAIL 28/81, delay 1130 | — | 22/22 |
| lc1_secant_growth | FAIL 314/1200, min HQ 0.87769, converged step 3033 | NOT_RUN | 0/5, 0/120 | FAIL 56/81, no stable suffix | 4 modes, HQ 0.36768 | NOT_RUN |
| lc2_calibrated_secant | FAIL NOT_CONVERGED, 0/4800 settling checks | NOT_RUN | 0/5 (6 modes, HQ 1), 0/120 | FAIL 0/81 | 6 modes, HQ 0.66406 | NOT_RUN |
| lc3_step_independent_secant | FAIL NOT_CONVERGED, 0/4800 settling checks | NOT_RUN | 0/5 (5 modes, HQ 1), 0/120 | FAIL 0/81 | 5 modes, HQ 0.99902 | NOT_RUN |

lc1 is the least-bad negative result because it did acquire 8 modes and held them for 314 checks. It is not a base. Its shift pre-hold had already collapsed, so 56/81 is not timely recovery.

## What the runs showed

All three keep K3P's learned particle prior, direct response, bounded sparse-latent rule, critic guard (5× after 200 Adam steps), and anchor decay 0.999. `latent.py` `197df6350f5295f7d396f7d3c821808be1d15168d6e5586a89ebfbd403586139`, `response.py` `7e71d60a343f9f47e1c16600279364f0482863ce116c00f4657355638615987d`, and `config.json` `a1475108a82f67a93e0cdcd793b920b0cc2b1e1ccf31285974adc3b341b2fca2` are byte copies of pinned K3P. The controller overwrites the Adam learning rate inside the step and restores the host value afterward. Mobility `group_lrs` and the probe's final optimizer rates are therefore the restored host schedule (network floor 4.25e-5, prior floor 4.25e-4 by step 3600). Applied rates are in each `mechanism-receipt.json` and in the `lc*` log lines.

**lc1 — absolute Malitsky ratio.** `eta = blend(min(growth, rms(dx)/(2 rms(dg))))`, clamped to `[floor, base]`, with non-positive directional curvature taking the growth branch. Mechanism `e22e7f9069848a27af9e9e10843a83d12bb8ae9c3d767ad0a97e5371a793c5d3`. The ratio stayed above the Adam cap, and about half the logged steps were non-positive, so generator and critic applied rates sat at 0.00425 for the whole run (prior 0.00826–0.0085). Mixing weight stayed 1. The EMA anchor never started (0 extra critic forwards, 3648 pure R1 calls on hold). Hold converged at step 3033 after 1255 settling failures, then lost precision at step 3348 (8 modes, HQ 0.87769). Post-failure diagnostic min HQ 0.84106. Shift stationary window min was 2 modes, HQ 0.09497. At the shift (step 2400) the ring had 7 modes, HQ 0.89380, then 0 modes on the moved target. Deadline 56/81 with no 5-check stable suffix. Both optimizers took 3600 Adam steps, so this was not a frozen generator. Full rate with the early penalty still on does not hold this ring.

**lc2 — same ratio divided by the mean of the first 100 positive secants.** Mechanism `d817b398e7158c3008d65dd4d28654e77a899fd2afbacf6871fbb300bdc0e5e9`. Calibration finished at critic step 141, generator 168, prior 175, with references 0.450, 0.854, and 0.736. The raw ratio scales with the step, so once eta fell, the multiplier fell with it and ratcheted every role to its floor (generator/critic 4.25e-5, prior 4.25e-4) by about step 400. Mixing went to 0 and the anchor ran (hold: 6129 extra critic forwards, 170 pure R1, 6130 blended). The ring locked a precise partial cover: 6 modes at HQ 1 by step 600, 7 modes at HQ 0.99976 at step 6300, and **zero** 8-mode settling checks in 4800. Shift stationary and continued windows were 6 modes at HQ 1 (0 passes). Deadline 0/81, worst deadline point 4 modes, HQ 0.42261. Final 6 modes, HQ 0.66406. Optimizers took 3600 updates. Floor rate plus the anchor freezes a short cover and does not recover.

**lc3 — calibrate `raw/eta` so the multiplier no longer tracks the step length.** Mechanism `74dbc6318c2ef4db15fd27fa189c5e1339b8f9f8982a61528fa8805a62d5aca9`. Blend per step is 0.05. References of the independent ratio: critic 105.87, generator 201.85, prior 86.27. Rates stayed inside the band instead of hitting the floor: hold minima were critic 7.86e-4, generator 6.92e-4, prior 6.21e-4, against bases 0.00425, 0.00425, and 0.0085. Mixing settled near 0.36–0.51, so the penalty was a blend (hold: 644 pure R1, 5656 blended, 5655 extra forwards). That partial rate never produced an 8-mode check. Last hold point, step 6300: **4 modes, HQ 1.0**. Shift stationary window was 5 modes at HQ 1. Continued hold touched 0 modes. At step 2400 the live ring had 7 modes, HQ 0.90112, then 0 on the moved target. Deadline 0/81, worst 2 modes, HQ 0.25659. By step 3200 the rise branch had reopened the critic mixing weight to 1 and lifted generator/prior applied rates to 0.00175/0.00201; by step 3600 they were back at 0.00122/0.000875 with s 0.651 and the live sample was 5 modes at HQ 0.99902. Updates stayed at 3600. Reopening happened, and it did not rebuild eight modes.

## Rules that were declared

- Step rule is per generator, critic, and prior group. Growth is `sqrt(1 + eta/eta_previous) * eta`. Floors stay 0.01 (network) and 0.05 (prior) of the initial Adam rate. Cap is that initial rate.
- Gradient memory is an EMA with new-sample weight 0.1. The secant divides the EMA difference by 0.1 so the filter speed is not the Lipschitz scale. This is finite memory, not an accumulating denominator.
- lc1 lets a non-positive directional derivative take the growth branch. lc2 and lc3 hold eta on that event, and grow only when the fast gradient RMS exceeds 1.5 times the slow RMS and a positive secant is not asking to shrink.
- Critic mixing is the K3P floor map of `eta_critic / base_critic`. It does not read the max-observed host learning rate. On lc1 that ratio stayed 1, so the anchor stayed off. On lc2 it fell to 0 with the ratchet. On lc3 it moved with the partial rate and did reopen when the rate rose.
- K3P input noise 0.5 over the first 0.1 of `noise_horizon=1200`, and output noise warming to 0.029 over 0.2 of that horizon, are unchanged. The probe rejects any other noise horizon. That inherited schedule is an intermediate ablation on every candidate here, not a horizon-free result.
- The controller does not read the training budget, horizon cap, scores, target identity, target center, or shift time. The host still writes the horizon learning rate; it is overwritten for the Adam update and restored after. A two-horizon prefix audit was **NOT_RUN**. Guard warmup 200, anchor decay 0.999, gradient EMA 0.1, and the 100-secant calibration are estimator memories, not a declared training length.

## NOT_RUN

Matched frozen recovery control, sensitive four (`mode_hold`, unequal mass, unequal width, stripes), the other transfer toys, native 7000-update coverage and accuracy, horizon-prefix equality, and the delayed/repeated-change stress. No candidate earned a passing hold plus 81/81 recovery, so that ladder was not opened. None of those scores are borrowed from K3P.

No seed sweeps, no coefficient grid, no metric feedback, no edits to the pinned parent or the frozen runtime. No pushes or comments.

## Replay

Environment on every benchmark process: `CUDA_VISIBLE_DEVICES=GPU-72c1b506-891d-b8bc-b353-e020585e1c47`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `/tmp/pr38-default-env/bin/python`.

```bash
BASE=/ml2/hypergan/gan-attempts/formulations-20260925T172651Z/k3p_local_curvature/20260925T172651Z-3715973/repo/reports/toy100/k3p-local-curvature
RUNTIME=/ml2/hypergan/gan-attempts/claude-pool-20260925T041936Z/qualify_a2_bounded_damp/20260925T041936Z-2509885/repo/reports/toy100/qualify-a2-attempt/prepared/repos/cuda
FIXTURE=/ml2/hypergan/ParticleGAN-epsilon-gan-followup/reports/toy100/direct-particle-base/initialization-fixtures/mode_hold/initial-values.pt

# Replace lc3 with lc1 or lc2. Hold is the convergence gate plus 300 post-window updates.
/tmp/pr38-default-env/bin/python -u "$BASE/candidates/lc3/hold.py" \
  --repo "$RUNTIME" --config "$BASE/candidates/lc3/config.json" \
  --task mode_hold --backend cuda --initial-state "$FIXTURE" \
  --output "$BASE/runs/lc3-hold-replay" \
  --network-floor 0.01 --prior-floor 0.05 --anneal-start 0.6 --post-window 300

/tmp/pr38-default-env/bin/python -u "$BASE/candidates/lc3/shift.py" \
  --repo "$RUNTIME" --config "$BASE/candidates/lc3/config.json" \
  --task mode_hold --backend cuda --initial-state "$FIXTURE" \
  --output "$BASE/runs/lc3-shift-replay" \
  --network-floor 0.01 --prior-floor 0.05 --anneal-start 0.6
```

Logs: `$BASE/logs/`. Raw results: `$BASE/runs/lc{1,2,3}-{hold,shift}/result.json`. Ledger: `tests.jsonl` beside this file (3 FAIL hold rows, 3 FAIL shift rows).

## Next mechanism

Do not rescale `rms(dx)/(2 rms(dg))` again. The absolute ratio cannot enter the Adam band (lc1). Dividing by an early mean ratchets to the floor because the ratio shrinks with the step (lc2). Dividing by the current step stops the ratchet and still leaves a partial rate with the early penalty only partly off, which freezes four or five precise modes (lc3). Negative directional curvature is too common on this ring to be a growth signal. The next controller should keep the early penalty until a gradient-magnitude quiet signal, not until this secant falls, and it should measure curvature on the preconditioned Adam direction rather than on a displacement that is proportional to the learning rate.
