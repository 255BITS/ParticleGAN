# Particle / generator / critic balance — pnb1, pnb2, pnb3

Parent is the pinned K3P bundle: 22/22 toys, ring hold 1200/1200, extension 300/300, target-shift recovery 28/81. Those parent scores were not rerun. No candidate here passed hold, extension, or recovery. Toy gates, the sensitive screen, the repeated-change stress test, and a matched frozen control were not run.

Ranked by hold + extension + timely recovery, then by how close the ring got to 8 modes:

| Candidate | Ring shift (3600 updates, shift at 2400) | Official hold + 300 extension | Final live | Anchor forwards |
|---|---|---|---|---|
| **pnb2** | **FAIL** 0/120 pre-shift, 0/120 recovery, deadline no | **FAIL** `NOT_CONVERGED`, streak 0, hold 0/1200, extension not started | shift 5 modes / HQ .660; hold ended at step 6300 with **7 modes / HQ 1.0** | 3598 / 3600 steps |
| pnb3 | **FAIL** 0/120 pre-shift (min HQ 0), 0/120 recovery | not run | 6 modes / HQ .994 (EMA 6 / .915) | 3598 |
| pnb1 | **FAIL** 0/120 pre-shift (min HQ 0), 0/120 recovery | not run | 7 modes / HQ .762 (EMA 7 / .809) | 3598 |

K3P on this same shift reaches 8 modes by update 500, holds them, and still only passes 28/81 deadline checks after the shift. All three attempts are behind that parent on the ring.

## What was learned

Critic successive-gradient cosine is negative while K3P is still acquiring the ring. Using it as a rate or penalty switch fires in the window where K3P goes from 4 modes (update 400) to 8 modes (update 500).

- **pnb1** turned the early penalty off by update ~130 (`s` hit 0) and updated a separate gain on every optimizer, then lifted the slow role to half the fastest gain. Applied rates swung (generator gain from ~0.06 to ~0.6 after update 1200). The ring never settled on 8 modes.
- **pnb2** kept K3P's early penalty and one shared gain, decayed `×0.997` once per critic step while that cosine was negative, and floored the gain at 0.05. Gain was already 0.59 at update 400 and 0.05 by update 2000. Modes stuck at 6 with HQ 1.0 through the pre-shift window (every check failed only the 8-mode rule). The separate 6300-update hold still never recorded an 8-mode check; it ended at 7 modes and HQ 1.0. The shift did not restore 8 modes (final 5 / .660).
- **pnb3** held the shared multiplier at 1, so generator, critic, and particles stayed at the recipe ratio (prior 2× networks) and ignored the host floor. With the 0.25 anchor on from the start, the ring still missed 8 modes, collapsed to 0 modes at update 2000, and returned to 6. Full rate did not buy a stable 8-mode hold.

The 0.25 EMA anchor was evaluated on every step after initialization (3598 extra critic forwards per 3600-update shift). It does not follow the learning-rate clock. On these runs it also did not reproduce K3P's 8-mode acquisition. K3P leaves that anchor off until the critic learning-rate ratio falls.

Optimizer steps continued through every shift: each role was updated 3600 times. Post-shift mode counts moved (pnb3: 6 → 5 → 6; pnb2 ended at 5). Stationary output is not a substitute for adaptation here; the outputs moved and still missed the 8-mode deadline.

## Rules

`latent.py`, `response.py`, and `config.json` are byte-identical to pinned K3P (`197df635…`, `7e71d60a…`, `a1475108…`). Only `mechanism.py` changes. A2 sparse latent damping and direct response are unchanged. On this ring both are inactive (latent scoped calls 0, direct-response calls 0), same as K3P.

Applied rate, restored to the host schedule before the step returns:

```text
lr = initial_group_lr * shared_or_role_gain * direct_response_gain
```

| | Penalty | Rate |
|---|---|---|
| pnb1 | `s*A+(1-s)*B+w*P`, `s` = critic cosine EMA, `w=0.25+0.75*(1-s)` | per-role cosine gain, then `max(gain, 0.5*fastest)`, clip `[0.02, 1]` |
| pnb2 | `A + 0.25*P` every step | one shared gain, `×0.997` if critic cosine EMA `< 0`, `×1.01` on a critic grad-RMS spike, clip `[0.05, 1]` |
| pnb3 | `A + 0.25*P` every step | shared gain fixed at 1 |

`A` is K3P's early R1 plus fake RMS cap. `P` is the EMA-critic gap, decay 0.999. The critic spike guard is K3P's (200 Adam steps, ratio 5). No candidate reads the training horizon, scores, targets, or the shift time.

**Remaining budget dependence.** Input and output noise still follow the driver `noise_horizon=1200` and the K3P fractions. That is an inherited scheduled component, so none of these is a horizon-independent final formulation. The host still writes its horizon learning-rate schedule; the update overwrites it and restores it. A two-horizon prefix equality check was not run. Logged `scheduled_lr` versus `applied_lr` in the balance lines is the applied-rate audit. The probe's `rate_ranges` are the host rates recorded before the step.

## Replay

Environment: `CUDA_VISIBLE_DEVICES=GPU-72c1b506-891d-b8bc-b353-e020585e1c47`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, one CPU thread, `/tmp/pr38-default-env/bin/python`. Repo: the frozen gap-fill CUDA tree `.../qualify-a2-attempt/prepared/repos/cuda`. Fixture: `.../initialization-fixtures/mode_hold/initial-values.pt`. Floors passed to the driver are the parent `.01/.05`; pnb overwrites the applied rate.

```bash
# pnb2 shift (same shape for pnb1 and pnb3; change the directory)
/tmp/pr38-default-env/bin/python -u candidates/pnb2/shift.py \
  --repo /ml2/hypergan/gan-attempts/claude-pool-20260925T041936Z/qualify_a2_bounded_damp/20260925T041936Z-2509885/repo/reports/toy100/qualify-a2-attempt/prepared/repos/cuda \
  --config candidates/pnb2/config.json --task mode_hold --backend cuda \
  --initial-state /ml2/hypergan/ParticleGAN-epsilon-gan-followup/reports/toy100/direct-particle-base/initialization-fixtures/mode_hold/initial-values.pt \
  --output runs/pnb2-shift --network-floor 0.01 --prior-floor 0.05 --anneal-start 0.6

# pnb2 official hold (ran to step 6300, NOT_CONVERGED)
/tmp/pr38-default-env/bin/python -u candidates/pnb2/hold.py \
  --repo /ml2/hypergan/gan-attempts/claude-pool-20260925T041936Z/qualify_a2_bounded_damp/20260925T041936Z-2509885/repo/reports/toy100/qualify-a2-attempt/prepared/repos/cuda \
  --config candidates/pnb2/config.json --task mode_hold --backend cuda \
  --initial-state /ml2/hypergan/ParticleGAN-epsilon-gan-followup/reports/toy100/direct-particle-base/initialization-fixtures/mode_hold/initial-values.pt \
  --output runs/pnb2-hold --network-floor 0.01 --prior-floor 0.05 --anneal-start 0.6 --post-window 300
```

Paths are under `/ml2/hypergan/gan-attempts/formulations-20260925T154444Z/particle_network_balance/20260925T154444Z-3603393/`.

## Next mechanism

Keep K3P's early penalty and its high early rate until 8 modes exist, and do not use critic cosine as that switch. The cosine is an acquisition-time signal on this ring. Balance the generator and critic floors with the particle floor only after that acquisition, with the EMA anchor already active, and test recovery against K3P's 28/81. A higher constant rate alone, measured here as pnb3, lost modes at update 2000.
