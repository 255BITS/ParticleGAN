# Lane C1 — capacity-balanced anchor mass

Path: **C1**. One change to the pre-start anchor objective. Same network, one Adam step per player, pre-G joint fit, rest on a nonconverged fit.

The pinned objective pulls every surplus particle onto the nearest centroid. Cold and own-2400 therefore emit counts `[1,1,1,1,1,1,1,5]`, mode-mass TV `.2917`, and coincident-particle spread `.029` against target sigma `.07`.

Replacement: group counts differ by at most one, and a two-particle group is placed at the centroid ± the current real minibatch's principal axis, with scale `sqrt(max(0, empirical variance − 0.029²))` capped at `2σ`. No mode-center oracle and no configured mode count.

## Receipts

Host: **neural**. PyTorch `2.14.0+cu130`, one CPU thread, seed 0, constant nominal rates. Logs: `BALANCED_UPDATE`, `COLD_PROGRESS`, `OWN_HOLD`.

| Check | Pinned #1 | This run |
| --- | --- | --- |
| Trajectory MSE | PASS `.000942662` | PASS `.000942668` (4.01 s) |
| Ring | 24/24, 8 modes, HQ 1, 87.66 s | 24/24, 8 modes, HQ 1, **97.06 s** |
| Particle counts | `[1,1,1,1,1,1,1,5]` | `[2,2,1,1,1,1,2,2]` at step 1200 and at 2400 |
| Mode-mass TV | `.2917` | **`.1667`** (Δ `−.125`) |
| Mean emitted spread | `.029` | **`.0442`** at 1200, **`.0448`** at 2400 |
| Own hold 1201–2400 | 1200/1200 | **1200/1200**, min modes 8, min HQ `.975`, final HQ 1 |

Cold ring selected `joint_fit` on 1199/1200 updates and rested once at step 950 (`fit=BUDGET`), then returned to a converged fit. The four singleton modes still emit only `.029`. Mean spread moves toward `.07` and does not reach it.

Raw rows: `continuous-evidence/lane-c1-balanced-mass/`.

## Rank

Under the board rubric this is still a coverage objective, now with an explicit mass/shape term. It keeps acquire and own-hold and **beats #1 on the admitted mass TV and emitted-spread gap**. It does not beat PR84 on purity, and it is slightly slower (97 s vs 87.66 s). Not a production or 22/22 claim.

## Keep / kill / next

**Keep** the balanced-quota objective. **Kill** nearest-centroid surplus collapse as the mass rule. **Next single bet:** the four single-particle modes cannot carry within-mode variance with one coincident particle; do not spend the next run on a seed sweep or a clip ladder. A 12k continuation is still open and was not required to measure this fidelity gap.
