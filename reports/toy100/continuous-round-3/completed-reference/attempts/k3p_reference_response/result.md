# k3p_reference_response

K3P stays the selected base. No candidate passed its own hold, the 300-update extension, and timely shift recovery together. `current-research-base.json` was not edited. The pinned parent sources were not edited.

Parent, reused and not rerun: 22/22 GPU toys, ring hold 1200/1200 (min HQ 0.90723), extension 300/300 (min HQ 0.98779), target-shift deadline **FAIL 28/81**, delay 1130. Mechanism `d2eb08ee932b288cbba25cd1e7be3a9572b129bd1baf0b79718be1eb37ba9391`.

Three proposals, one GPU worker. Round-2 anchor tracking was not repeated: critic cosine, Adam-step innovation, generator-gradient energy, from-start anchoring, and a constant peak rate with a slow anchor from step 200. The generator-collapse controller from the precision lane was not repeated.

## Leaderboard

Ranked by hold, extension, and timely recovery. None of the three is timely (81/81 by the 400-update deadline). Deadline counts are reported so the failures stay visible. Toy gates were not opened.

| Candidate | Own hold | Extension | Pre-shift | Deadline | Delay | Toys |
|---|---|---|---|---|---|---|
| K3P parent (not rerun) | 1200/1200 | 300/300 | published pass | FAIL 28/81 | 1130 | 22/22 |
| rr2 prox gap, slow reference | **1200/1200**, min HQ 0.91699, converged 1400 | **300/300**, min HQ 0.98291 | stationary 5/5, continued **120/120** | **FAIL 52/81** | 1140, stable 3540 | NOT_RUN |
| rr1 parameter-RMS rate | **1200/1200**, min HQ 0.92383, converged 1400 | **300/300**, min HQ 0.99658 | stationary 5/5, continued **120/120** | FAIL 29/81 | 1010, stable 3410 | NOT_RUN |
| rr3 prox gap, fast baseline leak | **1200/1200**, min HQ 0.91699, converged 1400 | **300/300**, min HQ 0.98291 | stationary 5/5, continued **120/120** | FAIL 0/81 | none | NOT_RUN |

rr2 is the strongest partial deadline count in this lane and still fails the raw shift verdict. 52/81 is not a pass. Its delay is worse than the parent's. Nothing here is promoted.

## What was measured

All three copy K3P `config.json` `a1475108…`, `latent.py` `197df635…`, and `response.py` `7e71d60a…`. Learned particle prior, direct response, and bounded sparse-latent damping are unchanged. Critic guard stays 5× Adam RMS after 200 steps. Cold path, shared until the anchor latches: pure `a_r1r2` and peak rates (G/D 0.00425, prior 0.0085) until critic-parameter RMS stays under a quarter of its post-warmup peak for 250 critic steps, then the mixing gain decays at 0.99. Mixing weight uses the K3P floor map on that gain, not on last/max critic LR. After the gain first hits 0 it stays 0, so the early penalty does not return. Step 600 was 8 modes on every run. The latch is at critic step 1200 with floors G/D 4.25e-5 and prior 4.25e-4.

Rates ignore step, horizon, and anneal. The multiplier is latched per completed step so the driver's post-step check sees the applied value. **Remaining budget dependence:** input noise 0.5 over the first 0.1 of `noise_horizon=1200`, and output noise warming to 0.029 over 0.2 of that horizon, are still the driver schedule. That inherited noise is a labeled ablation, not a horizon-free rule. A paired prefix under two declared horizons was **NOT_RUN**.

**rr1** (`086f3a39…`). After the latch, rate gain follows critic-parameter RMS over its decaying peak, capped at 0.2, and the reference step slows from 0.001 toward 0.0002 as that level rises. Hold and extension passed. On the shift the level only reached 0.38, inside the hold's own fluctuation, so gain never reached 0.2 (max post-shift gain 0.105). The ring fell to 1 mode at step 2410 and the deadline was 29/81. The prox gap, logged but not used for the rate, jumped from about 0.0007 on the hold to 0.033 after the shift. Both optimizers took 3600 Adam updates. Extra critic forwards: hold 2095, shift 2795.

**rr2** (`87dcf9ab…`). After the latch, rate gain follows the prox gap over a quiet prox baseline that does not chase spikes: gain 0 below 4×, gain 0.2 at or above 10×. The reference step moves from 0.001 to 0.0002 as that activity rises. The hold never crossed 4×, so the partial rate stayed off and the hold passed. At step 2449 the ratio was 52 and the applied rates were G/D **0.000884**, prior **0.00204**. Eight modes and HQ 0.949 were back at step 2500. Alpha 0.0002 then kept the ratio above 10 through step 3600, so the 0.2 rate never released. The deadline window passes 52 checks and then breaks; sustained recovery starts at step 3540 (delay 1140). Final live 8 modes, HQ 0.95654. Final optimizer rates were still the partial rate, with 3600 updates on both roles, so this was not a frozen generator. Extra critic forwards on the shift: 2795.

**rr3** (`fadcc121…`). Same prox opener, but the reference step stays 0.001, and while the gap is open the quiet baseline leaks toward the current prox at 0.002 per critic step. Hold passed with the same minima as rr2. The shift opened gain 0.2 at step 2449 (ratio 21) and the leak closed it by step 2499 (ratio 2.94), before 8 modes had returned (step 2500: 6 modes, HQ 0.421). The crept baseline then treated the still-large prox as normal, the ratio stayed near 1, and the rate sat on the floor. Deadline **0/81**. Final live 6 modes, HQ 0.64526. Both optimizers still took 3600 updates and ended on the floors.

## NOT_RUN

Matched frozen recovery, the sensitive four, the other 18 toys, native 7000-update coverage and accuracy, the two-horizon prefix, and delayed or repeated changes. No live shift returned UNCONFIRMED, so the frozen control was not started. None of these are inherited from K3P.

## Replay

```sh
export CUDA_VISIBLE_DEVICES=GPU-72c1b506-891d-b8bc-b353-e020585e1c47
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/tmp/pr38-default-env/bin/python
REPO=/ml2/hypergan/gan-attempts/claude-pool-20260925T041936Z/qualify_a2_bounded_damp/20260925T041936Z-2509885/repo/reports/toy100/qualify-a2-attempt/prepared/repos/cuda
FIX=/ml2/hypergan/ParticleGAN-epsilon-gan-followup/reports/toy100/direct-particle-base/initialization-fixtures/mode_hold/initial-values.pt
BASE=/ml2/hypergan/gan-attempts/formulations-20260925T172923Z/k3p_reference_response/20260925T172923Z-3719944/repo/reports/toy100/k3p-reference-response

$PY -u $BASE/rr2/hold.py --repo $REPO --config $BASE/rr2/config.json \
  --task mode_hold --backend cuda --initial-state $FIX \
  --output <fresh-dir> --network-floor 0.01 --prior-floor 0.05 --anneal-start 0.6

$PY -u $BASE/rr2/shift.py --repo $REPO --config $BASE/rr2/config.json \
  --task mode_hold --backend cuda --initial-state $FIX \
  --output <fresh-dir> --network-floor 0.01 --prior-floor 0.05 --anneal-start 0.6
```

rr1 and rr3 use the same commands with their own directories. Logs and raw results: `reports/toy100/k3p-reference-response/logs/` and `runs/`. Ledger: `tests.jsonl` beside this file.

## Next mechanism

Keep rr2's prox-gap opener. It is the signal that stays quiet on a passing hold and opens gain 0.2 on the shift; critic-parameter RMS does not separate those states. Keep the K3P reference step 0.001. Do not slow it to 0.0002: that pinned the gap, and the partial rate, through the rest of the run. Do not leak the baseline at 0.002: that released the rate before 8 modes returned and then hid the remaining gap. The untested middle is a slower release of the same 0.2 rate, long enough to finish the reacquisition rr2 already showed at step 2500, without a fixed dwell and without restoring the early penalty. That candidate was not run. Noise on horizon 1200 is still an ablation to remove before any horizon-free claim.
