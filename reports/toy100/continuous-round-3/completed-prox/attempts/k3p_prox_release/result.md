# k3p_prox_release

K3P stays the selected base. No candidate passed its own hold, the 300-update extension, and timely shift recovery together. `current-research-base.json` was not edited. The pinned parent sources were not edited. Nothing here is promoted.

Parent, reused and not rerun: 22/22 GPU toys, ring hold 1200/1200 (min HQ 0.90723), extension 300/300 (min HQ 0.98779), target-shift deadline **FAIL 28/81**, delay 1130. Mechanism `d2eb08ee932b288cbba25cd1e7be3a9572b129bd1baf0b79718be1eb37ba9391`.

Three proposals, one GPU worker. Each copies K3P `config.json` `a1475108…`, `latent.py` `197df635…`, and `response.py` `7e71d60a…`. Learned particle prior, direct response, and bounded sparse-latent damping are unchanged. Cold path matches the reference-response latch: pure `a_r1r2` and peak rates until critic-gradient RMS stays under a quarter of its post-warmup peak for 250 critic steps, then the mixing gain decays at 0.99. After that gain hits 0 it stays 0, so the early penalty does not return and the critic constraint does not read the LR clock. Reference step stays 0.001. The quiet prox baseline updates only while prox/quiet is under 4. Rates ignore step, horizon, and anneal.

**Remaining budget dependence:** input noise 0.5 over the first 0.1 of `noise_horizon=1200`, and output noise warming to 0.029 over 0.2 of that horizon, are still the driver schedule. That inherited noise is a labeled ablation on every candidate. A paired prefix under two declared horizons was **NOT_RUN**.

## Leaderboard

Ranked by hold, extension, and timely recovery. None is timely (81/81 from step 2800 through 3600). Toy gates were not opened.

| Candidate | Own hold | Extension | Pre-shift | Deadline | Delay | Toys |
|---|---|---|---|---|---|---|
| K3P parent (not rerun) | 1200/1200 | 300/300 | published pass | FAIL 28/81 | 1130 | 22/22 |
| px3 calm innovation | **1200/1200**, min HQ 0.91699, converged 1400 | **300/300**, min HQ 0.98291 | stationary 5/5, continued **120/120** | **FAIL 71/81** | 850, stable 3250 | NOT_RUN |
| px1 half-peak cutoff | **1200/1200**, min HQ 0.91699, converged 1400 | **300/300**, min HQ 0.98291 | stationary 5/5, continued **120/120** | FAIL 16/81 | none | NOT_RUN |
| px2 proportional gap | **1200/1200**, min HQ 0.91699, converged 1400 | **300/300**, min HQ 0.98291 | stationary 5/5, continued **120/120** | FAIL 0/81 | none | NOT_RUN |

px3 is the strongest deadline count in this lane and still fails. 71/81 is not a pass. Its delay is shorter than the parent's and the final live state is 8 modes, HQ 0.97705, with both optimizers at 3600 updates and the network rate back on the floor (`4.25e-5`). That is active adaptation followed by an early floor, not a frozen generator.

## What was measured

All three holds match RR2's minima. Step 600 is 8 modes at HQ 0.99243 on the logged runs. The post-latch rule never opens on the hold: prox/quiet stays under 2. Extra critic forwards follow the reference-response pattern, 2095 on hold and 2795 on shift for the logged receipts. Mixing weight `s` is 0 after the latch on every shift trace.

**px1** (`5e4e33bc…`), `reports/toy100/k3p-prox-release/px1/`. Mobility is 1 at the excursion peak and 0 once prox falls to half that peak. The shift peak was about 0.052 by step 2499, prox was already under half, and gain hit 0 while the ratio was still 25. Later positive innovations stayed under that peak, so the rate never returned. Deadline checks kept 8 modes only down to 7, min HQ 0.78223, and 16/81 passed. Final live 8 modes, HQ 0.86670, both roles on the floors. Hold 128s, shift 133s.

**px2** (`c32e4141…`), `px2/`. Removes the half-peak zero. Gain is `0.2 * opener * (prox/peak)`. A spike ratcheted the peak to about 0.122, and gain then sat near 0.03–0.07 (network LR about `1.4e-4` to `2.6e-4`) through step 3600. Deadline **0/81**, min modes 1, min HQ 0.07056. Final live 7 modes, HQ 0.89502. The moderate rate neither reacquired a stable ring nor sat on the floor. Hold 138s, shift 168s.

**px3** (`33e44766…`), `px3/`. Gain is the full 0.2 cap while a 0.01 smoother of prox is still moving faster than a quarter of its fastest speed in the excursion, then the floor. Reopen requires prox above four times the smoothed level at calm time. The cap was on at step 2449 (gain 0.2, G/D LR `8.84e-4`). The calmer latched by step 2499, while the smoother was still rising (`prox_s` 0.0109, later peaked near 0.0226) and before 8 modes had returned (step 2500: 7 modes, HQ 0.64233). It did not reopen: later prox stayed under `4 * 0.01089`. From the floor, HQ climbed through the deadline. All 81 deadline checks have 8 modes. The 10 failures are HQ under 0.90: steps 2800–2870 (0.86743 up to 0.89575) and 3230–3240 (0.89014, 0.89111). Stable suffix starts at 3250 (delay 850). Final live 8 modes, HQ 0.97705. Final optimizer rates are the floors, G/D `4.25e-5` and prior `4.25e-4`, with 3600 updates on both roles. Hold 134s, shift 154s.

The reference-response comparison that motivated the lane still stands: RR2's slowed reference pinned gain 0.2 and scored 52/81; RR3's baseline leak scored 0/81. px3's floor-after-motion is the best count here (71/81) and still misses the first eight deadline checks because the calm latch is a lull during the smoother's rise, about 300 updates before the ring is actually steady.

RP1's measured false reopen on a stationary native grid was not copied. px3's reopen did not fire on the shift plateau, and the hold ratio gate stayed shut, so this lane did not reproduce that late restart.

## NOT_RUN

Matched frozen recovery, `img_intensity2`, full 7000-update grid100 coverage and accuracy, the sensitive four, the other 18 toys, the two-horizon prefix, delayed or repeated changes, the 30000-update continuation, and native seeds 1235–1237. No live shift returned UNCONFIRMED, so the frozen control was not started. None of these are inherited from K3P or from RR2/RR3/RP1.

## Replay

```sh
export CUDA_VISIBLE_DEVICES=GPU-72c1b506-891d-b8bc-b353-e020585e1c47
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/tmp/pr38-default-env/bin/python
REPO=/ml2/hypergan/gan-attempts/claude-pool-20260925T041936Z/qualify_a2_bounded_damp/20260925T041936Z-2509885/repo/reports/toy100/qualify-a2-attempt/prepared/repos/cuda
FIX=/ml2/hypergan/ParticleGAN-epsilon-gan-followup/reports/toy100/direct-particle-base/initialization-fixtures/mode_hold/initial-values.pt
BASE=/ml2/hypergan/gan-attempts/formulations-20260925T180919Z/k3p_prox_release/20260925T180919Z-3766575/repo/reports/toy100/k3p-prox-release

$PY -u $BASE/px3/hold.py --repo $REPO --config $BASE/px3/config.json \
  --task mode_hold --backend cuda --initial-state $FIX \
  --output <fresh-dir> --network-floor 0.01 --prior-floor 0.05 --anneal-start 0.6

$PY -u $BASE/px3/shift.py --repo $REPO --config $BASE/px3/config.json \
  --task mode_hold --backend cuda --initial-state $FIX \
  --output <fresh-dir> --network-floor 0.01 --prior-floor 0.05 --anneal-start 0.6
```

px1 and px2 use the same commands with their own directories. Logs: `reports/toy100/k3p-prox-release/logs/`. Raw results: `reports/toy100/k3p-prox-release/runs/`. Ledger: `tests.jsonl` beside this file.

## Next mechanism

Keep px3's structure: full 0.2 cap while the excursion is moving, floor after it calms, and no reopen on the 2.5× prox wander already seen at the floor. Move the calm test off the first lull. At the step 2499 latch, smoothed velocity was still positive and `prox_s` kept rising until about step 2699. The next rule should arm the floor only after that smoother has peaked and its contraction has itself calmed, so the cap still covers the reacquisition RR2 showed by step 2500 and the floor is already on before step 2800. Do not return to a half-peak cutoff or a prox/peak proportional rate: those scored 16/81 and 0/81. Remove the horizon-1200 noise only on a candidate that already clears 81/81.
