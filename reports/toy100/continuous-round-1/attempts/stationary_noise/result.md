# Stationary noise lane

Parent K3P was not rerun. Published scores stay 22/22 toys, ring hold 1200/1200, extension 300/300, target-shift recovery 28/81. Pinned sources were copied and left unchanged (`reports/toy100/gap-fill-20260925/sources/k3p/`, mechanism sha256 `d2eb08ee932b288cbba25cd1e7be3a9572b129bd1baf0b79718be1eb37ba9391`).

This lane removed horizon-tied noise and the learning-rate clock on the critic anchor. No candidate passed hold plus extension plus timely recovery. Toy screens, the 22-gate suite, the frozen recovery control, and the delayed/repeated-change stress were not run.

## What changed

All three candidates share one noise rule and one anchor, and differ only in the rate map.

- Output noise is the declared peak 0.029 on every step. The warmup fraction is validated and ignored.
- Input noise is `0.5 * scale`, where `scale = clip((Adam ratio - 1) / 4, 0, 1)` and the ratio is critic gradient RMS over sqrt(bias-corrected Adam second moment). Before that ratio is finite, scale stays at 1. K3P's linear decay to zero is validated, then discarded.
- The critic penalty never reads the learning-rate ratio. Mixing weight `s = 0.5 * scale`, so the EMA anchor weight `1-s` is always in `[0.5, 1]`. Anchor decay stays 0.999. The spike guard stays at 5x after 200 critic steps.
- Host losses, architecture, seeds, the learned particle prior, direct response, and the sparse latent rule are the K3P copies. `observation_sigma` 0.03 stays a constant host term.
- Base rates are still multiplied by the frozen schedule function, but that function is called with network floor 1, prior floor 1, and anneal start 0. With floor 1 its multiplier is identically 1 for every horizon. SN2 and SN3 then replace the returned multiplier.

Active noise during training: adaptive input noise (it fell near zero once the Adam ratio settled, in the SN1 trace by step 100), constant output noise 0.029, and the constant host `observation_sigma`. No returned amplitude uses `total_steps`.

## Leaderboard

Ranked by hold, then extension, then timely recovery. Parent row is the published measurement.

| Candidate | Hold | Extension | Recovery deadline | Delay | Toys |
|---|---|---|---|---|---|
| K3P parent (published) | 1200/1200 PASS | 300/300 PASS, min HQ 0.988 | 28/81 FAIL | 1130 | 22/22 |
| sn3 asymmetric rate | 156 then FAIL at step 2679 | not a passed-hold extension; post-failure diagnostic 194/301 | 80/81 FAIL | 550 | NOT_RUN |
| sn1 constant full rate | 32 then FAIL at step 3818 | post-failure 28/301 | 44/81 FAIL | 890 | NOT_RUN |
| sn2 symmetric reversal | 23 then FAIL at step 1424 | post-failure 124/301 | NOT_RUN | | NOT_RUN |

Timely recovery means all 81 diagnostic checks from step 2800 through 3600 after the shift at 2400. One miss fails it.

## sn1 — constant full rate

Sources: `reports/toy100/stationary-noise-3603401/sn1/`. Mechanism sha256 `0900199253faf907ded5167ec825aca978892fef188935ddd1896a3f179b9ebf`. Config sha256 `9872c4824e5d69c00721a7771279a84b32121198af123cd7dadab70f7536dcf5`.

The anchor was on from penalty call 1 (`pure_a` 0, `blend` 1166, `pure_b` 2952, anchor started at call 2). Input noise was 0.5 at step 1 and 0 by step 100. Critic and generator rates stayed at 0.00425, prior at 0.0085, for every observed step. Optimizers still took 4118 and 3600 updates; the output was not a frozen copy.

Hold: converged at 3785, lost HQ at 3818 (8 modes, HQ 0.895), then fell to 0 modes. 185.4 s. Artifact `runs/sn1-hold/result.json`.

Shift: continued hold 52/120, min HQ 0.040. Deadline window 44/81, stable only from step 3290, delay 890. Final state was 8 modes at HQ 0.9998, after the deadline. 149.0 s. Rates remained exactly 0.00425 / 0.0085, with 3600 optimizer updates. Artifact `runs/sn1-shift/result.json`.

Full rate with a quiet noise rule and an active anchor can reacquire late. It does not hold.

## sn2 — symmetric reversal

Sources: `reports/toy100/stationary-noise-3603401/sn2/`. Mechanism sha256 `5e985e83533e2c586a6307f8fd95d83a67a17a57dd6b8d0aa783c94b0f8a6b11`.

`flip_ema` is a decay-0.9 average of `relu(-cosine)` of successive critic gradients. `calm = 1 - flip_ema`. Network multiplier `0.01 + 0.99*calm`, prior multiplier `0.05 + 0.95*calm`.

Hold: converged at 1400 with zero settling failures, the same acquisition step as K3P, then failed at 1424 (HQ 0.801). During the passing window the sampled critic cosine was about -0.8, but the critic rate was still 0.0024 to 0.0035, roughly 60 times the parent floor 4.25e-5. 69.8 s. Artifact `runs/sn2-hold/result.json`. Shift was not run. The hold had already failed and the rate had not reached the holding floor.

## sn3 — slow reversal, fast release

Sources: `reports/toy100/stationary-noise-3603401/sn3/`. Mechanism sha256 `5f235e5d55b5128cd5d6bf72feb3a060187db354a70001a3fb8fba3077a30755`. Config sha256 `37a4f4b18634a757a3ee27046f45011e6ef12596e50895e8ce1bef6eaab30629`.

Negative cosine accumulates with 0.999; a nonnegative cosine multiplies the memory by 0.99. The rate hits the parent floor only once `flip_ema` reaches 0.25. That level is the reversal memory SN2 still had while stepping too hard. It was not swept.

Hold: converged at 2522, 156 good hold updates, failed at 2679 (window min HQ 0.618, min modes 6). Post-failure diagnostic 194/301, ending at 8 modes and HQ 0.916, which does not reopen the gate. Critic rate samples stayed between 0.00293 and 0.00425. The fast release wiped the slow accumulation whenever a step was not a reversal, so the floor was never reached. Anchor calls: `pure_a` 0, `blend` 648, `pure_b` 2331. 113.9 s. Artifact `runs/sn3-hold/result.json`.

Shift: continued hold 57/120, min HQ 0.090. After the shift, modes went to 0 at step 2400 and the optimizers kept updating (3600/3600; critic rate still moved, 0.00269 to 0.00425). Deadline window 80/81. The single miss is step 2940. Stable suffix starts at 2950, delay 550. Final and EMA were 8 modes at HQ 1. Min HQ inside the deadline window was 0.916 and min modes 7. 129.8 s. Artifact `runs/sn3-shift/result.json`.

80/81 is still a failed deadline. No frozen control was run.

## Budget dependencies that remain

The applied noise amplitude and, for SN2/SN3, the applied rate multiplier do not change if `total_steps` or `network_lr_horizon_cap` change. These reads are still in the process:

- `hold.py` and `shift.py` still pass `noise_horizon=1200` into `NoisePolicy`. The wrappers validate with that integer and discard the annealed value.
- The frozen `policy_multipliers` is still called with `total_steps` and the cap 1600. SN1 uses its return, which is identically 1 at floor 1. SN2 and SN3 discard it and cache their own multiplier by step index.
- The cap remains in the config because the frozen policy requires it beside a network floor.
- The 200-step guard warmup and the cosine EMA decays are estimator memories, not a declared training horizon.
- A two-horizon training prefix was not run, so that independence is from the formulas and the rate traces, not from a matched pair of short trainings.
- Direct-response gain still multiplies direct particle steps from gradient alignment and then restores the base rate. On this ring the logged prior rate matched the schedule, so that gain was 1 at the sampled steps.

## Not run

Sensitive screens (`mode_hold`, unequal mass, unequal width, stripes), the other transfer toys, native 7000-update coverage and accuracy, the matched frozen control, and the long-hold / second-change stress. None of those were opened, because no candidate cleared its own hold and timely recovery.

## Replay

```sh
export CUDA_VISIBLE_DEVICES=GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
REPO=/ml2/hypergan/gan-attempts/claude-pool-20260925T041936Z/qualify_a2_bounded_damp/20260925T041936Z-2509885/repo/reports/toy100/qualify-a2-attempt/prepared/repos/cuda
FIX=/ml2/hypergan/ParticleGAN-epsilon-gan-followup/reports/toy100/direct-particle-base/initialization-fixtures/mode_hold/initial-values.pt
PY=/tmp/pr38-default-env/bin/python
CAND=reports/toy100/stationary-noise-3603401/sn3
$PY -u $CAND/hold.py --repo $REPO --config $CAND/config.json --task mode_hold --backend cuda \
  --initial-state $FIX --output /tmp/sn3-hold --network-floor 1 --prior-floor 1 --anneal-start 0
$PY -u $CAND/shift.py --repo $REPO --config $CAND/config.json --task mode_hold --backend cuda \
  --initial-state $FIX --output /tmp/sn3-shift --network-floor 1 --prior-floor 1 --anneal-start 0
```

Swap `sn3` for `sn1` or `sn2`. Logs and raw results are under `runs/`. Gate rows are in `tests.jsonl`.

## Next mechanism

Noise annealing was not the piece that kept the ring stable. With that anneal removed and the anchor forced on, the hold lasted only while the rate happened to be moderate, and it broke while the critic rate was still about 0.003. The parent holds at about 4.25e-5. A following attempt should make the calm rate actually sit on that floor under mixed gradient signs, and raise it only after a sustained aligned run that can move inside the 400-update deadline. Do not search noise amplitudes, and do not treat the step-2940 miss as a pass.
