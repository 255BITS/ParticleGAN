# k3p_critic_confidence

K3P stays the selected base. Nothing here is promoted. The pinned parent was not rerun and its mechanism hash is still `d2eb08ee932b288cbba25cd1e7be3a9572b129bd1baf0b79718be1eb37ba9391`. Three proposals each ran the canonical ring hold and the canonical target-shift. All six gates FAIL. Image screening, the 22-toy matrix, native 7000-update runs, the matched frozen control, the horizon-prefix audit, and the delayed/repeated stress are NOT_RUN.

Parent published scores, kept for comparison only: hold 1200/1200, extension 300/300, shift deadline 28/81, toys 22/22.

## Leaderboard

Ranked by hold, then extension, then timely recovery. Toy passes stay NOT_RUN because no candidate cleared the ring.

| Candidate | Hold | Extension | Shift deadline | Stationary / continued hold | Toys |
|---|---|---|---|---|---|
| K3P parent (published) | 1200/1200 | 300/300 | FAIL 28/81, delay 1130 | published pass | 22/22 |
| cc1 half-split coherence | FAIL, entered hold then lost it: converged step 3933, 153 good hold updates, first failure 4087 | NOT_RUN | FAIL 0/81 | 0/5 and 0/120 | NOT_RUN |
| cc3 frozen-baseline quiet | FAIL NOT_CONVERGED, longest streak 120, final 6 modes at step 6300 | NOT_RUN | FAIL 0/81 | 5/5 and 33/120 | NOT_RUN |
| cc2 margin uncertainty | FAIL NOT_CONVERGED, streak 0, final 6 modes at step 6300 | NOT_RUN | FAIL 0/81 | 0/5 and 0/120 | NOT_RUN |

cc1 is the only run that produced 200 consecutive 8-mode checks. The following 300 updates are a post-failure diagnostic (123/301, minimum 0 modes), not the canonical extension. cc3’s shift stationary window passed 5/5, then the continued pre-shift hold failed 33/120.

## What was tested

All three keep K3P’s learned particle prior, direct response, bounded sparse latent rule, `a_r1r2` / `b_cap` / EMA prox, anchor decay 0.999, and the 5× guard after 200 critic steps. `config.json`, `latent.py`, and `response.py` are byte copies of the pinned bundle. The mixing weight uses the K3P handover on a controller gain and does not read the applied learning rate. Network multiplier `0.01 + 0.99 * gain`, prior multiplier `0.05 + 0.95 * gain`. Host noise is unchanged on every candidate: the ring drivers still pass `noise_horizon=1200`. That noise is a labeled intermediate ablation. The rate rules do not read step, total, horizon, scores, task identity, centers, or the shift time.

### cc1 half-split coherence — hold FAIL, shift FAIL

Coherence `clip(1 - ||g1-g2||^2 / ||g1+g2||^2, 0, 1)` on Rp-logistic parameter gradients of the two batch halves (mean loss, so the ratio is batch-normalized and scale-free). Asymmetric EMA, rise weight 0.10 and fall weight 0.01. Gain is 0 at or below 0.08 and 1 at or above 0.35 after 200 steps. Each penalty call adds four critic forwards and two backwards.

Measured hold (202.1 s): EMA stayed in [0.536, 0.980], gain stayed 1, `s` stayed 1, anchor never started, critic LR stayed 0.00425. It reached an 8-mode streak at step 3933 and lost the hold at step 4087. The post-failure window ended at 2 modes, HQ 0.165. Shift (166.1 s): deadline 0/81 (minimum 5 modes, HQ 0.366), stationary 0/5, continued hold 0/120. Final live 6 modes at HQ 0.984. Applied rates were 0.00425 / 0.0085 for all 3600 critic, generator, and prior updates, so the output was not a frozen snapshot. The controller did not move after the shift. Extra shift evaluations: 14400 forwards and 7200 backwards.

Mechanism sha256 `9d9ad253d4943587f82e56de7d479890b07ca5a2f8b0f98ee9399a8da1e9a722`.

### cc2 margin uncertainty — hold FAIL, shift FAIL

`uncertainty = var(margin) / (var(margin) + mean(margin)^2)` over paired critic margins, confidence `1 - uncertainty`, gain `relu(2 * confidence - 1)` after a symmetric EMA of weight 0.05. Gain hits 0 when the batch standard deviation is at least the absolute mean. Two no-grad critic forwards per penalty call.

Measured hold (318.4 s): uncertainty sat near 0.5–0.7. Pure early penalty lasted 201 calls. The anchor started at call 202 and `s` reached 0 there. Longest 8-mode streak was 0. Final live and EMA were 6 modes at HQ 0.829, at the floor LR 4.25e-5. Confidence EMA ended at 0.346, so the brief low-uncertainty samples (raw minimum 0.092) did not reopen acquisition. A multi-modal batch does not share one margin, so this statistic floors during learning. Shift (165.1 s): deadline 0/81 (minimum 4 modes, HQ 0.102), stationary 0/5 (6 modes, HQ 0.804), continued hold 0/120. Final 6 modes at HQ 0.451. Final optimizer rates were the floor. 3600 updates ran. Extra hold forwards 18698, including the anchor forwards after step 202.

Mechanism sha256 `9f3f09972764da31fe7adc4e18f7b789ee1337c0f330f15a9c30046f3811e0ae`.

### cc3 frozen-baseline quiet — hold FAIL, shift FAIL

This uses the training critic-gradient RMS only. No probe forward or backward. While open, a slow EMA (weight 0.005) tracks that RMS. After 200 steps, 100 consecutive steps below half the EMA freeze the EMA and decay the gain by 0.98. Reopen requires 100 consecutive steps above 3× that frozen baseline, measured before the rate rises. The baseline is not updated after the close. That shape is a response to the offline RP1 grid trace: center RMS improved to 0.206 while the rate stayed at the floor, then a stationary reopen to multiplier 0.208 worsened it to 0.287. cc3 does not copy RP1’s 3-step surprise or its 0.2 reopen, and it does not read step 5672.

Measured hold (277.0 s): `close_count` 0, phase stayed `open`, gain stayed 1, `s` stayed 1, anchor never started, critic LR stayed 0.00425. Longest streak 120. Final live 6 modes at HQ 0.959. The RMS and its slow EMA moved together, so the half-baseline test did not hold for 100 steps. Shift (139.9 s): stationary 5/5 (minimum HQ 0.966), continued hold 33/120 (minimum 0 modes), deadline 0/81 (minimum 1 mode, HQ 0.077). Final 7 modes at HQ 1.000. Applied rates stayed 0.00425 / 0.0085 for all 3600 updates. Quiet counters in the trace were 0 or 1. The target change did not produce a sustained RMS rise against the slow EMA, and the controller had not closed, so there was no frozen baseline to reopen. Extra probe forwards: 0.

Mechanism sha256 `82836481f499e5888a5eadc81b7a897523faaa7e61066262b219e6b0acb2cf6f`.

## Budget dependencies that remain

On every candidate, `phase_multipliers` accepts step, total, anneal, floors, and the horizon cap and does not use them. The config still carries `network_lr_horizon_cap` because the host requires the field. Host noise still receives a horizon: 1200 on these ring drivers, and the task step count on transfer runs. Warmup 200, EMA weights, and cc3’s dwell of 100 are estimator constants. A paired two-horizon training prefix was NOT_RUN.

## Not run

`img_intensity2`, the other sensitive screens, the remaining toy gates, native grid100 / rotated100 / staggered100 at 7000 updates, seeds 1235–1237, the matched frozen control, the horizon-prefix audit, the delayed and repeated shift, and the 30000-update long continuation. No candidate cleared hold and the shift deadline together. K3P was not edited and was not rerun.

## Replay

```sh
export CUDA_VISIBLE_DEVICES=GPU-72c1b506-891d-b8bc-b353-e020585e1c47
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/tmp/pr38-default-env/bin/python
REPO=/ml2/hypergan/gan-attempts/claude-pool-20260925T041936Z/qualify_a2_bounded_damp/20260925T041936Z-2509885/repo/reports/toy100/qualify-a2-attempt/prepared/repos/cuda
FIX=/ml2/hypergan/ParticleGAN-epsilon-gan-followup/reports/toy100/direct-particle-base/initialization-fixtures/mode_hold/initial-values.pt
CAND=reports/toy100/k3p-critic-confidence-3766578/cc1
$PY -u $CAND/hold.py --repo $REPO --config $CAND/config.json --task mode_hold --backend cuda \
  --initial-state $FIX --output /tmp/cc1-hold --network-floor 0.01 --prior-floor 0.05 --anneal-start 0.6
$PY -u $CAND/shift.py --repo $REPO --config $CAND/config.json --task mode_hold --backend cuda \
  --initial-state $FIX --output /tmp/cc1-shift --network-floor 0.01 --prior-floor 0.05 --anneal-start 0.6
```

Swap `cc1` for `cc2` or `cc3`. Logs and `result.json` are under `reports/toy100/k3p-critic-confidence-3766578/runs/`. Gate rows are in the attempt `tests.jsonl`.

## Next mechanism

Keep full-rate acquisition long enough to form 8 modes: cc1 did that at step 3933, and cc3 had a streak of 120 while still fully open. Half-split gradient agreement does not later fall (cc1 EMA never left [0.536, 0.980]). Margin coefficient of variation is high on a mixture and floors at step 202 (cc2). Critic-gradient RMS divided by a tracking slow EMA also stays near 1, so a 100-step “below half” test never fires (cc3, `close_count` 0). The close needs a reference taken once and then frozen, and a reopen has to clear a bar that stationary fluctuations miss: the RP1 grid trace gained center RMS down to 0.206 at the floor and lost it after a stationary reopen to multiplier 0.208. Host noise is still on the 1200-step driver horizon and still has to be removed before any of these rules could be a final continuous formulation.
