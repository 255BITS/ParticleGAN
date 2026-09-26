# K3P noise and signal — round 2

K3P stays the selected base. Nothing was promoted. `current-research-base.json` was not edited. Parent measured scores remain 22/22 toys, ring hold 1200/1200, extension 300/300, target-shift deadline 28/81.

This lane removed horizon-tied noise and the learning-rate handover clock, starting from copied K3P files. Three proposals. No candidate passed hold, the 300-update extension, and the shift deadline together. Toy gates, the matched frozen control, the two-horizon prefix, and the delayed/repeated-change stress are NOT_RUN.

## Leaderboard

Ranked by hold, then extension, then timely recovery. Parent row is the published measurement, not a rerun.

| Candidate | Hold | Extension | Recovery deadline | Continued pre-shift | Toys |
|---|---|---|---|---|---|
| K3P parent (published) | 1200/1200 | 300/300 | FAIL 28/81, delay 1130 | pass (published) | 22/22 |
| ns3 fixed schedule + shock | 1200/1200 PASS, converged step 1881, min HQ 0.964 | FAIL 274/300, min modes 7, min HQ 0.918 | FAIL 0/81, suffix 0, min modes 6, min HQ 0.724 | FAIL 92/120, min HQ 0.673 | NOT_RUN |
| ns2 paced floor | FAIL NOT_CONVERGED, longest streak 15, 8 modes only at step 6300 | NOT_RUN | FAIL 13/81, delay 1110, min modes 6, min HQ 0.470 | FAIL 0/120, min modes 5 | NOT_RUN |
| ns1 cosine heat | FAIL NOT_CONVERGED, streak 0, max 7 modes | NOT_RUN | FAIL 0/81, min modes 1, min HQ 0.045 | FAIL 0/120 | NOT_RUN |

Deadline means the 81 checks from step 2800 through 3600 after the shift at 2400. One miss fails it. An 81/81 with a failed pre-shift would still fail; none of these reached 81/81.

## What was tested

All three copies keep K3P's `a_r1r2`, `b_cap`, EMA prox, anchor decay 0.999, 5× guard after 200 critic steps, learned particle prior, direct response, and sparse latent rule. CLI floors were 1 and anneal start 0 so the frozen schedule return is identically 1; each mechanism discards that return. Base rates stay 0.00425 and prior ×2. No seed sweep, no coefficient grid, no target centers, no change times, no metric feedback.

### ns1 cosine heat — FAIL / FAIL

Cosine of successive critic gradients drove one heat. Full rate until a slow EMA near −0.5, then the parent floors. Noise burst tracked positive cosine. Mix `s = min(1, heat/2)` did not read LR.

Measured: cosine kept flipping, slow EMA stayed near −0.25, heat stayed 0.97, applied critic LR stayed 0.00335–0.00425 for all 3600 shift steps, `s` stayed 1, anchor never started (`pure_a` 6300 on the hold, `blend` 0, `pure_b` 0). Hold settling 4800/4800 failed, max 7 modes. Shift deadline 0/81. Final live 6 modes at HQ 0.958. The output was not frozen: 3600 optimizer observations.

Mechanism sha256 `adee5e40acc607dfeabbc81ff7ea3351f9450ac216cdc4b60e9a0110dc43fb91`.

### ns2 paced floor — FAIL / FAIL

NS1 showed cosine never authorizes the floor. ns2 leaks heat after a fixed pace memory crosses 0.40 (~step 1020), floors with `heat *= 0.994`, and recharges on critic grad-RMS above 1.5× a rise-fast fall-slow envelope. Same `s` rule. Noise burst leaks at 0.98 and recharges on that shock.

Measured: the floor arrived while only 5 modes were covered (step 1000: 5 modes at full rate; step 1600: critic LR 2.7e-4 and still 5 modes). A later shock reopened burst to 0.96 and heat to 0.49 near step 2650 and collapsed the partial ring. Eight modes appeared at the settling budget's end (step 6300, HQ 0.941) with streak 15, not 200. Extension NOT_RUN. Shift deadline 13/81, delay 1110, continued hold 0/120. Critic LR range on the shift 1.08e-4–0.00425. Anchor blended (`blend` 2448) but never reached pure `b_cap` (`pure_b` 0).

Mechanism sha256 `d8d5261b5995e1389d0e894dd75663eb5e396228292b52eb624a4fc68afdba11`.

### ns3 fixed parent schedule + shock — hold PASS, extension FAIL, shift FAIL

ns1 never left full rate. ns2 floored before 8 modes existed. ns3 restores the parent's network cosine as constants, not as a read of `total_steps` or `network_lr_horizon_cap`: `learning_rate_scale(step, 1600, 0.6, 0.01)`. Input noise is `0.5 * max(0, 1 - step/120)`. Output noise is `0.029 * min(1, step/240)`. Those 120 and 240 steps are the ring probe's `0.1*1200` and `0.2*1200`, frozen as constants. Prior multiplier stays 1, which is the parent's prior multiplier through step 1600 on both the 7500-step hold budget and the 3600-step shift budget. After the scheduled multiplier is ≤ 0.05, boost = `0.98*boost + 0.02*shock` and the applied network multiplier is `scheduled + (1-scheduled)*boost`. Mix `s` uses the scheduled multiplier in the K3P handover, not the applied LR, so a reopen would not turn the anchor off.

Measured hold: converged at step 1881, then 1200/1200 with min HQ 0.964 and 8 modes. The next 300 updates passed only 274/300 (min modes 7, min HQ 0.918). That extension miss fails the conjunction. Around step 1900 a small boost of 0.020 lifted critic LR from the floor 4.3e-5 to 1.28e-4 while `s` was already 0. Anchor did engage (`pure_b` 1780 on the hold).

Measured shift: continued hold 92/120, min HQ 0.673 (the anneal dip around steps 1400–1650, HQ 0.86–0.90 and a 7-mode step). Deadline 0/81, passing suffix 0, min modes 6, min HQ 0.724. Final 7 modes at HQ 0.897. Final boost 0.004, final `s` 0, critic LR range 4.26e-5–0.00425, prior fixed at 0.0085, 3600 optimizer observations. The 1.5× grad-RMS shock did not treat the target change as a shock, so the rate never reopened. Extra anchor forwards: 317 blend + 1999 pure-b calls on the shift.

Mechanism sha256 `e3087c3150f04d859327cd823b2459c81a8b5b1b78044e23ed94a8c11bd5e472`.

## Budget dependencies that remain

Applied amplitudes do not read `total_steps`, `noise_horizon`, or `network_lr_horizon_cap`. The frozen validators are still called and their schedule returns are discarded. The cap remains in the copied config because the frozen policy requires it beside a floor.

ns3's 1600, 0.6, 0.01, 120, and 240 are fixed copies of the parent ring schedule. Changing the declared horizon would not change them. A matched two-horizon prefix was NOT_RUN, so that independence is from the formulas, not from a paired training. The parent's prior anneal still depends on `total_steps`; ns3 replaced it with a constant 1. Guard warmup of 200 and the 0.98/0.999 memories are estimator constants.

Direct-response gain still multiplies direct particle steps from particle-gradient alignment and restores the base rate.

## Not run

Sensitive screens, the other 18 toys, native 7000-update coverage and accuracy, the matched frozen recovery control, the two-horizon prefix, and the delayed/repeated-change stress. No candidate cleared hold and extension and the shift deadline. K3P was not rerun.

## Replay

```sh
export CUDA_VISIBLE_DEVICES=GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/tmp/pr38-default-env/bin/python
REPO=/ml2/hypergan/gan-attempts/claude-pool-20260925T041936Z/qualify_a2_bounded_damp/20260925T041936Z-2509885/repo/reports/toy100/qualify-a2-attempt/prepared/repos/cuda
FIX=/ml2/hypergan/ParticleGAN-epsilon-gan-followup/reports/toy100/direct-particle-base/initialization-fixtures/mode_hold/initial-values.pt
CAND=reports/toy100/k3p-noise-signal-3675742/ns3
$PY -u $CAND/hold.py --repo $REPO --config $CAND/config.json --task mode_hold --backend cuda \
  --initial-state $FIX --output /tmp/ns3-hold --network-floor 1 --prior-floor 1 --anneal-start 0
$PY -u $CAND/shift.py --repo $REPO --config $CAND/config.json --task mode_hold --backend cuda \
  --initial-state $FIX --output /tmp/ns3-shift --network-floor 1 --prior-floor 1 --anneal-start 0
```

Swap `ns3` for `ns1` or `ns2`. Logs and `result.json` are under `reports/toy100/k3p-noise-signal-3675742/runs/`. Gate rows are in the attempt `tests.jsonl`.

## Next mechanism

Keep ns3's fixed acquisition schedule. It is the only variant that produced a 1200-update 8-mode hold. Do not go back to a cosine floor (ns1) or an earlier pace floor (ns2). The shift was invisible to a 1.5× grad-RMS envelope: boost ended at 0.004 and the deadline was 0/81, with the anchor correctly left on (`s` 0). The next change is the reopen detector only, aimed at a sustained post-shift gradient that ordinary anneal spikes do not trip. The step-1900 boost of 0.02, which lifted LR to 1.3e-4, is the likely source of the extension's 26 misses and should not get easier to trigger during a settled ring.
