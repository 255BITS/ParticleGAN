# K3P signal noise — round 3

K3P stays the selected base. Nothing was promoted. `current-research-base.json` was not edited. The pinned parent was not rerun and was not modified. Parent scores remain 22/22 toys, ring hold 1200/1200, extension 300/300, target-shift deadline 28/81.

Three proposals. No candidate passed its own hold, so the 300-update extension was not entered (SKIPPED / NOT_RUN). Every candidate also ran the shift. Toy gates, the matched frozen control, the two-horizon prefix, and the delayed/repeated-change stress are NOT_RUN. No candidate cleared hold, extension, and the deadline together.

## Leaderboard

Ranked by hold, then extension, then timely recovery. Parent row is the published measurement.

| Candidate | Hold | Extension | Recovery deadline | Continued pre-shift | Toys |
|---|---|---|---|---|---|
| K3P parent (published) | 1200/1200 | 300/300 | FAIL 28/81, delay 1130 | pass (published) | 22/22 |
| sn2 absolute-gap noise | FAIL NOT_CONVERGED, streak 0, max 7 modes, max HQ 1.0 | NOT_RUN | FAIL 0/81, min modes 6, min HQ 0.811 | FAIL 0/120, min modes 0 | NOT_RUN |
| sn3 settle cosine | FAIL NOT_CONVERGED, streak 0, max 5 modes, HQ 0.999 at 5 modes | NOT_RUN | FAIL 0/81, min modes 1, min HQ 0.166 | FAIL 0/120, min modes 4, min HQ 0.687 | NOT_RUN |
| sn1 relative score gap | FAIL NOT_CONVERGED, streak 0, max well below 8, last live 2 modes HQ 0.165 | NOT_RUN | FAIL 0/81, min modes 0, min HQ 0 | FAIL 0/120, min modes 0 | NOT_RUN |

Deadline means the 81 checks from step 2800 through 3600 after the shift at 2400. One miss fails it. None reached 81/81. Shift runs executed 3600 Adam updates on both optimizers, so a frozen generator is not the failure.

## What was tested

Copied K3P files under `reports/toy100/k3p-signal-noise-3725471/`. Only `mechanism.py` differs. `a_r1r2`, `b_cap`, EMA prox, anchor decay 0.999, 5× guard after 200 critic steps, learned particle prior, direct response, and the sparse latent rule are the parent copies. CLI floors were 1 and anneal start 0 so the frozen schedule returns 1; each mechanism discards that return. Base rates stay 0.00425 and prior ×2. No seed sweep, no coefficient grid, no target centers, no change times, no HQ feedback.

Mix `s` is the K3P handover of `r = 0.01 + 0.99*anchor_heat`. `anchor_heat` rises only while rate-heat > 0.5 and otherwise decays by 0.995. Applied LR is not the mix clock. A moderate reopen cannot restore pure `a_r1r2`.

Each penalty also takes two extra critic forwards for `q = mean D(real) - mean D(fake)` on the training batch. That is a training signal, not a toy score. SN3's shift logged 7200 of those forwards.

### sn1 relative score gap — FAIL / FAIL

`rel = |q_fast - q_slow| / ema|q|`, quiet below 0.05. Heat decays only when that quiet run exceeds 40 and generator displacement is under 0.15 of its first-200-step peak. Noise uses the same quiet test.

Measured: `q` itself falls to ~0.01 by step 200, so `rel` stays large. `quiet_run` never lasts. Input std stays 0.5 and output std stays 0 for all 6300 hold steps. Heat stays 1, `s` stays 1, anchor never starts (`pure_a` 6300, blend 0, pure_b 0). Last live point is 2 modes at HQ 0.165. Shift deadline 0/81, continued hold 0/120, critic LR fixed at 0.00425 across 3600 observations.

Mechanism sha256 `fb7afaab46a499c91c72c1a48c9eed7ea6090244152fbed77ecc73e23a10583f`.

### sn2 absolute-gap noise — FAIL / FAIL

Noise burn-in uses absolute `|q - q_noise|` with `q_noise` decay 0.8. Deviation ≤ 0.08 multiplies `noise_base` by 0.955; a larger deviation pauses it. After `noise_base` < 0.05, fifteen high steps can reopen at most 0.08 of noise. Rate leak `heat *= 0.996` starts only when critic grad RMS falls below 0.15 of its first-200-step peak.

Measured hold: input noise is effectively off by step ~250 and output std is 0.029. That is the first horizon-free noise rule here that actually leaves the initial burst. Live coverage reached 7 modes at HQ 1.0 around steps 1350–1850, then collapsed. The grad-RMS gate never opened (`decaying` false, `g_peak` 0.00870, `g_fast` stayed above it). Heat stayed 1 and `s` stayed 1. Noise reopen pulsed input std back up to ~0.04 near the collapses (step ~2600: 1 live mode). Final live 0 modes. Extension NOT_RUN.

Measured shift: deadline 0/81, min modes 6, min HQ 0.811. Continued hold 0/120. Final point 7 modes at HQ 0.999, which is not the deadline window. Critic LR range 0.00197–0.00425 (a late partial leak, not the 1% floor). 3600 optimizer updates.

Mechanism sha256 `97ef074bc535d971c0f8d73328c0a29dedad6130abe667ddfe7f8e80b99530d9`.

### sn3 settle cosine — FAIL / FAIL

Same burn-in as sn2, reopen removed. The first time `noise_base` < 0.05 (measured settle step 161) starts an 800-step full-rate plateau and then a 600-step cosine of heat from 1 to 0. Network multiplier `0.01 + 0.99*heat`. Prior multiplier `0.05 + 0.95*heat`, so the prior floors with the network. After the cosine, heat can rise by 0.01 per step, capped at 0.18, only if critic grad RMS stays above 2.5× a falling 0.995 baseline for 30 steps.

Measured hold: noise matches sn2 without the reopen pulses (input ~0, output 0.029). Cosine reaches the network floor 4.25e-5 and the prior floor 4.25e-4 by step ~1560. Coverage stops at 5 modes with HQ 0.999 (EMA 5 modes, HQ 1.0 at step 6300). Streak 0. The prior is at 5% of its base rate while K3P's prior is still at full rate through this part of a ring run. `pure_a` 1402, `blend` 4898, `pure_b` 0: `s` falls to ~0 but the blend branch is `s > 0`, so the run never takes the pure anchor penalty. Extension NOT_RUN.

Measured shift: stationary 0/5 (min HQ 0.977, modes short of 8). Continued hold 0/120, min modes 4, min HQ 0.687. Deadline 0/81, min modes 1, min HQ 0.166. Final 2 modes at HQ 0.241. Critic LR hits the floor and stays there (range 4.25e-5–0.00425). At step 2700 `rate_hi` is 11, then the baseline has already risen and the counter resets, so the 30-step reopen never applies. Final heat 0. 3600 Adam updates at the floor, not a stopped optimizer.

Mechanism sha256 `07717e24f8d7e89753aff02e0caf2a0d8caa9b8c5804a6723f703a9d4485847f`.

## Budget dependencies that remain

Applied amplitudes do not read `total_steps`, `noise_horizon`, `network_lr_horizon_cap`, shift time, or toy HQ. The frozen validators are still called and their schedule returns are discarded. The cap remains in the copied config because the frozen policy requires it beside a floor.

SN1 and SN2 have no step-horizon constants. Their decays, the 10/30/200-step calibrations, and the deviation thresholds are estimator constants. A matched two-horizon prefix was NOT_RUN, so independence is from the formulas, not from a paired training.

SN3's 800 and 600 are fixed lengths counted from the noise-settle event (step 161 on both the hold and the shift). They are not the declared training budget. They are still a schedule. Changing the declared horizon would not change them. The paired prefix was NOT_RUN.

Direct-response gain still multiplies direct particle steps and restores the base rate. Guard warmup of 200 is the parent estimator.

## Not run

Sensitive screens, the other 21 toys, native 7000-update coverage and accuracy, the matched frozen recovery control, the two-horizon prefix, and the delayed/repeated-change stress. No candidate cleared hold and extension and the shift deadline.

## Replay

```sh
export CUDA_VISIBLE_DEVICES=GPU-72c1b506-891d-b8bc-b353-e020585e1c47
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/tmp/pr38-default-env/bin/python
REPO=/ml2/hypergan/gan-attempts/claude-pool-20260925T041936Z/qualify_a2_bounded_damp/20260925T041936Z-2509885/repo/reports/toy100/qualify-a2-attempt/prepared/repos/cuda
FIX=/ml2/hypergan/ParticleGAN-epsilon-gan-followup/reports/toy100/direct-particle-base/initialization-fixtures/mode_hold/initial-values.pt
CAND=reports/toy100/k3p-signal-noise-3725471/sn3
$PY -u $CAND/hold.py --repo $REPO --config $CAND/config.json --task mode_hold --backend cuda \
  --initial-state $FIX --output /tmp/sn3-hold --network-floor 1 --prior-floor 1 --anneal-start 0
$PY -u $CAND/shift.py --repo $REPO --config $CAND/config.json --task mode_hold --backend cuda \
  --initial-state $FIX --output /tmp/sn3-shift --network-floor 1 --prior-floor 1 --anneal-start 0
```

Swap `sn3` for `sn1` or `sn2`. Logs and `result.json` are under `reports/toy100/k3p-signal-noise-3725471/runs/`. Gate rows are in the attempt `tests.jsonl`.

## Next mechanism

Keep SN2/SN3's absolute-deviation burn-in and do not reopen noise. That is the only rule in this lane that turned input noise off without a horizon and still made a sharp partial ring. Drop the relative-gap test (SN1) and the grad-RMS floor gate (SN2); full rate keeps the RMS above the early peak, so the gate never fires.

Do not tie the prior multiplier to network heat. SN3 put the prior on its 5% floor by step ~1560 and coverage froze at 5 modes with HQ 0.999. K3P's prior is still at full rate through that window. Floor the network only, and not on a cosine that finishes while the ring is still at 5 modes.

Score-gap deviation is not a shift detector: `|q - q_noise|` stays large on a sharp ring. The 2.5× local RMS test also missed this shift (`rate_hi` peaked at 11, under the 30-step bar) because the baseline rose with the new RMS. The next reopen has to use a baseline that does not chase a sustained elevation, and it has to leave the prior rate alone.
