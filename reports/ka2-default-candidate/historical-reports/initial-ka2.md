# kal_asym attempt (20260926T071221Z-447231) — ASYMMETRIC stateful alpha on the R2 base

Lane: own asymmetric gain time constants on the R2 base. Sibling sur_kalman shapes use
symmetric instantaneous maps alpha=g(ratio); here alpha itself is stateful with different
attack/release speeds. Keep R2's W band for the blend (validated); only alpha dynamics change.
3/3 proposals used, all s LR-decoupled, no budget/shift/metric reads.

Parent (read-only prior, same frozen protocol): R2
`.../b3_release2/20260925T222617Z-4046392/cands/r2/mechanism.py` (sha256 59483b5e…):
shift-hold 114/120, recovery 72/81 delay 490. Do not rerun parent baselines.

Work dirs: `/tmp/kalasym/cands/ka{1,2,3}/` (mechanism differs; config/latent/response/drivers
byte-identical to K3P: a1475108/197df635/7e71d60a). Isolation: candidate mechanism runs from
`/tmp/kalasym/iso/ka*/` copies — never from sibling cands/ dirs. Runtime:
/tmp/pr38-default-env/bin/python, CUDA_VISIBLE_DEVICES=GPU-72c1b506-891d-b8bc-b353-e020585e1c47,
CUBLAS_WORKSPACE_CONFIG=:4096:8, OMP_NUM_THREADS=1, CUDA FP32 deterministic. Frozen CUDA repo:
.../qualify_a2_bounded_damp/20260925T041936Z-2509885/repo/.../prepared/repos/cuda. Fixtures:
direct-particle-base/initialization-fixtures. Floors .01/.05 everywhere.

## Proposals (3/3 used)

Common: instantaneous target t=g(ratio): 0 at ratio<=1, linear 1->3, saturated 1 above 3.0
(same band numbers as R2's validated W band). Per blended call: alpha <- alpha+(t-alpha)*k,
k=K_ATK if t>alpha else K_REL. EMA decay = 1-alpha*(1-0.90). EMA updates EVERY critic step
(no binary iff-W==1 gate); W band drives only the blend penalty. Reseed-60 retained.

- **KA1 `ka1/mechanism.py` (sha edb8df1a…):** fast-attack/slow-release. K_ATK=1.0
  (alpha jumps within 1-3 calls of rising surprise: B2-like urgency), K_REL=1/300
  (gain decays over hundreds of calls: no re-clamp flicker).
- **KA2 `ka2/mechanism.py` (sha 9f1d5eda…):** slow-attack/fast-release. K_ATK=1/60
  (false-alarm guard: needs sustained surprise; ignores R2's spurious pre-shift release),
  K_REL=0.5 (re-anchor within ~2-3 calls: agility).
- **KA3 `ka3/mechanism.py` (sha 5a564627…):** load-adaptive. p=consecutive ratio>3.0
  calls (cap 400), p_peak=episode max. k_atk=min(1,0.1+p/20); k_rel=0.3/(1+p_peak/30)
  floored 0.003.

LR-decoupling: s never reads LR/floors/total-steps; constant critic LR keeps surprise
dynamics (hence ratio/alpha/W) alive — K3P's LR-clock handover would pin s=1. Remaining
budget dependency (labeled ablation): inherited LR/noise schedules (floors .01/.05, cap 1600).

## Pre-gate: alpha step-response sketch (CPU, synthetic pulse 50x ratio=1, 300x ratio=5, 400x ratio=1)

| candidate | attack to alpha>=0.9 | overshoot | settle to alpha<=0.1 | final |
|---|---|---|---|---|
| KA1 | 1 call | 1.0 | NOT settled in 400 (0.263) | 0.263 |
| KA2 | 138 calls | 0.994 | 4 calls | 0.0 |
| KA3 | 7 calls | 1.0 | 8 calls | 0.0 |

KA1 latches (release too slow to re-anchor within any realistic window); KA2 guards then
releases fast; KA3 adapts (fast attack on sustained pulse, fast release on brief episode).
Script: `/tmp/kalasym/step_sketch.py`, results `/tmp/kalasym/step_response.json`.

## Measured gates (own-state, continuous, fixed drivers, isolated-dir runs)

| candidate | gate | status | key metrics |
|---|---|---|---|
| ka1 | shift | FAIL | hold 100/120 (fails 2210-2400, minHQ 0.821); deadline **0/81**; final 7 modes HQ 0.879; alpha latches 1.0, W==0 to end; ema_up 2775/reseed 19 |
| ka2 | shift | FAIL | hold **120/120** minHQ 0.919; deadline **50/81, delay 1120**; final 8 modes HQ 0.996; W==0 for 24/57 traces; ema_up 2775/reseed 19 |
| ka2 | shift_frozen (matched control) | FAIL (control) | same 120/120 hold then 0/81, final 0 modes — live 8-mode final is active adaptation (D/G 4.25e-05 at shift, 3600/3600 updates) |
| ka2 | toys mode_hold/unEqWidth/stripes | 3x PASS | 8 modes HQ 0.999; sw1 0.041; 2 modes HQ 0.969 |
| ka2 | toy vector_unequal_mass | FAIL (screen) | all 6 metrics PASS but only 4/5 terminal consecutive passes (confirmed_step null) — stability-window shortfall, not a metric failure |
| ka3 | shift | FAIL | hold **120/120** minHQ 0.921; deadline **0/81**; final 4 modes HQ 0.248; alpha latches 1.0, W==0 to end; ema_up 2775/reseed 19 |

Rank (hold+extension+timely recovery, then toys): **KA2 > KA3 = KA1 on hold (120 > 100),
KA2 > KA1 = KA3 on recovery (50 > 0 = 0)**; none passes recovery, so no survivor advances
to extension/22-toy/full stress. KA2 beats R2 on hold (120/120 vs 114/120, fixes the 1663
dip) but regresses recovery (50/81 vs R2 72/81, delay 1120 vs 490). KA1/KA3 both latch.

## Traces / audit

- KA1 vs R2 failure mode: R2 fails hold at 1663 (HQ 0.8992, fixed-0.5 blend through
  LR-anneal settling). KA1's fast attack fires even earlier/harder (alpha=1.0 by call 850
  in settled window) and its 1/300 release never re-anchors: pre-shift hold already broken
  2210-2400 (minHQ 0.821), then post-shift stall at 7 modes HQ 0.88. Urgency+patience corner
  fails both gates.
- KA2: slow attack (138 calls to 0.9) rides through the LR-anneal settling window that
  broke R2/KA1 — hold 120/120 minHQ 0.919. But the same guard delays post-shift release:
  recovery 50/81, delay 1120 (vs R2 72/81 delay 490). Guard+agility corner trades recovery
  speed for hold. Final alpha 0.9999, ratio 3.76, W==0 sustained — still released, just late.
- KA3: load-adaptive attacks in 7 calls and holds 120/120, but p_peak from the long
  pre-shift settling episode makes k_rel ~0.003 post-shift: alpha latches 1.0, EMA tracks
  the moving critic (decay 0.90), anchor never re-engages, final 4 modes HQ 0.248. The
  persistence memory confuses settling noise with shift load — worse than fixed KA2.
- Rates (all shifts): D/G floored 4.25e-05, prior 4.25e-04 at shift and end; optimizer
  2400/2400 at shift → 3600/3600 final; frozen twin 0/81. Active adaptation confirmed.
- Extra compute: 1 EMA-critic forward+input-grad per blended call, same as K3P/R2 (receipt
  extra_critic_forwards stays 0 by implementation — not zero cost).
- Horizon-prefix check NOT_RUN (no survivor; budget spent on 3 proposals + 9 gates).
  All proposals read no total-steps/eval/shift-time/target-center signals (code: only
  optimizer Adam stats + fixed constants + guard stats).

## Remaining failure / next mechanism

Failure gate: **target-shift recovery** (best 50/81 KA2; KA1/KA3 0/81). Timescale thesis
is supported asymmetrically: slow attack fixes R2's hold dip (KA2/KA3 120/120) but costs
recovery delay (KA2 1120 vs R2 490); fast attack (KA1) breaks hold even earlier. Next (one
coherent change on KA2): keep slow attack for the settling window but add a second,
higher-threshold fast lane for large ratios (e.g. ratio>5 → k=1.0 immediate release) so
genuine shift transients (ratios 3.8-9.2) release fast while settling noise (ratios 2.3-2.7)
stays guarded. Do NOT repeat: symmetric instantaneous alpha, pure fast-attack, or
persistence-scaled release (KA3 latch). Also required before claim: extension 300, full 22
toys for a survivor, long-hold/delayed-change + second-change stress with frozen protocol.
Do not rerun parent baselines.

## Replay

```
CUDAREPO=.../qualify_a2_bounded_damp/20260925T041936Z-2509885/repo/reports/toy100/qualify-a2-attempt/prepared/repos/cuda
FIX=.../direct-particle-base/initialization-fixtures/mode_hold/initial-values.pt
# mechanism.py MUST be isolated — never run from cands/ sibling dirs
CUDA_VISIBLE_DEVICES=GPU-72c1b506-891d-b8bc-b353-e020585e1c47 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
/tmp/pr38-default-env/bin/python -u /tmp/kalasym/iso/ka2/shift.py --repo $CUDAREPO --task mode_hold \
 --output /tmp/kalasym/out/ka2-shift --config /tmp/kalasym/iso/ka2/config.json --backend cuda \
 --initial-state $FIX --network-floor 0.01 --prior-floor 0.05
# same pattern with hold.py / shift_frozen.py / probe.py --task {mode_hold,vector_unequal_mass,vector_unequal_width,img_stripes2}
```

Artifacts: `/tmp/kalasym/cands/ka{1,2,3}/mechanism.py`, `/tmp/kalasym/out/{ka1-shift,ka2-shift,
ka2-frozen,ka2-toy-*,ka3-shift}/result.json`, `/tmp/kalasym/logs/*.log`, tests.jsonl rows below.
Code/artifact paths above are /tmp/kalasym canonical; candidate file copies owned by this
attempt live there (isolated checkout rule: never run from sibling cands/ dirs).
