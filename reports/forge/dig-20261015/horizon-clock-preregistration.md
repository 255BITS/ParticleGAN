# Preregistration: horizon-clock (written 2026-10-15 before any training)

Family: K3P (cheaper than KA2: 4 declared clock dependencies vs 5; no call-800 switch).
Seed 0 only, CUDA, all arms on ONE frozen source commit/runtime, same lease helper.

Arms (all start from the shipped `k3p` idea; only the listed fields differ):
- A `k3p` (step-schedule control, unchanged shipped K3P). On ring16 the task forces
  total_steps=400, so LR/noise schedules finish at update 400 and 1,200 updates run at the floor.
- B `k3p-hc-constant` (constant-rate control): lr_floor=1, network_lr_floor=1,
  network_lr_horizon_cap=None, input_noise_std=0, output_noise_warmup=0 (constant 0.029),
  d_guard_min_steps=0. No declared clock dependency.
- C `k3p-hc-progress` (candidate): progress_schedule="critic_order", progress_rate=0.001,
  progress_target=0.75, progress_ema=0.98, d_guard_min_steps=0. K3P's own schedule SHAPE
  (LR full until p=0.6 then cosine to floors 0.01/0.05; input noise 0.5 -> 0 over p in [0,0.1];
  output noise warmup over p in [0,0.2]) is driven by an internal progress value p instead of
  completed_steps/total_steps. p advances by at most 0.001 per update, scaled by
  clamp((0.75 - a)/(0.75 - 0.5), 0, 1), where a = EMA(0.98) of the critic's paired order accuracy
  mean[D(real_i) > D(fake_i)]. p never decreases.
- D `k3p-hc-progress-ungated` (diagnostic only): same as C but p advances 0.001 every update
  (an internal counter). Separates "anneal within the cap" from "progress gating".

Tests (unchanged task definitions): ring16_acquisition (1,600 updates, 96 checks, >=5 terminal
passes), clockfree_audit_measurement_v1. Tier 2 (gaussian1d_stability, vector_anisotropic) for
A/B/C only if C passes ring16.

Decision rule for C:
- SUPPORTED: ring16_acquisition sustained gate PASS, AND all four measured clock parity
  comparisons (step_label, horizon, evaluation_cadence, restart) agree. The official
  clockfree_audit verdict is reported separately; it is expected to FAIL by *declaration*
  because clockfree.source_audit keys on lr_floor!=1 / input_noise_std / output_noise_warmup
  without knowing about progress mode. I will NOT edit the gate; a declaration change is
  proposed for review instead.
- REFUTED: ring16 FAIL, or any parity comparison differs.
- INCONCLUSIVE: runner refusal / incomplete evidence.

Interpretation table (not part of the verdict): D PASS and C FAIL => annealing within the cap
works, the gate signal is the problem; C and D FAIL => K3P cannot get annealing's ring16 benefit
inside 1,600 updates with this schedule shape; the horizon is binding.

Stop rule: if C FAILs ring16, skip Tier 2.
