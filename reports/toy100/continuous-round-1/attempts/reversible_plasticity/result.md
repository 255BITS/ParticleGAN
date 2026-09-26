# Reversible plasticity: completed negative search

**No proposal qualifies. Keep the unchanged selected K3P parent.** Three proposals used; six canonical hold/shift runs completed, all FAIL. No seed sweeps, extra agents, detached jobs, baseline reruns, pushes or comments. One CUDA benchmark worker at a time. Search ended before the 45-minute cap after the proposal cap and gate failures.

## Leaderboard

The ranking criterion is hold + 300-update extension + timely recovery, then retained toy gates. No modified candidate passes that conjunction. Failed rows are ordered by their measured good hold length only; this is a diagnostic ordering, not a promotion.

| Formulation | Standard hold | Qualified extension | Recovery deadline | Retained declared toys |
|---|---|---|---|---|
| K3P pinned parent; existing evidence | PASS 1200/1200 | PASS 300/300 | FAIL 28/81; sustained delay 1130 | 22/22 PASS |
| `c02_constant_stationary` | FAIL: 523 good, failure on check 524/1200 | NOT_RUN | FAIL 8/81; no terminal sustained recovery | NOT_RUN: all 22 |
| `c03_energy_ratio` | FAIL: 50 good, failure on check 51/1200 | NOT_RUN | FAIL 0/81; no terminal sustained recovery | NOT_RUN: all 22 |
| `c01_coherence` | FAIL: no confirmation; 0/4800 settling checks pass | NOT_RUN | FAIL 0/81; no terminal sustained recovery | NOT_RUN: all 22 |

No candidate has a passing recovery to confirm with a frozen control. Sensitive transfers (mode_hold, unequal mass, unequal width, stripes), the remaining 18 toy gates including all three full-7000 native coverage **and** accuracy gates, and delayed/repeated-change GAN stress are NOT_RUN. None of the parent's passes is credited to a modification. These runs do not identify a sole cause of K3P's recovery failure.

## Quality and gate details

| Candidate | Confirmation update | First failing hold update | Worst HQ in checked hold | Worst deadline-window HQ | Following 300 diagnostic updates: pass / fail, worst HQ |
|---|---:|---:|---:|---:|---|
| `c02_constant_stationary` | 3254 | 3778 | 0.860595703125 | 0.427978515625 | 219 / 81; 0.295654297 |
| `c03_energy_ratio` | 4635 | 4686 | 0.89599609375 | 0.0 | 259 / 41; 0.880859375 |
| `c01_coherence` | none | n/a | n/a | 0.210693359375 | 0 / 300; 0.032958984 |

The 300 updates after a failed hold or failed settling period are **diagnostics**, not a qualified post-1200-hold extension. All use the original driver and fixed thresholds. C01 never confirmed: all 4800 settling checks failed, worst settling HQ 0.032470703125. C02 confirmed at 3254 and failed at 3778. C03 confirmed at 4635 and failed at 4686. No later passing streak replaces those first failures. All three shift runs also fail all 120 pre-change hold checks. The recovery deadline remains update 2800 and the full deadline window remains 81 checks through 3600.

Parent comparison only: hold minimum HQ 0.9072265625; extension minimum 0.98779296875; recovery minimum in the deadline window 0.66845703125. Parent source/config selection is unchanged.

## Formulations and fixed comparison

Every proposal retains relativistic-paired logistic GAN losses, architecture, learned particle prior, direct response, bounded sparse-latent damping, Adam choices and original auxiliary host-loss code. Response and latent sources are byte-identical to the parent. The auxiliary hosts were not newly qualified.

All three fix critic mixing at **s=0.1**, independent of the LR clock: penalty = 0.5 * [0.1*A + 0.9*(B+P)]. A is dimension-normalized real R1 plus the fake RMS cap; B is real/fake L2 caps; cap threshold and coefficient remain 1. P is the dimension-normalized input-gradient gap to the EMA critic. EMA decay is 0.999, initialized on the second penalty call; an extra EMA critic evaluation occurs from call 3 onward. Original critic guard remains C=5 after 200 optimizer updates; there is no generator guard. Generator EMA decay 0.995 and Adam bias correction remain estimation memory, with no final-step input.

Base rates are G/D 0.00425 and prior 0.0085. The inherited direct-particle alignment gain still applies after the controller, and the bounded sparse-latent rule is unchanged. Those particle mechanisms do not become new ring pass claims.

- **C01 coherence, intermediate only:** CUDA gradient mean m and gradient energy q use decay 0.95. c = clamp(sum(m²)/(q*(1-0.95^n)),0,1); LR multiplier f+(1-f)c, f=0.01 for networks and 0.05 for priors. Original input noise 0.5 anneals over 0.1 of the horizon; output noise 0.029 warms over 0.2. This remaining noise schedule makes C01 horizon dependent.
- **C02 constant control:** LR multiplier 0.1 throughout (G/D 0.000425; prior 0.00085 before inherited direct response). Input noise is zero; output noise is 0.029 immediately, without warmup. Explicit fixed critic mixing activates the EMA constraint despite constant rates.
- **C03 reversible energy rescue:** exactly C02's config/noise, critic, particle sources, hold and shift drivers; only `control.py` differs. Fast/slow gradient-energy EMAs decay at 0.95/0.999 and are bias corrected. r=fast/slow; innovation=clamp((r-1)/4,0,1). With C01 coherence c, multiplier=f+(1-f)*max(c,innovation), f=0.01 for networks and 0.05 for priors. It can reduce rates again and raise them at later ages. History tensors are CUDA FP32.

C01 versus C02 also changes noise and does not isolate a rate-only causal effect. C02 versus C03 is the controlled rate-rule comparison: config, fixed critic and particle hooks are identical.

## Applied rates, motion and compute

Measured critic mixing is 0.1 throughout all six canonical runs. The original driver `schedule`/`rate_ranges` fields describe **upstream overwritten schedules**; authoritative applied values are in `controller-trace.json`, mobility `group_lrs`, optimizer-final counters, and `applied-rate-summary.csv`.

Applied rates during updates **2401–2800** (after the change through the unchanged deadline):

| Candidate | Group | Min | Median | Max |
|---|---|---:|---:|---:|
| `c02_constant_stationary` | d | 0.000425 | 0.000425 | 0.000425 |
| `c02_constant_stationary` | g | 0.000425 | 0.000425 | 0.000425 |
| `c02_constant_stationary` | prior | 0.00085 | 0.00085 | 0.00085 |
| `c03_energy_ratio` | d | 0.00016985961 | 0.000413421667 | 0.000612471416 |
| `c03_energy_ratio` | g | 0.000103931668 | 0.000330832172 | 0.000838284134 |
| `c03_energy_ratio` | prior | 0.000889954182 | 0.0018548456 | 0.0034228599 |
| `c01_coherence` | d | 0.000128859294 | 0.000201732973 | 0.000260702271 |
| `c01_coherence` | g | 4.61599303e-05 | 7.59828327e-05 | 0.000262659033 |
| `c01_coherence` | prior | 0.000482494863 | 0.000940269978 | 0.00179944867 |

Each shift run advances both optimizers from 2400 to 3600 actual Adam updates. All 12 logged post-change displacement samples per optimizer are nonzero; neither frozen optimization nor stationary output is being credited as adaptation. The complete applied-rate traces are recorded at every step. Hold and shift rate/controller traces match exactly through update 2400 for each candidate.

| Candidate | Hold training updates | Shift training updates | Added EMA critic forward/input-gradient evaluations: hold / shift | Measured training seconds: hold / shift |
|---|---:|---:|---|---|
| `c02_constant_stationary` | 4078 | 3600 | 4076 / 3598 | 163.67 / 136.00 |
| `c03_energy_ratio` | 4986 | 3600 | 4984 / 3598 | 225.19 / 159.03 |
| `c01_coherence` | 6300 | 3600 | 6298 / 3598 | 237.19 / 129.05 |

The six canonical jobs use 26,164 training updates (52,328 G/D optimizer calls), 26,152 added EMA critic evaluations and 1050.14 measured training seconds. Gate stop times differ because first confirmation/failure differs; these hold totals are not equal-compute comparisons. The shift runs have equal 3600-update budgets, but controller reductions/history and EMA work add compute. At equal 3600 updates the new fixed blend adds 3598 EMA evaluations; the pinned parent starts its anchor at call 964, implying 2636 such evaluations, so the difference is +962. The parent's old zero `extra_critic_forwards` receipt is not zero cost. Shared-GPU elapsed times are not isolated throughput measurements.

## Horizon and controller audits

- C01: FAIL prefix equivalence when declaring horizons 1200 versus 2400, already by update 96. Parameters, Adam state, controller state, EMA and applied rates differ; random streams match. Scheduled noise remains a real dependency, not merely metadata.
- C02: exact prefix match at both 96 and **1100** updates under horizons 1200 versus 2400.
- C03: exact prefix match at **1100** updates under horizons 1200 versus 2400.
- The long checks cross the first horizon's LR decay and the alternative capped network decay onset. Comparisons include model/Adam state, critic EMA, controller/latent/response history, RNG, mixing, metrics and the entire applied-rate prefix. These are diagnostic passes, not cold-acquisition or full22 passes.
- Synthetic C03 input regression: mobility increases on gradient-energy pulses after 500 and 4000 preceding calls and returns to damping afterward. All six boundedness/device/response assertions pass. This is **not** a GAN target-change or repeated-recovery result.

No effective training-horizon dependency was found for C02/C03 in these ring prefixes. Remaining host budget uses are loop termination, canonical evaluation/checkpoint timing, validation/metadata and upstream LR calculations overwritten by the controller. C01 additionally retains effective horizon-dependent noise. Native/other-host prefix equivalence is NOT_RUN, so the ring audit is not a global qualification. Guard/EMA/Adam/controller estimator ages remain ordinary memory; there is no final-step schedule, shift-time reset or evaluation feedback. Fresh-process checkpoint restart equivalence remains unqualified.

Source/runtime audit: **58/58 PASS**, including all 11 parent pins, 1613 exact frozen runtime files, controller/driver consistency, positive actual rates, constant critic mixing and CUDA FP32 parameter/state evidence. All 19 retained initialization fixtures were hash verified. Python is `/tmp/pr38-default-env/bin/python`, torch `2.13.0+cu126`, physical GPU `GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69` as cuda:0; CUBLAS `:4096:8`, deterministic algorithms, TF32 off and one CPU thread. Environment settings are explicit on each benchmark subprocess.

## What the failures narrow down

C01's gradient-mean coherence rule can suppress useful acquisition: its generator median LR before the shift is 7.86e-5 while the prior median is 1.10e-3, and it never obtains a passing settling check. This association does not establish the sole cause.

The constant control eventually acquires good precision but loses retention; increased rates alone are not a solution. C03 also eventually acquires eight modes, but keeps only 50 good hold updates and loses the shifted distribution. Its energy detector reaches full base rate before the target changes: D for 15 updates, G/prior for 42, beginning at updates 1445 and 1442 respectively. A training-energy transient is therefore not specific evidence of new data. This does not prove that these bursts alone caused the failure.

**Recommendation:** keep K3P selected. Do not expand these controllers into full22 or tune a coefficient/seed grid. A useful next mechanism is a shared reversible mobility factor for G and its learned prior, preserving their base-rate ratio, with D control, critic constraint and stationary noise held fixed. That directly tests whether independently adapting group rates harms acquisition/retention. A separate critic/noise ablation would be needed to attribute parent-versus-candidate differences. No fourth proposal was executed.

## Test totals and preserved errors

Ledger: **38 rows: 14 PASS / 7 FAIL / 2 ERROR / 15 SKIPPED**. Canonical candidate gates: **0 PASS / 6 FAIL**. Regression/diagnostic checks: **14 PASS / 1 FAIL / 1 ERROR**. The separate interrupted setup run is ERROR; fifteen downstream-stage groups are SKIPPED. Diagnostic PASS rows do not qualify any candidate toy or hold.

Preserved setup error: the first C01 integration captured the host LR 0.002 before the frozen adapter installed 0.00425/0.0085. It was interrupted, sources/logs retained under `c01_setup_error`, corrected, and both canonical gates rerun in fresh directories. This was an implementation correction within proposal 1, not a fourth mechanism. Exact interrupted elapsed time was not recorded.

Preserved regression error: first prefix audit tried a byte view of a scalar Adam step tensor. Its ERROR result is retained; diagnostic v2 reshapes scalars before hashing. The training formulation was unchanged. One syntax/hash preflight was not separately timed; these ledger seconds fields are null rather than invented.

## Exact artifacts and replay

- Full declaration/workflow: `/ml2/hypergan/gan-attempts/formulations-20260925T154444Z/reversible_plasticity/20260925T154444Z-3603365/repo/reports/toy100/reversible-plasticity-3603365/README.md`
- Candidate source copies: `/ml2/hypergan/gan-attempts/formulations-20260925T154444Z/reversible_plasticity/20260925T154444Z-3603365/repo/reports/toy100/reversible-plasticity-3603365/candidates` (each `declaration.json`, `hashes.json`, `control.py`, copied mechanism/particle sources and drivers).
- Raw results: `/ml2/hypergan/gan-attempts/formulations-20260925T154444Z/reversible_plasticity/20260925T154444Z-3603365/repo/reports/toy100/reversible-plasticity-3603365/out`; each `<candidate>_hold/result.json` and `<candidate>_shift/result.json`, with `controller-trace.json`, `mechanism-receipt.json`, and applied-rate CSV.
- Machine-readable analysis: `/ml2/hypergan/gan-attempts/formulations-20260925T154444Z/reversible_plasticity/20260925T154444Z-3603365/repo/reports/toy100/reversible-plasticity-3603365/analysis.json`; source/rate audit: `/ml2/hypergan/gan-attempts/formulations-20260925T154444Z/reversible_plasticity/20260925T154444Z-3603365/repo/reports/toy100/reversible-plasticity-3603365/evidence-audit.json`; innovation trace summary: `/ml2/hypergan/gan-attempts/formulations-20260925T154444Z/reversible_plasticity/20260925T154444Z-3603365/repo/reports/toy100/reversible-plasticity-3603365/innovation-prechange-audit.json`.
- Exact subprocess arguments/environment/source hashes: `/ml2/hypergan/gan-attempts/formulations-20260925T154444Z/reversible_plasticity/20260925T154444Z-3603365/repo/reports/toy100/reversible-plasticity-3603365/commands`.
- Test ledger: `/ml2/hypergan/gan-attempts/formulations-20260925T154444Z/reversible_plasticity/20260925T154444Z-3603365/tests.jsonl`; required result: `/ml2/hypergan/gan-attempts/formulations-20260925T154444Z/reversible_plasticity/20260925T154444Z-3603365/result.md`.

Run from this attempt checkout; a replay tag creates fresh outputs. These are replay instructions, not additional executions:

```bash
cd /ml2/hypergan/gan-attempts/formulations-20260925T154444Z/reversible_plasticity/20260925T154444Z-3603365/repo
RP_RUN_TAG=replay01 python -u reports/toy100/reversible-plasticity-3603365/run_gate.py c01_coherence hold shift
RP_RUN_TAG=replay01 python -u reports/toy100/reversible-plasticity-3603365/run_gate.py c02_constant_stationary hold shift
RP_RUN_TAG=replay01 python -u reports/toy100/reversible-plasticity-3603365/run_gate.py c03_energy_ratio hold shift
RP_RUN_TAG=replay01 python -u reports/toy100/reversible-plasticity-3603365/run_prefix.py c03_energy_ratio 1100
tail -F reports/toy100/reversible-plasticity-3603365/live.log
```

All benchmark launchers read the supervisor before a batch and explicitly set the GPU/thread/determinism environment. Raw logs remain separately named per candidate and gate. No training process is intentionally left running.

Completed UTC: 2026-09-25T16:15:52.113505+00:00
