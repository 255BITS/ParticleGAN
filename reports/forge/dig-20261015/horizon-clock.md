# Dig: horizon and clock policy (lead 3), 2026-10-15

**Verdict: refuted.** The progress-driven schedule ran as designed and passed every measured clock comparison. It still **failed ring16_acquisition within 1,600 updates**. The constant-rate and step-schedule controls failed too. For K3P the binding constraint on ring16 is the **1,600-update horizon**, not the clock rule: no arm reached the per-cluster covariance limit, and the arm that never anneals came closest.

## Terms

- **Clock-free.** The learned state must not depend on the step label (`completed_steps`), the declared horizon (`total_steps`), how often evaluation samples are drawn, or a save-and-restore. `clockfree_audit_measurement_v1` measures this directly. From a 4-update warmup it runs 4 more updates under each perturbation and compares the hashes of the full trainer state. It then applies a *source declaration table* (`experiments/forge/clockfree.py:source_audit`). That table fails any recipe whose fields imply a step schedule, for example `lr_floor != 1`, a nonzero `input_noise_std`, or `output_noise_warmup`.
- **Annealing.** K3P lowers three things by the fraction of the horizon completed (step / horizon). Its learning rate is cosine-decayed from 60% of the horizon down to a floor of 0.01 (G/D) or 0.05 (prior). Its critic input noise falls from 0.5 to 0 over the first 10%. Its generator output noise rises from 0 to 0.029 over the first 20%.
- **Progress schedule (new, opt-in).** `Recipe.progress_schedule="critic_order"` replaces step/horizon in all three schedules with an internal value p in [0,1]. Each update p rises by up to `0.001 * clamp((0.75 - a) / 0.25, 0, 1)`, where a is the EMA (0.98) of the critic's paired order accuracy, mean[D(real_i) > D(fake_i)]. p moves fastest when the critic is at chance and stops when the critic orders 75% of pairs correctly. It never decreases. p is saved and restored with the trainer state. `"updates"` is a diagnostic variant in which p rises by 0.001 every update, an internal counter with no gate.
- **Sustained gate.** ring16 passes only if at least 5 *terminal* consecutive observations, out of 96, meet every threshold: modes ≥ 16, mass_tv ≤ 0.15, hq ≥ 0.85, component covariance error ≤ 0.85 and min eigen ratio ≥ 0.15.

## Hypothesis and preregistered rule

The rule was committed before any training (`829d6544`, `reports/forge/dig-20261015/horizon-clock-preregistration.md`).

- **Supported** if the progress arm (C) passes the ring16 sustained gate within 1,600 updates and all four measured parity comparisons agree.
- **Refuted** if ring16 fails or any comparison differs.
- The official audit verdict is reported separately. It was expected to FAIL by declaration, and I did not change the gate.
- Stop rule: if C fails ring16, skip Tier 2.

## Arms

All arms are K3P, seed 0, CUDA RTX A6000, on one frozen source digest `3cdf77a6…` (branch commit `4cac97b0`). Recipe differences:

| Arm | Card | Change from shipped K3P |
| --- | --- | --- |
| A step control | `k3p-hc-step-v1` | none. On ring16 the task-owned horizon is 400 updates, so annealing ends at update 400 and 1,200 updates run at the floor. |
| B constant control | `k3p-hc-constant-v1` | lr_floor = network floor = 1, input noise 0, constant output noise, spike guard from update 0 |
| C progress (candidate) | `k3p-hc-progress-v1` | `progress_schedule=critic_order` (rate 0.001, target 0.75, EMA 0.98), spike guard from update 0 |
| D ungated (diagnostic) | `k3p-hc-progress-ungated-v1` | `progress_schedule=updates` (rate 0.001), spike guard from update 0 |

## Commands

```sh
PY=/ml2/codex-storage/workspaces/2026-10-07/task-14/runtime/python312/bin/python; export OMP_NUM_THREADS=1
Q=/ml2/hypergan/dig-20261015/horizon-clock/queue
PYTHONPATH=. $PY reports/forge/dig-20261015/horizon-clock-make-arms.py      # idea cards
PYTHONPATH=. $PY reports/forge/dig-20261015/horizon-clock-make-studies.py   # studies (view dig-horizon-clock-diagnostic-v1)
$PY -m experiments.forge --root . --queue-root $Q enqueue k3p-hc-step-v1 --study dig-horizon-clock-k3p-hc-step-v1-study
$PY -m experiments.forge --root . --queue-root $Q enqueue k3p-hc-constant-v1 --study dig-horizon-clock-k3p-hc-constant-v1-study-r2
$PY -m experiments.forge --root . --queue-root $Q enqueue k3p-hc-progress-v1 --study dig-horizon-clock-k3p-hc-progress-v1-study-r2
$PY -m experiments.forge --root . --queue-root $Q enqueue k3p-hc-progress-ungated-v1 --study dig-horizon-clock-k3p-hc-progress-ungated-v1-study-r2
# each request, through the workflow GPU lease:
gpu-lease.sh bash -c '$PY -m experiments.forge --root . --queue-root $Q drain --gpus $LEASED_GPU --workers-per-gpu 1 --campaign dig-horizon-clock-v1 --request <id>'
# diagnostic trace of p / LR for arm C (not evidence; reproduces the graded endpoint exactly):
gpu-lease.sh hc_replay.sh <attempt> progress   # runs reports/forge/dig-20261015/horizon-clock-replay.py
```

Logs, all tail-able:

- `/ml2/hypergan/dig-20261015/horizon-clock/queue/events.jsonl`
- `.../queue/dig-horizon-clock-v1/<attempt>/run.log`
- `/ml2/hypergan/dig-20261015/horizon-clock/logs/`

## Results

ring16_acquisition (1,600 updates, 96 observations). Every arm has 0 passing observations and a terminal streak of 0. Covariance limit ≤ 0.85.

| Arm | ring16 | Final cov err | Min cov err (update) | Mean cov, last 16 obs | Final min-eig / hq / mass_tv | Obs passing all gates except covariance | Clock audit | Measured parity (label / horizon / cadence / restart) |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- |
| A step | FAIL | 2.950 | 2.873 (1334) | 3.210 | 0.536 / 0.812 / 0.047 | 0/96 | FAIL: `step_label` changed the state | ✗ / ✗ / ✓ / ✓ |
| B constant | FAIL | **1.337** | **1.285** (1517) | **1.374** | 0.447 / 0.876 / 0.033 | 39/96 | **PASS** | ✓ / ✓ / ✓ / ✓ |
| C progress | **FAIL** | 1.787 | 1.734 (1334) | 2.030 | 0.244 / 0.905 / 0.052 | 30/96 | FAIL by declaration only (3 stale step-schedule entries) | ✓ / ✓ / ✓ / ✓ |
| D ungated | FAIL | 1.895 | 1.851 (1334) | 2.038 | 0.276 / 0.878 / 0.039 | 47/96 | FAIL by declaration only | ✓ / ✓ / ✓ / ✓ |

Arm C progress trace from the diagnostic replay, which reproduces the graded endpoint bit for bit (cov 1.78698):

| Update | p | Order-accuracy EMA | G LR multiplier | Input noise |
| ---: | ---: | ---: | ---: | ---: |
| 200 | 0.098 | 0.765 | 1.00 | 0.012 |
| 400 | 0.128 | 0.639 | 1.00 | 0 |
| 800 | 0.382 | 0.575 | 1.00 | 0 |
| 1200 | 0.673 | 0.566 | 0.92 | 0 |
| 1400 | 0.826 | 0.554 | 0.40 | 0 |
| 1550 | 0.943 | 0.556 | 0.06 | 0 |

The gate held progress back while the critic was winning (around update 200, accuracy 0.77). After that it advanced at about 70% of the maximum rate. The learning rate was fully annealed by the end, so annealing did happen inside the cap. Annealing lowered covariance error only from about 2.0 to 1.79. The arm that never annealed (B) reached 1.34.

GPU cost:

- Training: 70.6 s for the four ring16 runs (17-18 s each). The four clock probes are a few seconds each.
- Total lease wall time: about 175 s, including the 20 s replay. That is well under the 2 GPU-hour budget.
- Tier 2 (gaussian1d_stability, vector_anisotropic) was skipped by the preregistered stop rule.

## Conclusion

1. **Refuted for K3P.** A progress-driven anneal is clock-free by measurement but does not deliver the long-horizon ring16 benefit inside 1,600 updates. The step schedule is worst (cov 2.95). Annealing over the full cap, gated (1.79) or ungated (1.90), is better than that. Not annealing at all is best (1.34). On ring16 at this budget, annealing *costs* covariance. The 6,400-update passes come from about 4× more training at high rate, and their annealing happens late. The 6,400 study is consistent with this: those runs still failed at the 1,600, 3,200 and 4,800 cuts, and K3P's last covariance failure was at update 5,300.
2. **The clock-free rule is not what blocks annealing.** Arms C and D pass all four measured comparisons. The audit fails them only through its declaration table, which keys on the `lr_floor`, `input_noise_std` and `output_noise_warmup` fields without knowing about progress mode. Even an internal update counter (D) is measurably clock-free. As written, the audit forbids dependence on the *external* step label and horizon, not annealing over a fixed internal duration.
3. **Constant-rate K3P (B) is a full clock-audit PASS** and the closest K3P comes to ring16 at 1,600 updates. It is still 0.49 above the covariance limit.

## Policy note: what the 1,600 cap and the clock-free rule exclude

| Rule | Excludes | Numbers |
| --- | --- | --- |
| ring16 cap of 1,600 updates (task-owned horizon 400) | K3P, KA2 and R1/R2 ring16 passes. GAN v3 also fails. | Full-schedule passes need 6,400 updates, 4× the cap. Terminal passes K3P 66, KA2 47, R1/R2 5 (the minimum). Last failures at updates 5,300, 5,617 and 6,317. All three fail at the 1,600/3,200/4,800 cuts. This dig: best K3P at 1,600 is cov 1.34 against a 0.85 limit, under any of four schedule policies. BCAP DualNorm acquires at update 684, so it is unaffected. |
| Ring16 as a Tier 1 requirement | Every Tier 2 result for these families. Tier 2 does not count unless Tier 1 passes. | K3P would have about 11/21 Tier 2 passes if counted (review estimate). K3P/KA2/R1R2 each sit at 20/22 Tier 1 view requirements; the missing two are ring16 and the clock audit. |
| Clock-free audit declaration table | Every step-scheduled family: K3P (4 declared entries), KA2 (5, including the call-800 penalty switch), R1/R2 (1), GAN v3. Also every `continuous_policy` controller (E22 DV1-DV12, by declaration) and any formulation extension. | Measured parity alone would pass progress-driven and internal-counter schedules (this dig, C and D). BCAP and BISPF pass both parts. |
| Cost of a longer ring16 horizon | — | About 18 s per K3P ring16 run at 1,600 updates on an A6000, so about 72 s at 6,400. Negligible against the 300 s task allowance per run. |

Options for the user, decision only, nothing changed here:

- **(a) Keep both rules.** The annealed families stay out of ring16 and therefore out of counted Tier 2.
- **(b) Add a declared 6,400-update ring16 variant** as a separate task identity. This admits K3P/KA2/R1R2 on ring. They would still fail the clock audit.
- **(c) Extend the clock-audit declaration table** to accept schedules driven only by saved internal state (the `progress_schedule` modes). The measured comparisons stay as they are. This would need reviewer sign-off because it changes a gate. It only helps if the horizon question is also addressed: these results show that clock-free annealing alone does not pass ring16 within 1,600 updates.

## Caveats

- One family (K3P) and one seed, as the rules require. KA2 and R1/R2 were not run.
- The progress hyperparameters (rate 0.001, target 0.75) were chosen once, before any run, and not tuned. A different rate cannot exceed the full-LR arm B, which already fails. Arm B is therefore an upper bound for any decay-only schedule of this shape. The exception is a schedule that also changes input or output noise in a way B does not.
- Arms B, C and D also set `d_guard_min_steps=0`, so that the critic spike guard is not a declared clock dependency. Arm A keeps 200. A–C differences include this.
- The clock audit probe is short: 4 warmup updates, 4 probe updates, horizon 128 vs 4,096. It cannot see the schedule's long-run shape. The parity result says only that the state does not read the label, horizon, cadence or restart.
- Two harness quirks: one coordinator per queue root, so drains ran one at a time. Enqueued studies are immutable, so three studies were cancelled before running (their control id changed) and re-issued with `-r2` ids. No result was discarded.
- Code is opt-in. Defaults, recipe serialization (via `recipe_compat` default projection), tasks, gates and budgets are unchanged. New unit tests are in `tests/test_progress_schedule.py`. Existing schedule, clock-free, E22 policy and API tests pass (90 tests).
