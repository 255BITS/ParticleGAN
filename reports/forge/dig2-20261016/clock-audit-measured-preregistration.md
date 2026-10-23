# Preregistration: dig2 clock-audit-measured (2026-10-16)

Written and committed before any training or GPU probe for this lead.

## Question

Can an annealing family (K3P or KA2) pass both ring16 at a 6,400-update cap and the clock-free
audit when the audit decides from measurement?

## Proposed audit revision (part 1; needs user/reviewer sign-off; counted results unchanged)

New task `clockfree_audit_measured_v2-clockdig`. Probe and four measured comparisons are identical to
`clockfree_audit_measurement_v1`, which stays as it is. The rule `measured_parity_v2`
(`experiments/forge/clockfree.py:measured_v2_decision`) works as follows:

1. If any measured comparison differs (step label, horizon, evaluation cadence, restart), the verdict is FAIL. Nothing overrides this.
2. The v1 declaration table is a pre-screen. It also gets one new entry: `progress_schedule="updates"`
   is a fixed internal update count. Each entry is then handled by class:
   - **S (step/horizon schedule: LR anneal, input-noise anneal, output-noise warmup).** A passing measurement overrides the entry only
     when the *power witness* shows that the probe could have detected a step- or horizon-driven version of that
     schedule. The witness evaluates the recipe's own schedule shapes with step/horizon drivers at the probe's
     reference, step-label (+8192) and horizon (4096) settings, and checks whether any value differs.
   - **C (fixed internal counter: critic-guard minimum Adam steps, KA2 call 800, progress `updates`).** The entry is overridden
     only under `internal_counter_policy="measured"`, and only if the counter is found in the saved learning state
     at the declared warmup.
   - Everything else stands: continuous policies, extensions, and the lazy-penalty `reg_every` counter.
3. The verdict is PASS when no entry stands. The research task uses `internal_counter_policy="measured"`. The strict
   variant (`"declared"`) is regraded offline from the same saved evidence.

## Arms (seed 0, one frozen source and runtime, CUDA A6000, one job per leased GPU)

Every arm sets `d_guard_min_steps=0` and `network_lr_horizon_cap=None`. Each family gets four arms:

| Arm | Change |
| --- | --- |
| step | shipped schedules over the task-owned 6,400 horizon (positive ring control; clock-dependent) |
| const | no annealing: LR floors 1, input noise 0, constant output noise (clock-free control) |
| count6400 | `progress_schedule="updates"`, rate 1/6400: an internal-counter anneal with the same shape as step |
| gated | `progress_schedule="critic_order"`, rate 0.001, target 0.75, EMA 0.98 (round-1 C, untuned) |

Tasks (view `dig2-clock-audit-measured-v1`):

- `ring16_acquisition_6400-clockdig`: the counted ring16 task with steps and schedule horizon set to 6,400 and 384
  observations, which keeps the original cadence. Gates and the 5-terminal-observation sustained rule are unchanged.
- `clockfree_audit_measurement_v1`: unchanged v1.
- `clockfree_audit_measured_v2-clockdig`: the proposed rule.

## Decision rule

An arm **dual-passes** when the ring16_6400 sustained verdict is PASS and the v2 (measured counters) verdict is PASS.
Verdicts come from the task graders only (sustained gates), never from endpoints.

- **Supported**: at least one annealing arm (count6400 or gated) of K3P or KA2 dual-passes.
- **Refuted**: no annealing arm dual-passes.
- Also reported, without changing the verdict:
  - The strict-counter verdict (`declared`) for each arm.
  - Whether the step control passes ring16_6400. If it does not, the variant does not reproduce the earlier 6,400 study, and "longer cap" conclusions are marked inconclusive.
  - Whether const passes ring16_6400. If it does, annealing is not needed at 6,400.
  - count6400 vs step: is |final cov err difference| <= 0.1? If so, the internal counter is behaving as a re-labelled horizon.
- Expected v1 outcome: FAIL for every annealing arm and for KA2 const (call-800 declaration). v1 is not changed.

## Budget and stop rule

The GPU budget is 1.5 GPU-hours of training. If any single ring16_6400 run exceeds 600 s, stop and report.
There are no seed sweeps and no hyperparameter tuning.
