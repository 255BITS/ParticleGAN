# Same-target model-error response probe (prepared, not run)

`pr84_model_error_recovery.py` prepares one bounded diagnostic after a
**passing, complete repaired cold ring**. It reuses the exact post-update
resumer and its original 1200-step noise horizon, constant G/D/prior rates,
full Adam/EMA/RNG restoration, and every-update host observations. The CLI
checks both cold host verdicts and all frozen source hashes before writing an
artifact. The repaired cold ring failed (3 modes, HQ 0.451416 at update 1200;
0/24 passing checks), so this response probe must not run on that result.

The single predeclared perturbation adds `(0.35, 0)` to the final generator
output bias **and its matching EMA parameter**, leaving D, prior, both Adam
states, target data, and all random streams untouched. It verifies the actual
12-particle clean output shift is 0.35 and hashes every other state component.
Two trained branches resume the same acquired snapshot for 50 updates: one
unperturbed and one perturbed. An evaluation-only frozen copy of the perturbed
generator uses the same absolute host evaluation/noise clock as a negative
control. The original 8-mode/HQ≥0.9 thresholds apply at every dense check.
The fixed local-response criterion is 50/50 passing unperturbed checks,
50/50 failing frozen checks, and a passing final five-check suffix for the
perturbed learner. Functional output displacement is reported directly; no
minimum movement is imposed on the unperturbed arm. A miss at 50 updates is
only a failed short-response filter, not evidence that recovery is impossible.

The module exposes `run_continuation(saved, recipe, noise, completed_steps,
target_steps, fail_fast)` for a separate source-bound 1201–2400 same-target
hold, without duplicating the optimizer/noise setup. Tests cover one-update
unperturbed full-state parity against the established resumer, isolated bias
and EMA changes, frozen clock accounting, fixed response grading, and the
failed-cold hard gate. `pytest -q tests/test_pr84_model_error_recovery.py
tests/test_pr84_critic_refinement_resume.py`: 9 passed. No 50-update
response or long-hold experiment was run.
