# Stability after convergence

The user prioritizes stability once learning has converged over smooth behavior
while learning. The old diagnostic began its hold at step1200 and failed on the
first bad observation. It did not establish dense convergence first.

`first_convergence_then_hold_v1` measures a separate research objective:

1. Restore the candidate's own completed cold1200 checkpoint with its exact
   policy, models, optimizer moments and random streams.
2. Allow at most4800 further settling updates. Require the first200 consecutive
   dense observations with8 modes and HQ>=0.90 to confirm quality convergence.
3. Starting with the following update, require1200 passing observations while
   ordinary training continues. Save the convergence checkpoint without resetting
   training. Fail on the first post-convergence violation and never re-arm.

Record time to convergence, early violations, actual work, and the post-convergence
hold separately. No confirmation window within the bound is NOT_CONVERGED; an
armed violation is POST_CONVERGENCE_FAIL. A passing finite hold is not proof of
permanent stability or mathematical game equilibrium. This metric never supplies
labels, targets, loss terms, schedules or freezes to training.

Existing frozen toy budgets and verdicts remain unchanged. An old ring or immediate
hold FAIL is retained even if this diagnostic later passes. Only1-2 promising
candidates per attempt should get the extra budget; original toy failures and
native100 still block release qualification.

```bash
# Use the launcher's pinned CPU/AVX2, single-thread environment.
python reports/toy100/h_stability/converged_probe.py --output NEW_OUTPUT
python reports/toy100/h_stability/converged_probe.py --candidate OWN_COLD_CANDIDATE --output NEW_OUTPUT
```

The shared `stability_runner.run(..., convergence_gate=ConvergenceGate())` also
supports adapters that already reconstruct their own policy. Its default behavior
is unchanged for historical replay. Use the same new criterion for every candidate;
do not tune confirmation or hold lengths to observed outcomes.
