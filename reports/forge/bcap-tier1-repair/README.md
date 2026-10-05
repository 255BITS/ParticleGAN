# BCAP Tier 1 repair

This PR audits the newly introduced acquisition tasks and repairs BCAP through
bounded whole-recipe comparisons. The target is all six acquisition/behavior
tasks and a separately scoped schedule-contract audit. Strict clock-free
qualification remains separate: scheduled BCAP cannot acquire that claim by
passing its operational schedule audit.

The current incumbent passes four of six acquisition/behavior tasks. Saved
Gaussian outputs show late location drift despite improving standardized shape.
Ring outputs acquire all 16 modes, but retain radial tails and thin component
cores. Oracle/destructive controls validate scorers, not acquisition budgets or
Tier 1 placement. Shared K3P failures motivate a task audit rather than establish
that either criteria or formulations are wrong.

Three parallel tracks inspect task/scorer calibration, add explicit
schedule-contract checks, and compare longer training with original schedule
horizons. The first BCAP rate search freezes eight new whole configurations,
retaining the public trainer, D multiplier2, cap1 and mechanism signature.
LR .002125/.0010625, prior multiplier .5/1 and coefficient1/2 test stability.
These lower rates avoid repeating the already measured .00425/prior1/coefficient1
configuration. Every candidate completes independent Tier 1 tasks even after a
numerical failure. No seed-only study or task-specific optimizer adaptation is
permitted.

The initial round reserves at most 36,000 GPU-seconds, including search,
diagnostics and any final retest. New rounds require their own finite declaration
and evidence-backed hypothesis. The first search reserves its exact full-task
allowance per configuration. Higher tiers and default adoption are outside this
round. A single configuration must satisfy every required gate; task-specific
winners cannot be combined into a full pass.

Run the preparer only before study admission, after all implementation changes
are integrated. Commit the reviewed source and frozen declarations before
execution. Forge owns training, reservations, source snapshots and grading.

```sh
.venv/bin/python reports/forge/bcap-tier1-repair/prepare.py
.venv/bin/python -m experiments.forge --queue-root runs/forge/bcap-tier1-repair/queue search plan bcap-tier1-repair-rates-v1
.venv/bin/python -m experiments.forge --queue-root runs/forge/bcap-tier1-repair/queue search enqueue bcap-tier1-repair-rates-v1
.venv/bin/python -m experiments.forge --queue-root runs/forge/bcap-tier1-repair/queue drain --gpus 0,1 --workers-per-gpu 1
tail -F runs/forge/bcap-tier1-repair/queue/events.jsonl
```

Bulk outputs stay in ignored local directories or the artifact archive. Commit
compact final metrics, reproduction sources, provenance and actual-observation
GIFs. Use the [single current leaderboard](../technique-inventory.md); this
readout introduces no second leaderboard. Preserve old evidence and explain any
revised task without retroactively changing its original failure.
