# Post-merge CUDA smoke inventory

This round reruns every registered idea and the 13 previously selected complete
family configurations against the new Gaussian acquisition smoke and CUDA
behavioral hosts. The Gaussian default keeps depth 2 and two Fourier frequencies:
the original recipe already demonstrated acquisition, the shallow alternative
also acquires but drifts, and removing Fourier features misses the smoke budget.

The frozen roster contains **52 declarations: 39 ideas and 13 configurations**.
Historical tuning grids are not repeated. The selected configurations retain
their exact recipe identities and are registered through one-point searches;
this round does not tune hyperparameters. Protocol seed 0 and the public
deterministic initializer apply throughout. Architecture, prior, target,
sampling law, seen batches, update budget and evaluation cadence are fixed per
task across trainer candidates. Each candidate supplies one global recipe.

All runnable Tier 1 peers finish before a failure blocks higher tiers. A whole
candidate must pass all **6 required Tier 1 tasks** before its eligible **20
required Tier 2 tasks** run. Task-specific checkpoint prerequisites still apply.
Diagnostic clock results do not substitute for required gates. Tier 3 is outside
this round. The Gaussian stability task restores its own candidate's exact
1,000-update smoke state, continues without resetting history and tests the
stationary and shifted goals.

Four old idea contracts and one selected configuration refuse resolution because their control maps do not bind
the new tasks. Their exact declarations and refusal reasons remain in the round;
they receive no attempt or scientific verdict. Draft and unregistered study
blockers are likewise retained. Independent runnable declarations proceed
through the ordinary Forge APIs without weakening those contracts.

Each declaration has a conservative 42,720-second complete allowance, for a
2,221,440-second maximum campaign reservation. These are task timeout ceilings,
not measured runtime. Scientific retries are zero. Software planning occurs
before the merged execution source is frozen and any training starts.

```sh
/usr/bin/python -u reports/forge/prepare_gaussian_smoke_inventory.py plan \
  --queue-root runs/forge/gaussian-smoke-inventory-v3
/usr/bin/python -u reports/forge/prepare_gaussian_smoke_inventory.py enqueue \
  --queue-root runs/forge/gaussian-smoke-inventory-v3
/usr/bin/python -u -m experiments.forge \
  --queue-root runs/forge/gaussian-smoke-inventory-v3 drain \
  --gpus 0,1 --campaign gaussian-smoke-inventory-v3
tail -F runs/forge/gaussian-smoke-inventory-v3/events.jsonl
```

The original v1 admission stopped before creating any submission, attempt or training update: the host validator conflated the 6,000-update continuation budget with its retained 1,000-update schedule horizon. [The refusal/archive receipt](admission-v1.json) preserves that source and zero cost. The v2 successor changes only this software validation and the registration identity; roster, recipes, task budgets and campaign ceilings remain fixed.

The v2 run then exposed an older generic vector-adapter omission: ring execution
stopped at its 400-update schedule horizon before the declared 1,600-update
allowance. Dispatch stopped, leaving active tasks to finish and retaining every
partial result and cost. The v3 software amendment passes the explicit task
execution allowance to the public trainer, preserving its recipe schedule.
The complete fixed roster runs under the repaired source because ordinary
qualification cannot splice task passes from different sources. This is an
execution repair, with no recipe selection, seed changes or retries of scientific
failures. The combined conservative executable allowance fits inside the
original 2,221,440-second goal ceiling.

The preparation interface launches no training. The parent runs the reviewed
campaign with one worker on each physical GPU, preserves raw stdout, traces,
states and original receipts in an artifact archive, and publishes compact
metrics and actual-training GIFs. No raw log is committed.

Publication uses the existing
[regeneration script](../regenerate_technique_inventory.py). New whole-row family
pins select fresh measured evidence; old policy/source outcomes remain archived.
The only generated goal leaderboard is
[technique-inventory](../technique-inventory.md). Results and recommendations will
be filled after this bounded round concludes.
