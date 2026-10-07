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
  --queue-root runs/forge/gaussian-smoke-inventory-v1
/usr/bin/python -u reports/forge/prepare_gaussian_smoke_inventory.py enqueue \
  --queue-root runs/forge/gaussian-smoke-inventory-v1
/usr/bin/python -u -m experiments.forge \
  --queue-root runs/forge/gaussian-smoke-inventory-v1 drain \
  --gpus 0,1 --campaign gaussian-smoke-inventory-v1
tail -F runs/forge/gaussian-smoke-inventory-v1/events.jsonl
```

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
