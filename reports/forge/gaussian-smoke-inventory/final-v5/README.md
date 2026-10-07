# Default truncation and serial-autograd CUDA rerun

This user-requested round reruns the unchanged 52-declaration roster after
merging numerical-rank truncation into the default DualNorm matrix update and
disabling autograd multithreaded backward scheduling project-wide. All 39
registered ideas and 13 preselected configuration identities remain frozen;
known unsupported/refused declarations remain visible without spending.

Each whole recipe retains six required Tier 1 tasks, twenty Tier 2 tasks and
two out-of-scope Tier 3 tasks. All runnable independent Tier 1 peers finish
before a failure blocks deeper work. Only whole-recipe six-task Tier 1 PASS
unlocks its eligible Tier 2 tasks, including same-source checkpoint dependencies.
One global recipe per candidate is used throughout; no seed study, parameter
search, task repair, cross-source pass pooling or scientific retry is allowed.
All workers use CUDA on RTX A6000 GPUs 0/1, one worker per GPU. Architecture,
target/data law, public seed0 initializer, priors, named streams, sampling,
update budgets, evaluation cadence and numerical bounds remain unchanged.

The original campaign ceiling is 2,221,440 charged seconds, with 42,720 seconds
per declaration. The executable new reservation upper bound remains 982,560
seconds; previous-source paid cost is separately retained at 9,845.240927
seconds. The combined executable upper bound remains below the original goal
ceiling. Reserved ceilings are not measured costs. No Tier 3 work is authorized.

Execution and publication are pending. Original v1–v4 evidence, grades, costs,
initialization contracts and archive identities remain unchanged. The only
current family leaderboard is [technique-inventory.md](../../technique-inventory.md);
a source-specific readout will retain every candidate and failed numerical bound.

Reproduce preparation from the committed post-merge source:

```sh
python -u reports/forge/prepare_gaussian_smoke_inventory.py enqueue \
  --round configs/forge/rounds/gaussian-smoke-inventory-v5.json \
  --queue-root runs/forge/gaussian-smoke-inventory-v5
```

Bulk stdout, JUnit, event/metric streams, checkpoints and tensor dumps remain
in the ignored local queue and a byte-exact artifact archive. Compact reports,
provenance receipts, reproduction sources and actual-training GIFs are published.
