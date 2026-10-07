# Default truncation and serialized autograd: CUDA qualification refresh

Both default changes are merged into develop: PR332 adds numerical-rank
truncation to DualNorm matrices; PR339 disables autograd multithreaded
backward scheduling throughout the public trainer, examples and benchmarks.
PR341 repairs the joint-word task's stale evaluator source pin. The repair
changes global source identity; interrupted V5 results and costs remain separate.

The corrected V6 cohort retains all 52 original candidate identities, 39 idea
cards and 13 preselected configurations. No hyperparameters, priors, architecture,
target/data law, seed0 deterministic initializer, named streams, sampling,
update budgets, evaluation cadence or numerical gates change. All training uses
CUDA on physical RTX A6000 GPUs 0/1, one worker per GPU. Whole recipes must pass
all six required Tier1 gates before their twenty required Tier2 tasks can run.
Two Tier3 tasks remain outside the campaign. Capability/declaration blockers
stay visible; no scientific retry or cross-source pass pooling occurs.

Executed origin: `45f056556503341bccf3ade0cd3365c5d0dadb91`.
Scientific source digest: `6269a18ac4f82564cb16ba19afa4b3dd2f836a2b4085fbe5aeb81a35a453e895`.
The 23 admitted candidates have a declared executable ceiling of 982,560 seconds.
Earlier measured cost, including interrupted V5, is 10,631.038402 seconds.
The combined bound stays below the original 2,221,440-second goal ceiling.
Reserved ceilings are not measured cost.

Execution and final publication are pending. Tail the current ignored progress:

```sh
tail -f /tmp/particlegan-default-tier-refresh/runs/forge/gaussian-smoke-inventory-v6/progress.jsonl
```

Reproduce admission from the exact committed source:

```sh
python -u reports/forge/prepare_gaussian_smoke_inventory.py enqueue \
  --round configs/forge/rounds/gaussian-smoke-inventory-v6.json \
  --queue-root runs/forge/gaussian-smoke-inventory-v6
```

The single current leaderboard remains
[technique-inventory.md](../../technique-inventory.md). Final compact scalars,
source/complete-state certificates, actual-training GIFs and byte-exact archive
identities will be linked here after all eligible work drains.
