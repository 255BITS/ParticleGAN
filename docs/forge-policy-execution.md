# Policy study execution ownership

`benchmarks.toy_audit.api_family_search.run_study` keeps the existing
Atlas/E22 cloud and served-policy cohort, its complete case denominator, and
its original numerical gates. It uses a dedicated coordinator because ordinary
Forge tasks currently refuse these policies. Policy evidence grants no learned
MoG qualification, calibration, promotion, adoption or speed credit.

The coordinator and `Queue.claim` share the main Git repository's `runs/forge`
admission ledger and short `queue.lock` transactions. Worktrees resolve the same
location through `--git-common-dir`. `PARTICLEGAN_FORGE_QUEUE` or the policy
runner's `--queue-root` overrides it; all cooperating runners must use the same
location. Both directions account for active CPU-thread and host-memory
reservations, and a policy GPU is exclusive across every Forge GPU slot.
Before admission, the existing core collection machinery recovers abandoned
reservations with no terminal receipt or execution lease. A live core collector
lease protects its claim-to-launch gap, and live execution leases remain fenced.
Optional study `resources.host_memory_mb` declares at least 512 MiB; the default
is 512 MiB. Torch thread count is one, matching the actual API child CLI, and
is bound to the retained runtime contract.

Compatible study requests with different names or output directories attach to
one canonical archive. Shared physical-attempt keys also cover configurations
that overlap across different studies. They bind the family, actual case and
sampling law, recipe overrides, frozen execution-source digest, runtime, full
timeout/export allowance and media contract. Attachments record their canonical
archive; they are projections, not new execution receipts. Changed scientific
bindings create different attempts. Unchanged interrupted/failed attempts never
retry automatically.

Physical CUDA indices are admission details. Matching GPU-model/runtime requests
reuse the same science across placements while retaining each original receipt's
actual device. An attached study resumes its canonical lane's original device.

Execution snapshots include package and benchmark sources, lazily loaded example
and published host Python sources, and the toy catalog. They exclude raw logs,
checkpoints and observation arrays. Every child launches with its snapshot as
both `cwd` and `PYTHONPATH`, checks source bytes and frozen origin metadata before
launch, and retains the planned Git origin in its receipt. Concurrent edits to
the checkout cannot change those imports. The expanded execution-source binding
creates a new cohort; archived outcomes and original reproduction sources are
not rewritten or requalified.

The study and physical attempt each have a kernel lease inherited by their
child. Recovery never replaces an attempt while that lease is held, including
after the coordinator dies. When the last holder exits without a central
terminal receipt, recovery charges the complete timeout plus export allowance
and retains INCOMPLETE. A central terminal receipt that survived a crash before
the study save is reused with its measured cost. Full allowances must fit both
family and candidate budgets before admission. Logical studies retain the
original measured/reserved cost of reused evidence conservatively; physical
cost is charged once in `policy_attempts` in the shared ledger.

Only an explicit `run` starts children. If another owner or admission is busy,
the call returns the current registration with a canonical attachment or
`coordinator.waiting_reason`; a later explicit `run` can resume unlaunched work.
Registration and publication use unique temporary files and atomic replacement.

```sh
OMP_NUM_THREADS=1 python -m benchmarks.toy_audit.api_family_search run STUDY.json \
  --family atlas --device cuda:0 --output runs/policy-study/atlas
tail -F runs/policy-study/atlas/atlas--CONFIG_ID/CASE_ID.log
```

Read `study.json` for the exact canonical output, command, log path, attempt key,
reservation, original receipt and measured cost. Keep logs and state under
ignored `runs/` or an artifact archive. Continue using the existing policy-family
leaderboard; this change generates no additional leaderboard.

This is one-host coordination for cooperating runners using Linux `flock` and
the same local queue root. Unmanaged processes and a different queue root can
still contend. Dedicated attempt costs live in the policy ledger and study
readout; ordinary Forge job telemetry is not silently reclassified to include
them. The coordinator does not establish scientific screening calibration.

Software controls in `tests/test_forge_policy_execution.py` exercise concurrent
submitters, overlapping-study reuse, surviving inherited leases, interrupted
charges, central-terminal recovery, actual frozen child imports, tamper rejection,
and both CPU/GPU directions of admission with real `Queue.claim` calls. Children
are software controls, with no GAN campaign or new scientific result.
