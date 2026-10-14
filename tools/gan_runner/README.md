# Bounded Codex / current Forge runner

`try-gan.sh` plans one study. `run-gan-pool.py` plans a finite list of independent
studies. Both require `--config`; both default to planning. Only an explicit
`--execute` permits model sessions or ordinary Forge execution. `--dry-run`
overrides `--execute` and makes no reservations, model calls or GPU allocations.
Legacy engine, time, worker and dollar flags are intentionally rejected.

The installed Codex0.161.0 contract was verified from `exec --help`,
`debug models --bundled` and its generated protocol schema. The model is
`gpt-6.1-sol`, effort `xhigh`, service tier `default` (Standard). The exact
`service_tier="default"` config was accepted by the installed read-only catalog
command. No fast/ultrafast selection, full access, approval bypass, ignored
rules or `approval_policy=never` is used. `workspace-write` and `--approve-for-me`
retain automatic approval review. User config is ignored; conflicting project
speed settings are refused. Live account/model access has not been tested.

See [official noninteractive guidance](https://learn.chatgpt.com/docs/non-interactive-mode)
and [configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).

## One shared budget and queue

The owner explicitly imports a reconciled ledger once:

```bash
python -m experiments.forge.codex_runner init-budget \
  --state /absolute/shared/runner-budget.json \
  --ledger /absolute/authoritative/current-ledger.json \
  --queue-root /absolute/shared/forge-queue \
  --semantic-history /absolute/known-semantic-keys.json
```

This imports the existing remaining queue/supervisor seconds without adding
funds. The exact historical ledger and chain hashes are retained. It never
rewrites a prior ledger, queue charge, receipt or scientific result. An owner
sidecar fences the baseline and queue against duplicate state initialization.
All pools/cohorts must share that state and queue. Outstanding reservations
remain fenced after crashes or uncertain submission/launch; there is no
automatic retry or history reset. Reconcile them explicitly before continuing.

Model allowance defaults to zero. An approved future model budget must name
both `--agent-seconds` and `--agent-sessions`. These bound aggregate process time
and admitted model calls separately from experiment time. Codex exposes no
verified hard dollar cap; dollar cost is unknown, never fabricated or replaced
by an old Claude dollar limit. A model turn that fails still costs its measured
time. Session admission counts are conservative caps, not proof of starts.

## Finite studies

Configuration version1 requires `budget_state`, `queue_root`, `runs_dir`,
`lease_root`, `slots` (1–12), `max_studies` (1–12), an exact committed
`source_commit`, and a finite `studies` list. Each study names:

- `id`, a concrete `hypothesis`, its `falsifier`, and its `cohort`;
- normalized `semantic_factors`, including declared method and parameters;
- original `task_ids`, including its own checkpoint/gate prerequisites;
- either an existing normal Forge `request_path`, or positive `agent_seconds`.

Names, arbitrary seed edits, source-only edits and repeated lanes do not create a fresh semantic key. An explicit registered `experimental_unit` can distinguish paired control/candidate roles and preregistered research seeds. Its protocol, frozen recipes, original task contracts, scientific implementation, view and exact source must bind the registration. Cosmetic registration names still deduplicate. Prior negatives and uncertain attempts must remain in the
shared history. Semantic-factor declarations complement Forge's exact
compatibility dedup and scientific review; they do not prove that arbitrary
different implementations are mathematically distinct. Read compiled experiment
memory before authoring another study. Do not pad twelve slots.

Ordinary requests bind the full current view and tier. A partial comparison
requires `scope: "scoped"`, an explicit registered `view_id`, current tier
labels and unchanged original tasks. It grants no whole-gauntlet qualification.
Registered paired research uses an explicit nonqualifying `research_diagnostic` view. Its diagnostic execution labels are separate from public qualification tiers; original numeric gates, samplers, task resources, full horizons and own-state dependencies remain unchanged. Protocol `screening` stays seed0. `resolve_idea(..., protocol_id=...)` accepts an alternate seed only with a matching `configs/forge/research-cohorts/<id>.json` registration. This is research, not promotion or whole-gauntlet qualification.

All source snapshots retain the frozen global revision10 declarations and
current queue/worker lease code. Normal Forge admission, source verification,
checkpoint dependencies, resource checks and campaign caps still apply.

The pool atomically reserves the entire finite cohort before its first launch, including every original execution group's complete timeout plus20 supervisor seconds per group. Prepared requests stream into ordinary admission and dispatch; it does not wait for the last request to be admitted before starting ready work. Forge v3 study review must be `READY`. Shared uninterrupted
groups are counted once, using the maximum original timeout; it never shortens horizons to fit. It
replenishes only from the supplied distinct list. Default dispatch window is
300seconds; `dispatch_window_seconds` can set another finite positive window.
STOP or SIGTERM/SIGINT prevents new dispatch and drains started attempts. Pending
cohorts/reservations remain visible for explicit reconciliation. The historical
`/ml2/hypergan/gan-attempts/STOP` is honored by default; extra `stop_files` can be
listed. No global STOP is removed by installation.

```bash
tools/gan_runner/try-gan.sh --config /absolute/one-study.json --dry-run
python tools/gan_runner/run-gan-pool.py --config /absolute/finite-studies.json --dry-run
```

## Leases, actual receipts and provenance

The existing twelve lease tokens (named ten GPU0/two GPU1) are opened without creation or
truncation. Their inode/mode/size must remain unchanged. A supervisor and its
training/evaluation descendants inherit the shared descriptor; coordinator
crashes cannot release a live child's slot. Closing the owner's descriptor
does not explicitly unlock a surviving child. Forge's coordinator lock and
per-submission claim fence prevent another owner’s queued work being dispatched.

Optional `gpu_allocation: {"0": 6, "1": 6}` explicitly maps those same twelve tokens onto six physical workers per GPU. Remapping requires an explicit prepared-work allocation within the twelve-token fleet; a smaller safe allocation still owns every token. All twelve tokens are acquired exclusively before any reservation or admission; an occupied token refuses the whole remapped fleet. Every supervisor, training child and evaluator inherits every fleet descriptor. Even after a coordinator crash, any surviving child holds the entire remapped fleet against other owners until the last child exits. No lease file is created, renamed, truncated or silently reassigned. `fleet-binding.json` records original token identity and actual physical GPU/slot. Native per-worker memory/model/CPU admission remains authoritative.

Optional `required_concurrent_workers` checks native capacity before admission. A shortage returns a zero-reservation blocker receipt. The caller can explicitly choose a safe smaller concurrency or allocation; task memory reservations are never reduced.

Each run keeps a tail-friendly local Codex JSONL log, original Forge logs,
agent start/terminal receipts, supervisor receipts, separately observed live
training-child receipts confirmed by new completed updates beyond the task's own prefix, observed simultaneous peak, source/candidate/task/lease identities, actual queue and supervisor
charges, and a compact `catalog.json`. An admitted agent turn is not a training
start or a scientific success. Unavailable stage evidence is `UNMEASURED`.
Raw logs/tensors/checkpoints remain outside Git.

Revision10 declarations and the stage reader are copied exactly from
[PR403](https://github.com/255BITS/ParticleGAN/pull/403) at its pinned head;
[provenance](../../docs/runner-contract-provenance.json) records every blob.
The revision9 view is archived byte-for-byte. Original task declarations,
samplers, numerical gates, horizons and historical result receipts are not
changed. Stage readouts use existing original runs, add zero updates and grant
no extra qualification. The incomplete rotation-event protocol stays unmeasured.

To install the two host entry points, retain exact backups first and symlink
them to this checkout's wrappers. `PARTICLEGAN_RUNNER_ROOT` can explicitly
select another verified installation; `GAN_PYTHON` selects its Python runtime.
Installation alone never starts a pool, initializes a live budget or removes
old jobs, leases, charges or artifacts.
