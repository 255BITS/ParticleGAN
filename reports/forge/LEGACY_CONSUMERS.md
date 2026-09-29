# Legacy consumers and GPU ownership audit

Read-only observation on **2026-09-28, approximately 21:12–21:16 America/Denver**
(2026-09-29 03:12–03:16 UTC), from the Forge feature worktree. No launcher,
queue, process, device allowance or training job was changed. This is an ownership
inventory, not an adoption approval or an exclusive resource reservation.

**A non-overlapping two-GPU pilot cannot start on this machine at this snapshot.**
Both A6000s are actively training under other owners. The retained launchers also
have consumers beyond Forge's initial toy qualification scope. Keep them until
their requests, outputs and consumers have an explicit migration decision.

## Observed local ownership

| Device / resource | Observed owner | Evidence and implication |
| --- | --- | --- |
| GPU 0, `GPU-ed080e41-3193-3755-6756-f3d46c433331`, RTX A6000 | User `martyn`; HyperGAN queue PID **14015**, active training child **53852** | 91% utilization; 3,126 / 49,140 MiB device memory used. Child command is `python -m hypergan train .../cifar-tiny-transformer-resnet-features-pg214-lr4e-4-noise.toml ... --server --checkpoint-every 1000 --preview-every 100 --progress-every 20`. Its parent is `bash ./run-queue.sh queue-gpu0.txt`, running since 13:54 local. |
| GPU 1, `GPU-548116b7-9dbe-de58-b3d9-a6e27b0f74ce`, RTX A6000 | User `martyn`; NPC MisGAN child **135282**, `uv` parent **135274** | 99% utilization; 4,533 / 49,140 MiB device memory used. Command: `tools/train_npc_misgan.py --config configs/npc_misgan_s1_encES_pg070_b2k_lazy1_gxhop3_dres3_gmblock_pack16_200k.toml --device cuda:1`, started 20:54 local. |
| GPU 1 desktop clients | Brave GPU process **11101**, ModOrganizer **62189**, other graphics use | These remain outside Forge. A training reservation is not permission to stop desktop applications. |
| HyperGAN pending queue | `/home/martyn/dev/hypergan/training-runs/queue-gpu0.txt` | Three pending launchers: `start-cifar-tiny-transformer-resnet-features-pg215-lr4e-4-noise.sh`, `...-pg217-lr4e-4-noise.sh`, and `...-pg08-lr4e-4-h200k.sh`. Finishing the current child will not release ownership: the persistent queue can launch the next item. |
| Main-checkout Forge queue | `/home/martyn/dev/ParticleGAN/runs/forge/queue/state.json` | One **concluded** onboarding submission, one terminal job, 22 pending downstream definitions. Those pending definitions are not 22 active requests. |
| Isolated Forge implementation pilot | `<feature-worktree>/runs/forge/implementation-pilot/queue/state.json` | One **blocked** K3P CPU submission, one terminal job, 22 pending downstream definitions after the smoke failure. |

The process scan found no active ParticleGAN `run_grid.py`, `follow_grid.py`,
Forge drain/worker, or LR-free `harness/pool.py` consumer. This is a local process
observation only. The historical roots `/ml2/hypergan/lrfree-20260926` and
`/ml2/hypergan/gan-attempts/gap-fill-20260925` do not exist here; their remote or
previous-machine queues have not been inspected or declared drained.

The GPU 0 queue's working directory is
`/home/martyn/dev/hypergan/training-runs`. Its script consumes the first queue
line before running it, logs each start/exit, and continues after a child failure.
It supports a stop-after-current marker, but **this audit did not create it**.
Any later reconciliation must combine the pending file with the running child
and queue log; copying only the pending file would omit the active request.

Tail-friendly existing logs:

```sh
tail -F /home/martyn/dev/hypergan/training-runs/queue-gpu0.txt.log
tail -F /home/martyn/dev/hypergan/training-runs/logs/current.log
tail -F /home/martyn/dev/games/mods/3.random_ai_npcs_and_presets/ml/out/npc-misgan/npc_misgan_s1_encES_pg070_b2k_lazy1_gxhop3_dres3_gmblock_pack16_200k/stdout.log
tail -F /home/martyn/dev/ParticleGAN/runs/forge/events.jsonl
```

## Launcher and consumer mapping

The source scan used tracked Python/shell paths and process-launch/entrypoint
signals, followed by source reads. The worktree contains **22**
`experiments/train_*.py` entrypoints, **334** Python/shell files under
`reports/toy100/` (108 with entrypoint/process signals), and **122** under
`reports/transfer_suite/` (24 with those signals). These are inventory counts,
not independent experiment counts; archived copies and helpers overlap.

| Existing entrypoint / consumer | Current contract and artifacts | Forge coverage and migration decision |
| --- | --- | --- |
| [`experiments/run_grid.py`](../../experiments/run_grid.py), [`experiments/follow_grid.py`](../../experiments/follow_grid.py), [runner guide](../../docs/experiment-runner.md) | TOML/YAML or explicit config manifest → selected `--trainer`; defaults are frozen; per-output advisory locks; `requested_config.yaml`, `summary.json`, `run_grid_complete.json`, `log.txt`; previous attempts preserved in `.run_grid_history`. `follow_grid` invokes the runner and combines logs. Default runner concurrency is five workers per GPU. | Forge implements immutable requests, source snapshots, exact task reuse, budgets, ownership and central logs for its registered adapters. It is **not** a generic `--trainer` replacement. Do not translate a completed grid manifest into a new qualification PASS, or run both owners against the same output directory. Keep these entrypoints for uncovered domains. |
| [`experiments/sparse_pipeline.sh`](../../experiments/sparse_pipeline.sh) → config generator → `run_grid.py --trainer experiments/train_sparse.py` → analyzer | `recipe`, `ucd`, `sparse`, `discrete`, `champion`, `fewshot` stages; exact manifests; `results/sparse/PIPELINE.log`; namespace/config generation and analysis depend on legacy paths. | No sparse task/adapter or result-contract replacement in the initial 29 Forge cards. Retain pipeline and analyzer. Historical seed-list support does not authorize new seed-only screening experiments. |
| The 22 `experiments/train_*.py` programs | 100-Gaussian and denoising; CIFAR DDGAN / particle DDGAN / particle AE; MoG VAE, VAE stability, encoder fit, autoencoder; sparse / trajectory / transition; ten `train_gym_*` control or finetuning entrypoints. Config/result conventions are consumed by legacy analyzers and reports. | Forge covers particular toy/native measurements, not these entire training products. CIFAR, Gym, sparse, autoencoder/VAE and application checkpoint consumers need explicit domain adapters, metrics, API/parity and artifact decisions before retirement. |
| [`benchmarks.toy100`](../../benchmarks/toy100/__main__.py), [`benchmarks.toy_suite`](../../benchmarks/toy_suite.py), [`constraint_screen`](../../benchmarks/toy100/constraint_screen.py) | Native coverage/accuracy commands; common-recipe full 22-task replay; predeclared fixed-budget constraint screens. Original gates read events, sample clouds, holdout and config. Native `--steps` affects the original learner's schedule, so a shorter run is not automatically an equivalent prefix. | Forge reuses graders and supports the initial 19 transfer tasks, three native 7k tasks, ring endurance and separate continuations. API/default prior/RNG identities differ where declared; original replay/CLI behavior remains necessary for archived reproduction. Keep legacy commands usable. |
| [CI trained gate](../../.github/workflows/toy100.yml) | On merged primary branches/manual dispatch: CPU torch 2.13, pinned AVX2 dispatch, 45-minute job; runs `python -u -m benchmarks.toy_suite run --output artifacts/toy-suite-ci`, uploads `common22-trained-gate`. | This is an actual automated consumer that has **not** migrated to Forge. Do not replace its whole-suite protection with an uncalibrated smoke profile. Any CI change needs the CI maintainer's declared equivalent coverage and retained artifacts. |
| [`benchmarks.transfer_suite`](../../benchmarks/transfer_suite/README.md), public-default verification, formulation builders and solver searches | Live sustained verdicts, separate required/ranking/diagnostic importance, host budgets; formulation reports rebuilt through `reports.transfer_suite.formulations.build`; solver witnesses preserve their own policy. | Forge adapters share behavioral/scalar host logic and independent verdict reconstruction. Frozen solver searches, reserved-family claims and historical policy-specific scripts are not all active Forge tasks. Keep original evidence and separate historical views. |
| [`benchmarks.locked_shared`](../../benchmarks/locked_shared/__main__.py), [`benchmarks.paired_error_2d`](../../benchmarks/paired_error_2d/run.py) | Standalone reference/parity suites; locked-shared package/wheel checks; paired-error task × arm × cloud matrix using a process pool and complete state/artifact comparisons. | Some behavioral hosts are reused. The full reference/wheel parity and paired-error protocol are not declared Forge task replacements. Preserve these validation entrypoints; add a task only with a named purpose and original contract. |
| [`reports/toy100/gap-fill-20260925/run_gap_fill.py`](../toy100/gap-fill-20260925/run_gap_fill.py), [`run_supplement.py`](../toy100/gap-fill-20260925/run_supplement.py) | Archived immutable-command manifests, hard-coded `/ml2/...` root and different GPU UUIDs; per-job logs, `progress.jsonl`, `status.json`, `completed.json`; memory-based pool admission. Supplement adds P1 own-state ring evidence. | Outcomes are historical inputs, not jobs to relaunch on this workstation. Forge covers analogous tasks under frozen new identities and own-state dependencies. No automatic import of an old machine's pending work or ownership. |
| [`gpu-leaderboard/batch.py`](../toy100/gpu-leaderboard/batch.py), [`gpu-known-winner-control/batch.py`](../toy100/gpu-known-winner-control/batch.py), [`formulation-round-20260924`](../toy100/formulation-round-20260924/) drivers and direct-particle launchers | Candidate matrices, per-candidate probes and control replays; ledgers and generated boards. Some intentionally run every supported toy despite quality failures. Some carry copied implementations, fixed paths/devices and special initializer controls. | Initial task families map to Forge; archived exact implementations do not become interchangeable public-package recipes. Preserve source/package/fixture identity. Replace new idea orchestration only after each still-used driver consumer is identified and its API capabilities are represented. |
| [`reports/transfer_suite`](../transfer_suite/) search/driver families | `valid_search`, `rare_focus`, solvability and stress reference scripts; several alter optimizer/host policy in local drivers and save protocol/source/evidence beside results. | Historical import supports recall. New implementations should declare reusable public API bindings; do not silently adopt driver monkeypatches as Forge capabilities or claim host parity from matching names. |
| Pinned #155 [LR-free harness](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/harness/README.md): `submit.py`, `pool.py`, `screen.py`, `wait.py`, `lrlib.py` | `queue/*.json` → screen children; package/config hashes; live `pool-config.json`; `ledger.jsonl`, `LEADERBOARD.md`, `pool.log`, `runs/<candidate>/<task>/`; parity refusals and clean/noisy/EMA receipts. Archived documented pool is seven slots per GPU. | Forge adopts the submit/drain/ledger pattern and preserves evidence through import. No live bridge currently adopts that daemon's claims or mutates its queue. Old package-copy candidates need public capability migration/parity or remain historical/BLOCKED. Its queue must be reconciled at the actual owning machine before cutover. |
| External HyperGAN queue and NPC MisGAN application | Active production/application-like consumers of ParticleGAN environments in separate repositories. HyperGAN owns pending PR215/217/0.8 application trials; NPC uses its own training/checkpoint/log directory. | These consumers are outside the Forge toy scheduler's current adapters. Preserve their ownership and package environments. They are resource conflicts to coordinate, not jobs Forge can cancel or replace. |

## Safe adoption and cutover

| Step | Required concrete record | Owner |
| --- | --- | --- |
| Inventory each still-used consumer | Launcher path, host/repository, effective config/package revision, pending/running/completed IDs, output/checkpoint paths, current scheduler and log path. Include the active child already removed from a destructive queue. | Existing queue/application owner, with Forge coordinator |
| Establish coverage | Map the actual host/objective to a supported task or declare a gap. Verify public API mechanisms, MoG/cloud law, initialization/RNG, scoring and checkpoint parity. Keep historical scopes separate. | Task/domain maintainer |
| Calibrate adoption | Meet unchanged false-accept/reject, denominator and cost criteria in the matching current cohort. CPU smoke failures and software tests alone do not approve a screen. | Protocol maintainer / Forge coordinator |
| Reserve the pilot | Record allowed GPU UUIDs, time window, one scheduler/resource owner and bounded campaign. Recheck processes and both device utilization and ownership immediately before claims. | Device/queue owners and Forge coordinator |
| Reconcile pending work | Ask the old owner to stop **new claims**, let active work drain or explicitly checkpoint it, preserve manifests/receipts, and record which owner has each remaining request. Never allow both schedulers to launch it. | Old scheduler owner |
| Transfer supported requests | Resolve new immutable Forge requests; import compatible old evidence with exact provenance; unsupported requests stay with their named owner. Verify deduplication and account for already spent cost. | Forge coordinator plus domain maintainer |
| Enable the entrypoint | Point new idea users at [EXPERIMENTATION.md](../../EXPERIMENTATION.md), keep compatibility commands for unmigrated consumers, and complete the fresh-checkout walkthrough. Retire a launcher only when its consumers and artifacts are replaced. | Repository/CI maintainers |

A two-GPU pilot must contain explicitly budgeted, GPU-applicable work on both
devices. The initial three behavioral smoke tasks are CPU hosts; passing `--gpus
0,1` does not make those tasks a two-GPU execution test. Verify cancellation,
restart, per-device placement, exact reuse and the single cost owner from receipts;
do not reinterpret contended wall time as an efficiency comparison. Preserve
the existing failed CPU pilots and their unlaunched downstream tasks.

## Rollback

Keep legacy scripts, pending manifests, frozen package/environment references and
output directories intact during the pilot. If Forge ownership must be rolled
back, pause its campaign claims first; reconcile worker leases and terminal
receipts; use its explicit cancellation path only for requests owned by that
campaign when necessary. Retain the failed/incomplete result and its cost.
Return only unexecuted, explicitly handed-back requests to the old owner, with
the same config/output identity. A completed Forge task or a scientific failure
must not be silently requeued as fresh work. Resume the legacy owner only after
the ownership ledger proves there is no duplicate live request.

## Concrete coordination still needed

**Who will reserve a joint GPU window around the existing HyperGAN GPU 0 queue
(current PG214 run plus the three pending launchers) and the NPC MisGAN GPU 1
run?** The safe default is to wait for their owners to release both devices, or
agree on one shared resource owner and an explicit contention policy. Available
VRAM does not answer that question. This audit neither stopped those jobs nor
assumed authorization to change their queues.

Inspection commands were `nvidia-smi` device/compute-process queries, targeted
`ps` and `/proc/<pid>/{cwd,fd}` reads, tracked-path/source searches, `git show` of
the pinned #155 harness, and direct reads of the two Forge state files and the
identified HyperGAN queue script/file/log. No training commands were executed.

Related: [migration status](MIGRATION.md), [accepted migration checklist](../../docs/better-experiment-automation-plan-2026-09-28.md),
[continuation review](CONTINUATION_REVIEW.md), [current calibration protocol](CURRENT_CALIBRATION_PROTOCOL.md).
