# Prepared external Codex search: run after compaction

**Preparation only. No external model request or GPU experiment was launched.**
The next supervisor starts three distinct approaches with the existing
`/ml2/hypergan/try-gan.sh`, using `gpt-6-astra` and reasoning effort `max`.
Each receives an isolated worktree from public API commit
`fa511ce010120b502f494d717d01b14b8551eed8` (PR195), one GPU worker,
and no hard timeout (`--minutes 0`). A three-proposal cap provides a review
point per attempt; it is not a total search budget. No token or dollar budget
was supplied. Continue reviewed attempts until qualification or user stop.

| Approach | First mechanism lead | GPU |
|---|---|---|
| `reversible_precision` | PX3 successor: autonomous closing/reopening with all horizon coupling removed | 0 |
| `data_drift_mobility` | Separate real-data drift from generator/game instability | 1 |
| `constant_rate_stability` | Repair KA2 memory/update instability while all LRs stay constant | 0 |

`COMMON.md` is the governing experiment brief. Recovery is arrival plus later
stability, with no 81/81 deadline requirement. The learner cannot know the
planned ending or data-change times. All schedules, noise and API continuation
behavior matter, not just the critic controller. Fixed initialization and
smoothing counts alone do not disqualify a policy. Research-host scores alone
cannot qualify an API default. Neither PR155 nor PR195 should be merged here.

[Current evaluation declarations](evaluation-protocols.json) specify the shared
4600-update single shift, 7500-update stationary run, and 9000/30000-update
delayed/repeated-change evaluations. Budgets belong only to the evaluator.
These declarations supersede the old stress protocol's 81/81 and extra-seed
instructions for new work, while preserving that historical evidence unchanged.

## Preview and offline verification

```bash
cd /ml2/hypergan/ParticleGAN-k3p-continuous-search
python reports/toy100/continuous-eligibility/launch/verify_launch.py
python reports/toy100/continuous-eligibility/launch/launch.py
```

The default invocation is a read-only preview. It validates the API commit and
shared driver hash, prints prompts and exact launch commands, and counts active
shared-driver workers. The check script runs the driver's `--dry-run`; it does
not request a model response or create worktrees. Local CLI inspected during
preparation: `codex-cli 0.157.0`; its help exposes the used exec flags. This
does not test account/model availability, which only a real later launch can
establish. The shared driver explicitly supplies `model_reasoning_effort="max"`.

## Start after compaction

Read the eligibility audit and these briefs, inspect live jobs and current GPU
memory, then resume this new search. Existing user authorization covers this
launch; do not ask for approval again. The September 25
`continuous-round-3/search-stop.json` is preserved historical evidence. The
latest user requested a new search after compaction; that old round's stop must
not silently prevent the newly authorized run.

The wrapper honors `/ml2/hypergan/gan-attempts/STOP` and
`/ml2/hypergan/gan-attempts/continuous-api-20260926/STOP`. It never deletes them.
If present from preparation, the supervisor must read them and explicitly
resume only this authorized search after compaction. Preserve a marker if a
newer stop supersedes the user's launch request. Never remove unrelated stops.

```bash
python reports/toy100/continuous-eligibility/launch/launch.py --launch-after-compaction
```

This fills idle lanes up to three active shared-driver agents AND three GPU
workers in total, counting other `try-gan.sh` attempts as capacity reservations.
It does not kill or modify unrelated training jobs. It uses the existing
`continuous-launch.lock`; other launchers that ignore that lock still require
supervisor coordination. Each lane is registered with the existing monitor.
The wrapper returns after launching; it is not a blocking model orchestrator.

## Review and replenish until qualified

Tail the `launcher.log` path printed for each lane; nested attempt directories
also contain `codex.log`, `result.md`, `tests.jsonl`, sources, and worker logs.
The live registry is
`/ml2/hypergan/gan-attempts/continuous-api-20260926/batch.json`.
Save compact progress and failed evidence so context compaction loses nothing.

As a lane exits, inspect its actual results. Exit zero means execution finished,
not a win. Update the lane brief if the hypothesis should change, and write a
concise review note containing the measured failure, promising successor, and
any shared comparison to reuse. Then refill that lane while others continue:

```bash
python reports/toy100/continuous-eligibility/launch/launch.py \
  --lane reversible_precision --review-note /absolute/path/to/review.md \
  --launch-after-compaction
```

Replenishment refuses to launch without a review note after any previous attempt
in that lane. Prior attempt paths and that review are added to the next prompt.
The supervisor can select another lane or omit `--lane` to fill all idle slots;
the same review note then applies to every new slot. Keep at most three active
approaches, one worker each; no nested models, seed sweeps or unchanged reruns.
This review/refill loop is deliberate: an unattended process must not recycle
failed ideas indefinitely or interpret an agent's exit as successful research.

Before stopping for success, audit the SAME candidate through actual API cold
acquisition, long stationary retention, delayed and repeated changes, reversible
rates if applicable, budget-independent prefixes, complete checkpoint state,
matched K3P comparison, and its own broader 22-task quality checks. Record
arrival, departures and subsequent stability for every change. Finite tests
support a continuous-learning claim; they cannot prove stability forever.

To stop, write a STOP marker for this search and write `STOP: save results and
exit` into each running attempt's `supervisor.md`. The existing agent checks
that file before experiment batches. STOP markers prevent new launches; they
do not themselves terminate already running agents. Monitor graceful exit and
stop only this search's process groups if necessary. Preserve all artifacts.
