# Forge queue and public adapter review — 2026-09-28

This is an implementation review, not scientific qualification evidence. No
training campaign or GPU experiment was launched by this review. Bounded CPU
update tests and synthetic process fixtures exercise software contracts only.

## Findings sent to the coordinator

| Area | Finding and concrete consequence | Resolution observed during review |
| --- | --- | --- |
| Ordered smoke | `Queue._eligible` originally skipped a running required task and selected the next pending task in the same tier. Multiple slots could spend the entire smoke budget before the cheapest screen failed. | Coordinator changed required-task scheduling to wait for the preceding result and added a regression test. |
| Retry cost ownership | `cost_owner` changed on retry while aggregate `charged_seconds` retained prior attempts. Candidate budgets could attribute earlier campaign spending to the new payer. | Coordinator added immutable per-attempt `charges` and uses them for candidate accounting. |
| Frozen grading | Collector originally imported the checkout's current evaluator after a frozen job finished. Editing a grader during an attempt could change its verdict. | New `evaluate.py` subprocess imports the source snapshot under the same supervisor deadline. Collector verifies its raw-result hash and source digest. |
| Cancellation authorization | Every request subscribed to all jobs, including tasks above its requested tier cap. Cancelling a Tier 3 request could leave Tier 2 running for a completed Tier 1-only subscriber. | Reproduced without launching a process: completed Tier 1 subscriber retained `t2`, and no `cancel.json` was created. Coordinator is adding cap-aware subscriber filtering. |
| Grouped endurance authorization | Retiering ring hold below its grouped extension could authorize the hold while the shared adapter also executed the additional 300 updates above the requested cap. | Coordinator assigned validation that keeps an execution group in one tier. A group remains one uninterrupted run and one charge. |
| Central progress | Adapter observations appeared in attempt `run.log`, while `forge logs --follow` originally displayed only submission/claim/completion events. | Coordinator is adding collector ingestion of worker progress. Attempt log paths already appear in claim events. |
| Evaluator runtime | New independent evaluator initially omitted the runner's thread and deterministic settings. Its native sample audit could use a different thread policy and consume unexpected evaluation time. | Sent to coordinator: apply the same numerical policy before grading. |
| Evaluation cancellation | Supervisor's evaluation phase originally labeled a cancellation-file request as `timeout` unless a signal also set its local flag. | Sent to coordinator: classify either cancellation source as cancellation. |
| Orphan recovery | If supervisor and runtime group leader both die while a forked descendant retains the inherited lease, recovery must terminate the remaining process group. The initial recovery code killed only when the leader's recorded PID identity was still alive. | Sent to coordinator as a recovery corner case; preserve lease fencing while cleaning descendants. |

Files reviewed: `experiments/forge/queue.py`, `worker.py`, `planning.py`,
`runtime.py`, `evaluate.py`, `sources.py`, `__main__.py`, task grading in
`views.py`, and bounded worker/queue tests. Hardware cohort binding now includes
CPU/CUDA selection and GPU model/driver/compute capability. Queue slot selection
checks the chosen GPU model; physical GPU index does not define scientific
compatibility.

## Public adapter checks

`experiments/forge/adapters.py` uses public `GANTrainer` updates for vector,
image, ring, and native 100-mode hosts. Candidate optimizer settings remain in
the resolved recipe; host definitions contribute data, architecture, resource
sizes, and explicit schedule horizons. Native runs save real sample arrays for
the existing coverage/accuracy artifact grader, including its independent
100,000-sample holdout. A short software test cannot satisfy the original gate.

Ring hold and extension share one state and one uninterrupted update sequence.
`GANTrainer.max_steps` bounds execution independently from the unchanged recipe
schedule horizon. Dense observations stop at the first failed hold; the adapter
cannot select a later successful window. Grouped result rows charge execution
once. Unit tests use a synthetic observation sequence to test this policy;
their PASS values certify only the test fixture, not a candidate.

The explicit image cloud exception enumerates component indices through public
`GANTrainer.sample(fixed_first_n=True)`. It retains the formulation's output
noise and records that sampling law. It is not aliased to a historical
noise-free image receipt. MoG enumeration retains mixture noise; cloud sigma
zero plus output noise zero consumes no evaluation RNG.

Behavioral integration uses `behavior_adapters.py` and public component hooks
inside the existing host loops. A nested conditional-view/noise wrapper exposed
a public critic-penalty limitation during integration: EMA mapping handled only
one wrapper. `CriticPenalty._ema_view` now maps nested wrappers and all matched
critic descendants, with a bounded public API regression test. Behavioral
tests and a complete initial smoke run remain the owning integration agent's
responsibility.

Clock-free audit, target-shift, and 14k continuation capabilities cannot be
claimed by the existing scheduled formulation. Unsupported dispatch is an
explicit pretraining capability blocker, not a scientific failure or a claimed
implementation of a clock-free learner.

## Validation observed

- Public API/adapters plus existing trainer, K3P, MoG, AE, and initialization
  regressions: **143 passed** before the final enumeration/wrapper tests.
- Updated public API and K3P wrapper regressions: **45 passed**.
- Queue, real supervised-worker fixtures, public API, and scalar adapters:
  **51 passed in 4.71 seconds** at the review checkpoint.

These counts describe bounded software tests at their respective worktree
checkpoints. The coordinator's final integrated run must cover subsequent
changes, cancellation-cap regressions, grouped-tier validation, and complete
smoke execution before any operational completion claim.
