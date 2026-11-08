# Verified results from existing uninterrupted runs

BISPF's recorded `ring_hold` run already measured the immediately following
`ring_extension`. The reporting path had required a restart checkpoint for that
extension, despite both tasks belonging to one uninterrupted execution group.

The recovery verifies the original source snapshot, frozen task and recipe
identities, request/result/raw hashes, complete saved curve and provenance
artifact manifest. It uses the frozen source's full grader for both tasks.
The hold passes, and **all 300 extension checks pass**. The main leaderboard
now exposes that result as a separate diagnostic measurement. Ordinary selected
qualification retains its original lower-tier requirements and scores.

The compact JSON certificate links the existing producer and the original
blocked extension attempt. Neither original receipt was rewritten. The
provenance-only checkpoint remains ineligible for restarting training.
Recovery adds **zero optimizer updates and zero sampling draws**; the producer's
2,900 completed updates are attributed to that producer once.

Reconcile a stopped or otherwise unowned coverage state without training:

```sh
python -m experiments.forge.attempt_coverage reconcile --state-root RUN_DIRECTORY
```

The command takes the coordinator lock and never starts GPU work. The regular
runner performs the same check before scheduling a grouped child. Repeating
reconciliation preserves the existing recovered result. Failed parents,
mismatched bindings, changed artifact bytes and incomplete curves cannot gain
credit; a numerically failed extension remains FAIL.
