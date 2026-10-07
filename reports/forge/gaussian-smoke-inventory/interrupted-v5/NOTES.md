# Why V5 stopped

V5 executed source `df4539bbdda7bec5d20f95afd0bdb3fc283f41d2`, frozen digest
`3bf51eab2a80eef3645ca5c7df9fa0583cc7053332bc94b19ec874e2e19f98a0`. The word
task's evaluator source pin still named the helper bytes from before PR339 added
the project-wide serial-autograd wrapper. Its preflight therefore blocked the
word task before training. PR341 changes only the current word task's source
metadata; its architecture, target, prior, streams, budgets, cadence and numerical
bounds remain unchanged. See the [source-pin audit](../../word-runtime-pins/README.md).

The repaired word declaration is also captured as an evaluator support file in
the catalog-wide source manifest. Consequently the global digest, candidate
revisions and every job compatibility key change, including nonword jobs whose
numerical code is byte-identical. V5 evidence remains valid under its original
identity; it cannot automatically fill V6 cells or receive new-source credit.
The interruption requires no new seed, recipe tuning or outcome-based retry.

The coordinator and monitor stopped before preservation. All 74 existing V5
coordinator, queue and worker locks were idle; no workers or reservations remained.
The queue retains 72 terminal jobs, 572 pending jobs, 12 blocked submissions and
11 queued submissions, byte-exact. Missing work remains visible in the
[strict partial readout](readout.json). No original verdict, queue status or
durable result was rewritten. All completed task states passed the saved-byte
complete-state certificate checks; no checkpoints or neural models were loaded.

The 12 registered search summaries were refreshed from their frozen request keys,
then [the stop receipt](interruption.json) bound the actual state bytes and
785.797474547755 paid seconds. The archive retains 2,480 original files
(465,226,515 uncompressed bytes), including the complete frozen source snapshot,
queue, request/evidence/result files, worker logs, streams and checkpoints. The
[receipt](archive.json) binds the 105,352,732-byte ignored archive and every member
hash. It has `finalized: false`, `new_source_credit: false` and
`qualification_input: false`.

Previous cohorts' separate paid cost is 9,845.240926956409 seconds; combined paid
cost through this cut is 10,631.038401504164 seconds. Successor
`gaussian-smoke-inventory-v6` has origin
`45f056556503341bccf3ade0cd3365c5d0dadb91`, digest
`6269a18ac4f82564cb16ba19afa4b3dd2f836a2b4085fbe5aeb81a35a453e895`.
V6 is the corrected source-preflight rerun of the same authorized roster.

Four metadata-only archive checks and fifteen existing collector checks pass.
The default collector still refuses this incomplete cut before writing output;
collection requires explicit `--allow-interrupted` and the exact archive receipt.
Preservation launched no training and changed no scientific source files, task
definitions, gates, protocol, policy or leaderboard.

After this archive was verified, the generic archive helper also allowed
`UNKNOWN` trial status only in explicit interrupted mode. It still requires the
exact original attempt set and registered request. V5 already passed these
checks under the preceding helper revision; its archive and receipts were not
rewritten or recreated.

```sh
python reports/forge/gaussian-smoke-inventory/archive_final.py \
  --archive artifacts/gaussian-smoke-inventory-v5-interruption.tar.gz --verify-only
```

Tail local preparation logs at
`runs/software/gaussian-smoke-inventory-v5-preservation/`.
