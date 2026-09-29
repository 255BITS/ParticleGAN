# Bounded physical multi-GPU pilot

Status: prepared, not executed. The [legacy-consumer audit](LEGACY_CONSUMERS.md)
found both A6000s actively training. Reserve a joint window after those owners
release the devices; the HyperGAN parent queue can launch another child, so an
idle instant is insufficient. Do not stop or adopt those jobs as part of this
pilot. The coordinator has requested the missing capacity window.

## Work and budget

After the CPU smoke diagnostic readout, use the same frozen current-calibration
cohort and register exactly two independent reference cells: `img_blobs4` for
`k3p` and `forge-no-critic-penalty`. Each uses the existing 600-update task, live
grader and explicit finite-centre particle-cloud exception. This pilot tests
physical placement and workflow; it does not supply learned-MoG coverage or the
complete calibration denominator. The two ideas differ by a real public
mechanism ablation, with the same screening seed and initializer.

The diagnostic registration permits no qualification reuse. Reserve 1,800 seconds
per attempt, at most 3,600 per candidate and 5,400 for the campaign. This covers
two first attempts plus one bounded repair of the deliberate cancellation below;
it is a ceiling, not a runtime estimate. No additional downstream task is selected.
If the CPU study identifies an implementation error, repair it and declare a new
cohort before registering this pilot. Do not spend on an invalid frozen design.

## Acceptance procedure

1. Recheck the physical device IDs, GPU owners, device model and available memory.
   Record the agreed exclusive window or common resource owner in the registration
   readout. Freeze source, cohort, selected cells and budgets before enqueue.
2. Enqueue from the implementation checkout and a fresh validation clone, using
   one explicit queue root. Submit one identical request from both; verify one
   request/execution identity and one cost reservation. The second substantive
   idea provides the second runnable workload.
3. Drain with `--gpus 0,1 --workers-per-gpu 1 --campaign <registered-id>`. Record
   overlapping real child intervals and distinct physical device assignments.
   A CPU worker, fabricated slot or unused second device does not pass this item.
4. After both children start, interrupt only the Forge coordinator. Restart its
   drain; the same live worker leases must be recovered without duplicate launch
   or reservation. Preserve both centralized event streams and per-attempt logs.
5. Cancel one Forge request while its child is running. Verify its process group
   exits and resources/cost reservations reconcile. Resubmit the same frozen request/campaign to reattach its subscriber, then
   register the execution repair with `retry <compatibility-key> --reason ...`. Keep the cancelled attempt and
   its cost. Never retry a scientific FAIL. At most one repair is authorized.
6. Drain the selected work, record every verdict/error and publish both readouts.
   Check `stats`: unique paid attempts, measured elapsed time/memory, observed
   physical concurrency, reuse, final zero reservations and no live Forge child.
7. Demonstrate retiering/reduction against the completed receipts without another
   worker launch. Check that diagnostic evidence remains ineligible for ordinary
   qualification under every view.

If an attempt finishes before a planned restart/cancel action, record the missed
operational check. Do not quietly launch extra scientific repeats to obtain it.
Prepare a separately bounded follow-up only if the missing acceptance check still
requires real GPU execution. Keep source snapshots and durable receipts, plus
explicit locations/hashes for bulk artifacts that are not portable in Git.

The pilot alone cannot authorize cutover. Current calibration must meet its
unchanged frozen criteria; reconcile the covered legacy consumers and pending
requests, retain rollback receipts, and then make the adoption decision described
in [MIGRATION.md](MIGRATION.md).
