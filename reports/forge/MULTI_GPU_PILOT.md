# Bounded physical multi-GPU pilot

## Current-source registration — 2026-09-29

The merged-develop pilot is
[`develop-20260929-gpu-pilot-v1`](calibration-lanes/develop-20260929-gpu-pilot-v1/registration.json),
against the [new quick-screen cohort](QUICK_SCREEN_STUDY.md). Its
[contract](../../configs/forge/campaigns/develop-20260929-gpu-pilot-v1.json)
keeps two `vector_two_broad` cells (K3P and the critic-penalty ablation), the
5,400-second campaign ceiling, and at most one cancellation repair. Clean public
sampling and retained full-component vector gates are pinned before execution.
The acceptance procedure below still applies. Both registrations are unlaunched;
use the new one when validating the merged feature branch.

The latest [ownership observation](LEGACY_CONSUMERS.md#ownership-update--2026-09-29)
found a new NPC training job on GPU 0, with only desktop clients on GPU 1.
A one-device baseline screen cannot satisfy the physical two-GPU acceptance test.

## Preserved earlier-source registration

Status: [registered](calibration-lanes/current-k3p-mog-gpu-pilot-v2/registration.json),
not enqueued or executed. The exact [selection and caps](../../configs/forge/campaigns/current-k3p-mog-gpu-pilot-v2.json)
are frozen against current profile v2. The [legacy-consumer audit](LEGACY_CONSUMERS.md)
found both A6000s actively training. Reserve a joint window after those owners
release the devices; the HyperGAN parent queue can launch another child, so an
idle instant is insufficient. Do not stop or adopt those jobs as part of this
pilot. The coordinator has requested the missing capacity window.

Subsequent coordinator fixes changed the feature checkout's source digest.
The registered pilot still pins source `c673226c`; its read-only plan is prepared
using the preserved execution checkout `runs/forge/fresh-checkout-v2` and the
current coordinator CLI. Use that explicit `--root` and the common queue, as
shown in the [quick-screen preparation](QUICK_SCREEN_STUDY.md). Do not relabel
newer scientific source as this registered cohort.

## Work and budget

After the CPU smoke diagnostic readout, use the same frozen current-calibration
cohort and register exactly two independent reference cells: `vector_two_broad` for
`k3p` and `forge-no-critic-penalty`. Each uses the existing 1,200-update task, live
grader and learned-MoG prior. This pilot tests
physical placement and workflow; it does not supply the complete calibration denominator. The two ideas differ by a real public
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
