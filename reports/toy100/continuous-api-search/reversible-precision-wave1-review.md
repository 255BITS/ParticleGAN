# Review before refilling reversible precision

API-RP2 is a useful continuous-learning result, but is rejected as the default
because its frozen image stability test fails. Preserve every success and failure.
This review authorizes a fresh three-proposal attempt after the predecessor exits;
COMMON.md, one GPU worker, fixed declared seeds and all evaluation gates still apply.

Predecessor:
`/ml2/hypergan/gan-attempts/continuous-api-20260926/20260926T211209Z-1398466/reversible_precision/20260926T211209Z-1398474`

Read its final report, per-update precision/rate traces, source ZIPs and declarations
under `repo/reports/reversible-precision/`. Root's audited evidence is beside this
review. Do not repeat the failed configuration unchanged.

## What worked

The public API owns both reversible precision and eager Adam initialization.
Startup and reopening use the same upper rate; the controller closes and reopens
without target-change notifications or a learner horizon. Its independent reference
gradient gap and update activity control the decision. Ordinary public construction
reproduces the preceding CUDA-eager diagnostic exactly.

- Single change: original arrival640, retention177/177; changed arrival delay500,
  then171/171.
- Stationary7500: 687/687 after arrival, no departures.
- Uninterrupted30000: original arrival640, then537/537. Changes after6000,7800,27000
  lead to arrival delays360,420,320 and retention145/145,1879/1879,269/269 respectively.
- Horizon-prefix2400 matches across evaluator budgets. Same-process checkpoint
  branches at each long-run change match100 actual batches and complete states.

These are this candidate's own public API scores, not ancestor evidence. They
remain valuable even though the candidate fails broader quality.

## Why it fails

Frozen `img_intensity2`: only observations450 and600 pass among24. After its first
pass it fails475,500,525,550,575; minimum subsequent HQ is .375. Final passing
suffix1 is below the existing requirement5. Final HQ .90625 alone is not a pass.
The precision controller stays open for all600 updates and never settles.

The independent immutable-source audit confirms the correct residual_upsample16
model, seed, actual paired-noise stream, current public controller, and frozen
quality rules. All package files equal the successful ring source. This is a
real transfer failure, not a legacy-runner substitution or wrong image model.
The other21 tasks and matched K3P were NOT_RUN after that failure.

The final API-RP3 proposal adds generator-update cancellation and serial backward.
Its own ring screen passes: initial610,180/180; changed arrival+460,175/175.
Its image test passes0/24, ends at one quality mode/HQ.40625, and still never
closes. Keep these scores separately; longer qualification is NOT_RUN.

Choose a new general controller mechanism that addresses stochastic/high-frequency
game motion as well as large distribution changes. Explain what the recorded
image gradients/activity show and why the proposed observation/control rule
responds. A threshold grid, task-specific branch, evaluator-selected stop or
shortened fixed initialization window is not a justified mechanism. The other
lanes own explicit real-data drift sensing and constant-rate implicit game updates.

A concrete next lead is to combine RP2's demonstrated long retention control with
C6's implicit game correction, which passes the frozen image task but fails long
stationary retention. This addresses complementary measured failures. Read both
implementations and define one coherent joint update/controller under serial
execution; do not simply splice private training loops or inherit either score.
The constant lane will separately examine its critic-reference memory/implicit
model, so a controlled autonomous-retention hybrid belongs here.

## Execution and early evaluation order

The earlier ordinary API-RP1 partial failure is valid: CPU scalar Adam counters
are normal metadata. Moving them changes arithmetic; never discard a run solely
because they are on CPU. RP2's eager initialization is a distinct library option.

Cross-process replay of RP2 remains unpassed despite equal immediate state.
The data-drift lane isolated higher-order autograd priority differences and built
a scoped `serial_backward=True` correction. Use the independently reviewed
standalone patch from that lane, explicitly declare the execution change, record
copied hashes/diffs, and earn new quality scores. Do not transfer RP2's successes
to a new runtime. Keep the caller's context and checkpoint mode contract intact.

Run the cheap sensitive frozen image gate before another expensive long run.
Reuse the audited `rp2-img_intensity2/source.zip` evaluator with a copy receipt,
changing only the candidate API construction and required receipt fields. The
task spec comes from `plans/default_comparison.json`: residual_upsample,width16,
600updates,32particles,z8,batch32,24observations and five final passing checks.
Evaluation noise uses seed402+step+1901. Preserve full host initialization/scoring;
record Torch/CUDA/device and snapshot all imported helpers/declarations.

A passing image screen is not full22. The route map beside this review documents
the remaining API integrations. Current GANTrainer can cover fourteen hosts after
adaptation; eight need faithful component-controller binding. No LegacyRecipe
fallback, removed auxiliary objectives or copied ancestor passes.

## Shared ownership

C6 failed stationary retention, so no comparator is currently justified. Root
will assign one surviving lane the exact public K3P comparator; do not duplicate
it or run it for an already rejected candidate. The reference identity audit beside this review
explains why eager-state K3P or a current substituted kernel is another algorithm.
Use ordinary archived public K3P with its own counters and schedules in the same
runtime when eventually comparing a survivor.

Use unique API-RP4 onward names. No seed sweeps, no PR merges, no default promotion
without the unchanged candidate's complete qualification.
