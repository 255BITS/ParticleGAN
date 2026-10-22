# Three BISPF repair hypotheses: completed matched diagnostic comparison

This study tests three global trainer changes against the selected
`bispf-optimized-quintic-v1` recipe on one common committed source. The user
requested separate agents for output allocation, settling and local fidelity,
followed by a comparison. Prior investigations are PRs #463–473. The
[current technique inventory](../technique-inventory.md) remains the single
goal leaderboard; this is a bounded research comparison, not qualification.

**All 60 runs are complete. None of the three recipes repairs all eleven
failures.** All four original passing controls survive under every recipe.

- **Allocation: 6/15 PASS, two new sustained repairs.** Anisotropic mixtures and
  corner blobs pass. Rare occupancy improves from 0 to 81/4096 at the endpoint,
  but rare-axis shape remains unstable. Width and overlap improve without their
  required terminal retention. See the [measured allocation report](allocation/MEASURED.md).
- **Settling: 4/15 PASS, no new gate repairs.** Fourteen tasks exactly reproduce
  the control's retained numerical trajectories and final model/optimizer
  states, including Gaussian and overlap. Grid changes and improves precision
  from .30497 to .35668, still failing with worse covariance bias. See the
  [settling readout](settling-results.md) and its negative observer controls.
- **Fidelity: 5/15 PASS, one new sustained repair.** Overlap passes with twenty
  consecutive final checks. Width covariance error improves 9.68690 to 2.13772
  but remains above .85. Image fidelity worsens: intensity has 9/32 clean images
  versus 19/32, while blobs saturate toward blank output. See the
  [fidelity readout](fidelity/README.md).

The [complete numerical appendix](numerical-results.md) gives every task/arm
verdict, continuous measurements and actual-training GIF. The
[compact result](comparison.json) binds original request/result/certificate
hashes. All 45 changed-arm/control pairs match initial models and priors,
sampling, update counts and consumed non-evaluation streams. The fresh control
exactly reproduces all 15 archived final metric dictionaries and gate verdicts;
its distinct source/runtime cohort supplies the comparison, not new historical
qualification.

Paid cost is **6,604.419 seconds across 60 unique attempts**, independently
equal to the queue ledger, within the frozen 112,080-second reservation. Each
arm stays within its 28,020-second cap. Timing is potentially contended and
does not establish relative speed.

**Recommendation:** retain energy's two scoped gains as the strongest partial
result, and keep the existing representative recipe. The next mechanism to
isolate is a cooling signal that handles finite-sample noise while preserving
incomplete mode acquisition, alongside saved-state rare-shape diagnostics.
Moving an inactive factor cannot resolve the Gaussian/overlap failures; a
naive energy replacement also fails explicit missing-mode controls. Quartic's
overlap win does not justify its global image/tail tradeoffs. Do not combine
per-task winners into an unmeasured recipe. No paid successor was launched.

| Arm | Only global trainer delta | Intended question |
| --- | --- | --- |
| Matched control | All new mechanisms disabled | What does the original recipe do on this same source/runtime? |
| Allocation | `output_energy_weight=1` | Does a direct energy-distance distribution witness repair output probabilities? |
| Settling | `bispf_thermal_application=displacement` | Does cooling applied finite G/prior moves repair settling and retain shift response? |
| Fidelity | `fidelity_transport_weight=1` | Does normalized fixed-frame quartic transport repair tails and local detail? |

The allocation and fidelity arms keep learned prior locations, fixed kernel width
and mixture weights, and SBNP regularization at 0.01. Their extra objectives use
the same existing generator-phase fake and real batches with no new random
draws. They retain the original adversarial loss, critic penalty, optimizer,
thermal behavior and protected-loss geometry. These direct objectives do not
pass through the critic-logit thermal hook; that is part of the declared delta.

The settling arm retains the residual observer/reheating and critic update. It
removes generator-phase adversarial and SBNP gradient cooling, then scales the
actual combined optimizer movement once. SBNP's coefficient/objective remains,
but its gradient-cooling and applied prior-update policies explicitly change.
It adds no controller state or sampling draws. Inactive options retain the
older serialized recipe/checkpoint contract; unsupported component hosts refuse
these mechanisms before training.

## Frozen scope and decision

The [protocol](protocol.json) declares 15 unchanged tasks: all 11 selected
failures, plus Gaussian smoke, ring16 acquisition, mode hold and image stripes.
Each arm runs the entire panel, rather than choosing a different recipe for
each question. Task architectures, target/data laws, initial priors/capacities,
public seed-0 initialization, named RNG streams, sampling, update allowances,
observation schedules and numerical gates remain fixed. Each Gaussian stability
run must continue its own terminal 1,000-update smoke checkpoint after the
acquisition gate passes; the terminal state can fail output bounds. A missing parent stays
blocked. Historical source-bound outcomes are not substituted for this control.

The comparison uses one explicitly separate research-diagnostic view: every task
is diagnostic in its first tier. It grants no ordinary qualification and keeps
failed gates visible. All four first-round studies share campaign
`bispf-three-theme-repair-round1-v1`, with **28,020 reserved seconds per arm and
112,080 seconds total**. These are the sum of the original full-task allowances,
not a forecast of paid time. Hardware/contention costs are reported separately;
this is not a speed comparison. Each study freezes exactly one round.

The control study is the matched ablation that removes the allocation objective;
the other three studies compare their declared mechanism against that same
original recipe. This explicitly binds the necessary fresh control, rather
than admitting an unchanged run as a new repair idea. Predictions/falsifiers in
the study files are administrative research signatures; they do not replace
scientific task gates.

A complete repair requires **all 11 failure tasks and all four regression
controls to PASS**. Report every arm's task verdict, failed bounds, terminal
checks and continuous metrics. A missing/blocked/error outcome earns no pass.
Report repaired cells and regressions relative to the contemporaneous control;
an endpoint improvement alone does not establish a sustained repair. The three
theme groups overlap and their counts are not independent trials.

At most one substantive successor per theme may be declared after a first-round
readout identifies an actual mechanism to change. Any such study needs a new
source/ID, explicit finite cap and preserved controls. No automatic threshold
sweep, seed study, continuation, architecture change, gate relaxation or
public-default adoption follows.

## Execution and evidence

Run declarations are under `configs/forge/studies/*-three-theme-study-v1.json`.
Raw logs, full receipts, per-update events, checkpoints and sample tensors stay
under `/mnt/ml7tb/ParticleGAN-forge/three-theme-repairs-20261022/`. Tail progress:

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/three-theme-repairs-20261022/progress.log
tail -F /mnt/ml7tb/ParticleGAN-forge/three-theme-repairs-20261022/queue/events.jsonl
```

Each task executes through public `GANTrainer` and the existing Forge adapters
and graders. Actual-training GIFs are rendered from the saved observations with
no additional training or sampling draws. Only compact results, provenance,
readouts, reproduction sources and selected media enter Git. Reporting must
preserve all earlier selected measurements and qualifications.

## Software and publication verification

The executed scientific source is
`2a2b662bdfa6249076b27dbea4c136e23d3e7a80`, manifest
`8bfcc01cd7482b1b348edb2162567adfdcdaa7d308f75fc834774302c6ac4acb`.
The publisher verifies all 1,289 retained source files and all 60 certified
receipts. Four concluded research-diagnostic records retain the complete failed
revisions in Forge memory; none grants qualification or default adoption.

Before execution, 208 numerical/software controls passed. Additional checks
pass for actual-develop checkpoint compatibility (63), the metadata adapter
(3), generated family reporting (77) and the initializer registry (3). The
original archived comparator retains its exact bytes and assertions; a scoped
adapter checks the new inactive metadata defaults in both interpreters.

After all 60 jobs finished, a separate API guard rejected
`displacement + strict_progress`: interpolating an accepted nonconvex step can
invalidate its protected finite-step guarantee. The measured recipe uses
`direction_blend` throughout and is unaffected. The post-execution settling,
consumer and adapter checks pass 29/29. This software guard is not substituted
for the frozen scientific source and does not trigger an unchanged rerun.

The [telemetry limit](settling.md#execution-telemetry-limit) records that applied
thermal-movement histories were not retained. Optional optimizer post-hooks
would observe proposals before interpolation; no claim of a measured cooled
movement-ratio history is made. The retained tensor comparisons support the
narrower conclusions above.

See [reproduction and artifact verification](REPRODUCE.md) for the exact frozen
commands, log paths and zero-training publication workflow. Bulk execution
streams remain in the artifact archive; this report commits final metrics,
compact provenance, reproduction sources and actual-training media.
