# Three BISPF repair hypotheses: matched diagnostic comparison

This study tests three global trainer changes against the selected
`bispf-optimized-quintic-v1` recipe on one common committed source. The user
requested separate agents for output allocation, settling and local fidelity,
followed by a comparison. Prior investigations are PRs #463–473. The
[current technique inventory](../technique-inventory.md) remains the single
goal leaderboard; this is a bounded research comparison, not qualification.

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
run must continue its own valid passing smoke checkpoint; a missing parent stays
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

Status: implementations and numerical software controls prepared; scientific
measurements pending. No hypothesis is yet established as a repair.
