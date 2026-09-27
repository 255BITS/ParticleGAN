# KA2 through the public trainer, with constant learning rates

**KA2 is not ready to become the release default on this evidence.** Constant
rates adapt quickly but lose stability. The decay diagnostic preserves the
original distribution and adapts more slowly. Neither run demonstrates
automatic decay followed by automatic restoration of learning rates.

The accepted requirement is now **constant rates or automatic, reversible
decay**: the learner may reduce rates when settled and increase them again
when further learning is needed. A hand-timed or permanently decayed schedule
does not establish that capability. The PR remains a draft targeting `develop`.

This experiment checks the proposed default through `get_recipe()` and the
unmodified `GANTrainer.step()`. The question is whether KA2 can retain the
original distribution, reach a changed distribution, and remain stable with
constant learning rates. Recovery is measured by arrival time and stability
afterward; there is no fixed 81/81 requirement.

## Results

The target shifts after update 2,400; both runs finish at 4,600. An observation
passes when all eight modes are present and at least 90% of samples are within
the evaluator's target radius. Fractions below describe observations, not a
new all-checks release gate.

| Public KA2 run | Pre-shift retention | First arrival after shift | Passing checks from first arrival | Final uninterrupted run |
|---|---:|---:|---:|---|
| Constant rates | 61/120 | 120 updates | 126/209 | 4460–4600: 15 checks |
| Decay diagnostic | 120/120 | 1,690 updates | 48/52 | 4380–4600: 23 checks |

Constant KA2 first learns the original distribution at 580 and passes every
sampled observation through 1740, then collapses by 1750. It fails continuously
from 1750 through 2320 and again at 2380. After shifting, it first reaches the
new target at 2520, but has 83 failing observations after that arrival. Later
departures include 3620–3940 and 4300–4450. Its final HQ of 98.61% hides these
failures if reported alone. All three learning rates were verified unchanged
on every one of the 4,600 updates.

Only after this failure was the permitted decay diagnostic launched. It
retains the original target throughout all 120 observations, reaches the new
target at 4090 and fails four observations at 4340–4370. Final HQ is 91.75%.
This shows a retention/adaptation tradeoff under the tested schedules; it does
not prove that every constant-rate setting fails or that decay alone solves
continuous learning. Both frozen controls pass zero of 220 post-shift checks.

The pair has identical initial model, optimizer and RNG-state hashes, source
files, noise timings and model/data settings. Only `lr_floor` (1 → .05) and
`network_lr_floor` (1 → .01) differ. Both use PyTorch 2.13.0+cu126 on an RTX A6000.

For context, the archived public K3P run retains 120/120 observations and has
not reached the shifted target by 3600 (final HQ 89.09%). At that same endpoint,
constant KA2 has arrived but already failed retention; decayed KA2 has not
arrived (HQ 76.20%). K3P used a different runtime and prior-decay horizon, so
these historical numbers cannot establish which formulation wins a controlled
comparison. No K3P rerun or seed variants were launched.

## Release decision and next verification

The existing `NetworkLRTransition` only allows a custom loop's caller to start
one-way decay. `GANTrainer` uses the recipe's timed schedule. KA2's internal
controller adjusts critic memory and anchor strength; it does not choose LRs.
Its surprise rises after both a changed target and self-induced collapse, so
raising LR on that signal alone is not a validated solution.

Before release, a proposed automatic policy must demonstrate when it lowers
rates, when it raises them again, and retention/arrival/stability afterward.
The policy must operate without target-shift notifications or benchmark quality
scores. Test a stationary continuation plus delayed and repeated target changes
with the same existing seed; compare against K3P on the same runtime/protocol.
Checkpoint continuation must preserve the policy state. These are remaining
checks, not completed results. No new adaptive algorithm was added in this run.

## Protocol

- One existing seed (0), with no seed variants or parameter search.
- Public defaults: 20,000 particles, latent dimension 2, batch size 2,048.
- Existing public ring architecture: three hidden layers of width 96 and
  critic Fourier scale 3. Networks initialize on CPU before moving to CUDA.
- Generator and critic LR 0.00425; prior LR 0.0085. Both public LR floors are
  set to 1. Every update records and checks all three applied learning rates.
- The original public shift experiment's noise timings stay fixed: input
  noise reaches zero at 360 and output noise reaches full strength at 720.
  Constant learning rates do not imply constant noise or no acquisition phase.
- The eight-mode target moves by (1, 0) after update 2,400. Training continues
  through 4,600 to expose later departures. Observations occur every 10 updates
  using 4,096 samples and an isolated evaluation stream.
- A passing observation covers all eight modes with HQ at least 0.90.
  Record pre-shift retention, first arrival, every subsequent departure and
  the final uninterrupted passing suffix. A frozen copy at the shift checks
  whether progress requires further training.
- Run constant KA2 first. Run the optional decay comparison only after the
  constant run demonstrates a failure. If needed, decay uses identical model
  initialization, data, noise and training budget, changing only the two LR
  floors to their stock values (network 0.01, prior 0.05). Its prior decay spans
  4,600 updates; it is not a replay of the older 3,600-update decay schedule.

This tests 120 sampled pre-shift retention observations and a single target
change. It does not substitute for the separate 7,500-step stationary protocol
or the full 22-task suite. The archived public K3P baseline used a different
PyTorch version, so its scores provide context rather than a controlled
formulation comparison.

## Reproduction

From the repository root, use a fresh output directory:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python reports/ka2-default-candidate/constant-lr-api/worker.py \
  --output /tmp/ka2-constant-api --device cuda:0 > /tmp/ka2-constant-api.log 2>&1
tail -f /tmp/ka2-constant-api.log
```

The worker snapshots its sources before training and saves full checkpoints,
initial state receipts, per-update learning rates, evaluation metrics and
controller diagnostics. Logs flush as observations arrive. Raw results use
`COMPLETE` to mean execution finished, independently of the quality assessment.

The retained `evidence/constant` and `evidence/decay` directories contain
lossless compressed metrics and LR histories, resolved recipes, initial receipts,
final summaries, periodic state hashes and source snapshots. `manifest.json`
pins their hashes and lists full checkpoints retained outside Git. Verify the
retained evidence without PyTorch or training:

```bash
python reports/ka2-default-candidate/constant-lr-api/verify_evidence.py
```
