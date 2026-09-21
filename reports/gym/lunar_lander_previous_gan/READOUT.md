# Previous actions plus GAN training did not recover imitation performance

The new scratch three-generator GAN landed **7/50** fresh test worlds. The
original previous-action imitation fine-tune landed **50/50** on those same
worlds. Keeping previous actions and action MSE while jointly training all heads
adversarially did not reproduce the successful controller under this recipe.

The existing state-only joint GAN remains the validation-selected GAN baseline.
The playable default stays unchanged. No further training or weight sweep was run.

## What was trained

```text
prior -> z -> G1 -> st
           -> G2 -> at
           -> G3 -> st+1

E(st, at-1, terrain) -> z -> G1 / G2 / G3
playback: E(st, at-1, terrain) -> z -> G2 -> at -> actual simulator
```

One previous-action encoder supplies all three independent generators. GAN losses
reach E through its decoded triples as well as the prior-generated path. This
differs from the historical joint fine-tune, whose separate control encoder
received only action MSE while prior/paired-encoder paths received GAN feedback.

All Gs, E, D, and MoG particles start from scratch. Training uses all 9,297
transitions and action labels from 47 expert training episodes. Previous actions
are expert commands aligned within each episode before shuffling, beginning with
engines-off `[-1, 0]`. During control, the learner feeds back its own commands.
Current actions and successors are targets, never encoder inputs. State and
action scalers are fit on these training records; no pretrained scaler is used.

Every update includes joint D, action D, and one shared state D with a
current/successor role indicator. Each role averages prior and encoded fake
paths. Generator loss is joint GAN + mean of three marginal GAN roles, action
MSE, G1/G3 continuous MSE + contact BCE, and one full-table prior regularizer.
All head/adversarial weights are 1. No detached probes, synthetic cycle, second
encoder, missing-action masks, reward loss, or simulator training calls are used.

MoG has 1,024 particles, z32, width128 Gs/E, bounded offsets, straight-through
routing, relativistic GAN loss, bcap, neural LR 0.0006, prior LR 100x, cosine
decay after 60%, and EMA 0.995. One 2,500-update run used batch256 on GPU1.
The [frozen plan](../../../docs/gym-previous-gan-plan.md) and `protocol.json`
pin the exact configuration, sources, data, and evaluation references.

## Fresh paired GAN-only leaderboard

Validation worlds are 1191000–1191019; test worlds are 1291000–1291049, disjoint
from data collection and earlier evaluations. Checkpoint selection uses validation
landing rate, mean return, then earlier update. Test outcomes do not choose the
checkpoint or default. The table is ordered by validation, not test performance.

| GAN controller | Labeled episodes | Validation landings | Test landings | Wilson 95% | Mean test return |
| --- | ---: | ---: | ---: | --- | ---: |
| Existing state-only joint | 5 | 13/20 | 27/50 | 40.4%–67.0% | 130.67 |
| Existing state-only joint + marginals | 5 | 7/20 | 14/50 | 17.5%–41.7% | -4.12 |
| Legacy pretrained joint + marginals | 47 | 6/20 | 28/50 | 42.3%–68.8% | 159.83 |
| **New previous-action joint + marginals** | **47** | **4/20** | **7/50** | **7.0%–26.2%** | **-4.12** |

The non-GAN imitation reference is excluded from this ranking: **20/20 validation,
50/50 test, mean test return 285.92**. It was initialized from the world-model
GAN and then fine-tuned only with action MSE through E_control/G2. Its frozen
G1/G3/prior and separate encoder history differ from the new scratch model.

The new GAN has 39 crashes, 4 out-of-bounds failures, and no timeouts. Against
the existing joint GAN it wins 2 paired landing outcomes and loses 22; it wins
14/50 paired returns, with mean difference -134.79. Against the imitation
reference it loses 43 landing outcomes and ties 7, with mean return difference
-290.04. The two marginal models' mean returns both round to -4.12, but they
have different outcomes; the new model's mean is 0.003 lower.

These references differ in labels, encoders, initialization, losses, and record
draws. This is a benchmark of the requested recipe, not an isolated test of
previous-action inputs, marginal critics, or GAN versus MSE. Confidence intervals
describe finite evaluation worlds, not training-seed or label-subset variation.

| New-model update | Validation landings | Mean validation return |
| ---: | ---: | ---: |
| 250 | 1/20 | -93.80 |
| 1000 | 1/20 | -103.75 |
| 2500 | 4/20 | 35.05 |

The selected checkpoint is the final update, 2,500; its test rollout is reused
for the final score. Earlier saved checkpoints did not do better on validation.

## Predictions and action disagreement

On 2,444 held-out expert transitions, the new model has physical action MSE
0.013328, standardized action MSE 0.081177, main-engine agreement 98.8%, and
lateral-engine agreement 90.8%. The imitation reference's physical MSE is
0.006348. Both receive expert previous commands for this diagnostic.

G1 continuous standardized MSE is 0.026479; G3 is 0.037161. Persistence (copy
the current continuous state) scores 0.022230 and remains better on that metric.
G3 contact Brier is 0.006275 versus persistence 0.008183. Prediction metrics do
not establish a good policy or alternative-action simulator dynamics.

Posthoc heuristic labels on the saved learner traces reveal a larger gap:

| Fixed input dataset | Evaluated model | Physical action MSE | Main agreement | Side agreement |
| --- | --- | ---: | ---: | ---: |
| New GAN's visited states/previous commands | New GAN | 0.332475 | 81.2% | 33.7% |
| Same new-GAN inputs | Imitation | 0.139397 | 91.3% | 86.9% |
| Imitation's visited states/previous commands | New GAN | 0.065689 | 89.7% | 76.7% |
| Same imitation inputs | Imitation | 0.047833 | 90.0% | 82.5% |

The imitation model better matches heuristic commands on both fixed datasets.
Within each dataset, states, previous commands, and terrain are identical across
models, so the gap is not solely an artifact of comparing different visited
states. Between expert and learner datasets, both states and previous commands
change; this does not isolate previous-action feedback as the cause. Heuristic
labels are recommendations, not demonstrated successful recovery. No diagnostic
labels were used in training, normalization, or selection. The
[action diagnostic](action_diagnostics.md) makes zero simulator calls.

## Cost, verification, and recommendation

Training took **99.18 seconds on GPU1**, with 355,925 trainable parameters:
G 68,754; E 44,096; prior 32,768; D 210,307. Action inference uses 99,266
parameters and averaged 0.678 ms/step on CPU. Shared-machine timings are
descriptive. Training timing includes optimization/logging/batch hashing and
excludes setup/checkpoint writes. Two batch streams draw 640,000 records each,
1.28 million draws total; training calls the simulator zero times.

The full suite passed **353 tests and 27 subtests**, with four opt-in CUDA
integration tests skipped and two existing warnings. Focused tests verify
previous-action alignment, input exclusion of current targets, nonzero previous-
action gradients, GAN gradients through every G/prior and encoded E, D-only
gradient isolation, and saved inference parity. A three-update GPU1 smoke passed
before the frozen full run. Checkpoint evaluation verifies trained finite D
weights, all component changes, exact sources/data/configuration, and hashes.

The 12 unique evaluation sets contain 390 episodes, 68,266 explicit simulator
steps, and 68,656 including resets. Selected/final aliasing adds no rollouts.
Offline diagnostics add no simulator resets or steps; their own-action replay
checks match the saved actions. No training-seed repeats or post-result sweeps
were run.

Keep the current joint GAN as the GAN baseline. A useful next controlled test
would increase **action MSE weight from 1 to 10** in this same scratch
previous-action recipe, leaving joint/marginal adversarial training and the
state losses active. That tests whether stronger action supervision protects
control when the shared representation also serves generative objectives.
It is a proposal, not a demonstrated remedy, and has not been run. The present
result does not isolate which objective or initialization difference caused the
gap to imitation.

Artifacts: `leaderboard.json`, `selection.json`, `evaluations/`, `traces/`,
`diagnostics.json`, `action_diagnostics.json`, and `training/`. The selected
checkpoint remains at
`results/gym/lunar_lander_previous_gan/previous_marginals/best.pt`.
See the [guide](../../../docs/gym-previous-gan.md) for reproduction and log paths.
