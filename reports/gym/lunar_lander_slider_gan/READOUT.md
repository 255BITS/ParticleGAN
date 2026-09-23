# All-heads slider-error supervision did not beat L2

Replacing all paired MSE/contact-BCE losses with the sliders error game produced
**6/50 landings**, versus **11/50** for the matched previous-action L2 GAN.
The existing state-only joint GAN landed 37/50 and remains the validation-selected
GAN baseline. The separate non-GAN imitation reference landed 50/50.

This is one negative result for the all-heads adaptation, not a general verdict
on paired-error GANs. No second scope, weight sweep, or training-seed repeat was run.

## Formulation and implementation

The source is the [Anima concept-sliders paired-error game](https://huggingface.co/ntc-ai/anima-concept-sliders#the-paired-error-game).
Our graph remains:

```text
E(st, at-1, terrain) -> z -> G1 -> st
                         -> G2 -> at
                         -> G3 -> st+1

prior -> z -> G1 / G2 / G3
```

The new critic R receives either Gaussian noise or that same noise plus the
normalized error of the encoded 18-coordinate transition. The generator tries
to make these pairs indistinguishable. Real/fake noise is shared within a pair;
G and D draw independent noise streams. Contact logits become probabilities for
this residual; the existing joint/marginal critics keep binary sampled contacts.

R uses the unmodified pinned global-mix implementation: eight width 48 tokens,
one four-head attention layer, and bounded score 8*tanh(score/8). These are
coordinate mixtures, not trajectory tokens. Its exact autograd gradient cap is
evaluated every fourth update with coefficient 1, threshold 1, and lazy multiplier 4.

The source's frozen neutral teacher is replaced by a fixed training-target mean.
For normalized target y and prediction y_hat, delta_T=y-mean_train and
delta_S=y_hat-mean_train, so their difference is y_hat-y. Scale each coordinate
by training-only sample std(delta_T), floored at 1e-4, then multiply by the median
row RMS of delta_T/std. Median normalized target-edit RMS is one. There is one
normalization group rather than diffusion-position groups.

The calibrated RMS is 1.031844 and starting noise is **3.685158**. Noise uses
the source's geometric curve toward 0.03 with absolute hold 1.0, adapted to a
2,500-update horizon and index k-1. It reaches the hold during training and
stays at 1.0; it does not reach0.03 in this run.

`slider_scope: all` replaces action MSE and both state heads' continuous MSE
and contact BCE. Those losses are logged under no-grad for diagnostics only.
The optimized objective is existing joint/marginal GAN loss + paired-error GAN
loss (weight1) + one default full-table MoG prior regularizer. All four critic
networks train throughout: joint, action, shared state, and paired error.

The configurable `slider_scope: action` instead replaces only action MSE and
retains G1/G3 MSE/BCE. Both scopes passed correctness tests; only `all` was run
at the full research budget. We keep the existing MoG routing, 1,024 particles,
optimizer/EMA recipe, and prior regularizer. This adapts the sliders error game,
not its LoRA architecture or frozen diffusion model. The vendored MIT source,
license, provenance, and exact revision are under `lib/vendor/concept_slider_core/`.

## Matched comparison and GAN-only leaderboard

Both previous-action models use all 9,297 labeled expert transitions from 47
training episodes, with expert previous actions during training and learner
previous actions during rollout. Initial G/E/prior/joint/marginal parameters
and complete training minibatch-stream hashes match exactly. The scaler,
2,500 updates, batch 256, default MoG recipe, and 1.28 million real-record draws
also match. The sliders arm changes paired supervision and adds R capacity/compute.

No L2 model was retrained. Both models' saved updates 250/1000/2500 were reselected
on the same 20 fresh validation worlds 1391000–1391019. Selection uses landing
rate, mean return, then earlier update. Both select 2500. Test uses the same 50
fresh worlds 1491000–1491049, disjoint from data collection and prior evaluations.
The table is ranked by validation, and test outcomes do not choose the default.

| GAN controller | Validation landings | Test landings | Wilson95% | Mean test return | Crash / bounds / timeout |
| --- | ---: | ---: | --- | ---: | --- |
| Existing state-only joint | 15/20 | 37/50 | 60.4%–84.1% | 186.88 | 13 / 0 / 0 |
| Previous-action L2 | 2/20 | 11/50 | 12.8%–35.2% | 13.95 | 34 / 5 / 0 |
| **Previous-action sliders, all heads** | **1/20** | **6/50** | **5.6%–23.8%** | **-26.40** | **41 / 3 / 0** |

The **non-GAN imitation reference** is excluded from ranking:20/20 validation,
50/50 test, mean test return278.56. It and the state-only joint reference differ
in supervision and training history from the matched previous-action models.

Sliders wins1 paired landing outcome against L2, loses6, and ties43. It wins15/50
paired returns, with mean difference **-40.35**. The finite-world landing intervals
overlap; this single run does not characterize training-run variability. Both
previous-action recipes remain substantially below the existing joint GAN.

| Update | Sliders validation landings | Sliders mean return | L2 validation landings | L2 mean return |
| ---: | ---: | ---: | ---: | ---: |
| 250 | 0/20 | -135.70 | 0/20 | -128.11 |
| 1000 | 0/20 | -117.22 | 0/20 | -117.19 |
| 2500 | 1/20 | -25.47 | 2/20 | 15.23 |

The selected and final slider checkpoint are identical, so their test rollout
is reused. The existing joint GAN's37/50 and L2's11/50 differ from prior reports
because this round uses fresh worlds, not because those models were retrained.

## Prediction and fixed-input diagnostics

On the same 2,444 held-out expert transitions and common training scaler:

| Metric | L2 | Sliders, all heads |
| --- | ---: | ---: |
| Physical action MSE | 0.013328 | 0.021207 |
| Standardized action MSE | 0.081177 | 0.114571 |
| G1 continuous MSE | 0.026479 | 0.036829 |
| G3 continuous MSE | 0.037161 | 0.047174 |
| G3 contact Brier | 0.006275 | 0.009244 |

Sliders is worse on each metric here. Persistence continuous MSE is 0.022230,
better than either G3; persistence contact Brier is 0.008183, between the models.
These are diagnostic scores, not losses optimized by the all-heads sliders model.

Posthoc comparisons against heuristic commands on fixed saved inputs:

| Input trace | Evaluated model | Physical action MSE | Main agreement | Side agreement |
| --- | --- | ---: | ---: | ---: |
| Sliders learner | Sliders | 0.354068 | 80.6% | 35.0% |
| Same sliders inputs | L2 | 0.289661 | 83.7% | 50.4% |
| L2 learner | Sliders | 0.288700 | 84.9% | 43.1% |
| Same L2 inputs | L2 | 0.283049 | 85.0% | 40.8% |

L2 has lower physical error on both fixed input datasets, although lateral regime
agreement slightly favors sliders on the L2 trace. Each comparison holds current
state, previous command, and terrain fixed. Heuristic labels are recommendations,
not demonstrated recovery actions. These diagnostics do not isolate a causal
failure mechanism; no labels were used for training or selection. See
[action diagnostics](action_diagnostics.md).

## Training behavior, cost, and verification

At update 2500, R's paired D loss is 0.65203, paired G loss0.76088, and cap0.00209.
The ordinary joint/marginal G loss is 1.81947. These adaptive adversarial scores
are not accuracy measures, and their scales should not rank models. The logged
action MSE decreased to 0.08092 despite not being optimized directly. Full
role losses and noise values are archived in `training/metrics.jsonl`.

Training took **137.44 seconds on GPU1**, versus 99.18 seconds for the recorded
L2 run (1.39x). R adds 26,689 parameters, giving 382,614 trainable parameters.
Action inference remains 99,266 parameters and averaged 0.698 ms/step on CPU;
R and the other discriminators are not used for control. Shared-machine timings
are descriptive and exclude setup/checkpoint writes from training time.

There were zero simulator calls during training. The12 unique scored evaluation
sets contain 360 episodes, 54,325 explicit steps, and 54,685 including resets.
Offline action/prediction diagnostics add zero simulator calls. These counts
exclude correctness checks; there were no training-seed repeats or extra full runs.

The suite passed **356 tests and 27 subtests**, with four opt-in CUDA tests
skipped and two existing warnings. New tests verify training-only normalization,
paired zero-error behavior, noise indexing/hold, exact lazy-cap scaling and
attention double backward, all-heads gradient equality to the paired-error loss,
GAN gradients through all Gs/E/prior, unchanged base initialization, both config
scopes, and checkpoint inference/calibration parity. A GPU1 smoke exercised the
fourth-update cap before freezing the full run. Evaluation verifies every trained
critic, source/data/checkpoint hashes, matching base initial weights and batch
streams, and identical simulator starts across paired policies.

Keep the existing joint GAN as the baseline. The next bounded test is
**`slider_scope: action`**: replace only action L2 while restoring G1/G3 paired
supervision, retaining all GAN critics. That separates the action-loss experiment
from dropping auxiliary reconstruction. It is a recommendation, not a run or
a promise of improvement. The current result does not establish whether error
normalization, critic capacity, loss weighting, or removing auxiliary losses
caused the weaker control.

See the [guide](../../../docs/gym-slider-gan.md) and
[frozen plan](../../../docs/gym-slider-gan-plan.md). Artifacts include
`protocol.json`, `leaderboard.json`, `selection.json`, `evaluations/`, `traces/`,
`diagnostics.json`, `action_diagnostics.json`, and `training/`. Checkpoints remain
under ignored `results/gym/lunar_lander_slider_gan/sliders_all/`.
