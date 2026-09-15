# Broader geometry helps; temporal D trades validity for more variation

All eight runs completed: four 1k viability scouts and four 10k comparisons,
using both RTX A6000 GPUs, one worker per GPU. Total training: **8.19 GPU
minutes** across the eight runs. No failures or startup warnings. Both GPUs
are free; nothing further is queued. Branch: `experiment/trajectory-diversity`.
Core DDGAN formulation and default model/data selections remain unchanged.

[Interactive comparison](confirm_10k/index.html) ·
[Best SW1 animation](confirm_10k/mlp_continuous/futures.gif) ·
[Machine-readable results](confirm_10k/leaderboard.json) ·
[Design and commands](PLAN.md)

## Matched 10k results

Each run: batch128, 1.28M examples per optimizer phase / 2.56M real draws total,
constant LR, same evaluation samples and source fingerprint. SW1 is lower-better;
variance ratio targets 1 and uses valid generated paths only. All interpolation
variance groups are present except temporal/discrete (only 5 of 12).

| D / training geometry | Test SW1 ↓ | Route TV ↓ | Interp. valid ↑ | Extrap. valid ↑ | Test collision ↓ | Interp. variance ratio | Training s |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP / continuous | **.0390** | .0655 | **99.8%** | **80.3%** | 1.4% | .35 | 99.0 |
| Temporal / continuous | .0418 | .0588 | 99.0% | 53.5% | **0.5%** | **.72** | 119.9 |
| MLP / discrete control | .0468 | **.0458** | 99.1% | 1.8% | 10.2% | .31 | 100.9 |
| Temporal / discrete | .0963 | .0495 | 18.0% | 0.0% | 36.5% | 1.08* | 124.7 |

*Temporal/discrete's ratio describes a small surviving subset, not successful
diversity. Only 5/12 interpolation route groups have enough valid samples;
training-reference validity is 1.4%, with only 1/16 variance groups present.
Reject this configuration despite its near-1 ratio and reasonable route TV.

Real-vs-real floor: SW1 .0070, route TV .0156, validity100%, collisions0%.
The test set has three interpolation geometries and one extrapolation geometry,
each with two preferences. These percentages do not establish broad OOD ability.

## What the comparison tells us

**Training coverage matters.** With the same MLP, continuous geometry reduces
pooled SW1 by 16.6%, reduces collisions, and improves unseen-scene validity.
It recovers valid mass in all 16 test context/route pairs versus 12/16 for the
current control. But route probability calibration worsens and variance remains
near .35: broader geometry alone does not solve missing within-route variation.

Continuous geometry samples start height [-.2,.2], obstacle height [-.12,.12],
radius [.22,.32] independently. These are the old coordinate bounds, not the old
four-point joint distribution or its exact convex hull. The extrapolation scene
(.3,.18,.35) remains outside every training coordinate range. Interpolation is
within-distribution prediction at previously unsampled exact contexts, not OOD.
The `train` evaluation split remains the original four reference geometries;
it does not measure an average over the new continuous training distribution.

**D architecture affects variation, but is not a universal fix.** On continuous
geometry, temporal D increases the median within-route variance ratio from
.35 to .72, while retaining 99.0% interpolation validity. All 12 variance groups
are represented. It also reduces collisions and improves pooled route TV. These
are useful gains, but pooled SW1 and extrapolation validity are worse than MLP.
Training-reference validity is 86.3% versus MLP99.5%, showing trouble at some
reference geometries. Training costs about 21% more despite only 2.2% more D
parameters. The model has more spread, not demonstrably correct full densities.

Temporal D on the four discrete geometries fails badly at 10k. It was the best
pooled SW1 scout at 1k; the early rank again did not predict the 10k result.
This factorial interaction means we cannot call temporal D generally better or
infer the exact reason for its failure. No intermediate checkpoint selection,
extra training losses, schedule changes, or hyperparameter search were used.

The split metrics also disagree in useful ways: temporal/continuous has slightly
better extrapolation SW1 (.0653 vs .0677), yet lower thresholded validity
(53.5% vs80.3%). Interpolation SW1 favors MLP (.0294 vs .0340). A single pooled
number or collision rate is not enough to choose the model.

## Reproducibility caveat

The prior round's default had SW1 .0466 and extrapolation validity21.4%; this
round's control has .0468 and1.8%. Same seed, batch, model and optimizer settings,
GPU1, PyTorch2.14.0+cu130. New config fields select the original behavior.
A direct old-source/current-source CPU audit found bitwise equal G/D initial
state, sampled batches, G outputs and D logits. No default formulation change
was found. GPU training does not enforce deterministic execution; numerical
trajectory divergence is plausible, but its exact cause was not isolated.
Do not claim repeatability or statistical significance from these runs.
The continuous MLP beats both observed controls on SW1/extrapolation validity,
but the magnitude of that validity improvement needs this caveat. No seed-only
repeats were run, in accordance with repository instructions.

## Recommendation

Use **MLP + continuous geometry as the next experimental baseline** for overall
path fidelity/generalization. Keep temporal + continuous as a diversity-focused
candidate. Do not promote temporal D universally or describe the toy as solved.
No no-argument defaults have been changed in this branch.

A focused next architecture test would combine the MLP's full-path features
with a modest temporal feature branch, on continuous geometry, at a comparable
parameter budget. That tests whether preserving global shape information can
retain the spread improvement. It is an idea, not a queued experiment. Keep
Particle DDGAN/UCD/Gaussian noise, exact bcap, particles and constant LR.

## Implementation and verification

- `geometry_mode: discrete|continuous`, `d_architecture: mlp|temporal`, and
  `d_temporal_width` are YAML settings. Eight full configs are in
  `configs/trajectory/diversity/{scout_1k,confirm_10k}`. All unrelated settings
  match. Default remains discrete/MLP.
- G remains102,402 parameters; particles640,000. D is204,040 for MLP and208,456
  for temporal. Temporal D uses kernel5 convolutions, 64/128/128 channels,
  four ordered pooled bins per stage, physical position channel and the same
  continuous context. No c/t inputs to joint-UCD backbone. The gradient penalty
  differentiates original candidate coordinates through the full feature stack.
- Nine unit tests pass, including real support/bounds under continuous geometry,
  unchanged holdouts, joint-label exclusion, input gradients, and temporal-D
  double backward with nonzero finite parameter gradients. Existing sampler,
  loss and schedule modules remain unchanged.
- All eight certificates and source fingerprints verified; CUDA environments
  confirm both GPUs were used. Trainer/lib sources stayed fixed across stages.
- Analyzer additionally exports split SW1/TV, variance group counts, reference
  train variance, support/boundary errors, and route coverage. It now copies
  the best-SW1 run's GIF regardless of its name.
- Both continuous-arm plots and the failed temporal/discrete plot were visually
  inspected. Generated gallery references checked and exported GIFs decoded;
  no new browser rendering test was performed. Existing viewer code unchanged.
- Summaries/configs/provenance/certificates/logs/plots/viewers are preserved in
  this report. Raw samples, source archives and EMA checkpoints remain under
  ignored `results/trajectory/diversity/`. [All eight results](all_runs.json).

Re-export:

```sh
.venv/bin/python experiments/analyze_trajectory.py \
  --roots results/trajectory/diversity/confirm_10k \
  --out reports/trajectory/diversity/confirm_10k
```
