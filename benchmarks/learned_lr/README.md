# Learned learning-rate adapter research

This experiment trains a small causal LR controller using ParticleGAN training
runs as its outer-loop objective. It is an optional research module; it does not
change `Recipe`, `GANTrainer`, or any default.

The policy is a two-output linear network in log-LR space, with one six-weight
row for G/prior and another for D. Every 20 completed updates it observes:

| Input | Meaning |
| --- | --- |
| Bias | Constant 1 |
| Progress | Completed updates / training budget |
| Gradient ratio | Log gradient RMS / first-observation gradient RMS |
| Gradient alignment | Cosine similarity to the previous observed gradient |
| Adam direction ratio | Log proposed Adam direction RMS / initial direction RMS |
| Parameter ratio | Log parameter RMS / initial parameter RMS |

Ratios are clipped to `[-2, 2]`. The weighted sum predicts a log-LR multiplier,
clipped to `[log(0.05), log(2)]` and smoothed halfway from the preceding action.
This multiplier scales the original optimizer learning rates; parameter-group
ratios are preserved. Inputs contain no formulation names, task identities,
mode counts, distributional distances, or other evaluation metrics. Adam
directions are computed from the current gradient and existing optimizer state,
before the update. Only LR changes; penalties, loss definitions, and optimizer
betas remain fixed.

```python
import json
from benchmarks.learned_lr.controller import OptimizerLRAdapter

policy = json.load(open("policy.json"))
adapter = OptimizerLRAdapter(policy, total_steps=1200)

# Inside an existing GAN training loop, after each backward():
adapter.step(opt_d, completed_updates, role="d")
opt_d.step()
# ... generator backward() ...
adapter.step(opt_g, completed_updates, role="g")
opt_g.step()
```

This research version supports ordinary dense Adam without weight decay,
AMSGrad, or maximize. Online adapter state is not checkpointed. It measures
attributes every 20 updates, including device-to-host scalar copies; CUDA
overhead requires separate measurement. It should not yet replace a production
scheduler.

## Training and evaluation separation

The training distributions are a four-mode ring and a 3×3 Gaussian grid. They
use the existing mode-hold host networks and update mechanics: hidden width 96,
three hidden layers, critic Fourier width 3, latent width 4, 12 particles,
batch 128, 1,200 updates, seed 0, Adam `(0, .99)`, LR `.0017`, `b_cap` coefficient
3 / κ 1.25, no particle L2, and prior spread weight .05.

Outer-loop cross-entropy search fits 32 proposed policies: four generations of
eight, each evaluated on both training distributions. The three best in each
generation update the search distribution. Its RNG is separate from GAN
initialization; every GAN run uses seed 0. This is parameter search, not a seed
sweep or supervised imitation of cosine.

The objective is the mean over training tasks of half the 24-checkpoint mean
normalized sliced Wasserstein distance and half the last-quarter mean distance.
There are 16 fixed projection directions and independent fixed evaluation RNGs.
Mode/HQ gates and EMA never select a policy. The first search population includes
constant LR and a simple time-only exponential policy.

After the selected policy is frozen, evaluate it on the untouched eight-mode
ring, the same ring at half scale, and an alternate R1+R2 penalty. Compare equal
update budgets with constant LR, the existing delayed cosine, and a time-only
ablation of the learned policy that zeros all four feedback features while
keeping bias, progress, coefficients, and action smoothing. The full nine-toy
behavioral suite is also held out from fitting; its evaluator is separate.
These are architecture-sharing synthetic tasks, so successful transfer would
still be limited evidence of broader generalization.

```bash
python -u -m benchmarks.learned_lr.study --output /tmp/learned-lr \
  > /tmp/learned-lr.log 2>&1
tail -f /tmp/learned-lr.log
```

The output retains every policy, episode curve, action/feature trace, final
live/EMA metric, objective, runtime, package versions, and source hashes.
`policy.json` appears only after training selection is complete. Training and
held-out reports are separate, and completed results are never overwritten.

## Mathematical form

For each optimizer role, the six observations form `h`. With learned row `w`,
the controller uses `target = clip(w @ h, log(.05), log(2))`, then
`log_scale = .5 * log_scale + .5 * target`. Each parameter group receives
`lr = original_lr * exp(log_scale)`. This is a two-output linear network followed
by bounded, stateful smoothing; all twelve coefficients are learned from GAN
rollouts. The outer trainer uses black-box search, rather than differentiating
through 1,200 Adam updates.

The broader idea of fitting optimizer rules and checking their transfer appears
in [Learning to learn by gradient descent by gradient descent](https://arxiv.org/abs/1606.04474)
and [Learned Optimizers that Scale and Generalize](https://proceedings.mlr.press/v70/wichrowska17a.html).
This small GAN experiment uses its own policy and training procedure; those
papers do not establish the results of this implementation.
