# Circle tracing with a transition encoder

**Collaborator start here.** This is a fresh circle experiment on
`experiment/circle-transition`, based on `particle-finetune/base` at `2bb99b3`
(the merge of [PR #18](https://github.com/255BITS/ParticleGAN/pull/18)).
The explicit-memory requirement and previous memory runs are dropped from this
experiment. There are no inherited circle scores, checkpoints or winning configs.

**Status:** sampler, trainer, and frozen evaluator are implemented. The first
learned arm is paired-error RpGAN at `adv_weight=1` on `E_control`+G2. Scores
live on the [circle leaderboard](../reports/circle-transition/README.md).

## Model and runtime

Use the existing [transition-encoder design](transition-gan-encoder.md) and
[PR #18's controller objective](gym-particle-finetune.md) as the starting points:

```text
current observed state + task context -> E_control -> z -> G2 -> action
action -> environment step -> next observed state -> E_control -> ...

optional transition heads from z:
G1 -> reconstructed current state
G3 -> predicted next state
```

No GRU, learned memory writer, persistent latent, real starting prefix or memory
handoff is required. Recompute z from the current observation every step. Temporal
continuity comes from the evolving environment state fed back through the encoder.
This can provide implicit state through feedback; a feedforward encoder does not
recover arbitrary hidden history absent from its input.

Make the first task fully observed: supply current 2D position, circle center,
target radius and signed angular step as state/context. Signed speed identifies
both rate and direction; a position alone cannot distinguish two circles or
clockwise from counterclockwise travel. Do not feed target next position, current
expert action, future observations or an absolute phase clock into E_control.
No previous action is necessary for this first state contract.

Start with the action meaning 2D displacement. The environment applies
`position_next = position + action`, then returns that measured position.
G3's prediction is evaluated separately and never substituted for the environment
observation. Using G3 directly for playback would be a different experiment.

## Independent local training data

Sample circle geometry, signed speed and phase independently for each training
row. Build `(state, context, expert_action, next_state)` directly from those
samples, and shuffle rows. Do not execute trajectories to collect training data,
generate replay data or compute losses. No temporal unroll or backpropagation
through a generated trajectory is needed.

A concrete first sampler can use centers in [-0.75, 0.75]^2, radii in [0.6, 1.4],
and angular-step magnitudes in [0.12, 0.40], with balanced directions. Sample
phase uniformly over a full turn. These are proposed task settings, not inherited
results or a requirement to reproduce an earlier study.

Include independent off-circle local states so the model can learn recovery
from its own small errors without training on its own trajectories. For example,
let q be position relative to the center, in target-radius units; sample its
radius rho in [0.8, 1.2], and define the expert successor by rotating q through
the signed angular step and moving rho a quarter of the way toward 1:

```text
rho_next = rho + 0.25 * (1 - rho)
q_next = rotation(angular_step) * (q / rho) * rho_next
expert_action = target_radius * (q_next - q)
```

The analytic rule supplies training labels and an evaluation oracle. It must not
be hard-coded into the learned policy's output or used to snap playback onto a
circle. This is a controlled representation/control toy with known dynamics.

## Objectives to transfer

The transition guide and PR #18 describe different stages, not one identical loss:

1. The transition encoder learns a joint representation with independent G1/G2/G3
   heads, adversarial objectives, and paired reconstruction/prediction supervision.
   If adapting that pretraining, E_pair may see the current action; E_control
   must only receive information available before choosing it.
2. PR #18 fine-tunes E_control and G2 using **paired-error RpGAN at weight 1**,
   freezing G1/G3, E_pair, the prior and transition D. It does not use an action
   MSE term in that controller update. Keep that scope for a faithful first port;
   training all heads from scratch would be a separately named mechanism.

For the paired-error critic, `real = noise` and
`fake = noise + (predicted_action - target_action) / scale`. Fit the scale from
`target - frozen_initial_action`, with the existing paired-edit normalization.
Use the existing Rp logistic objectives and sample-point b_cap every fourth
update, including its factor-of-four compensation. Keep `adv_weight=1`; log
actual adversarial gradients and cap applications. Diagnostic MSE is allowed
and should stay outside this controller loss.

PR #18 is a useful implementation reference, not evidence that this circle task
already works. Its 2D gate adjusts a scalar action sign around a supplied expert
law; the circle model must learn its action from observations with real networks.
Its Lunar Lander scores are not circle leaderboard entries.

## Implementation starting points

| Purpose | Existing source |
|---|---|
| Particle-routed transition encoder and independent heads | [lib/transition.py](../lib/transition.py) |
| Transition training and checkpointing | [experiments/train_transition.py](../experiments/train_transition.py) |
| Paired-error objective, normalization, cap and controller scope | [lib/gym_particle_finetune.py](../lib/gym_particle_finetune.py) |
| Standalone paired-error gate | [examples/yue2_particle_2d.py](../examples/yue2_particle_2d.py) |
| Gate implementation and its limits | [lib/yue2_particle_toy.py](../lib/yue2_particle_toy.py) |

The gym-specific tensor dimensions and pretrained checkpoints are not a circle
API. Adapt the reusable components; a fresh clone should not need Lunar Lander
data or old circle checkpoints to run the new toy.

```bash
git clone --branch experiment/circle-transition https://github.com/255BITS/ParticleGAN.git ParticleGAN-circle-transition
cd ParticleGAN-circle-transition
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[experiments,dev]'

# Existing PR #18 CPU gate; this is not circle training.
.venv/bin/python -u examples/yue2_particle_2d.py

# Circle toy. Fresh directory, flushed log, paired-error controller at adv_weight 1.
.venv/bin/python -u examples/circle_transition.py
# equivalent:
.venv/bin/python -u experiments/train_circle_transition.py --config configs/circle/paired_error.yaml
```

Run artifacts go under `results/circle_transition/<run>/`. The stable log is
`results/circle_transition/live.log` (the run directory links the same file):

```bash
tail -F results/circle_transition/live.log
```

Protocol v1 partitions circle parameters by a 5×5×4×4 cell hash of center,
radius, and speed magnitude (buckets 0–13 train, 14–16 validation, 17–19 test).
Direction is balanced inside each split and is not a separate held-out axis.
The main panel starts at normalized radius 1. The recovery panel starts at 0.8
and 1.2; its extra radial readout uses a 64-step recovery window. Positive
angular step is counterclockwise. Full-trace success thresholds are unchanged:
radial RMSE below 0.1, mean absolute signed-step error below 0.03 rad/step,
direction agreement above 95%, and at least one requested turn.

## First experiment and acceptance

Implement and verify the independent sampler and frozen evaluator first. Check
that the analytic expert passes the circle protocol, a zero-action policy fails
progress, and reversed motion fails the requested direction. Verify that shuffling
training rows preserves aligned state/action/target tuples, and that the encoder
has no hidden state carried between rows or rollout episodes.

Then port the transition-encoder plus paired-error controller recipe for one
fresh learned baseline. Evaluate full trajectories **only with frozen weights
after training**. Compare local action/prediction quality and actual closed-loop
circle fidelity separately. Fix the benchmark before selecting variants; no
seed-only repeats. Any later comparison should change one mechanism and report
the training budget and compute cost.

Publish metrics, a leaderboard, interpretation and a next recommendation after
completion. Do not import the old memory scores, warm-prefix protocol or saved
winner as controls. No explicit-memory investigation is required to begin.
