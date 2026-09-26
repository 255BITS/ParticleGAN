# KA2: the selected default formulation

Every `get_recipe()` model family, `Recipe()` and `GANTrainer` uses the same
KA2 formulation on this branch. Model names select the model/prior/encoder,
not a different training method. Public examples use those same factories.
The released 0.8.0 package still uses [K3P](k3p.md).

KA2 was selected for full pre-shift retention together with reaching the new
distribution and stability afterward. Arrival time and subsequent stability
are measured separately; the old 81-check deadline score is not a selection
requirement. Later dropouts and incomplete 22-toy coverage remain documented.
[Selection rationale and evidence](../reports/ka2-default-candidate/README.md).

## Critic penalty and adaptive memory

Let `d` be the input dimension, `c = reg_coeff`, `κ = reg_kappa`, and
`D̄` the critic's parameter EMA:

```text
A = mean(||∇D(real)||² / d) + mean(relu(||∇D(fake)|| / √d − κ)²)
B = mean(relu(||∇D(real)|| − κ)²) + mean(relu(||∇D(fake)|| − κ)²)
P = mean(||∇D(real) − ∇D̄(real)||² / d)
penalty = c/2 · (s·A + (1−s)·(B + W·reg_anchor_weight·P))
```

The first 799 applied penalty calls bypass the blend and use `c/2·A`; call
800 starts the fixed `s=.5` blend. This preserves the exact frozen KA2 source,
including its warmup boundary. Diagnostics report `s=.5` throughout; use
`phase='a'` versus `phase='blend'` to distinguish the two phases. The blend
no longer follows the learning rate.

After each critic Adam update, surprise is the upper median across parameter
tensors of gradient RMS divided by bias-corrected Adam second-moment RMS.
The gradient is the one used by Adam after the spike guard. Each blended
penalty call records that surprise. After 25 observations, the upper median
of the latest 24 is divided by the median of the first 24 to form `ratio`.
The anchor gate `W` switches off above 3 and back on below 1.75.

EMA tracking is adaptive even while the gate is off. Its target tracking gain
is `clamp((ratio−1)/2, 0, 1)`. The actual gain `alpha` moves toward the target
by 1/60 of the gap when rising and 1/2 when falling. On a critic step the EMA
decay is `1−alpha·(1−reg_anchor_min_decay)`, with the default minimum .90;
alpha zero freezes memory. After 60 consecutive blended calls with the gate
off and ratio above 3, the next critic step copies the current critic into
the anchor and clears the streak.

The gate reads optimizer history, not a target-shift label or training
budget. The overall learner still uses LR and noise schedules and a fixed
warmup; this is not a claim of horizon-independent training.

`reg_every=k` applies the penalty every kth critic step at k times its
coefficient. The controller counts applied penalty calls; Adam observation
and anchor updates happen on critic steps. Several penalty calls in one step
share one critic's history and advance its penalty-call counter separately.

## Shared settings

The loss is RpGAN logistic. Generator updates retain A2 sparse particle-row
damping and direct-particle response. Adam uses `(0,.999)`, G/D rates are
.00425, and the particle rate is .0085. G/D anneal over the capped 1600-step
horizon to 1%; the prior anneals over the full budget to 5%. Input/output
noise, G/prior EMA and the critic spike guard keep their existing defaults.
See the [complete recipe fields](api.md#regularization-factories).

`reg_anchor_min_decay` replaces K3P's fixed `reg_anchor_decay`; the latter is
rejected rather than silently ignored. `reg_anchor_weight=0` removes the
anchor penalty as an explicit ablation. These settings do not select a
second default formulation.

## Caller-controlled learning-rate decay

For a caller-owned loop, the caller can start G/D decay when its own validation
metric plateaus. `NetworkLRTransition` keeps G/D at full LR until marked, then
cosine-decays them over the chosen duration to `network_lr_floor`. The particle
prior retains its normal full-budget schedule. KA2's blend remains fixed after its warmup; this changes the learning
rates, not the penalty's blend rule. The caller owns the plateau rule and
must save the transition state with its
optimizer checkpoint:

```python
from particlegan import NetworkLRTransition, scale_learning_rates

transition = NetworkLRTransition(decay_steps=40_000)
# After validation at 120,000 completed updates meets a declared plateau rule:
transition.mark_plateau(120_000)
# Before the next optimizer update, using the number of completed updates:
network_scale, prior_scale = scale_learning_rates(
    120_000, recipe, (opt_g, opt_d), base_lrs, prior,
    network_transition=transition)
checkpoint["network_transition"] = transition.state_dict()
# On resume, recreate the same transition duration and restore its state:
transition.load_state_dict(checkpoint["network_transition"])
```

Passing no transition retains the recipe's fixed network horizon. The
caller-marked step remains fixed after marking; a different step raises an
error. This API does not inspect validation data or choose checkpoints.

## API and checkpoints

```python
import copy
from particlegan import get_recipe

recipe = get_recipe()
opt_g, opt_d = recipe.make_optimizers(G, D, prior, ema_critic=copy.deepcopy(D))
penalty = recipe.make_critic_penalty(opt_d)
d_loss = adv_d + penalty(D, real, fake.detach())
opt_d.zero_grad(); d_loss.backward(); opt_d.step()
opt_g.zero_grad(); g_loss.backward(); opt_g.step()
```

Each critic optimizer owns its controller, guard and EMA. Extra critics use
`recipe.make_critic_optimizer`; conditioning, tuple outputs, critic submodules
and noise wrappers use the same [public API](api.md#regularization-factories).
There are no process-wide optimizer hooks or shared mutable controller state.
Anchor forwards preserve the live and EMA buffers; floating EMA buffers
follow the adaptive decay and integer buffers are copied.

Save both optimizer `state_dict()` values along with models and RNGs. The
critic optimizer saves surprise history, baseline, gate, alpha, reseed streak,
counters and EMA, so a checkpoint does not restart adaptation. `GANTrainer`
uses schema 4 and validates before mutating the live trainer. Older trainer
schemas and K3P optimizer checkpoints are rejected; resume those with the
release that wrote them, or start a fresh KA2 run.

Frozen-source parity and API tests establish implementation correctness.
They do not establish performance on unmeasured hosts or replace the missing
release qualification.
